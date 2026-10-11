from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import (
    batch_generate,
    response_token_logprobs,
    score_reward_pairs,
)
from common.logging_utils import save_json, set_seed, wall_timer
from common.metrics import sample_entropy, sampled_kl
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from task2_ppo.ppo import (
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
    value_mse_loss,
)


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.mkdir(parents=True, exist_ok=True)

    policy = bundle["policy"]
    value_model = bundle["value_model"]
    reward_model = bundle["reward_model"]
    reward_tokenizer = bundle["reward_tokenizer"]
    tokenizer = bundle["tokenizer"]
    prompts_data = bundle["prompt_rows"]
    policy_optimizer = bundle["policy_optimizer"]
    value_optimizer = bundle["value_optimizer"]

    num_updates = int(cfg.get("updates", 20))
    prompts_per_update = int(cfg.get("prompts_per_update", 1))
    ppo_epochs = int(cfg.get("ppo_epochs", 2))
    eps = float(cfg.get("clip_epsilon", 0.20))
    beta_kl = float(cfg.get("kl_beta", 0.10))
    gamma = float(cfg.get("gamma", 1.0))
    gae_lambda = float(cfg.get("gae_lambda", 0.95))
    value_coef = float(cfg.get("value_coef", 0.50))
    missing_eos_penalty = float(cfg.get("missing_eos_penalty", 1.0))
    max_prompt_len = int(cfg.get("max_prompt_length", 256))
    max_resp_len = int(cfg.get("max_response_length", 512))
    reward_max_len = int(cfg.get("reward_max_length", 1280))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))

    gen_cfg = cfg.get("generation", {})
    temperature = float(gen_cfg.get("temperature", 0.7))
    top_p = float(gen_cfg.get("top_p", 0.9))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    timer = wall_timer()

    print("=" * 65)
    print(f"STARTING PPO CONTINUATION: run_name={run_name}")
    print(f"Updates: {num_updates} | Epsilon: {eps} | Beta KL: {beta_kl} | PPO Epochs: {ppo_epochs}")
    print("=" * 65)

    history: list[dict] = []

    for u in range(num_updates):
        batch_prompts = []
        for i in range(prompts_per_update):
            idx = (u * prompts_per_update + i) % len(prompts_data)
            batch_prompts.append(prompt_messages(prompts_data[idx]))

        policy.eval()
        gen_out = batch_generate(
            model=policy,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            max_prompt_length=max_prompt_len,
            max_new_tokens=max_resp_len,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

        with torch.no_grad():
            old_logp, _ = response_token_logprobs(
                policy,
                gen_out["sequences"],
                gen_out["attention_mask"],
                gen_out["prompt_width"],
                gen_out["response_ids"],
            )

        with reference_mode(policy):
            with torch.no_grad():
                ref_logp, _ = response_token_logprobs(
                    policy,
                    gen_out["sequences"],
                    gen_out["attention_mask"],
                    gen_out["prompt_width"],
                    gen_out["response_ids"],
                )

        # 4. Score with reward model
        with torch.no_grad():
            task_rewards = score_reward_pairs(
                reward_model,
                reward_tokenizer,
                batch_prompts,
                gen_out["responses"],
                max_length=reward_max_len,
            )
            if missing_eos_penalty > 0:
                for b, term in enumerate(gen_out["terminated_with_eos"]):
                    if not term:
                        task_rewards[b] -= missing_eos_penalty

        with torch.no_grad():
            full_values = token_values(value_model, gen_out["sequences"], gen_out["attention_mask"])
            pw = gen_out["prompt_width"]
            rw = gen_out["response_ids"].shape[1]
            values = full_values[:, pw - 1 : pw - 1 + rw]

        shaped_r = shaped_rewards(task_rewards, old_logp, ref_logp, gen_out["response_mask"], beta_kl=beta_kl)
        advantages, returns = compute_gae(shaped_r, values, gen_out["response_mask"], gamma=gamma, lam=gae_lambda)
        norm_adv = normalize_advantages(advantages, gen_out["response_mask"])

        policy.train()
        value_model.train()

        ep_pol_loss = 0.0
        ep_val_loss = 0.0
        ep_entropy = 0.0
        ep_clip_frac = 0.0
        ep_kl = 0.0
        last_pol_grad_norm = 0.0
        last_val_grad_norm = 0.0

        for ppo_ep in range(ppo_epochs):
            # Forward policy
            new_logp, _ = response_token_logprobs(
                policy,
                gen_out["sequences"],
                gen_out["attention_mask"],
                gen_out["prompt_width"],
                gen_out["response_ids"],
            )
            pol_loss, _, clip_frac = ppo_policy_loss(
                new_logp=new_logp,
                old_logp=old_logp.detach(),
                advantage=norm_adv.detach(),
                mask=gen_out["response_mask"],
                eps=eps,
            )

            policy_optimizer.zero_grad()
            pol_loss.backward()
            if max_grad_norm > 0:
                pol_gn = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_grad_norm)
            else:
                pol_gn = torch.tensor(0.0)
            policy_optimizer.step()
            last_pol_grad_norm = float(pol_gn.item() if hasattr(pol_gn, "item") else pol_gn)

            # Forward value model
            pred_full_values = token_values(value_model, gen_out["sequences"], gen_out["attention_mask"])
            pred_values = pred_full_values[:, pw - 1 : pw - 1 + rw]
            val_loss = value_mse_loss(pred_values, returns.detach(), gen_out["response_mask"])

            value_optimizer.zero_grad()
            (value_coef * val_loss).backward()
            if max_grad_norm > 0:
                val_gn = torch.nn.utils.clip_grad_norm_(trainable_parameters(value_model), max_grad_norm)
            else:
                val_gn = torch.tensor(0.0)
            value_optimizer.step()
            last_val_grad_norm = float(val_gn.item() if hasattr(val_gn, "item") else val_gn)

            with torch.no_grad():
                cur_entropy = sample_entropy(new_logp, gen_out["response_mask"])
                cur_kl = sampled_kl(new_logp, ref_logp, gen_out["response_mask"])

            ep_pol_loss += float(pol_loss.item())
            ep_val_loss += float(val_loss.item())
            ep_entropy += float(cur_entropy.item())
            ep_clip_frac += float(clip_frac.item())
            ep_kl += float(cur_kl.item())

        ep_pol_loss /= ppo_epochs
        ep_val_loss /= ppo_epochs
        ep_entropy /= ppo_epochs
        ep_clip_frac /= ppo_epochs
        ep_kl /= ppo_epochs

        cur_reward = float(task_rewards.mean().item())
        cur_length = float(np.mean(gen_out["response_lengths"]))
        elapsed = timer()
        peak_vram = float(torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0

        record = {
            "update": u + 1,
            "reward": cur_reward,
            "kl": ep_kl,
            "policy_loss": ep_pol_loss,
            "value_loss": ep_val_loss,
            "entropy": ep_entropy,
            "clip_fraction": ep_clip_frac,
            "policy_grad_norm": last_pol_grad_norm,
            "value_grad_norm": last_val_grad_norm,
            "response_length": cur_length,
            "wall_clock_time": elapsed,
            "peak_vram_gib": peak_vram,
        }
        history.append(record)

        print(
            f"Update {u + 1:2d}/{num_updates} | "
            f"Reward: {cur_reward:+.3f} | "
            f"KL: {ep_kl:.4f} | "
            f"Pol Loss: {ep_pol_loss:.4f} | "
            f"Val Loss: {ep_val_loss:.4f} | "
            f"Clip Frac: {ep_clip_frac:.3f} | "
            f"Len: {cur_length:.0f} | "
            f"VRAM: {peak_vram:.2f}GB"
        )

    # 8. Save policy adapter, value model, and training results
    print(f"\nSaving continued PPO policy adapter and tokenizer to {out}...")
    policy.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    val_out = out / "value"
    val_out.mkdir(parents=True, exist_ok=True)
    value_model.save_pretrained(str(val_out))

    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"train_{run_name}.json"
    save_json(out_file, {
        "run_name": run_name,
        "updates": num_updates,
        "clip_epsilon": eps,
        "kl_beta": beta_kl,
        "history": history,
        "total_wall_clock_time": timer(),
        "peak_vram_gib": float(torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0,
    })
    print(f"Training trajectory saved to {out_file}\n")

    return {
        "output_dir": str(out),
        "history": history,
        "total_wall_clock_time": timer(),
        "peak_vram_gib": float(torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
