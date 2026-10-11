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
from common.metrics import safe_corr
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo", run_name: str = "standard"):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    out = repo_path(output or cfg["output"])
    out.mkdir(parents=True, exist_ok=True)

    policy = bundle["policy"]
    reward_model = bundle["reward_model"]
    reward_tokenizer = bundle["reward_tokenizer"]
    tokenizer = bundle["tokenizer"]
    prompts_data = bundle["prompt_rows"]
    optimizer = bundle["optimizer"]

    num_updates = int(cfg.get("updates", 20))
    prompts_per_update = int(cfg.get("prompts_per_update", 1))
    num_generations = int(cfg.get("num_generations", 4))
    policy_epochs = int(cfg.get("policy_epochs", 1))
    eps = float(cfg.get("clip_epsilon", 0.20))
    beta = float(cfg.get("kl_beta", 0.10))
    max_prompt_len = int(cfg.get("max_prompt_length", 256))
    max_comp_len = int(cfg.get("max_completion_length", 512))
    mask_truncated = bool(cfg.get("mask_truncated_completions", True))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))

    gen_cfg = cfg.get("generation", {})
    temperature = float(gen_cfg.get("temperature", 0.7))
    top_p = float(gen_cfg.get("top_p", 0.9))

    device = next(policy.parameters()).device

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    timer = wall_timer()

    print("=" * 65)
    print(f"STARTING GRPO CONTINUATION: run_name={run_name}, loss_type={loss_type}")
    print(f"Updates: {num_updates} | K: {num_generations} | Epsilon: {eps} | Beta KL: {beta}")
    print("=" * 65)

    history: list[dict] = []

    for u in range(num_updates):
        # 1. Select prompts and expand K completions per prompt
        batch_prompts = []
        group_ids_list = []
        for i in range(prompts_per_update):
            idx = (u * prompts_per_update + i) % len(prompts_data)
            p_msg = prompt_messages(prompts_data[idx])
            for _ in range(num_generations):
                batch_prompts.append(p_msg)
                group_ids_list.append(i)

        group_ids = torch.tensor(group_ids_list, device=device, dtype=torch.long)

        # 2. Rollout generation (policy in eval mode)
        policy.eval()
        gen_out = batch_generate(
            model=policy,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            max_prompt_length=max_prompt_len,
            max_new_tokens=max_comp_len,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

        # 3. Mask truncated sequences if configured
        token_mask = gen_out["response_mask"]
        if mask_truncated:
            token_mask = mask_truncated_sequences(token_mask, gen_out["truncated"])

        # 4. Compute rollout log-probs (old) and frozen reference log-probs
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

        # 5. Score with reward model
        with torch.no_grad():
            rewards = score_reward_pairs(
                reward_model,
                reward_tokenizer,
                batch_prompts,
                gen_out["responses"],
            )

        # 6. Group informativeness diagnostics
        group_stds = []
        for gid in torch.unique(group_ids):
            g_r = rewards[group_ids == gid]
            std_g = g_r.std(unbiased=False).item() if len(g_r) > 1 else 0.0
            group_stds.append(std_g)

        mean_within_group_std = float(np.mean(group_stds)) if group_stds else 0.0
        uninformative_frac = float(np.mean([1.0 if s <= 1e-6 else 0.0 for s in group_stds])) if group_stds else 0.0

        # 7. Compute within-group relative advantages
        adv = group_relative_advantages(rewards, group_ids)

        # 8. Policy optimization
        policy.train()
        for epoch in range(policy_epochs):
            new_logp, _ = response_token_logprobs(
                policy,
                gen_out["sequences"],
                gen_out["attention_mask"],
                gen_out["prompt_width"],
                gen_out["response_ids"],
            )

            loss, loss_metrics = grpo_policy_loss(
                new_logp=new_logp,
                old_logp=old_logp.detach(),
                seq_adv=adv.detach(),
                token_mask=token_mask,
                ref_logp=ref_logp.detach(),
                eps=eps,
                beta=beta,
                loss_type=loss_type,
                max_completion_length=max_comp_len,
            )

            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm > 0:
                gn = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_grad_norm)
            else:
                gn = torch.tensor(0.0)
            optimizer.step()
            grad_norm_val = float(gn.item() if hasattr(gn, "item") else gn)

        lengths = [float(l) for l in gen_out["response_lengths"]]
        with torch.no_grad():
            seq_contrib = (torch.minimum(torch.exp(new_logp - old_logp) * adv[:, None], (1.0 + eps) * adv[:, None]) * token_mask).sum(-1)
            len_loss_corr = safe_corr(lengths, seq_contrib.cpu().numpy())

        cur_reward = float(rewards.mean().item())
        cur_length = float(np.mean(lengths))
        elapsed = timer()
        peak_vram = float(torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0

        record = {
            "update": u + 1,
            "reward": cur_reward,
            "kl": float(loss_metrics["sampled_kl"].item()),
            "within_group_reward_std": mean_within_group_std,
            "uninformative_group_fraction": uninformative_frac,
            "policy_loss": float(loss.item()),
            "gradient_norm": grad_norm_val,
            "entropy": float(loss_metrics["sample_entropy"].item()),
            "clip_fraction": float(loss_metrics["clip_fraction"].item()),
            "response_length": cur_length,
            "length_loss_correlation": len_loss_corr if not np.isnan(len_loss_corr) else 0.0,
            "wall_clock_time": elapsed,
            "peak_vram_gib": peak_vram,
        }
        history.append(record)

        print(
            f"Update {u + 1:2d}/{num_updates} | "
            f"Reward: {cur_reward:+.3f} | "
            f"KL: {record['kl']:.4f} | "
            f"Grp Std: {mean_within_group_std:.3f} | "
            f"Uninf: {uninformative_frac:.0%} | "
            f"Loss: {record['policy_loss']:.4f} | "
            f"Len: {cur_length:.0f} | "
            f"VRAM: {peak_vram:.2f}GB"
        )

    # 9. Save adapter and logs
    print(f"\nSaving GRPO policy adapter and tokenizer to {out}...")
    policy.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    results_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"train_{run_name}.json"
    save_json(out_file, {
        "run_name": run_name,
        "loss_type": loss_type,
        "updates": num_updates,
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
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()
