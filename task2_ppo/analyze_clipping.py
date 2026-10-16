from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    prompt_messages,
    read_jsonl,
    repo_path,
)
from common.logging_utils import load_json, save_json
from common.models import clear_gpu, load_policy, load_tokenizer
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import run_evaluation
from task2_ppo.ppo import (
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
)


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def evaluate_cached_rollouts(cfg: dict, rows: list[dict], eps_values: list[float]) -> dict:
    print(f"\n[Part 1] Evaluating cached rollout batch ({len(rows)} samples)...")
    eval_prompts_map = {r["prompt_id"]: r for r in read_jsonl(cfg["paths"]["rl_prompt_eval"])}

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=False)
    device = next(policy.parameters()).device

    max_len = max(len(r["old_logprobs"]) for r in rows)
    batch_size = len(rows)

    old_logp_tensor = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)
    ref_logp_tensor = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)
    values_tensor = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)
    mask_tensor = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)
    task_rewards = torch.zeros((batch_size,), dtype=torch.float32, device=device)

    examples = []
    max_tot_len = int(cfg.get("max_prompt_length", 256)) + int(cfg.get("max_response_length", 512))

    for i, r in enumerate(rows):
        L = len(r["old_logprobs"])
        old_logp_tensor[i, :L] = r["old_logprobs"].to(device)
        ref_logp_tensor[i, :L] = r["ref_logprobs"].to(device)
        values_tensor[i, :L] = r["values"].to(device)
        mask_tensor[i, :L] = 1.0
        task_rewards[i] = float(r["effective_terminal_reward"])

        p_row = eval_prompts_map[r["prompt_id"]]
        p_msgs = prompt_messages(p_row)
        ex = encode_prompt_response(tokenizer, p_msgs, r["response"], max_tot_len)
        examples.append(ex)

    new_logp_tensor = torch.zeros_like(old_logp_tensor)
    micro_batch_size = 4

    with torch.no_grad():
        for start_idx in range(0, len(examples), micro_batch_size):
            chunk = examples[start_idx : start_idx + micro_batch_size]
            padded = pad_batch(tokenizer, chunk)
            padded = {k: v.to(device) for k, v in padded.items()}

            outputs = policy(
                input_ids=padded["input_ids"],
                attention_mask=padded["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
            logits = outputs.logits[:, :-1, :]
            labels = padded["input_ids"][:, 1:]
            resp_mask = padded["response_mask"][:, 1:]
            logp = F.log_softmax(logits.float(), dim=-1)
            tok_logp = torch.gather(logp, -1, labels.unsqueeze(-1)).squeeze(-1)

            for b in range(len(chunk)):
                idx = start_idx + b
                L = int(mask_tensor[idx].sum().item())
                row_resp_toks = tok_logp[b][resp_mask[b].bool()]
                matched_len = min(L, len(row_resp_toks))
                new_logp_tensor[idx, :matched_len] = row_resp_toks[:matched_len]

    beta_kl = float(cfg.get("kl_beta", 0.10))
    gamma = float(cfg.get("gamma", 1.0))
    gae_lambda = float(cfg.get("gae_lambda", 0.95))

    shaped_r = shaped_rewards(task_rewards, old_logp_tensor, ref_logp_tensor, mask_tensor, beta_kl=beta_kl)
    adv, ret = compute_gae(shaped_r, values_tensor, mask_tensor, gamma=gamma, lam=gae_lambda)
    norm_adv = normalize_advantages(adv, mask_tensor)

    cached_metrics = {}
    for eps in eps_values:
        loss, ratio, clip_frac = ppo_policy_loss(new_logp_tensor, old_logp_tensor, norm_adv, mask_tensor, eps=eps)
        cached_metrics[str(eps)] = {
            "epsilon": eps,
            "surrogate_loss": float(loss.item()),
            "clip_fraction": float(clip_frac.item()),
        }
        print(f"  Cached Batch eps={eps:.2f} | Surrogate Loss: {loss.item():.4f} | Clip Fraction: {clip_frac.item():.4f}")

    clear_gpu(policy)
    return cached_metrics


def run_clipping_study(config_path: str, skip_train: bool = False):
    cfg = load_yaml(config_path)
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    eps_values = [float(e) for e in cfg.get("clip_values", [0.05, 0.20, 0.50])]
    fork_updates = int(cfg.get("fork_updates", 8))
    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("TASK 2 STEP 2: CLIPPING STUDY")
    print(f"Epsilon values to test: {eps_values}")
    print(f"Fork update budget: {fork_updates} updates")
    print("=" * 65)

    cached_batch_results = evaluate_cached_rollouts(cfg, rows, eps_values)

    print(f"\n[Part 2] Running matched short forks ({fork_updates} updates each)...")
    fork_summaries = []

    for eps in eps_values:
        run_name = f"clip_{eps}"
        output_dir = f"outputs/task2_ppo/{run_name}"
        adapter_path = repo_path(output_dir)

        if not skip_train or not (adapter_path / "adapter_config.json").exists():
            print(f"\n--- Training PPO fork with eps={eps} for {fork_updates} updates ---")
            run_ppo(
                config_path=config_path,
                output=output_dir,
                updates=fork_updates,
                clip_epsilon=eps,
                run_name=run_name,
            )
            clear_gpu()
        else:
            print(f"\nFound existing fork checkpoint for eps={eps}; skipping training.")

        # Compute training stability statistics
        train_log = results_dir / f"train_{run_name}.json"
        loss_std, grad_std = 0.0, 0.0
        if train_log.exists():
            t_data = load_json(train_log)
            losses = [h["policy_loss"] for h in t_data.get("history", [])]
            g_norms = [h["policy_grad_norm"] for h in t_data.get("history", [])]
            loss_std = float(np.std(losses)) if losses else 0.0
            grad_std = float(np.std(g_norms)) if g_norms else 0.0

        # Evaluate on held-out evaluation set
        eval_metrics = run_evaluation(
            config_path=config_path,
            adapter=output_dir,
            name=run_name,
        )
        clear_gpu()

        c_info = cached_batch_results.get(str(eps), {})
        fork_summaries.append({
            "epsilon": eps,
            "cached_clip_fraction": c_info.get("clip_fraction"),
            "cached_surrogate_loss": c_info.get("surrogate_loss"),
            "reward_score_mean": eval_metrics.get("reward_score_mean"),
            "kl_from_reference": eval_metrics.get("kl_from_reference"),
            "response_length_mean": eval_metrics.get("response_length_mean"),
            "response_length_std": eval_metrics.get("response_length_std"),
            "stability_loss_std": loss_std,
            "stability_grad_norm_std": grad_std,
        })

    # Save consolidated results
    out_file = results_dir / "clipping_analysis.json"
    save_json(out_file, {
        "cached_batch_results": cached_batch_results,
        "fork_results": fork_summaries,
    })
    print(f"\nClipping study results saved to {out_file}")

    # Print summary table
    print("\n" + "=" * 80)
    print("PPO CLIPPING STUDY SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Epsilon':<8} | {'Cached Clip %':<14} | {'Held-out RM':<12} | {'KL (ref)':<10} | {'Len (mean)':<10} | {'Loss Std':<10}"
    print(header)
    print("-" * 80)
    for row in fork_summaries:
        eps_str = f"{row['epsilon']:.2f}"
        c_clip = f"{row['cached_clip_fraction']*100:.2f}%" if row['cached_clip_fraction'] is not None else "N/A"
        rm_str = f"{row['reward_score_mean']:+.4f}" if row['reward_score_mean'] is not None else "N/A"
        kl_str = f"{row['kl_from_reference']:.4f}" if row['kl_from_reference'] is not None else "N/A"
        len_str = f"{row['response_length_mean']:.1f}" if row['response_length_mean'] is not None else "N/A"
        l_std = f"{row['stability_loss_std']:.4f}"
        print(f"{eps_str:<8} | {c_clip:<14} | {rm_str:<12} | {kl_str:<10} | {len_str:<10} | {l_std:<10}")
    print("=" * 80 + "\n")

    return fork_summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--skip-train", action="store_true", help="Skip training if checkpoints already exist.")
    args = ap.parse_args()
    run_clipping_study(args.config, skip_train=args.skip_train)


if __name__ == "__main__":
    main()
