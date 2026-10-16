from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json
from common.metrics import safe_corr
from common.models import clear_gpu
from task3_grpo.continue_train import run_grpo
from task3_grpo.evaluate import run_evaluation


def run_normalization_comparison(config_path: str, skip_train: bool = False):
    cfg = load_yaml(config_path)
    fork_updates = int(cfg.get("fork_updates", 8))
    results_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    conditions = [
        {"loss_type": "grpo", "label": "Canonical GRPO (1/Tk)", "run_name": "norm_canonical"},
        {"loss_type": "dr_grpo", "label": "Dr. GRPO (1/Tmax)", "run_name": "norm_dr_grpo"},
    ]

    print("=" * 70)
    print("TASK 3 STEP 3: LENGTH-NORMALIZATION STUDY (CANONICAL GRPO vs Dr. GRPO)")
    print(f"Fork budget: {fork_updates} updates | Conditions: {[c['loss_type'] for c in conditions]}")
    print("=" * 70)

    comparison_results = []

    for cond in conditions:
        loss_type = cond["loss_type"]
        run_name = cond["run_name"]
        output_dir = f"outputs/task3_grpo/{run_name}"
        adapter_path = repo_path(output_dir)

        train_record = None
        if not skip_train or not (adapter_path / "adapter_config.json").exists():
            print(f"\n--- Training short GRPO fork: {cond['label']} ({fork_updates} updates) ---")
            train_record = run_grpo(
                config_path=config_path,
                output=output_dir,
                updates=fork_updates,
                loss_type=loss_type,
                run_name=run_name,
            )
            clear_gpu()
        else:
            print(f"\nFound existing checkpoint for {cond['label']} at {output_dir}; skipping training.")

        print(f"\nEvaluating {cond['label']} on held-out test prompts...")
        eval_metrics = run_evaluation(
            config_path=config_path,
            adapter=output_dir,
            name=run_name,
        )
        clear_gpu()

        # Compute length-conditioned statistics
        # 1. From training trajectory: average length-loss correlation
        avg_len_loss_corr = None
        train_log = results_dir / f"train_{run_name}.json"
        history = []
        if train_record and train_record.get("history"):
            history = train_record["history"]
        elif train_log.exists():
            history = load_json(train_log).get("history", [])

        if history:
            corrs = [h["length_loss_correlation"] for h in history if "length_loss_correlation" in h]
            if corrs:
                avg_len_loss_corr = float(np.mean(corrs))

        # 2. From evaluation samples: length-reward and length-KL correlations
        samples = eval_metrics.get("qualitative_samples", [])
        len_reward_corr = None
        len_kl_corr = None
        if samples:
            sample_lens = [float(s["length"]) for s in samples]
            if "reward" in samples[0]:
                sample_rewards = [float(s["reward"]) for s in samples]
                c_r = safe_corr(sample_lens, sample_rewards)
                len_reward_corr = float(c_r) if not np.isnan(c_r) else 0.0
            if "kl" in samples[0]:
                sample_kls = [float(s["kl"]) for s in samples]
                c_k = safe_corr(sample_lens, sample_kls)
                len_kl_corr = float(c_k) if not np.isnan(c_k) else 0.0

        summary_item = {
            "loss_type": loss_type,
            "label": cond["label"],
            "run_name": run_name,
            "updates": fork_updates,
            "held_out_reward_mean": eval_metrics.get("reward_score_mean"),
            "held_out_kl_from_reference": eval_metrics.get("kl_from_reference"),
            "held_out_entropy_mean": eval_metrics.get("entropy_mean"),
            "held_out_length_mean": eval_metrics.get("response_length_mean"),
            "held_out_length_std": eval_metrics.get("response_length_std"),
            "held_out_length_iqr": eval_metrics.get("response_length_iqr"),
            "train_length_loss_correlation_mean": avg_len_loss_corr,
            "eval_length_reward_correlation": len_reward_corr,
            "eval_length_kl_correlation": len_kl_corr,
            "qualitative_samples": samples,
        }
        comparison_results.append(summary_item)

    # Save consolidated results for report
    out_file = results_dir / "normalization_comparison.json"
    save_json(out_file, {"normalization_comparison": comparison_results})
    print(f"\nConsolidated normalization comparison saved to: {out_file}")

    # Print summary table
    print("\n" + "=" * 95)
    print("GRPO NORMALIZATION COMPARISON SUMMARY TABLE")
    print("=" * 95)
    header = f"{'Method':<25} | {'Reward':<12} | {'KL (ref)':<10} | {'Length (mean±std)':<18} | {'Len-Loss r':<12} | {'Len-Reward r':<12}"
    print(header)
    print("-" * 95)
    for row in comparison_results:
        m_str = row["label"]
        r_str = f"{row['held_out_reward_mean']:+.4f}" if row['held_out_reward_mean'] is not None else "N/A"
        kl_str = f"{row['held_out_kl_from_reference']:.4f}" if row['held_out_kl_from_reference'] is not None else "N/A"
        len_str = f"{row['held_out_length_mean']:.1f} ± {row['held_out_length_std']:.1f}" if row['held_out_length_mean'] is not None else "N/A"
        c_loss = f"{row['train_length_loss_correlation_mean']:+.4f}" if row['train_length_loss_correlation_mean'] is not None else "N/A"
        c_rw = f"{row['eval_length_reward_correlation']:+.4f}" if row['eval_length_reward_correlation'] is not None else "N/A"
        print(f"{m_str:<25} | {r_str:<12} | {kl_str:<10} | {len_str:<18} | {c_loss:<12} | {c_rw:<12}")
    print("=" * 95 + "\n")

    return comparison_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--skip-train", action="store_true", help="Skip training if checkpoints already exist.")
    args = ap.parse_args()
    run_normalization_comparison(args.config, skip_train=args.skip_train)


if __name__ == "__main__":
    main()
