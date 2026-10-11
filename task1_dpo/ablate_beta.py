from __future__ import annotations

import argparse
from pathlib import Path

from common.data import load_yaml, repo_path
from common.logging_utils import save_json
from common.models import clear_gpu
from task1_dpo.evaluate import run_evaluation
from task1_dpo.train import run_training


def run_beta_ablation(config_path: str, skip_train: bool = False):
    cfg = load_yaml(config_path)
    betas = cfg.get("betas", [0.03, 0.10, 0.30])
    max_examples = int(cfg.get("short_ablation_examples", 600))
    results_dir = repo_path(cfg.get("results_dir", "results/task1_dpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"TASK 1 STEP 2: REGULARIZATION-STRENGTH STUDY (BETA ABLATION)")
    print(f"Betas to test: {betas}")
    print(f"Short ablation budget per condition: {max_examples} examples")
    print("=" * 60)

    ablation_summary: list[dict] = []

    for beta in betas:
        run_name = f"beta_{beta}"
        output_dir = f"outputs/task1_dpo/{run_name}"
        adapter_path = repo_path(output_dir)

        # 1. Train short-budget condition from original initialization
        if not skip_train or not (adapter_path / "adapter_config.json").exists():
            print(f"\n[1/2] Training DPO fork with beta={beta} on {max_examples} examples...")
            run_training(
                config_path=config_path,
                run_name=run_name,
                output_path=output_dir,
                beta=float(beta),
                max_examples=max_examples,
            )
            clear_gpu()
        else:
            print(f"\nFound existing checkpoint for beta={beta} at {output_dir}; skipping training.")

        # 2. Evaluate under common evaluation protocol
        print(f"\n[2/2] Evaluating beta={beta} on held-out test set...")
        eval_metrics = run_evaluation(
            config_path=config_path,
            adapter=output_dir,
            name=run_name,
            beta=float(beta),
        )
        clear_gpu()

        ablation_summary.append({
            "beta": beta,
            "budget_examples": max_examples,
            "dpo_loss": eval_metrics["dpo_loss"],
            "preference_accuracy": eval_metrics["preference_accuracy"],
            "margin_mean": eval_metrics["margin_mean"],
            "kl_from_reference": eval_metrics.get("kl_from_reference"),
            "reward_score_mean": eval_metrics.get("reward_score_mean"),
            "response_length_mean": eval_metrics.get("response_length_mean"),
            "response_length_std": eval_metrics.get("response_length_std"),
            "response_length_iqr": eval_metrics.get("response_length_iqr"),
        })

    # Save consolidated results for report
    out_file = results_dir / "beta_ablation.json"
    save_json(out_file, {"ablation_results": ablation_summary})
    print(f"\nConsolidated beta ablation results saved to: {out_file}")

    # Print summary comparison table
    print("\n" + "=" * 80)
    print("BETA REGULARIZATION-STRENGTH STUDY SUMMARY")
    print("=" * 80)
    header = f"{'Beta':<8} | {'DPO Loss':<10} | {'Pref Acc (%)':<14} | {'KL (ref)':<10} | {'RM Score':<10} | {'Length (mean±std)':<18}"
    print(header)
    print("-" * 80)
    for row in ablation_summary:
        b_str = str(row["beta"])
        loss_str = f"{row['dpo_loss']:.4f}"
        acc_str = f"{row['preference_accuracy'] * 100:.2f}%"
        kl_str = f"{row['kl_from_reference']:.4f}" if row['kl_from_reference'] is not None else "N/A"
        rm_str = f"{row['reward_score_mean']:.4f}" if row['reward_score_mean'] is not None else "N/A"
        len_str = f"{row['response_length_mean']:.1f} ± {row['response_length_std']:.1f}" if row['response_length_mean'] is not None else "N/A"
        print(f"{b_str:<8} | {loss_str:<10} | {acc_str:<14} | {kl_str:<10} | {rm_str:<10} | {len_str:<18}")
    print("=" * 80 + "\n")

    return ablation_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--skip-train", action="store_true", help="Skip training if checkpoints already exist.")
    args = ap.parse_args()
    run_beta_ablation(args.config, skip_train=args.skip_train)


if __name__ == "__main__":
    main()
