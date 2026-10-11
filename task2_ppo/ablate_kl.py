from __future__ import annotations

import argparse
from pathlib import Path

from common.data import load_yaml, repo_path
from common.logging_utils import save_json
from common.models import clear_gpu
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import run_evaluation


def run_kl_ablation(config_path: str, skip_train: bool = False):
    cfg = load_yaml(config_path)
    kl_values = [float(k) for k in cfg.get("kl_values", [0.0, 0.10, 0.20])]
    fork_updates = int(cfg.get("fork_updates", 8))
    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("TASK 2 STEP 3: REWARD-OVEROPTIMIZATION STUDY (KL PRESSURE ABLATION)")
    print(f"KL beta conditions: {kl_values}")
    print(f"Fork update budget: {fork_updates} updates")
    print("=" * 65)

    ablation_summaries = []

    for beta_kl in kl_values:
        run_name = f"kl_{beta_kl}"
        output_dir = f"outputs/task2_ppo/{run_name}"
        adapter_path = repo_path(output_dir)

        if not skip_train or not (adapter_path / "adapter_config.json").exists():
            print(f"\n--- Training PPO fork with kl_beta={beta_kl} for {fork_updates} updates ---")
            run_ppo(
                config_path=config_path,
                output=output_dir,
                updates=fork_updates,
                kl_beta=beta_kl,
                run_name=run_name,
            )
            clear_gpu()
        else:
            print(f"\nFound existing checkpoint for kl_beta={beta_kl}; skipping training.")

        print(f"\nEvaluating kl_beta={beta_kl} on held-out test prompts...")
        eval_metrics = run_evaluation(
            config_path=config_path,
            adapter=output_dir,
            name=run_name,
        )
        clear_gpu()

        ablation_summaries.append({
            "kl_beta": beta_kl,
            "reward_score_mean": eval_metrics.get("reward_score_mean"),
            "kl_from_reference": eval_metrics.get("kl_from_reference"),
            "entropy_mean": eval_metrics.get("entropy_mean"),
            "response_length_mean": eval_metrics.get("response_length_mean"),
            "response_length_std": eval_metrics.get("response_length_std"),
            "response_length_iqr": eval_metrics.get("response_length_iqr"),
            "qualitative_samples": eval_metrics.get("qualitative_samples", []),
        })

    # Save consolidated results for report
    out_file = results_dir / "kl_ablation.json"
    save_json(out_file, {"kl_ablation_results": ablation_summaries})
    print(f"\nConsolidated KL ablation results saved to: {out_file}")

    # Print summary comparison table
    print("\n" + "=" * 80)
    print("REWARD-OVEROPTIMIZATION (KL PRESSURE) SUMMARY TABLE")
    print("=" * 80)
    header = f"{'KL Beta':<10} | {'Held-out RM':<14} | {'KL (ref)':<12} | {'Entropy':<12} | {'Length (mean±std)':<18}"
    print(header)
    print("-" * 80)
    for row in ablation_summaries:
        b_str = f"{row['kl_beta']:.2f}"
        rm_str = f"{row['reward_score_mean']:+.4f}" if row['reward_score_mean'] is not None else "N/A"
        kl_str = f"{row['kl_from_reference']:.4f}" if row['kl_from_reference'] is not None else "N/A"
        ent_str = f"{row['entropy_mean']:.4f}" if row['entropy_mean'] is not None else "N/A"
        len_str = f"{row['response_length_mean']:.1f} ± {row['response_length_std']:.1f}" if row['response_length_mean'] is not None else "N/A"
        print(f"{b_str:<10} | {rm_str:<14} | {kl_str:<12} | {ent_str:<12} | {len_str:<18}")
    print("=" * 80 + "\n")

    return ablation_summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--skip-train", action="store_true", help="Skip training if checkpoints already exist.")
    args = ap.parse_args()
    run_kl_ablation(args.config, skip_train=args.skip_train)


if __name__ == "__main__":
    main()
