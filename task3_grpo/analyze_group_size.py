from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt: dict[str, list[dict]], k: int) -> list[dict]:
    """Return K-sized groups while keeping total cached completions fixed.

    Each prompt's 8 completions are partitioned into chunks of size K (e.g., K=4 yields 2 groups per prompt,
    K=2 yields 4 groups per prompt). This holds the total completions fixed at 192 across all K.
    """
    groups = []
    for pid, completions in by_prompt.items():
        for start_idx in range(0, len(completions), k):
            sub_group = completions[start_idx : start_idx + k]
            if len(sub_group) == k:
                groups.append({
                    "prompt_id": pid,
                    "rows": sub_group,
                    "rewards": [float(r["reward"]) for r in sub_group],
                })
    return groups


def compute_group_metrics(groups: list[dict], eps: float = 1e-6) -> dict:
    if not groups:
        return {
            "num_groups": 0,
            "informative_rate": 0.0,
            "uninformative_rate": 1.0,
            "mean_reward_std": 0.0,
            "advantage_variance": 0.0,
        }

    stds = []
    advantages = []
    informative_count = 0

    for g in groups:
        r = np.array(g["rewards"], dtype=float)
        mean_r = np.mean(r)
        std_r = np.std(r, ddof=0)
        stds.append(std_r)

        if std_r > eps:
            informative_count += 1
            adv = (r - mean_r) / (std_r + eps)
        else:
            adv = np.zeros_like(r)

        advantages.extend(adv.tolist())

    inf_rate = informative_count / len(groups)
    return {
        "num_groups": len(groups),
        "informative_rate": float(inf_rate),
        "uninformative_rate": float(1.0 - inf_rate),
        "mean_reward_std": float(np.mean(stds)),
        "advantage_variance": float(np.var(advantages)),
    }


def run_group_size_analysis(config_path: str):
    cfg = load_yaml(config_path)
    by_prompt = load_k8_cache(cfg["group_cache"])
    group_sizes = [int(k) for k in cfg.get("group_sizes", [2, 4, 8])]
    results_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("TASK 3 STEP 2: GROUP-SIZE STUDY (EQUAL GENERATION BUDGET)")
    print(f"Cached prompts: {len(by_prompt)} (Total generations: {len(by_prompt) * 8})")
    print(f"Group sizes to evaluate: {group_sizes}")
    print("=" * 65)

    # 1. Define prompt difficulty binning rule once:
    # Use the median average reward of each prompt's 8 completions to split into Hard vs Easy bins.
    prompt_avg_rewards = {
        pid: float(np.mean([r["reward"] for r in comps]))
        for pid, comps in by_prompt.items()
    }
    median_reward = float(np.median(list(prompt_avg_rewards.values())))
    hard_prompt_ids = set(pid for pid, avg in prompt_avg_rewards.items() if avg <= median_reward)
    easy_prompt_ids = set(pid for pid, avg in prompt_avg_rewards.items() if avg > median_reward)

    print(f"Prompt Difficulty Binning (Median reward split = {median_reward:.4f}):")
    print(f"  Hard prompts (avg reward <= median): {len(hard_prompt_ids)} prompts")
    print(f"  Easy prompts (avg reward > median):  {len(easy_prompt_ids)} prompts\n")

    summary_by_k = []

    for k in group_sizes:
        all_groups = regroup_equal_generation_budget(by_prompt, k)
        hard_groups = [g for g in all_groups if g["prompt_id"] in hard_prompt_ids]
        easy_groups = [g for g in all_groups if g["prompt_id"] in easy_prompt_ids]

        overall_metrics = compute_group_metrics(all_groups)
        hard_metrics = compute_group_metrics(hard_groups)
        easy_metrics = compute_group_metrics(easy_groups)

        record = {
            "group_size_k": k,
            "total_completions": len(all_groups) * k,
            "overall": overall_metrics,
            "hard_bin": hard_metrics,
            "easy_bin": easy_metrics,
        }
        summary_by_k.append(record)

    # 2. Extract qualitative examples (one informative group and one uninformative group)
    qualitative_samples = {}
    k2_groups = regroup_equal_generation_budget(by_prompt, 2)
    for g in k2_groups:
        r = np.array(g["rewards"], dtype=float)
        std_r = np.std(r, ddof=0)
        if "uninformative_example" not in qualitative_samples and std_r <= 1e-6:
            qualitative_samples["uninformative_example"] = {
                "group_size_k": 2,
                "prompt_id": g["prompt_id"],
                "rewards": g["rewards"],
                "completions": [row["completion"][:200] + "..." for row in g["rows"]],
                "description": "Both completions received identical reward; within-group std is 0.0, so advantage collapses to 0.0 (no gradient update).",
            }
        elif "informative_example" not in qualitative_samples and std_r > 0.5:
            mean_r = np.mean(r)
            advs = ((r - mean_r) / (std_r + 1e-6)).tolist()
            qualitative_samples["informative_example"] = {
                "group_size_k": 2,
                "prompt_id": g["prompt_id"],
                "rewards": g["rewards"],
                "relative_advantages": advs,
                "completions": [row["completion"][:200] + "..." for row in g["rows"]],
                "description": "Completions received distinct rewards, producing non-zero positive and negative advantages for relative policy updates.",
            }
        if "uninformative_example" in qualitative_samples and "informative_example" in qualitative_samples:
            break

    out_file = results_dir / "group_size_analysis.json"
    save_json(out_file, {
        "difficulty_median_threshold": median_reward,
        "results_by_k": summary_by_k,
        "qualitative_samples": qualitative_samples,
    })
    print(f"Group-size analysis results saved to {out_file}")

    # Print summary comparison table
    print("\n" + "=" * 80)
    print("EQUAL-GENERATION GROUP-SIZE STUDY SUMMARY TABLE")
    print("=" * 80)
    header = f"{'K':<4} | {'Groups':<7} | {'Overall Inf %':<14} | {'Overall Std':<12} | {'Hard Inf %':<12} | {'Easy Inf %':<12} | {'Adv Var':<8}"
    print(header)
    print("-" * 80)
    for rec in summary_by_k:
        k = rec["group_size_k"]
        ng = rec["overall"]["num_groups"]
        inf = f"{rec['overall']['informative_rate']*100:.1f}%"
        std = f"{rec['overall']['mean_reward_std']:.4f}"
        h_inf = f"{rec['hard_bin']['informative_rate']*100:.1f}%"
        e_inf = f"{rec['easy_bin']['informative_rate']*100:.1f}%"
        v = f"{rec['overall']['advantage_variance']:.4f}"
        print(f"{k:<4} | {ng:<7} | {inf:<14} | {std:<12} | {h_inf:<12} | {e_inf:<12} | {v:<8}")
    print("=" * 80 + "\n")

    return summary_by_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    run_group_size_analysis(args.config)


if __name__ == "__main__":
    main()
