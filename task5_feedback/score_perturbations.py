from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from common.models import clear_gpu
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward


EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}

PERTURBATION_CONDITIONS = [
    {
        "name": "reasoning_corruption",
        "description": "Corrupt reasoning while keeping final answer correct (Tests S_reason)",
        "better_variant": "clean_correct",
        "perturbed_variant": "corrupt_reasoning_correct_final",
        "metric_tag": "S_reason",
    },
    {
        "name": "outcome_error",
        "description": "Coherent reasoning while designated final is wrong (Tests S_outcome)",
        "better_variant": "clean_correct",
        "perturbed_variant": "good_reasoning_wrong_final",
        "metric_tag": "S_outcome",
    },
    {
        "name": "persuasive_filler",
        "description": "Correct answer augmented with persuasive filler",
        "better_variant": "clean_correct",
        "perturbed_variant": "persuasive_filler_correct",
        "metric_tag": "filler_susceptibility",
    },
    {
        "name": "gold_distractor",
        "description": "Gold number mentioned as distractor but wrong final answer",
        "better_variant": "clean_correct",
        "perturbed_variant": "gold_distractor_wrong_final",
        "metric_tag": "distractor_robustness",
    },
]


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def evaluate_exact_verifier(groups: dict[str, dict[str, dict]]) -> dict:
    results = {}
    for cond in PERTURBATION_CONDITIONS:
        cname = cond["name"]
        better_v = cond["better_variant"]
        pert_v = cond["perturbed_variant"]

        better_count = 0
        tie_count = 0
        wrong_count = 0
        n_probs = len(groups)

        for pid, variants in groups.items():
            gold = str(variants[better_v]["gold_final"]).strip()
            r_better = exact_reward(variants[better_v]["response"], gold)
            r_pert = exact_reward(variants[pert_v]["response"], gold)

            if r_better > r_pert:
                better_count += 1
            elif r_better == r_pert:
                tie_count += 1
            else:
                wrong_count += 1

        results[cname] = {
            "name": cname,
            "metric_tag": cond["metric_tag"],
            "better_response_rate": float(better_count / n_probs),
            "tie_rate": float(tie_count / n_probs),
            "wrong_preference_rate": float(wrong_count / n_probs),
            "total_pairs": n_probs,
        }

    return results


def evaluate_ai_judge(groups: dict[str, dict[str, dict]], judge: PairwiseAIJudge) -> dict:
    results = {}
    for cond in PERTURBATION_CONDITIONS:
        cname = cond["name"]
        better_v = cond["better_variant"]
        pert_v = cond["perturbed_variant"]

        better_count = 0
        tie_count = 0
        wrong_count = 0
        n_probs = len(groups)

        for pid, variants in groups.items():
            question = variants[better_v]["question"]
            resp_better = variants[better_v]["response"]
            resp_pert = variants[pert_v]["response"]

            pref = judge.compare(problem=question, a=resp_better, b=resp_pert)
            if pref == "A":
                better_count += 1
            elif pref == "TIE":
                tie_count += 1
            else:
                wrong_count += 1

        results[cname] = {
            "name": cname,
            "metric_tag": cond["metric_tag"],
            "better_response_rate": float(better_count / n_probs),
            "tie_rate": float(tie_count / n_probs),
            "wrong_preference_rate": float(wrong_count / n_probs),
            "total_pairs": n_probs,
        }

    return results


def run_perturbation_study(config_path: str, skip_judge: bool = False):
    cfg = load_yaml(config_path)
    diag_path = cfg["paths"]["task5_diagnostics"]
    groups = load_diagnostic_groups(diag_path)
    outdir = repo_path(cfg.get("results_dir", "results")) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("TASK 5 STEP 2: REWARD SENSITIVITY AND ROBUSTNESS STUDY")
    print(f"Total problems: {len(groups)} (5 variants per problem = {len(groups) * 5} responses)")
    print("=" * 75)

    # 1. Exact Verifier Evaluation (runs instantly without GPU)
    print("\nEvaluating Exact Verifier (RLVR) on diagnostic pairs...")
    verifier_results = evaluate_exact_verifier(groups)

    # 2. AI Pairwise Judge Evaluation
    judge_results = None
    if not skip_judge:
        cache_path = outdir / "judge_diagnostics_cache.json"
        try:
            print(f"\nLoading Pairwise AI Judge ({cfg['ai_judge_model']})...")
            judge = PairwiseAIJudge(cfg, cache_path=cache_path)
            judge_results = evaluate_ai_judge(groups, judge)
            clear_gpu()
        except Exception as e:
            print(f"Warning: Could not run PairwiseAIJudge ({e}). Skipping AI judge scoring.")

    # 3. Extract S_reason and S_outcome
    s_reason_verifier = verifier_results["reasoning_corruption"]["better_response_rate"]
    s_outcome_verifier = verifier_results["outcome_error"]["better_response_rate"]

    s_reason_judge = judge_results["reasoning_corruption"]["better_response_rate"] if judge_results else None
    s_outcome_judge = judge_results["outcome_error"]["better_response_rate"] if judge_results else None

    # 4. Extract qualitative diagnostic examples
    qualitative_samples = {}
    sample_pid = next(iter(groups.keys()))
    sample_group = groups[sample_pid]
    for cond in PERTURBATION_CONDITIONS:
        cname = cond["name"]
        better_v = cond["better_variant"]
        pert_v = cond["perturbed_variant"]
        qualitative_samples[cname] = {
            "problem_id": sample_pid,
            "question": sample_group[better_v]["question"],
            "gold_final": sample_group[better_v]["gold_final"],
            "better_variant": better_v,
            "better_response": sample_group[better_v]["response"],
            "perturbed_variant": pert_v,
            "perturbed_response": sample_group[pert_v]["response"],
            "condition_description": cond["description"],
        }

    # 5. Save consolidated JSON
    summary_output = {
        "num_problems": len(groups),
        "total_responses": len(groups) * 5,
        "s_reason": {
            "verifier": s_reason_verifier,
            "ai_judge": s_reason_judge,
        },
        "s_outcome": {
            "verifier": s_outcome_verifier,
            "ai_judge": s_outcome_judge,
        },
        "exact_verifier_breakdown": verifier_results,
        "ai_judge_breakdown": judge_results,
        "qualitative_samples": qualitative_samples,
    }

    out_file = outdir / "perturbation_scores.json"
    save_json(out_file, summary_output)
    print(f"\nDiagnostic perturbation results saved to {out_file}\n")

    # 5. Print summary comparison table
    print("=" * 88)
    print("CONTROLLED REWARD DIAGNOSTIC COMPARISON TABLE")
    print("=" * 88)
    header = f"{'Perturbation Category':<25} | {'Mechanism':<14} | {'Prefers Better':<16} | {'Tie Rate':<12} | {'Wrong Pref':<12}"
    print(header)
    print("-" * 88)
    for cond in PERTURBATION_CONDITIONS:
        cname = cond["name"]
        v_res = verifier_results[cname]
        v_b = f"{v_res['better_response_rate']*100:.1f}%"
        v_t = f"{v_res['tie_rate']*100:.1f}%"
        v_w = f"{v_res['wrong_preference_rate']*100:.1f}%"
        print(f"{cname:<25} | {'Exact Verifier':<14} | {v_b:<16} | {v_t:<12} | {v_w:<12}")

        if judge_results:
            j_res = judge_results[cname]
            j_b = f"{j_res['better_response_rate']*100:.1f}%"
            j_t = f"{j_res['tie_rate']*100:.1f}%"
            j_w = f"{j_res['wrong_preference_rate']*100:.1f}%"
            print(f"{'':<25} | {'AI Judge':<14} | {j_b:<16} | {j_t:<12} | {j_w:<12}")
        print("-" * 88)
    print("=" * 88 + "\n")

    return summary_output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--skip-judge", action="store_true", help="Score only exact verifier")
    args = ap.parse_args()
    run_perturbation_study(args.config, skip_judge=args.skip_judge)


if __name__ == "__main__":
    main()
