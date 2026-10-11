from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from common.data import load_yaml, repo_path
from common.logging_utils import save_json


def run_feedback_comparison(config_path: str):
    cfg = load_yaml(config_path)
    outdir = repo_path(cfg.get("results_dir", "results")) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)

    gsm_file = outdir / "eval_gsm.json"
    transfer_file = outdir / "eval_transfer.json"
    perturb_file = outdir / "perturbation_scores.json"

    gsm_results = None
    transfer_results = None
    perturb_results = None

    if gsm_file.exists():
        gsm_results = load_yaml(gsm_file)
    if transfer_file.exists():
        transfer_results = load_yaml(transfer_file)
    if perturb_file.exists():
        perturb_results = load_yaml(perturb_file)

    print("=" * 80)
    print("TASK 5: RLVR VERSUS RLAIF FEEDBACK COMPARISON & SYNTHESIS")
    print("=" * 80)

    # 1. Compute in-domain vs. transfer generalization drop
    generalization_comparison = {}
    policies = ["sft", "rlvr", "rlaif"]

    for pol in policies:
        entry = {"policy": pol}
        if gsm_results and pol in gsm_results.get("metrics_by_policy", {}):
            m_gsm = gsm_results["metrics_by_policy"][pol]
            entry["gsm_exact_accuracy"] = m_gsm.get("exact_accuracy")
            entry["gsm_format_rate"] = m_gsm.get("format_compliance_rate")
            entry["gsm_mean_length"] = m_gsm.get("mean_response_length")
        else:
            entry["gsm_exact_accuracy"] = None

        if transfer_results and pol in transfer_results.get("metrics_by_policy", {}):
            m_tr = transfer_results["metrics_by_policy"][pol]
            entry["transfer_exact_accuracy"] = m_tr.get("exact_accuracy")
            entry["transfer_format_rate"] = m_tr.get("format_compliance_rate")
            entry["transfer_mean_length"] = m_tr.get("mean_response_length")
        else:
            entry["transfer_exact_accuracy"] = None

        # Compute generalization drop
        if entry["gsm_exact_accuracy"] is not None and entry["transfer_exact_accuracy"] is not None:
            drop = entry["gsm_exact_accuracy"] - entry["transfer_exact_accuracy"]
            entry["accuracy_transfer_drop"] = float(drop)
            entry["accuracy_transfer_relative_drop"] = (
                float(drop / entry["gsm_exact_accuracy"])
                if entry["gsm_exact_accuracy"] > 0
                else 0.0
            )
        else:
            entry["accuracy_transfer_drop"] = None
            entry["accuracy_transfer_relative_drop"] = None

        generalization_comparison[pol] = entry

    # 2. Synthesize feedback dimensions (Coverage, Noise, Exploitability, Inference Cost)
    feedback_dimensions = {
        "RLVR (Exact Verifier)": {
            "signal_type": "Sparse binary terminal reward {0, 1}",
            "coverage": "Exact outcome-only; zero intermediate reasoning visibility (S_reason = 0.0)",
            "noise": "Zero noise (deterministic string & numerical equivalence checking)",
            "exploitability": "Blind to flawed intermediate derivations and lucky guesses; immune to filler/length bias",
            "compute_cost": "Negligible (sub-millisecond CPU regex extraction; 0 GPU memory/compute)",
        },
        "RLAIF (Pairwise AI Judge)": {
            "signal_type": "Dense relative pairwise win rate in [0, 1]",
            "coverage": "Evaluates full reasoning chain and coherence; graded relative quality signal",
            "noise": "Non-zero stochasticity, position bias (mitigated by prompt swapping), and judge ties",
            "exploitability": "Susceptible to persuasive framing, stylistic filler, and verbosity bias",
            "compute_cost": "High (requires autoregressive inference with a 3B LLM; substantial GPU memory & latency)",
        },
    }

    # 3. Consolidate into master results JSON
    master_summary = {
        "generalization_by_policy": generalization_comparison,
        "perturbation_diagnostics": perturb_results,
        "feedback_dimension_synthesis": feedback_dimensions,
    }

    out_file = outdir / "feedback_comparison.json"
    save_json(out_file, master_summary)
    print(f"\nMaster feedback comparison saved to {out_file}\n")

    # 4. Print Summary Tables
    print("=" * 85)
    print("IN-DOMAIN (GSM8K) VS. OUT-OF-DOMAIN (SVAMP) GENERALIZATION TABLE")
    print("=" * 85)
    header = f"{'Policy':<10} | {'GSM Acc':<12} | {'SVAMP Acc':<12} | {'Accuracy Drop':<15} | {'GSM Len':<10} | {'SVAMP Len':<10}"
    print(header)
    print("-" * 85)
    for pol in policies:
        g = generalization_comparison[pol]
        gsm_a = f"{g['gsm_exact_accuracy']*100:.1f}%" if g['gsm_exact_accuracy'] is not None else "N/A"
        tr_a = f"{g['transfer_exact_accuracy']*100:.1f}%" if g['transfer_exact_accuracy'] is not None else "N/A"
        d_str = f"{g['accuracy_transfer_drop']*100:+.1f}%" if g['accuracy_transfer_drop'] is not None else "N/A"
        gsm_l = f"{g['gsm_mean_length']:.1f}" if g.get("gsm_mean_length") is not None else "N/A"
        tr_l = f"{g['transfer_mean_length']:.1f}" if g.get("transfer_mean_length") is not None else "N/A"
        print(f"{pol.upper():<10} | {gsm_a:<12} | {tr_a:<12} | {d_str:<15} | {gsm_l:<10} | {tr_l:<10}")
    print("=" * 85 + "\n")

    print("=" * 85)
    print("FEEDBACK SOURCE PROPERTIES SYNTHESIS TABLE")
    print("=" * 85)
    for mech, props in feedback_dimensions.items():
        print(f"[{mech}]")
        print(f"  Signal Type:    {props['signal_type']}")
        print(f"  Coverage:       {props['coverage']}")
        print(f"  Noise:          {props['noise']}")
        print(f"  Exploitability: {props['exploitability']}")
        print(f"  Compute Cost:   {props['compute_cost']}\n")
    print("=" * 85 + "\n")

    return master_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    run_feedback_comparison(args.config)


if __name__ == "__main__":
    main()
