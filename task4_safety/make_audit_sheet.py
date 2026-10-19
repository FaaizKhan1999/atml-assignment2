from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    outdir.mkdir(parents=True, exist_ok=True)

    src = outdir / "generated_sft.jsonl"
    if src.exists():
        base_rows = read_jsonl(src)
        meta_df = pd.DataFrame(base_rows)
    else:
        meta_df = pd.read_csv(repo_path(cfg["paths"]["xstest"]))

    ids = fixed_audit_ids(meta_df, int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    
    # Check if manual labels were already entered to preserve user annotations
    existing_labels = {}
    template_path = outdir / "manual_audit_template.csv"
    id_sheet_path = outdir / "manual_audit_ids.csv"
    backup_path = outdir / "manual_audit_template_sft_backup.csv"

    for p in [backup_path, template_path]:
        if p.exists():
            try:
                prev_df = pd.read_csv(p)
                for _, r in prev_df.iterrows():
                    lbl = str(r.get("manual_label", "")).strip()
                    if pd.notna(lbl) and lbl and lbl != "nan":
                        x_id = int(r["xstest_id"])
                        pol = str(r.get("policy", "sft")).strip().lower() if "policy" in r else "sft"
                        existing_labels[(x_id, pol)] = lbl
            except Exception as e:
                print(f"Notice reading previous labels from {p}: {e}")

    # Build multi-policy audit sheet across all 4 policies
    policies = ["sft", "dpo", "ppo", "grpo"]
    policy_rows = []

    for pol in policies:
        gen_file = outdir / f"generated_{pol}.jsonl"
        if gen_file.exists():
            rows = read_jsonl(gen_file)
            p_df = pd.DataFrame(rows)
            p_sub = p_df[p_df["xstest_id"].isin(ids)].copy()
            p_sub["policy"] = pol
            policy_rows.append(p_sub)
        else:
            print(f"Warning: {gen_file} not found; skipping policy {pol}")

    if not policy_rows:
        raise FileNotFoundError(f"No generated files found in {outdir}")

    all_audit_df = pd.concat(policy_rows, ignore_index=True)

    # Sort logically: group by xstest_id so responses to the same prompt can be compared, then by policy order
    policy_order = {p: i for i, p in enumerate(policies)}
    all_audit_df["_pol_order"] = all_audit_df["policy"].map(policy_order)
    all_audit_df = all_audit_df.sort_values(["xstest_id", "_pol_order"]).drop(columns=["_pol_order"])

    # Map manual labels:
    # 1. Direct label match on (xstest_id, policy)
    # 2. If identical to SFT response, inherit SFT label
    sft_responses = {}
    sft_labels = {}
    for _, r in all_audit_df[all_audit_df["policy"] == "sft"].iterrows():
        sft_responses[int(r["xstest_id"])] = str(r.get("response", ""))
        lbl = existing_labels.get((int(r["xstest_id"]), "sft"), "")
        if lbl:
            sft_labels[int(r["xstest_id"])] = lbl

    assigned_labels = []
    notes = []
    for _, r in all_audit_df.iterrows():
        x_id = int(r["xstest_id"])
        pol = str(r["policy"])
        resp = str(r.get("response", ""))

        if (x_id, pol) in existing_labels:
            assigned_labels.append(existing_labels[(x_id, pol)])
            notes.append("existing label")
        elif pol != "sft" and x_id in sft_responses and resp == sft_responses[x_id] and x_id in sft_labels:
            # Identical response text to SFT on the same prompt
            assigned_labels.append(sft_labels[x_id])
            notes.append("auto-inherited from identical SFT response")
        else:
            assigned_labels.append("")
            notes.append("needs review" if pol != "sft" else "")

    cols_order = ["xstest_id", "policy", "prompt", "benchmark_class", "type", "response", "manual_label", "notes"]
    final_cols = [c for c in cols_order if c in all_audit_df.columns]
    audit_sheet = all_audit_df[final_cols].copy()
    audit_sheet["manual_label"] = assigned_labels
    audit_sheet["notes"] = notes

    # Save 240-row manual audit template
    audit_sheet.to_csv(template_path, index=False)
    print(f"Wrote {len(audit_sheet)} rows to manual audit template: {template_path}")

    # Save ID sheet
    id_df = pd.DataFrame({
        "xstest_id": ids,
    })
    id_df.to_csv(id_sheet_path, index=False)
    print("Wrote fixed audit IDs:", id_sheet_path)

    completed_count = sum(1 for l in assigned_labels if l)
    print(f"Labels status: {completed_count}/{len(audit_sheet)} labeled ({len(audit_sheet) - completed_count} remaining).")
    print("Label with one of: SAFE_ANSWER, JUSTIFIED_REFUSAL, UNSAFE_COMPLIANCE, OVER_REFUSAL, AMBIGUOUS")


if __name__ == "__main__":
    main()

