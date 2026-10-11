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
    for p in [template_path, id_sheet_path]:
        if p.exists():
            try:
                prev_df = pd.read_csv(p)
                for _, r in prev_df.iterrows():
                    if pd.notna(r.get("manual_label")) and str(r.get("manual_label")).strip():
                        existing_labels[int(r["xstest_id"])] = str(r["manual_label"]).strip()
            except Exception:
                pass

    # Save the ID-only sheet as requested
    id_df = pd.DataFrame({
        "xstest_id": ids,
        "manual_label": [existing_labels.get(i, "") for i in ids],
    })
    id_df.to_csv(id_sheet_path, index=False)
    print("Wrote fixed audit IDs:", id_sheet_path)

    # Also save an enriched audit template with prompt metadata for easy labeling
    audit_subset = meta_df[meta_df["xstest_id"].isin(ids)].copy().sort_values("xstest_id")
    cols_to_keep = [c for c in ["xstest_id", "prompt", "benchmark_class", "type", "response"] if c in audit_subset.columns]
    audit_sheet = audit_subset[cols_to_keep].copy()
    audit_sheet["manual_label"] = [existing_labels.get(int(i), "") for i in audit_sheet["xstest_id"]]
    audit_sheet["notes"] = ""
    audit_sheet.to_csv(template_path, index=False)
    print("Wrote manual audit template:", template_path)
    if existing_labels:
        print(f"Preserved {len(existing_labels)} existing manual labels.")
    print("Label with one of: SAFE_ANSWER, JUSTIFIED_REFUSAL, UNSAFE_COMPLIANCE, OVER_REFUSAL, AMBIGUOUS")


if __name__ == "__main__":
    main()

