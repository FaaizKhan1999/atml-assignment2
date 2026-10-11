from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json
from common.models import clear_gpu, load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def generate_math_responses(
    cfg: dict,
    policy_name: str,
    rows: list[dict],
    batch_size: int = 4,
) -> list[dict]:
    specs = policy_specs(cfg)
    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)

    max_new_tokens = int(cfg.get("math_max_new_tokens", 512))
    records = []

    print(f"Generating math responses for '{policy_name}' (adapter={adapter}, total={len(rows)})...")

    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        prompts = [prompt_messages(r) for r in chunk]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )

        for r, resp, n_tok in zip(chunk, gen["responses"], gen["response_lengths"]):
            gold = str(r.get("gold_final", "")).strip()
            pred = extract_designated_final(resp)
            acc = exact_reward(resp, gold)
            format_ok = pred is not None

            records.append({
                "source_index": r.get("source_index", r.get("prompt_id", "")),
                "question": r.get("question", ""),
                "gold_final": gold,
                "policy": policy_name,
                "response": resp,
                "response_tokens": int(n_tok),
                "extracted_final": pred,
                "format_compliance": bool(format_ok),
                "exact_accuracy": float(acc),
            })

        if (start + batch_size) % 40 == 0 or (start + len(chunk)) == len(rows):
            print(f"  Generated {min(start + len(chunk), len(rows))}/{len(rows)}")

    return records


def run_math_evaluation(
    config_path: str,
    dataset: str = "gsm",
    policy_filter: str = "all",
    batch_size: int = 4,
    skip_gen: bool = False,
    run_judge: bool = True,
):
    cfg, rows, _ = load_math_evaluation(config_path, dataset)
    outdir = repo_path(cfg.get("results_dir", "results")) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)

    policies = ["sft", "rlvr", "rlaif"] if policy_filter == "all" else [policy_filter]
    generated_map = {}

    print("=" * 75)
    print(f"TASK 5: MATH EVALUATION ON {dataset.upper()} ({len(rows)} problems)")
    print(f"Policies to evaluate: {policies}")
    print("=" * 75)

    # 1. Generate or load responses for each policy
    for pol in policies:
        gen_file = outdir / f"generated_{dataset}_{pol}.jsonl"
        if skip_gen and gen_file.exists():
            print(f"Loading existing generations for '{pol}' from {gen_file}...")
            records = read_jsonl(gen_file)
        else:
            adapter = policy_specs(cfg).get(pol)
            if adapter is not None:
                ad_path = repo_path(adapter)
                if not (ad_path / "adapter_config.json").exists() and not (ad_path / "config.json").exists():
                    print(f"[Warning] Checkpoint for '{pol}' not found at {ad_path}. Skipping generation.")
                    if gen_file.exists():
                        records = read_jsonl(gen_file)
                    else:
                        continue
                else:
                    records = generate_math_responses(cfg, pol, rows, batch_size=batch_size)
                    write_jsonl(gen_file, records)
                    clear_gpu()
            else:
                records = generate_math_responses(cfg, pol, rows, batch_size=batch_size)
                write_jsonl(gen_file, records)
                clear_gpu()

        generated_map[pol] = records

    # 2. Compute individual policy performance metrics
    metrics_by_policy = {}
    for pol, records in generated_map.items():
        if not records:
            continue
        accs = [r["exact_accuracy"] for r in records]
        formats = [1.0 if r["format_compliance"] else 0.0 for r in records]
        lengths = [float(r["response_tokens"]) for r in records]

        metrics_by_policy[pol] = {
            "num_evaluated": len(records),
            "exact_accuracy": float(np.mean(accs)),
            "format_compliance_rate": float(np.mean(formats)),
            "mean_response_length": float(np.mean(lengths)),
            "std_response_length": float(np.std(lengths)),
        }

    # 3. AI Pairwise Judge comparisons against SFT (if SFT and candidate exist)
    pairwise_against_sft = {}
    if run_judge and "sft" in generated_map and len(generated_map) > 1:
        sft_records = {r["source_index"]: r for r in generated_map["sft"]}
        cache_file = outdir / f"judge_cache_{dataset}.json"
        try:
            judge = PairwiseAIJudge(cfg, cache_path=cache_file)
            print(f"\nEvaluating pairwise AI judge comparisons against SFT using {cfg['ai_judge_model']}...")

            for pol in ["rlvr", "rlaif"]:
                if pol not in generated_map:
                    continue

                pol_records = {r["source_index"]: r for r in generated_map[pol]}
                common_ids = [k for k in pol_records if k in sft_records]

                wins = 0
                ties = 0
                losses = 0
                verifier_agreements = 0
                verifier_strict_pairs = 0

                for pid in common_ids:
                    q = pol_records[pid]["question"]
                    p_resp = pol_records[pid]["response"]
                    s_resp = sft_records[pid]["response"]
                    p_acc = pol_records[pid]["exact_accuracy"]
                    s_acc = sft_records[pid]["exact_accuracy"]

                    pref = judge.compare(problem=q, a=p_resp, b=s_resp)
                    if pref == "A":
                        wins += 1
                    elif pref == "B":
                        losses += 1
                    else:
                        ties += 1

                    # Verifier-judge agreement on strict preference pairs
                    if p_acc > s_acc:  # Verifier strictly prefers candidate
                        verifier_strict_pairs += 1
                        if pref == "A":
                            verifier_agreements += 1
                    elif s_acc > p_acc:  # Verifier strictly prefers SFT
                        verifier_strict_pairs += 1
                        if pref == "B":
                            verifier_agreements += 1

                total_comp = len(common_ids)
                win_rate = float((wins + 0.5 * ties) / total_comp) if total_comp > 0 else 0.0
                agreement_rate = (
                    float(verifier_agreements / verifier_strict_pairs)
                    if verifier_strict_pairs > 0
                    else 0.0
                )

                pairwise_against_sft[pol] = {
                    "total_comparisons": total_comp,
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "pairwise_win_rate_vs_sft": win_rate,
                    "verifier_strict_pairs": verifier_strict_pairs,
                    "verifier_judge_agreements": verifier_agreements,
                    "verifier_judge_agreement_rate": agreement_rate,
                }
            clear_gpu()
        except Exception as e:
            print(f"Warning: Could not run PairwiseAIJudge ({e}). Skipping pairwise comparisons.")

    # 4. Save consolidated results
    summary = {
        "dataset": dataset,
        "metrics_by_policy": metrics_by_policy,
        "pairwise_against_sft": pairwise_against_sft,
    }
    out_file = outdir / f"eval_{dataset}.json"
    save_json(out_file, summary)
    print(f"\nSaved {dataset.upper()} evaluation summary to {out_file}\n")

    # 5. Print summary table
    print("=" * 85)
    print(f"MATH EVALUATION SUMMARY: {dataset.upper()}")
    print("=" * 85)
    header = f"{'Policy':<10} | {'Exact Acc':<12} | {'Format %':<10} | {'Length (mean±std)':<18} | {'Win Rate vs SFT':<16} | {'Verifier Agree':<14}"
    print(header)
    print("-" * 85)
    for pol in ["sft", "rlvr", "rlaif"]:
        if pol not in metrics_by_policy:
            continue
        m = metrics_by_policy[pol]
        pw = pairwise_against_sft.get(pol, {})
        acc_str = f"{m['exact_accuracy']*100:.1f}%"
        fmt_str = f"{m['format_compliance_rate']*100:.1f}%"
        len_str = f"{m['mean_response_length']:.1f} ± {m['std_response_length']:.1f}"
        wr_str = f"{pw['pairwise_win_rate_vs_sft']*100:.1f}%" if "pairwise_win_rate_vs_sft" in pw else ("Baseline" if pol == "sft" else "N/A")
        ag_str = f"{pw['verifier_judge_agreement_rate']*100:.1f}%" if "verifier_judge_agreement_rate" in pw else "-"
        print(f"{pol.upper():<10} | {acc_str:<12} | {fmt_str:<10} | {len_str:<18} | {wr_str:<16} | {ag_str:<14}")
    print("=" * 85 + "\n")

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--policy", choices=["sft", "rlvr", "rlaif", "all"], default="all")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--skip-gen", action="store_true", help="Skip generation if jsonl exists")
    ap.add_argument("--skip-judge", action="store_true", help="Skip pairwise AI judge")
    args = ap.parse_args()

    run_math_evaluation(
        config_path=args.config,
        dataset=args.dataset,
        policy_filter=args.policy,
        batch_size=args.batch_size,
        skip_gen=args.skip_gen,
        run_judge=not args.skip_judge,
    )


if __name__ == "__main__":
    main()
