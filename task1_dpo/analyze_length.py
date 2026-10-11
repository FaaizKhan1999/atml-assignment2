from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate
from common.logging_utils import save_json
from common.metrics import parse_word_limit, word_count, word_limit_compliance
from common.models import clear_gpu, load_policy, load_tokenizer
from task1_dpo.evaluate import evaluate_pairs
from task1_dpo.train import run_training


def evaluate_word_limits(policy, tokenizer, word_limit_rows: list[dict], cfg: dict) -> dict:
    prompts = [r["messages"] for r in word_limit_rows]
    gen_cfg = cfg.get("generation", {})
    temperature = float(gen_cfg.get("temperature", 0.7))
    top_p = float(gen_cfg.get("top_p", 0.9))
    do_sample = bool(gen_cfg.get("do_sample", True))
    max_gen_tokens = int(cfg.get("max_generation_tokens", 256))
    max_seq_len = int(cfg.get("max_sequence_length", 768))
    max_prompt_len = max(32, max_seq_len - max_gen_tokens)

    gen_out = batch_generate(
        policy,
        tokenizer,
        prompts,
        max_prompt_length=max_prompt_len,
        max_new_tokens=max_gen_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
    )

    compliances = []
    word_counts = []
    records = []

    for row, response, token_len in zip(word_limit_rows, gen_out["responses"], gen_out["response_lengths"]):
        prompt_text = row["messages"][-1]["content"]
        limit = parse_word_limit(prompt_text)
        w_count = word_count(response)
        comp = word_limit_compliance(prompt_text, response)
        compliances.append(comp if comp is not None else 0.0)
        word_counts.append(w_count)
        records.append({
            "prompt_id": row.get("prompt_id"),
            "prompt": prompt_text,
            "target_limit": limit,
            "actual_word_count": w_count,
            "token_length": int(token_len),
            "compliant": bool(comp == 1.0) if comp is not None else False,
            "response": response,
        })

    return {
        "compliance_rate": float(np.mean(compliances)) if compliances else 0.0,
        "mean_word_count": float(np.mean(word_counts)) if word_counts else 0.0,
        "mean_token_length": float(np.mean(gen_out["response_lengths"])) if gen_out["response_lengths"] else 0.0,
        "records": records,
    }


def run_length_analysis(config_path: str, skip_train: bool = False):
    cfg = load_yaml(config_path)
    results_dir = repo_path(cfg.get("results_dir", "results/task1_dpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    length_train_path = cfg["paths"]["dpo_length_train"]
    stratified_eval_path = cfg["paths"]["dpo_length_eval"]
    word_limit_path = cfg["paths"]["word_limit_prompts"]
    standard_output = cfg["standard_output"]
    length_output = cfg["length_output"]

    stratified_rows = read_jsonl(stratified_eval_path)
    word_limit_rows = read_jsonl(word_limit_path)

    print("=" * 65)
    print("TASK 1 STEP 3: LENGTH-CONFOUNDING STUDY")
    print(f"Length-balanced train dataset: {length_train_path}")
    print(f"Length-stratified eval dataset: {stratified_eval_path} ({len(stratified_rows)} pairs)")
    print(f"Word-limit prompts: {word_limit_path} ({len(word_limit_rows)} prompts)")
    print("=" * 65)

    # 1. Train length-balanced DPO model if needed
    len_adapter_dir = repo_path(length_output)
    if not skip_train or not (len_adapter_dir / "adapter_config.json").exists():
        print("\n[1/3] Training length-balanced DPO model for 1 epoch...")
        run_training(
            config_path=config_path,
            run_name="length_balanced",
            dataset_path=length_train_path,
            output_path=length_output,
        )
        clear_gpu()
    else:
        print(f"\nFound existing length-balanced checkpoint at {length_output}; skipping training.")

    tokenizer = load_tokenizer(cfg["base_model"])
    beta = float(cfg.get("beta", 0.10))
    max_len = int(cfg.get("max_sequence_length", 768))

    strata_names = ["preferred_longer", "length_matched", "rejected_longer"]
    models_to_eval = [
        ("standard", standard_output),
        ("length_balanced", length_output),
    ]

    analysis_results = {
        "stratified_eval": {},
        "word_limit_eval": {},
    }

    # 2. Evaluate both models on length-stratified eval pairs
    print("\n[2/3] Evaluating standard vs length-balanced DPO across strata...")
    for model_name, adapter_path in models_to_eval:
        print(f"\n--- Loading and evaluating model: {model_name} ({adapter_path}) ---")
        policy = load_policy(cfg, adapter_path=adapter_path, trainable=False)

        # Overall evaluation
        overall_metrics = evaluate_pairs(policy, tokenizer, stratified_rows, beta, max_length=max_len)

        # Per-stratum evaluation
        strata_metrics = {}
        for stratum in strata_names:
            stratum_rows = [r for r in stratified_rows if r.get("length_stratum") == stratum]
            s_res = evaluate_pairs(policy, tokenizer, stratum_rows, beta, max_length=max_len)
            strata_metrics[stratum] = {
                "num_pairs": len(stratum_rows),
                "preference_accuracy": s_res["preference_accuracy"],
                "dpo_loss": s_res["dpo_loss"],
                "margin_mean": s_res["margin_mean"],
            }
            print(f"  [{model_name}] Stratum '{stratum}': Acc = {s_res['preference_accuracy']*100:.2f}%, Margin = {s_res['margin_mean']:.4f}")

        analysis_results["stratified_eval"][model_name] = {
            "overall_accuracy": overall_metrics["preference_accuracy"],
            "overall_loss": overall_metrics["dpo_loss"],
            "overall_margin": overall_metrics["margin_mean"],
            "strata": strata_metrics,
        }

        # 3. Evaluate word-limit compliance
        print(f"  [{model_name}] Evaluating word-limit compliance on {len(word_limit_rows)} prompts...")
        wl_res = evaluate_word_limits(policy, tokenizer, word_limit_rows, cfg)
        analysis_results["word_limit_eval"][model_name] = wl_res
        print(f"  [{model_name}] Word-limit compliance rate: {wl_res['compliance_rate']*100:.2f}%, Mean words: {wl_res['mean_word_count']:.1f}")

        clear_gpu()

    # Save results to JSON
    out_file = results_dir / "length_analysis.json"
    save_json(out_file, analysis_results)
    print(f"\nLength analysis results saved to: {out_file}")

    # Print summary table
    print("\n" + "=" * 80)
    print("LENGTH-CONFOUNDING STUDY SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Model':<16} | {'Overall Acc':<11} | {'Pref-Longer':<11} | {'Matched':<11} | {'Rej-Longer':<11} | {'Word-Limit Comp':<15}"
    print(header)
    print("-" * 80)
    for model_name, _ in models_to_eval:
        s_eval = analysis_results["stratified_eval"][model_name]
        wl_eval = analysis_results["word_limit_eval"][model_name]
        ov_acc = f"{s_eval['overall_accuracy']*100:.2f}%"
        pl_acc = f"{s_eval['strata']['preferred_longer']['preference_accuracy']*100:.2f}%"
        mat_acc = f"{s_eval['strata']['length_matched']['preference_accuracy']*100:.2f}%"
        rl_acc = f"{s_eval['strata']['rejected_longer']['preference_accuracy']*100:.2f}%"
        wl_comp = f"{wl_eval['compliance_rate']*100:.2f}%"
        print(f"{model_name:<16} | {ov_acc:<11} | {pl_acc:<11} | {mat_acc:<11} | {rl_acc:<11} | {wl_comp:<15}")
    print("=" * 80 + "\n")

    return analysis_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--skip-train", action="store_true", help="Skip training if length-balanced checkpoint already exists.")
    args = ap.parse_args()
    run_length_analysis(args.config, skip_train=args.skip_train)


if __name__ == "__main__":
    main()
