from __future__ import annotations

import argparse
import pandas as pd

from common.data import load_yaml, repo_path
from common.generation import batch_generate
from common.models import load_policy, load_tokenizer


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    df = load_xstest(cfg)
    records = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for (_, row), response, n_tok in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"]):
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": response,
                "response_tokens": int(n_tok),
            })
    return records


from common.data import load_yaml, repo_path, write_jsonl
from common.models import clear_gpu, load_policy, load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policy", choices=["sft", "dpo", "ppo", "grpo", "all"], default="all")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    outdir.mkdir(parents=True, exist_ok=True)

    policies = ["sft", "dpo", "ppo", "grpo"] if args.policy == "all" else [args.policy]
    specs = policy_specs(cfg)

    print("=" * 65)
    print("TASK 4 STEP 1: GENERATE DETERMINISTIC RESPONSES ON XSTEST")
    print(f"Policies to generate: {policies}")
    print(f"Output directory: {outdir}")
    print("=" * 65)

    for pol in policies:
        adapter = specs.get(pol)
        if adapter is not None:
            adapter_p = repo_path(adapter)
            if not (adapter_p / "adapter_config.json").exists() and not (adapter_p / "config.json").exists():
                print(f"\n[Warning] Adapter for '{pol}' not found at {adapter_p}. Skipping.")
                continue

        out_file = outdir / f"generated_{pol}.jsonl"
        print(f"\nGenerating responses for '{pol}' (adapter={adapter})...")
        records = generate_for_policy(cfg, pol, batch_size=args.batch_size)
        write_jsonl(out_file, records)
        print(f"Saved {len(records)} responses to {out_file}")
        clear_gpu()

    print("\nGeneration step complete.\n")


if __name__ == "__main__":
    main()

