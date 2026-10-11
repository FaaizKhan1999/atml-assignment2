from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import (
    batch_generate,
    response_token_logprobs,
    score_reward_pairs,
)
from common.logging_utils import save_json
from common.metrics import sample_entropy, sampled_kl
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def run_evaluation(
    config_path: str,
    adapter: str,
    name: str = "standard",
    dataset_path: str | None = None,
    batch_size: int = 4,
    max_samples: int = 10,
) -> dict:
    cfg = load_yaml(config_path)
    eval_path = dataset_path or cfg["paths"]["rl_prompt_eval"]
    rows = read_jsonl(eval_path)
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)

    try:
        reward_model, reward_tokenizer = load_reward_model(cfg)
    except Exception as e:
        print(f"Warning: Could not load reward model ({e}). Skipping reward scoring.")
        reward_model, reward_tokenizer = None, None

    gen_cfg = cfg.get("generation", {})
    temperature = float(gen_cfg.get("temperature", 0.7))
    top_p = float(gen_cfg.get("top_p", 0.9))
    do_sample = bool(gen_cfg.get("do_sample", True))
    max_comp_len = int(cfg.get("cache_generation_cap", 768))
    max_prompt_len = int(cfg.get("max_prompt_length", 256))

    prompts = [prompt_messages(r) for r in rows]
    all_responses = []
    all_lengths = []
    all_rewards = []
    all_kls = []
    all_entropies = []

    print(f"Evaluating GRPO policy '{name}' on {len(prompts)} prompts from {eval_path}...")

    for i in range(0, len(prompts), batch_size):
        batch_p = prompts[i : i + batch_size]
        gen_out = batch_generate(
            policy,
            tokenizer,
            batch_p,
            max_prompt_length=max_prompt_len,
            max_new_tokens=max_comp_len,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )

        with torch.no_grad():
            pol_tok_logp, _ = response_token_logprobs(
                policy,
                gen_out["sequences"],
                gen_out["attention_mask"],
                gen_out["prompt_width"],
                gen_out["response_ids"],
            )

        with reference_mode(policy):
            with torch.no_grad():
                ref_tok_logp, _ = response_token_logprobs(
                    policy,
                    gen_out["sequences"],
                    gen_out["attention_mask"],
                    gen_out["prompt_width"],
                    gen_out["response_ids"],
                )

        for b_idx in range(len(batch_p)):
            mask_b = gen_out["response_mask"][b_idx : b_idx + 1]
            pol_b = pol_tok_logp[b_idx : b_idx + 1]
            ref_b = ref_tok_logp[b_idx : b_idx + 1]

            kl_b = sampled_kl(pol_b, ref_b, mask_b)
            ent_b = sample_entropy(pol_b, mask_b)
            all_kls.append(float(kl_b.item()))
            all_entropies.append(float(ent_b.item()))

        all_lengths.extend([int(l) for l in gen_out["response_lengths"]])
        all_responses.extend(gen_out["responses"])

        if reward_model is not None:
            with torch.no_grad():
                rm_scores = score_reward_pairs(
                    reward_model,
                    reward_tokenizer,
                    batch_p,
                    gen_out["responses"],
                )
            all_rewards.extend([float(s.item()) for s in rm_scores])

    mean_reward = float(np.mean(all_rewards)) if all_rewards else None
    mean_kl = float(np.mean(all_kls)) if all_kls else 0.0
    mean_entropy = float(np.mean(all_entropies)) if all_entropies else 0.0
    mean_length = float(np.mean(all_lengths)) if all_lengths else 0.0
    std_length = float(np.std(all_lengths)) if all_lengths else 0.0
    iqr_length = float(np.percentile(all_lengths, 75) - np.percentile(all_lengths, 25)) if all_lengths else 0.0

    samples = []
    for i in range(min(max_samples, len(prompts))):
        s = {
            "prompt": prompts[i],
            "response": all_responses[i],
            "length": all_lengths[i],
            "kl": all_kls[i],
            "entropy": all_entropies[i],
        }
        if all_rewards:
            s["reward"] = all_rewards[i]
        samples.append(s)

    results = {
        "name": name,
        "adapter": str(adapter),
        "dataset": str(eval_path),
        "num_examples": len(prompts),
        "reward_score_mean": mean_reward,
        "kl_from_reference": mean_kl,
        "entropy_mean": mean_entropy,
        "response_length_mean": mean_length,
        "response_length_std": std_length,
        "response_length_iqr": iqr_length,
        "qualitative_samples": samples,
    }

    results_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"eval_{name}.json"
    save_json(out_file, results)
    print(f"Results saved to {out_file}")

    print("\n" + "=" * 55)
    print(f"GRPO EVALUATION SUMMARY: {name}")
    if mean_reward is not None:
        print(f"  Mean Reward Score:   {mean_reward:+.4f}")
    print(f"  Sampled KL (ref):    {mean_kl:.4f}")
    print(f"  Policy Entropy:      {mean_entropy:.4f}")
    print(f"  Length (mean±std):   {mean_length:.1f} ± {std_length:.1f} (IQR: {iqr_length:.1f})")
    print("=" * 55 + "\n")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    run_evaluation(args.config, args.adapter, args.name)


if __name__ == "__main__":
    main()
