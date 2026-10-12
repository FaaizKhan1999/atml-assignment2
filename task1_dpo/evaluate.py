from __future__ import annotations

import argparse
import numpy as np
import torch
import torch.nn.functional as F

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import (
    batch_generate,
    response_sequence_logprobs,
    response_token_logprobs,
    score_reward_pairs,
)
from common.logging_utils import save_json
from common.metrics import sampled_kl
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["dpo_standard_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def evaluate_pairs(
    policy,
    tokenizer,
    rows: list[dict],
    beta: float,
    max_length: int = 768,
    batch_size: int = 4,
) -> dict:
    device = next(policy.parameters()).device
    all_margins = []
    all_accuracies = []
    all_losses = []
    all_pol_margins = []
    all_ref_margins = []

    # Filter out examples where prompt alone exceeds max_length
    valid_rows = []
    for r in rows:
        prompt = prompt_messages_from_preference(r)
        p_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
        if len(p_ids) < max_length - 8:
            valid_rows.append(r)

    if len(valid_rows) < len(rows):
        print(f"Filtered {len(rows) - len(valid_rows)}/{len(rows)} eval examples where prompt >= {max_length - 8} tokens.")

    for i in range(0, len(valid_rows), batch_size):
        batch_rows = valid_rows[i : i + batch_size]
        chosen_exs = []
        rejected_exs = []
        for r in batch_rows:
            prompt = prompt_messages_from_preference(r)
            yc, yr = preference_responses(r)
            chosen_exs.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected_exs.append(encode_prompt_response(tokenizer, prompt, yr, max_length))

        batch_c = pad_batch(tokenizer, chosen_exs)
        batch_r = pad_batch(tokenizer, rejected_exs)
        batch_c = {k: v.to(device) for k, v in batch_c.items()}
        batch_r = {k: v.to(device) for k, v in batch_r.items()}

        with reference_mode(policy):
            with torch.no_grad():
                ref_c_logp, _, _ = response_sequence_logprobs(policy, batch_c)
                ref_r_logp, _, _ = response_sequence_logprobs(policy, batch_r)

        with torch.no_grad():
            pol_c_logp, _, _ = response_sequence_logprobs(policy, batch_c)
            pol_r_logp, _, _ = response_sequence_logprobs(policy, batch_r)

        pol_margin = pol_c_logp - pol_r_logp
        ref_margin = ref_c_logp - ref_r_logp
        dpo_margin = pol_margin - ref_margin
        loss = -F.logsigmoid(beta * dpo_margin)
        acc = (dpo_margin > 0).float()

        all_margins.extend(dpo_margin.cpu().tolist())
        all_accuracies.extend(acc.cpu().tolist())
        all_losses.extend(loss.cpu().tolist())
        all_pol_margins.extend(pol_margin.cpu().tolist())
        all_ref_margins.extend(ref_margin.cpu().tolist())

    return {
        "dpo_loss": float(np.mean(all_losses)) if all_losses else 0.0,
        "preference_accuracy": float(np.mean(all_accuracies)) if all_accuracies else 0.0,
        "margin_mean": float(np.mean(all_margins)) if all_margins else 0.0,
        "policy_margin_mean": float(np.mean(all_pol_margins)) if all_pol_margins else 0.0,
        "ref_margin_mean": float(np.mean(all_ref_margins)) if all_ref_margins else 0.0,
        "sample_margins": all_margins,
    }


def evaluate_generation(
    policy,
    tokenizer,
    reward_bundle,
    prompts: list[list[dict]],
    cfg: dict,
    batch_size: int = 4,
    max_samples: int = 10,
) -> tuple[dict, list[dict]]:
    gen_cfg = cfg.get("generation", {})
    temperature = float(gen_cfg.get("temperature", 0.7))
    top_p = float(gen_cfg.get("top_p", 0.9))
    do_sample = bool(gen_cfg.get("do_sample", True))
    max_gen_tokens = int(cfg.get("max_generation_tokens", 256))
    max_seq_len = int(cfg.get("max_sequence_length", 768))
    max_prompt_len = max(32, max_seq_len - max_gen_tokens)

    all_responses = []
    all_lengths = []
    all_kls = []
    all_rewards = []

    rm_model, rm_tok = reward_bundle if reward_bundle is not None else (None, None)

    for i in range(0, len(prompts), batch_size):
        batch_p = prompts[i : i + batch_size]
        gen_out = batch_generate(
            policy,
            tokenizer,
            batch_p,
            max_prompt_length=max_prompt_len,
            max_new_tokens=max_gen_tokens,
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
            mask_b = gen_out["response_mask"][b_idx]
            kl_b = sampled_kl(
                pol_tok_logp[b_idx : b_idx + 1],
                ref_tok_logp[b_idx : b_idx + 1],
                mask_b.unsqueeze(0),
            )
            all_kls.append(float(kl_b.item()))

        all_lengths.extend([int(l) for l in gen_out["response_lengths"]])
        all_responses.extend(gen_out["responses"])

        if rm_model is not None:
            rm_scores = score_reward_pairs(rm_model, rm_tok, batch_p, gen_out["responses"])
            all_rewards.extend([float(s.item()) for s in rm_scores])

    length_mean = float(np.mean(all_lengths)) if all_lengths else 0.0
    length_std = float(np.std(all_lengths)) if all_lengths else 0.0
    length_iqr = float(np.percentile(all_lengths, 75) - np.percentile(all_lengths, 25)) if all_lengths else 0.0
    kl_mean = float(np.mean(all_kls)) if all_kls else 0.0
    reward_mean = float(np.mean(all_rewards)) if all_rewards else None

    samples = []
    for i in range(min(max_samples, len(prompts))):
        s = {
            "prompt": prompts[i],
            "response": all_responses[i],
            "length": all_lengths[i],
            "kl": all_kls[i],
        }
        if all_rewards:
            s["reward"] = all_rewards[i]
        samples.append(s)

    metrics = {
        "kl_mean": kl_mean,
        "reward_mean": reward_mean,
        "length_mean": length_mean,
        "length_std": length_std,
        "length_iqr": length_iqr,
    }
    return metrics, samples


def run_evaluation(
    config_path: str,
    adapter: str,
    name: str = "standard",
    dataset_path: str | None = None,
    eval_generation_prompts: bool = True,
    beta: float | None = None,
) -> dict:
    cfg = load_yaml(config_path)
    eval_path = dataset_path or cfg["paths"]["dpo_standard_eval"]
    rows = read_jsonl(eval_path)
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)

    beta_val = beta if beta is not None else float(cfg.get("beta", 0.10))

    reward_bundle = None
    if eval_generation_prompts:
        try:
            reward_bundle = load_reward_model(cfg)
        except Exception as e:
            print(f"Warning: Could not load reward model ({e}). Skipping reward model scoring.")

    print(f"Evaluating {name} on {len(rows)} pairs from {eval_path} (beta={beta_val})...")
    pair_metrics = evaluate_pairs(
        policy=policy,
        tokenizer=tokenizer,
        rows=rows,
        beta=beta_val,
        max_length=int(cfg.get("max_sequence_length", 768)),
        batch_size=int(cfg.get("batch_size", 2)),
    )

    gen_metrics = {}
    samples = []
    if eval_generation_prompts:
        print(f"Evaluating generation, KL, and RM scores for {name}...")
        prompts = [prompt_messages_from_preference(r) for r in rows]
        gen_metrics, samples = evaluate_generation(
            policy=policy,
            tokenizer=tokenizer,
            reward_bundle=reward_bundle,
            prompts=prompts,
            cfg=cfg,
            batch_size=int(cfg.get("batch_size", 2)),
        )

    results = {
        "name": name,
        "adapter": str(adapter),
        "dataset": str(eval_path),
        "dpo_loss": pair_metrics["dpo_loss"],
        "preference_accuracy": pair_metrics["preference_accuracy"],
        "margin_mean": pair_metrics["margin_mean"],
        "policy_margin_mean": pair_metrics["policy_margin_mean"],
        "ref_margin_mean": pair_metrics["ref_margin_mean"],
        "kl_from_reference": gen_metrics.get("kl_mean"),
        "reward_score_mean": gen_metrics.get("reward_mean"),
        "response_length_mean": gen_metrics.get("length_mean"),
        "response_length_std": gen_metrics.get("length_std"),
        "response_length_iqr": gen_metrics.get("length_iqr"),
        "qualitative_samples": samples,
    }

    results_dir = repo_path(cfg.get("results_dir", "results/task1_dpo"))
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"eval_{name}.json"
    save_json(out_file, results)
    print(f"Results saved to {out_file}")

    print("\n" + "=" * 50)
    print(f"EVALUATION SUMMARY: {name}")
    print(f"  Preference Accuracy: {pair_metrics['preference_accuracy'] * 100:.2f}%")
    print(f"  DPO Loss:            {pair_metrics['dpo_loss']:.4f}")
    if gen_metrics:
        print(f"  Sampled KL:          {gen_metrics.get('kl_mean', 0.0):.4f}")
        if gen_metrics.get("reward_mean") is not None:
            print(f"  Reward Score:        {gen_metrics.get('reward_mean'):.4f}")
        print(f"  Length (mean±std):   {gen_metrics['length_mean']:.1f} ± {gen_metrics['length_std']:.1f} (IQR: {gen_metrics['length_iqr']:.1f})")
    print("=" * 50 + "\n")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--beta", type=float, default=None)
    args = ap.parse_args()
    run_evaluation(args.config, args.adapter, args.name, beta=args.beta)


if __name__ == "__main__":
    main()
