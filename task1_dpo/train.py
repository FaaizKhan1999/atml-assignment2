from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import save_json, set_seed
from common.models import load_policy, load_tokenizer, reference_mode, trainable_parameters
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    loader = bundle["loader"]
    optimizer = bundle["optimizer"]
    beta_val = bundle["beta"]

    output = repo_path(output_path or cfg["standard_output"])
    output.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    epochs = int(cfg.get("epochs", 1))
    grad_accum_steps = int(cfg.get("grad_accum_steps", 1))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))

    history: list[dict[str, float | int]] = []
    total_steps = len(loader)
    optimizer.zero_grad()

    print(f"Starting DPO training: run_name={run_name}, epochs={epochs}, steps_per_epoch={total_steps}, beta={beta_val}")

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0

        for step, (batch_chosen, batch_rejected) in enumerate(loader):
            batch_chosen = {k: v.to(device) for k, v in batch_chosen.items()}
            batch_rejected = {k: v.to(device) for k, v in batch_rejected.items()}

            with reference_mode(model):
                with torch.no_grad():
                    ref_chosen_logp, _, _ = response_sequence_logprobs(model, batch_chosen)
                    ref_rejected_logp, _, _ = response_sequence_logprobs(model, batch_rejected)

            policy_chosen_logp, _, _ = response_sequence_logprobs(model, batch_chosen)
            policy_rejected_logp, _, _ = response_sequence_logprobs(model, batch_rejected)

            loss, metrics = dpo_loss(
                policy_chosen_logp=policy_chosen_logp,
                policy_rejected_logp=policy_rejected_logp,
                ref_chosen_logp=ref_chosen_logp,
                ref_rejected_logp=ref_rejected_logp,
                beta=beta_val,
            )

            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == total_steps:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            step_loss = float(loss.item())
            step_acc = float(metrics["preference_accuracy"].item())
            epoch_loss += step_loss
            epoch_acc += step_acc

            history.append({
                "epoch": epoch + 1,
                "step": step + 1,
                "loss": step_loss,
                "preference_accuracy": step_acc,
                "policy_margin": float(metrics["policy_margin_mean"].item()),
            })

            if (step + 1) % 10 == 0 or (step + 1) == total_steps:
                print(
                    f"Epoch {epoch + 1}/{epochs} | Step {step + 1}/{total_steps} | "
                    f"Loss: {step_loss:.4f} | Preference Acc: {step_acc:.4f}"
                )

        mean_epoch_loss = epoch_loss / max(1, total_steps)
        mean_epoch_acc = epoch_acc / max(1, total_steps)
        print(f"Epoch {epoch + 1} finished | Mean Loss: {mean_epoch_loss:.4f} | Mean Acc: {mean_epoch_acc:.4f}")

    print(f"Saving policy adapter and tokenizer to {output}...")
    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))

    results_dir = repo_path(cfg.get("results_dir", "results/task1_dpo"))
    results_dir.mkdir(parents=True, exist_ok=True)
    save_json(results_dir / f"train_{run_name}.json", {
        "run_name": run_name,
        "epochs": epochs,
        "beta": beta_val,
        "history": history,
        "final_loss": history[-1]["loss"] if history else None,
        "final_accuracy": history[-1]["preference_accuracy"] if history else None,
    })

    return {
        "output_dir": str(output),
        "history": history,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
