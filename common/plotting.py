"""Plotting utilities for ATML Assignment 2 report figures.
Reads saved machine-readable results and generates publication-quality PDF figures in report/figs/.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set clean publication style
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "lines.linewidth": 1.75,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

FIG_DIR = Path("report/figs")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_task2_ppo_trajectories():
    """Plot PPO continuation dynamics over 20 updates."""
    with open("results/task2_ppo/train_standard.json") as f:
        data = json.load(f)
    history = data["history"]
    updates = [h["update"] for h in history]
    rewards = [h["reward"] for h in history]
    kls = [h["kl"] * 1e4 for h in history]  # scale to 1e-4 nats
    pol_loss = [h["policy_loss"] for h in history]
    val_loss = [h["value_loss"] for h in history]
    entropy = [h["entropy"] for h in history]
    pol_grad = [h["policy_grad_norm"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True)

    # (a) Reward and KL
    ax1 = axes[0, 0]
    color = "#1f77b4"
    ax1.plot(updates, rewards, color=color, marker="o", markersize=3, label="Reward")
    ax1.set_ylabel("Reward Score", color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_title("(a) Policy Reward & Reference Drift")

    ax1_twin = ax1.twinx()
    color_twin = "#d62728"
    ax1_twin.plot(updates, kls, color=color_twin, linestyle="--", marker="s", markersize=3, label=r"Token KL ($\times 10^{-4}$)")
    ax1_twin.set_ylabel(r"KL ($\times 10^{-4}$ nats)", color=color_twin)
    ax1_twin.tick_params(axis="y", labelcolor=color_twin)
    ax1_twin.grid(False)

    # (b) Policy & Value Loss
    ax2 = axes[0, 1]
    ax2.plot(updates, pol_loss, color="#2ca02c", marker="^", markersize=3, label="Policy Loss")
    ax2.set_ylabel("Policy Loss", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.set_title("(b) Loss Trajectories")

    ax2_twin = ax2.twinx()
    ax2_twin.plot(updates, val_loss, color="#9467bd", linestyle="--", marker="d", markersize=3, label="Value Loss")
    ax2_twin.set_ylabel("Value Critic MSE", color="#9467bd")
    ax2_twin.tick_params(axis="y", labelcolor="#9467bd")
    ax2_twin.grid(False)

    # (c) Entropy
    ax3 = axes[1, 0]
    ax3.plot(updates, entropy, color="#e377c2", marker="o", markersize=3)
    ax3.set_ylabel("Token Policy Entropy")
    ax3.set_xlabel("Continuation Update Step")
    ax3.set_title("(c) Policy Entropy Dynamics")

    # (d) Policy Gradient Norm
    ax4 = axes[1, 1]
    ax4.plot(updates, pol_grad, color="#ff7f0e", marker="s", markersize=3)
    ax4.set_ylabel(r"$\|\nabla_\theta \mathcal{L}\|_2$")
    ax4.set_xlabel("Continuation Update Step")
    ax4.set_title("(d) Policy Gradient Norm")

    plt.tight_layout()
    out_path = FIG_DIR / "task2_ppo_trajectories.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_task3_grpo_informativeness():
    """Plot GRPO group informativeness across group size K and difficulty bins."""
    with open("results/task3_grpo/group_size_analysis.json") as f:
        data = json.load(f)

    k_vals = [r["group_size_k"] for r in data["results_by_k"]]
    overall_info = [r["overall"]["informative_rate"] * 100 for r in data["results_by_k"]]
    easy_info = [r["easy_bin"]["informative_rate"] * 100 for r in data["results_by_k"]]
    hard_info = [r["hard_bin"]["informative_rate"] * 100 for r in data["results_by_k"]]

    overall_std = [r["overall"]["mean_reward_std"] for r in data["results_by_k"]]
    easy_std = [r["easy_bin"]["mean_reward_std"] for r in data["results_by_k"]]
    hard_std = [r["hard_bin"]["mean_reward_std"] for r in data["results_by_k"]]

    x = np.arange(len(k_vals))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8))

    # (a) Informative Rate
    ax1.bar(x - width, overall_info, width, label="Overall", color="#1f77b4")
    ax1.bar(x, easy_info, width, label="Easy Prompts", color="#2ca02c")
    ax1.bar(x + width, hard_info, width, label="Hard Prompts", color="#d62728")
    ax1.set_ylabel("Informative Groups (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"K={k}" for k in k_vals])
    ax1.set_ylim(80, 105)
    ax1.set_title("(a) Informative Group Rate (%)")
    ax1.legend(loc="lower left", framealpha=0.8)

    # (b) Within-group Std
    ax2.bar(x - width, overall_std, width, label="Overall", color="#1f77b4")
    ax2.bar(x, easy_std, width, label="Easy Prompts", color="#2ca02c")
    ax2.bar(x + width, hard_std, width, label="Hard Prompts", color="#d62728")
    ax2.set_ylabel(r"Within-Group Reward Std ($\bar{s}_r$)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"K={k}" for k in k_vals])
    ax2.set_title(r"(b) Group Reward Spread ($\bar{s}_r$)")
    ax2.legend(loc="upper left", framealpha=0.8)

    plt.tight_layout()
    out_path = FIG_DIR / "task3_grpo_informativeness.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_task5_perturbations():
    """Plot Exact Verifier vs Pairwise AI Judge on controlled diagnostic perturbations."""
    with open("results/task5_feedback/perturbation_scores.json") as f:
        data = json.load(f)

    cats = ["reasoning_corruption", "outcome_error", "persuasive_filler", "gold_distractor"]
    labels = ["Reasoning\nCorruption", "Outcome\nError", "Persuasive\nFiller", "Gold\nDistractor"]

    v_break = data["exact_verifier_breakdown"]
    j_break = data["ai_judge_breakdown"]

    v_better = [v_break[c]["better_response_rate"] * 100 for c in cats]
    v_tie = [v_break[c]["tie_rate"] * 100 for c in cats]
    v_wrong = [v_break[c]["wrong_preference_rate"] * 100 for c in cats]

    j_better = [j_break[c]["better_response_rate"] * 100 for c in cats]
    j_tie = [j_break[c]["tie_rate"] * 100 for c in cats]
    j_wrong = [j_break[c]["wrong_preference_rate"] * 100 for c in cats]

    x = np.arange(len(cats))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)

    # Verifier
    ax1.bar(x, v_better, width=0.5, label="Preferred Clean", color="#2ca02c")
    ax1.bar(x, v_tie, width=0.5, bottom=v_better, label="Tied", color="#7f7f7f")
    ax1.bar(x, v_wrong, width=0.5, bottom=np.array(v_better) + np.array(v_tie), label="Preferred Corrupted", color="#d62728")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Outcome Distribution (%)")
    ax1.set_title("Exact Verifier (Rule-Based)")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)

    # AI Judge
    ax2.bar(x, j_better, width=0.5, label="Preferred Clean", color="#2ca02c")
    ax2.bar(x, j_tie, width=0.5, bottom=j_better, label="Tied", color="#7f7f7f")
    ax2.bar(x, j_wrong, width=0.5, bottom=np.array(j_better) + np.array(j_tie), label="Preferred Corrupted", color="#d62728")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_title("Pairwise AI Judge (3B LLM)")

    plt.tight_layout()
    out_path = FIG_DIR / "task5_perturbations.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    plot_task2_ppo_trajectories()
    plot_task3_grpo_informativeness()
    plot_task5_perturbations()
    print("All report figures successfully generated.")


if __name__ == "__main__":
    main()
