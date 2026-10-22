"""Plot training rewards and periodic evaluation success rates."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return np.full_like(values, np.nan, dtype=np.float64)
    result = np.full(len(values), np.nan, dtype=np.float64)
    result[window - 1 :] = np.convolve(
        values, np.ones(window) / window, mode="valid"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    training_rows = read_csv(args.run_directory / "train_log.csv")
    episodes = np.array([int(row["episode"]) for row in training_rows])
    rewards = np.array([float(row["reward"]) for row in training_rows])

    figure, axes = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)
    axes[0].plot(episodes, rewards, alpha=0.25, label="Episode reward")
    axes[0].plot(
        episodes,
        moving_average(rewards, args.window),
        label=f"{args.window}-episode average",
    )
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Training reward")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    evaluation_path = args.run_directory / "evaluation_log.csv"
    if evaluation_path.exists():
        evaluation_rows = read_csv(evaluation_path)
        steps = [int(row["total_steps"]) for row in evaluation_rows]
        success_rates = [float(row["success_rate"]) for row in evaluation_rows]
        axes[1].plot(steps, success_rates, marker="o")
        axes[1].set_ylim(0.0, 1.0)
    else:
        axes[1].text(0.5, 0.5, "No periodic evaluation data", ha="center")
    axes[1].set_xlabel("Environment steps")
    axes[1].set_ylabel("Success rate")
    axes[1].grid(alpha=0.3)

    output = args.output or args.run_directory / "training_result.png"
    figure.savefig(output, dpi=160)
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    main()
