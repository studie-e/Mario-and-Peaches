"""Plot learning curves for multiple Mario PPO runs.

 Example:
    python scripts/ve_do_thi.py runs/1-1/gamma09_seed1 runs/1-1/gamma99_seed1 \
        --output figures/so-sanh-gamma.png

Each run directory needs progress.csv; evals.csv is optional. Repeat --label
to name each run in the legend, in the same order as run directories.
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = (
    ("ep_return", "Lợi tức trung bình (100 lượt gần nhất)"),
    ("x_pos", "Vị trí x trung bình (100 lượt gần nhất)"),
    ("flag_rate", "Tỷ lệ chạm cờ (100 lượt gần nhất)"),
    ("entropy", "Entropy"),
    ("approx_kl", "Approx KL"),
    ("clipfrac", "Clip fraction"),
)


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "global_step" not in rows[0]:
        raise ValueError(f"{path} rỗng hoặc thiếu cột global_step")
    return rows


def series(rows, column):
    steps, values = [], []
    for row in rows:
        try:
            step = float(row["global_step"])
            value = float(row[column])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(step) and math.isfinite(value):
            steps.append(step)
            values.append(value)
    return steps, values


def mean_x_pos(rows):
    steps, values = [], []
    for row in rows:
        try:
            step = float(row["global_step"])
            positions = [float(item) for item in row["x_pos"].split()]
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        if positions and math.isfinite(step):
            steps.append(step)
            values.append(sum(positions) / len(positions))
    return steps, values


def plot(runs, labels, output):
    fig, axes = plt.subplots(4, 2, figsize=(15, 16), layout="constrained")
    for run, label in zip(runs, labels):
        progress = read_csv(run / "progress.csv")
        evaluations = read_csv(run / "evals.csv") if (run / "evals.csv").is_file() else []
        color = None
        for ax, (key, title) in zip(axes.flat, METRICS):
            steps, values = series(progress, key)
            if steps:
                line, = ax.plot(steps, values, label=label, linewidth=1.5)
                color = line.get_color()
            ax.set_title(title)
            ax.set_xlabel("Bước huấn luyện")
            ax.grid(alpha=0.25)
        for ax, (steps, values), title in (
            (axes[3, 0], series(evaluations, "flag_rate"), "Tỷ lệ chạm cờ (đánh giá 5 lượt)"),
            (axes[3, 1], mean_x_pos(evaluations), "Vị trí x trung bình (đánh giá 5 lượt)"),
        ):
            if steps:
                ax.plot(steps, values, marker="o", markersize=3, label=label,
                        color=color, linewidth=1.5)
            ax.set_title(title)
            ax.set_xlabel("Bước huấn luyện")
            ax.grid(alpha=0.25)
    for ax in axes.flat:
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize="small")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="Run directories")
    parser.add_argument("--label", action="append", default=[], help="Legend label, in run order")
    parser.add_argument("--output", type=Path, default=Path("figures/duong-cong-hoc.png"))
    args = parser.parse_args(argv)
    if args.label and len(args.label) != len(args.runs):
        parser.error("Number of --label values must match number of runs")
    labels = args.label or [str(run) for run in args.runs]
    print(f"Saved {plot(args.runs, labels, args.output)}")


if __name__ == "__main__":
    main()
