from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from longtail.config import load_config
from longtail.data import img_num_per_cls
from longtail.metrics import class_split_indices


def plot_distribution(
    train_counts: list[int],
    test_counts: list[int],
    out_path: Path,
    title: str,
) -> None:
    groups = class_split_indices(train_counts)
    colors = ["#4575b4"] * len(train_counts)
    for idx in groups["medium"]:
        colors[idx] = "#91bfdb"
    for idx in groups["few"]:
        colors[idx] = "#d73027"

    fig, ax1 = plt.subplots(figsize=(12, 4.8))
    ax1.bar(range(len(train_counts)), train_counts, color=colors, label="Train count")
    ax1.plot(range(len(test_counts)), test_counts, color="#fdae61", linewidth=2, label="Imbalanced test count")
    ax1.set_title(title)
    ax1.set_xlabel("Class index sorted by training frequency")
    ax1.set_ylabel("Number of images")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(loc="upper right")
    ax1.text(
        0.01,
        0.92,
        f"many={len(groups['many'])}, medium={len(groups['medium'])}, few={len(groups['few'])}",
        transform=ax1.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85},
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def save_counts_csv(train_counts: list[int], test_counts: list[int], out_path: Path) -> None:
    groups = class_split_indices(train_counts)
    group_map = {}
    for name, indices in groups.items():
        for idx in indices:
            group_map[idx] = name
    pd.DataFrame(
        {
            "class_id": list(range(len(train_counts))),
            "train_count": train_counts,
            "imbalanced_test_count": test_counts,
            "group": [group_map[i] for i in range(len(train_counts))],
        }
    ).to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--all_protocols", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    num_classes = int(data_cfg.get("num_classes", 100))
    max_train = int(data_cfg.get("max_train_per_class", 500))
    max_test = int(data_cfg.get("max_test_per_class", 100))
    ratio = float(data_cfg.get("imbalance_ratio", 100))
    out_dir = Path(args.out_dir or cfg.get("output_dir", "results/distribution"))
    out_dir.mkdir(parents=True, exist_ok=True)

    protocols = ["exp", "step"] if args.all_protocols else [data_cfg.get("imbalance_type", "exp")]
    for protocol in protocols:
        train_counts = img_num_per_cls(num_classes, max_train, protocol, ratio)
        test_counts = img_num_per_cls(num_classes, max_test, protocol, ratio)
        stem = f"distribution_{protocol}_ir{int(ratio)}"
        plot_distribution(
            train_counts,
            test_counts,
            out_dir / f"{stem}.png",
            f"CIFAR-100-LT {protocol} imbalance ratio={ratio:g}",
        )
        save_counts_csv(train_counts, test_counts, out_dir / f"{stem}.csv")


if __name__ == "__main__":
    main()
