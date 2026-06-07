from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_KEYS = [
    "top1",
    "top5",
    "macro_f1",
    "weighted_f1",
    "balanced_acc",
    "many_acc",
    "medium_acc",
    "few_acc",
    "ece",
]

HISTORY_METRIC_KEYS = [
    "loss",
    "top1",
    "top5",
    "macro_f1",
    "weighted_f1",
    "balanced_acc",
    "many_acc",
    "medium_acc",
    "few_acc",
    "ece",
]

HISTORY_COLUMNS = [
    "epoch",
    "lr",
    "train_loss",
    "train_top1",
    "train_top5",
    *[f"{prefix}_{key}" for prefix in ["balanced_test", "imbalanced_test"] for key in HISTORY_METRIC_KEYS],
]


def load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def include_metrics_file(path: Path) -> bool:
    lowered = [part.lower() for part in path.parts]
    if any("diag" in part or "smoke" in part for part in lowered):
        return False
    try:
        metrics = load_metrics(path)
    except Exception:
        return False
    dataset = metrics.get("config", {}).get("data", {}).get("dataset", "cifar100")
    return dataset == "cifar100"


def pct(value: Any) -> Any:
    if value is None:
        return None
    try:
        return float(value) * 100.0
    except Exception:
        return value


def flatten_metrics(metrics: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    method = metrics.get("method", source.parent.name)
    cfg = metrics.get("config", {})
    data_cfg = cfg.get("data", {})
    loss_cfg = cfg.get("loss", {})
    for split in ["train", "balanced_test", "imbalanced_test"]:
        if split not in metrics:
            continue
        row = {
            "method": method,
            "split": split,
            "source": str(source),
            "dataset": data_cfg.get("dataset", "cifar100"),
            "imbalance_type": data_cfg.get("imbalance_type"),
            "imbalance_ratio": data_cfg.get("imbalance_ratio"),
            "loss": loss_cfg.get("name"),
        }
        for key in METRIC_KEYS:
            row[key] = pct(metrics[split].get(key))
        rows.append(row)
    return rows


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    display = df.copy()
    numeric_cols = [c for c in METRIC_KEYS if c in display.columns]
    for col in numeric_cols:
        display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    headers = list(display.columns)
    rows = [[str(value) for value in row] for row in display.fillna("").to_numpy()]
    widths = [
        max(len(str(header)), *(len(row[i]) for row in rows)) if rows else len(str(header))
        for i, header in enumerate(headers)
    ]
    lines = [
        "| " + " | ".join(str(header).ljust(widths[i]) for i, header in enumerate(headers)) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    lines.extend("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_history_csv(path: Path) -> pd.DataFrame:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return pd.DataFrame()
    header = rows[0]
    body = rows[1:]
    max_width = max([len(header), *(len(row) for row in body)] or [0])
    if max_width > len(header):
        columns = HISTORY_COLUMNS[:max_width]
        if len(columns) < max_width:
            columns.extend(f"extra_{idx}" for idx in range(len(columns), max_width))
    else:
        columns = header
    normalized = [row[: len(columns)] + [""] * max(0, len(columns) - len(row)) for row in body]
    hist = pd.DataFrame(normalized, columns=columns)
    for column in hist.columns:
        converted = pd.to_numeric(hist[column], errors="coerce")
        if converted.notna().any():
            hist[column] = converted
    return hist


def plot_overall_bars(df: pd.DataFrame, out_dir: Path) -> None:
    test = df[df["split"].isin(["balanced_test", "imbalanced_test"])].copy()
    if test.empty:
        return
    for metric in ["top1", "macro_f1", "balanced_acc", "many_acc", "medium_acc", "few_acc"]:
        if metric not in test.columns or test[metric].dropna().empty:
            continue
        pivot = test.pivot_table(index="method", columns="split", values=metric, aggfunc="max")
        ax = pivot.plot(kind="bar", figsize=(10, 4.8), width=0.78)
        ax.set_title(f"{metric} comparison")
        ax.set_ylabel(f"{metric} (%)")
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="")
        plt.tight_layout()
        plt.savefig(out_dir / f"comparison_{metric}.png", dpi=220)
        plt.close()


def plot_history(results_dir: Path, out_dir: Path) -> None:
    for history_path in results_dir.rglob("history.csv"):
        try:
            hist = load_history_csv(history_path)
        except Exception:
            continue
        if hist.empty:
            continue
        method = history_path.parent.name
        for metric in ["train_loss", "train_top1", "balanced_test_top1", "balanced_test_balanced_acc"]:
            if metric not in hist.columns or hist[metric].dropna().empty:
                continue
            plt.figure(figsize=(8, 4.5))
            plt.plot(hist["epoch"], hist[metric], marker="o", linewidth=1.8)
            plt.title(f"{method} {metric}")
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.grid(alpha=0.25)
            plt.tight_layout()
            plt.savefig(out_dir / f"history_{method}_{metric}.png", dpi=220)
            plt.close()


def plot_per_class(metrics_files: list[Path], out_dir: Path) -> None:
    for path in metrics_files:
        metrics = load_metrics(path)
        counts = metrics.get("cls_num_list")
        class_names = metrics.get("class_names")
        for split in ["balanced_test", "imbalanced_test"]:
            split_metrics = metrics.get(split)
            if not split_metrics or "per_class_acc" not in split_metrics:
                continue
            per_acc = np.asarray(split_metrics["per_class_acc"], dtype=float) * 100
            plt.figure(figsize=(12, 4.5))
            if counts:
                plt.bar(np.arange(len(counts)), counts, alpha=0.25, label="train count")
                plt.twinx()
            plt.plot(np.arange(len(per_acc)), per_acc, linewidth=1.6, color="#c23b22", label="class acc")
            plt.ylim(0, 100)
            plt.title(f"{metrics.get('method', path.parent.name)} {split} per-class accuracy")
            plt.xlabel("Class index")
            plt.ylabel("Accuracy (%)")
            plt.grid(alpha=0.2)
            plt.tight_layout()
            safe_name = path.parent.name.replace(" ", "_")
            plt.savefig(out_dir / f"per_class_{safe_name}_{split}.png", dpi=220)
            plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--out", default="results/summary.csv")
    parser.add_argument("--figures_dir", default="report/figures/generated")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_path = Path(args.out)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metric_files = [path for path in sorted(results_dir.rglob("metrics.json")) if include_metrics_file(path)]
    rows: list[dict[str, Any]] = []
    for path in metric_files:
        rows.extend(flatten_metrics(load_metrics(path), path))
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    if not df.empty:
        write_markdown_table(df, out_path.with_suffix(".md"))
        plot_overall_bars(df, figures_dir)
        plot_history(results_dir, figures_dir)
        plot_per_class(metric_files, figures_dir)
    print(df)


if __name__ == "__main__":
    main()
