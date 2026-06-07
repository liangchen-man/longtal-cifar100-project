from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def plot_confusion(metrics: dict[str, Any], split: str, out_path: Path, normalize: bool = True) -> None:
    split_metrics = metrics.get(split, {})
    cm = np.asarray(split_metrics.get("confusion_matrix"), dtype=np.float64)
    if cm.size == 0:
        return
    if normalize:
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() if cm.max() > 0 else 1)
    ax.set_title(f"{metrics.get('method', 'model')} {split} confusion matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(0, cm.shape[1], 10))
    ax.set_yticks(np.arange(0, cm.shape[0], 10))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def reliability_from_metrics(metrics: dict[str, Any], split: str, n_bins: int = 15) -> pd.DataFrame:
    split_metrics = metrics.get(split, {})
    if split_metrics.get("calibration_bins"):
        return pd.DataFrame(split_metrics["calibration_bins"])
    scores = np.asarray(split_metrics.get("scores", []), dtype=np.float64)
    labels = np.asarray(split_metrics.get("y_true", []), dtype=np.int64)
    if scores.size == 0 or labels.size == 0:
        return pd.DataFrame()
    probs = softmax(scores)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (conf > lo) & (conf <= hi)
        rows.append(
            {
                "bin": i,
                "lo": lo,
                "hi": hi,
                "count": int(mask.sum()),
                "confidence": float(conf[mask].mean()) if mask.any() else 0.0,
                "accuracy": float(correct[mask].mean()) if mask.any() else 0.0,
            }
        )
    return pd.DataFrame(rows)


def plot_reliability(metrics: dict[str, Any], split: str, out_path: Path, n_bins: int = 15) -> None:
    df = reliability_from_metrics(metrics, split, n_bins=n_bins)
    if df.empty:
        return
    centers = (df["lo"] + df["hi"]) / 2
    width = 1.0 / n_bins * 0.9
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.bar(centers, df["accuracy"], width=width, color="#4c78a8", alpha=0.85, label="Accuracy")
    ax.plot([0, 1], [0, 1], color="#d62728", linestyle="--", linewidth=1.8, label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{metrics.get('method', 'model')} {split} reliability diagram")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    df.to_csv(out_path.with_suffix(".csv"), index=False)
    plt.close(fig)


def parse_scale(method: str, source: str) -> float | None:
    text = f"{method} {source}".lower()
    match = re.search(r"_s([0-9]+(?:_[0-9]+)?)(?:_|$)", text)
    if match:
        return float(match.group(1).replace("_", "."))
    if "ldam_drw_resnet32" in text and "_s" not in text:
        return 30.0
    return None


def plot_ldam_scale(summary_path: Path, out_path: Path) -> None:
    if not summary_path.exists():
        return
    df = pd.read_csv(summary_path)
    df = df[(df["split"] == "balanced_test") & df["method"].str.contains("ldam", case=False, na=False)].copy()
    df = df[df["source"].str.contains("_short", case=False, na=False)]
    if df.empty:
        return
    df["scale"] = [parse_scale(row["method"], row["source"]) for _, row in df.iterrows()]
    df = df.dropna(subset=["scale"]).sort_values("scale")
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for metric, marker in [("top1", "o"), ("macro_f1", "s"), ("few_acc", "^")]:
        if metric in df.columns:
            ax.plot(df["scale"], df[metric], marker=marker, linewidth=2.0, label=metric)
    ax.set_xscale("log")
    ax.set_xlabel("LDAM scale s")
    ax.set_ylabel("Balanced test metric (%)")
    ax.set_title("LDAM scale ablation")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    df.to_csv(out_path.with_suffix(".csv"), index=False)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--figures_dir", default="report/figures/generated")
    parser.add_argument("--summary", default="results/summary.csv")
    parser.add_argument("--splits", nargs="+", default=["balanced_test", "imbalanced_test"])
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    for metrics_path in sorted(path for path in results_dir.rglob("metrics.json") if include_metrics_file(path)):
        metrics = load_metrics(metrics_path)
        safe_name = metrics_path.parent.name.replace(" ", "_")
        for split in args.splits:
            plot_confusion(metrics, split, figures_dir / f"confusion_{safe_name}_{split}.png")
            plot_reliability(metrics, split, figures_dir / f"reliability_{safe_name}_{split}.png")
    plot_ldam_scale(Path(args.summary), figures_dir / "ldam_scale_ablation.png")


if __name__ == "__main__":
    main()
