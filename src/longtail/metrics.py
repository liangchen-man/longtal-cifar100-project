from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def class_split_indices(
    cls_num_list: list[int],
    many_threshold: int = 100,
    few_threshold: int = 20,
) -> dict[str, list[int]]:
    many = [i for i, n in enumerate(cls_num_list) if n >= many_threshold]
    medium = [i for i, n in enumerate(cls_num_list) if few_threshold <= n < many_threshold]
    few = [i for i, n in enumerate(cls_num_list) if n < few_threshold]
    return {"many": many, "medium": medium, "few": few}


def per_class_accuracy_from_confusion(cm: np.ndarray) -> np.ndarray:
    denom = cm.sum(axis=1)
    return np.divide(
        np.diag(cm),
        np.maximum(denom, 1),
        out=np.zeros(cm.shape[0], dtype=np.float64),
        where=denom > 0,
    )


def grouped_accuracy(per_class_acc: np.ndarray, cls_num_list: list[int] | None) -> dict[str, float | None]:
    if cls_num_list is None:
        return {"many": None, "medium": None, "few": None}
    groups = class_split_indices(cls_num_list)
    out: dict[str, float | None] = {}
    for name, indices in groups.items():
        out[name] = float(np.mean(per_class_acc[indices])) if indices else None
    return out


def topk_accuracy_from_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    topk: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    if scores.ndim != 2:
        raise ValueError("scores must have shape [N, C]")
    max_k = min(max(topk), scores.shape[1])
    top = np.argsort(-scores, axis=1)[:, :max_k]
    metrics: dict[str, float] = {}
    for k in topk:
        k_eff = min(k, scores.shape[1])
        metrics[f"top{k}"] = float(np.mean([label in row[:k_eff] for label, row in zip(y_true, top)]))
    return metrics


def softmax_np(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def expected_calibration_error(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bins: int = 15,
) -> float:
    bins = calibration_bins(y_true, scores, n_bins=n_bins)
    return float(sum(row["fraction"] * abs(row["accuracy"] - row["confidence"]) for row in bins))


def calibration_bins(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bins: int = 15,
) -> list[dict[str, float | int]]:
    probs = softmax_np(scores)
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    correct = (prediction == y_true).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float | int]] = []
    for idx, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (confidence > lo) & (confidence <= hi)
        rows.append(
            {
                "bin": idx,
                "lo": float(lo),
                "hi": float(hi),
                "count": int(mask.sum()),
                "fraction": float(mask.mean()),
                "confidence": float(confidence[mask].mean()) if mask.any() else 0.0,
                "accuracy": float(correct[mask].mean()) if mask.any() else 0.0,
            }
        )
    return rows


def compute_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray | None = None,
    y_score: np.ndarray | None = None,
    cls_num_list: list[int] | None = None,
    num_classes: int = 100,
) -> dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    if y_pred is None:
        if y_score is None:
            raise ValueError("Either y_pred or y_score must be provided")
        y_pred_arr = np.asarray(y_score).argmax(axis=1).astype(np.int64)
    else:
        y_pred_arr = np.asarray(y_pred, dtype=np.int64)

    labels = list(range(num_classes))
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    per_class_acc = per_class_accuracy_from_confusion(cm)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        zero_division=0,
    )

    metrics: dict[str, Any] = {
        "top1": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "many_acc": grouped_accuracy(per_class_acc, cls_num_list)["many"],
        "medium_acc": grouped_accuracy(per_class_acc, cls_num_list)["medium"],
        "few_acc": grouped_accuracy(per_class_acc, cls_num_list)["few"],
        "per_class_acc": per_class_acc,
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "support": support,
        "confusion_matrix": cm,
    }

    if y_score is not None:
        score_arr = np.asarray(y_score, dtype=np.float64)
        metrics.update(topk_accuracy_from_scores(y_true_arr, score_arr, topk=(1, 5)))
        metrics["ece"] = expected_calibration_error(y_true_arr, score_arr)
        metrics["calibration_bins"] = calibration_bins(y_true_arr, score_arr)
    else:
        metrics["top5"] = None
        metrics["ece"] = None
    return metrics


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
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
    return {key: metrics.get(key) for key in keys if key in metrics}


def per_class_table(
    metrics: dict[str, Any],
    class_names: list[str] | None = None,
    cls_num_list: list[int] | None = None,
) -> list[dict[str, Any]]:
    per_acc = np.asarray(metrics.get("per_class_acc", []), dtype=np.float64)
    support = np.asarray(metrics.get("support", np.zeros_like(per_acc)), dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for class_id, acc in enumerate(per_acc):
        train_count = None if cls_num_list is None else int(cls_num_list[class_id])
        if train_count is None:
            group = None
        elif train_count >= 100:
            group = "many"
        elif train_count >= 20:
            group = "medium"
        else:
            group = "few"
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id] if class_names and class_id < len(class_names) else str(class_id),
                "train_count": train_count,
                "test_support": int(support[class_id]) if class_id < support.shape[0] else 0,
                "group": group,
                "accuracy": float(acc),
            }
        )
    return rows


def save_per_class_csv(
    metrics: dict[str, Any],
    path: str | Path,
    class_names: list[str] | None = None,
    cls_num_list: list[int] | None = None,
) -> None:
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_class_table(metrics, class_names, cls_num_list)).to_csv(path, index=False)


def evaluate_model(
    model,
    loader,
    device: str = "cuda",
    criterion=None,
    cls_num_list: list[int] | None = None,
    num_classes: int = 100,
) -> dict[str, Any]:
    import torch

    model.eval()
    y_true: list[int] = []
    scores: list[np.ndarray] = []
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            if criterion is not None:
                loss = criterion(logits, y)
                total_loss += float(loss.item()) * y.numel()
            total_items += y.numel()
            y_true.extend(y.cpu().tolist())
            scores.append(logits.detach().cpu().numpy())
    y_score = np.concatenate(scores, axis=0)
    metrics = compute_metrics(
        y_true,
        y_score=y_score,
        cls_num_list=cls_num_list,
        num_classes=num_classes,
    )
    if criterion is not None and total_items:
        metrics["loss"] = total_loss / total_items
    return metrics
