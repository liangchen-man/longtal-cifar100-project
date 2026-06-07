from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from longtail.config import load_config
    from longtail.data import build_array_dataset_bundle, build_cifar100lt, seed_everything
    from longtail.metrics import compute_metrics, save_json, save_per_class_csv
else:
    from .config import load_config
    from .data import build_array_dataset_bundle, build_cifar100lt, seed_everything
    from .metrics import compute_metrics, save_json, save_per_class_csv


def _tensor_or_pil_to_uint8(image: Any) -> np.ndarray:
    if isinstance(image, np.ndarray):
        arr = image
    elif hasattr(image, "detach"):
        arr = image.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] in {1, 3}:
            arr = arr.transpose(1, 2, 0)
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / max(float(arr.max() - arr.min()), 1e-6)
        arr = arr * 255.0
    else:
        arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    return arr


def extract_hog_features_from_arrays(
    images: np.ndarray,
    labels: np.ndarray,
    max_items: int | None = None,
    desc: str = "HOG",
) -> tuple[np.ndarray, np.ndarray]:
    n_items = len(labels) if max_items is None else min(max_items, len(labels))
    features: list[np.ndarray] = []
    y: list[int] = []
    for i in tqdm(range(n_items), desc=desc, leave=False):
        arr = _tensor_or_pil_to_uint8(images[i])
        features.append(
            hog(
                arr,
                orientations=9,
                pixels_per_cell=(4, 4),
                cells_per_block=(2, 2),
                channel_axis=-1,
                feature_vector=True,
            )
        )
        y.append(int(labels[i]))
    return np.asarray(features, dtype=np.float32), np.asarray(y, dtype=np.int64)


def extract_hog_features_from_dataset(
    dataset,
    max_items: int | None = None,
    desc: str = "HOG",
) -> tuple[np.ndarray, np.ndarray]:
    n_items = len(dataset) if max_items is None else min(max_items, len(dataset))
    features: list[np.ndarray] = []
    labels: list[int] = []
    for i in tqdm(range(n_items), desc=desc, leave=False):
        image, label = dataset[i]
        arr = _tensor_or_pil_to_uint8(image)
        features.append(
            hog(
                arr,
                orientations=9,
                pixels_per_cell=(4, 4),
                cells_per_block=(2, 2),
                channel_axis=-1,
                feature_vector=True,
            )
        )
        labels.append(int(label))
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def _scores_for_all_classes(model: Pipeline, x: np.ndarray, num_classes: int) -> np.ndarray:
    if hasattr(model, "decision_function"):
        raw = model.decision_function(x)
    else:
        raw = model.predict_proba(x)
    raw = np.asarray(raw)
    if raw.ndim == 1:
        raw = raw[:, None]

    classes = getattr(model[-1], "classes_", np.arange(raw.shape[1]))
    scores = np.full((x.shape[0], num_classes), -1e9, dtype=np.float64)
    for col, class_id in enumerate(classes):
        if int(class_id) < num_classes:
            scores[:, int(class_id)] = raw[:, col]
    return scores


def evaluate_classifier(
    model: Pipeline,
    x: np.ndarray,
    y: np.ndarray,
    cls_num_list: list[int],
    num_classes: int,
) -> dict[str, Any]:
    scores = _scores_for_all_classes(model, x, num_classes)
    return compute_metrics(
        y_true=y,
        y_score=scores,
        cls_num_list=cls_num_list,
        num_classes=num_classes,
    )


def build_feature_sets(cfg: dict[str, Any], max_train_items: int | None):
    data_cfg = cfg.get("data", {})
    dataset = data_cfg.get("dataset", "cifar100").lower()
    if dataset in {"synthetic", "synthetic_cifar100"}:
        bundle = build_array_dataset_bundle(cfg)
        x_train, y_train = extract_hog_features_from_arrays(
            *bundle.train,
            max_items=max_train_items,
            desc="HOG train",
        )
        x_bal, y_bal = extract_hog_features_from_arrays(*bundle.balanced_test, desc="HOG balanced test")
        x_imb, y_imb = extract_hog_features_from_arrays(*bundle.imbalanced_test, desc="HOG imbalanced test")
        return x_train, y_train, x_bal, y_bal, x_imb, y_imb, bundle.train_counts, bundle.class_names

    train_set, balanced_test, imbalanced_test, counts, _, class_names = build_cifar100lt(
        root=data_cfg.get("root", "./data"),
        imb_type=data_cfg.get("imbalance_type", "exp"),
        imb_ratio=float(data_cfg.get("imbalance_ratio", 100)),
        seed=int(cfg.get("seed", 42)),
        augment=False,
        download=bool(data_cfg.get("download", True)),
    )
    x_train, y_train = extract_hog_features_from_dataset(train_set, max_items=max_train_items, desc="HOG train")
    x_bal, y_bal = extract_hog_features_from_dataset(balanced_test, desc="HOG balanced test")
    x_imb, y_imb = extract_hog_features_from_dataset(imbalanced_test, desc="HOG imbalanced test")
    return x_train, y_train, x_bal, y_bal, x_imb, y_imb, counts, class_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_train_items", type=int, default=None)
    parser.add_argument("--output_subdir", default="classical_hog_svm")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    num_classes = int(cfg.get("model", {}).get("num_classes", cfg.get("data", {}).get("num_classes", 100)))
    out_dir = Path(cfg["output_dir"]) / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_bal, y_bal, x_imb, y_imb, counts, class_names = build_feature_sets(
        cfg,
        args.max_train_items,
    )
    n_components = min(
        int(cfg.get("classical", {}).get("pca_components", 256)),
        x_train.shape[0] - 1,
        x_train.shape[1],
    )
    if n_components < 2:
        raise RuntimeError("Too few samples/features for PCA; increase the smoke dataset size")

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=int(cfg.get("seed", 42)))),
            (
                "svm",
                SGDClassifier(
                    loss="hinge",
                    class_weight="balanced",
                    max_iter=int(cfg.get("classical", {}).get("max_iter", 3000)),
                    tol=1e-4,
                    n_jobs=-1,
                    random_state=int(cfg.get("seed", 42)),
                ),
            ),
        ]
    )
    clf.fit(x_train, y_train)
    joblib.dump(clf, out_dir / "model.joblib")

    metrics = {
        "method": "HOG+PCA+LinearSVM",
        "config": cfg,
        "cls_num_list": counts,
        "class_names": class_names,
        "train": evaluate_classifier(clf, x_train, y_train, counts, num_classes),
        "balanced_test": evaluate_classifier(clf, x_bal, y_bal, counts, num_classes),
        "imbalanced_test": evaluate_classifier(clf, x_imb, y_imb, counts, num_classes),
    }
    save_json(metrics, out_dir / "metrics.json")
    for split in ["train", "balanced_test", "imbalanced_test"]:
        save_per_class_csv(metrics[split], out_dir / f"per_class_{split}.csv", class_names, counts)

    summary = []
    for split in ["train", "balanced_test", "imbalanced_test"]:
        row = {"method": metrics["method"], "split": split}
        for key in ["top1", "top5", "macro_f1", "balanced_acc", "many_acc", "medium_acc", "few_acc", "ece"]:
            row[key] = metrics[split].get(key)
        summary.append(row)
    pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)
    print(pd.DataFrame(summary))


if __name__ == "__main__":
    main()
