from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from longtail.config import load_config
from longtail.data import make_loaders, seed_everything
from longtail.losses import build_loss
from longtail.metrics import evaluate_model, save_json, save_per_class_csv
from longtail.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["output_dir"])
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else out_dir / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    data, train_loader, balanced_loader, imbalanced_loader = make_loaders(cfg)
    _, _, _, counts, test_counts, class_names = data
    num_classes = int(cfg.get("model", {}).get("num_classes", len(counts)))

    model = build_model(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    epochs = int(cfg.get("optim", {}).get("epochs", checkpoint.get("epoch", 0) + 1))
    criterion = build_loss(cfg, counts, epochs - 1).to(device)
    final_metrics = {
        "method": cfg.get("method", cfg.get("loss", {}).get("name", "resnet32")),
        "config": cfg,
        "cls_num_list": counts,
        "imbalanced_test_cls_num_list": test_counts,
        "class_names": class_names,
        "train": evaluate_model(model, train_loader, device, criterion, counts, num_classes),
        "balanced_test": evaluate_model(model, balanced_loader, device, criterion, counts, num_classes),
        "imbalanced_test": evaluate_model(model, imbalanced_loader, device, criterion, counts, num_classes),
    }
    save_json(final_metrics, out_dir / "metrics.json")
    for split in ["train", "balanced_test", "imbalanced_test"]:
        save_per_class_csv(final_metrics[split], out_dir / f"per_class_{split}.csv", class_names, counts)
    print(out_dir / "metrics.json")


if __name__ == "__main__":
    main()
