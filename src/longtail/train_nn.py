from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from longtail.config import load_config
    from longtail.data import make_loaders, seed_everything
    from longtail.losses import build_loss
    from longtail.metrics import compact_metrics, evaluate_model, save_json, save_per_class_csv
    from longtail.models import build_model
else:
    from .config import load_config
    from .data import make_loaders, seed_everything
    from .losses import build_loss
    from .metrics import compact_metrics, evaluate_model, save_json, save_per_class_csv
    from .models import build_model


def _require_torch():
    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError(
            "Neural-network experiments require PyTorch. Install torch/torchvision "
            "or run the classical/synthetic smoke pipeline first."
        ) from exc


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


def append_history_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def train_one_epoch(model, loader, criterion, optimizer, device: str) -> dict[str, float]:
    torch = _require_torch()
    model.train()
    total_loss = 0.0
    total_items = 0
    correct1 = 0
    correct5 = 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch = y.numel()
        total_items += batch
        total_loss += float(loss.item()) * batch
        _, pred = logits.topk(min(5, logits.shape[1]), dim=1)
        correct = pred.eq(y.view(-1, 1))
        correct1 += int(correct[:, :1].sum().item())
        correct5 += int(correct[:, : min(5, logits.shape[1])].sum().item())
    return {
        "loss": total_loss / max(total_items, 1),
        "top1": correct1 / max(total_items, 1),
        "top5": correct5 / max(total_items, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    torch = _require_torch()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    data, train_loader, balanced_loader, imbalanced_loader = make_loaders(cfg)
    _, _, _, counts, test_counts, class_names = data
    num_classes = int(cfg.get("model", {}).get("num_classes", len(counts)))

    model = build_model(cfg).to(device)
    optim_cfg = cfg.get("optim", {})
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(optim_cfg.get("lr", 0.1)),
        momentum=float(optim_cfg.get("momentum", 0.9)),
        weight_decay=float(optim_cfg.get("weight_decay", 2e-4)),
        nesterov=bool(optim_cfg.get("nesterov", False)),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(optim_cfg.get("milestones", [160, 180])),
        gamma=float(optim_cfg.get("gamma", 0.01)),
    )

    start_epoch = 0
    best_balanced_acc = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_balanced_acc = float(checkpoint.get("best_balanced_acc", -1.0))

    epochs = int(optim_cfg.get("epochs", 200))
    eval_every = args.eval_every or int(cfg.get("eval", {}).get("eval_every", 10))
    history_path = out_dir / "history.csv"

    for epoch in range(start_epoch, epochs):
        criterion = build_loss(cfg, counts, epoch).to(device)
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        should_eval = (epoch + 1) % eval_every == 0 or epoch + 1 == epochs
        row: dict[str, Any] = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "train_top5": train_metrics["top5"],
        }
        if should_eval:
            eval_criterion = build_loss(cfg, counts, epoch).to(device)
            balanced_metrics = evaluate_model(
                model,
                balanced_loader,
                device,
                eval_criterion,
                counts,
                num_classes,
            )
            imbalanced_metrics = evaluate_model(
                model,
                imbalanced_loader,
                device,
                eval_criterion,
                counts,
                num_classes,
            )
            for prefix, metrics in [
                ("balanced_test", balanced_metrics),
                ("imbalanced_test", imbalanced_metrics),
            ]:
                for key, value in compact_metrics(metrics).items():
                    row[f"{prefix}_{key}"] = value
            if balanced_metrics["balanced_acc"] > best_balanced_acc:
                best_balanced_acc = float(balanced_metrics["balanced_acc"])
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                        "best_balanced_acc": best_balanced_acc,
                        "cfg": cfg,
                        "cls_num_list": counts,
                        "class_names": class_names,
                    },
                    out_dir / "best.pt",
                )
        append_history_csv(history_path, row)
        print(row)

    checkpoint_path = out_dir / "best.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
    else:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epochs - 1,
                "best_balanced_acc": best_balanced_acc,
                "cfg": cfg,
                "cls_num_list": counts,
                "class_names": class_names,
            },
            checkpoint_path,
        )

    final_criterion = build_loss(cfg, counts, epochs - 1).to(device)
    final_metrics = {
        "method": cfg.get("method", cfg.get("loss", {}).get("name", "resnet32")),
        "config": cfg,
        "cls_num_list": counts,
        "imbalanced_test_cls_num_list": test_counts,
        "class_names": class_names,
        "train": evaluate_model(model, train_loader, device, final_criterion, counts, num_classes),
        "balanced_test": evaluate_model(model, balanced_loader, device, final_criterion, counts, num_classes),
        "imbalanced_test": evaluate_model(model, imbalanced_loader, device, final_criterion, counts, num_classes),
    }
    save_json(final_metrics, out_dir / "metrics.json")
    for split in ["train", "balanced_test", "imbalanced_test"]:
        save_per_class_csv(final_metrics[split], out_dir / f"per_class_{split}.csv", class_names, counts)


if __name__ == "__main__":
    main()
