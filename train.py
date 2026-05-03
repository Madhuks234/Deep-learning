"""
AyuLeafNet Training Script
============================
Run:  python train.py
      python train.py --synthetic      ← quick test without real data
      python train.py --backbone efficientnet_b0 --epochs 50
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from config import (
    EPOCHS, LEARNING_RATE, WEIGHT_DECAY, LR_STEP_SIZE, LR_GAMMA,
    EARLY_STOP_PAT, MODEL_DIR, LOG_DIR, BACKBONE, NUM_CLASSES, PRETRAINED,
    DROPOUT_RATE, BATCH_SIZE
)
from dataset import get_dataloaders, get_synthetic_loaders
from model import AyuLeafNet
from utils import (
    AverageMeter, compute_accuracy, save_checkpoint,
    load_checkpoint, plot_training_curves, set_seed, EarlyStopping
)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model, loader, criterion, optimizer, device, epoch, writer
) -> dict:
    model.train()
    loss_meter  = AverageMeter("Loss")
    top1_meter  = AverageMeter("Top-1 Acc")

    for step, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        logits, confidence = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        acc = compute_accuracy(logits, labels)
        loss_meter.update(loss.item(), images.size(0))
        top1_meter.update(acc, images.size(0))

        if step % 20 == 0:
            print(f"  Epoch {epoch:03d} | Step {step:04d} | "
                  f"Loss {loss_meter.avg:.4f} | Acc {top1_meter.avg:.2f}%")

    writer.add_scalar("Train/Loss", loss_meter.avg, epoch)
    writer.add_scalar("Train/Acc",  top1_meter.avg, epoch)
    return {"loss": loss_meter.avg, "acc": top1_meter.avg}


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, writer, split="Val") -> dict:
    model.eval()
    loss_meter = AverageMeter("Loss")
    top1_meter = AverageMeter("Top-1 Acc")

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits, _ = model(images)
        loss = criterion(logits, labels)
        acc  = compute_accuracy(logits, labels)
        loss_meter.update(loss.item(), images.size(0))
        top1_meter.update(acc, images.size(0))

    writer.add_scalar(f"{split}/Loss", loss_meter.avg, epoch)
    writer.add_scalar(f"{split}/Acc",  top1_meter.avg, epoch)
    return {"loss": loss_meter.avg, "acc": top1_meter.avg}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🌿 AyuLeafNet Training  |  Device: {device}\n")

    # ── Directories ──────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    if args.synthetic:
        train_loader, val_loader, test_loader = get_synthetic_loaders(args.batch_size)
    else:
        train_loader, val_loader, test_loader = get_dataloaders(
            batch_size=args.batch_size
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AyuLeafNet(
        backbone     = args.backbone,
        num_classes  = NUM_CLASSES,
        dropout_rate = DROPOUT_RATE,
        pretrained   = PRETRAINED,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model: AyuLeafNet | Backbone: {args.backbone} | Params: {total_params:,}\n")

    # ── Loss & Optimiser ─────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    # ── Resume checkpoint ────────────────────────────────────────────────────
    start_epoch = 1
    best_acc    = 0.0
    if args.resume:
        start_epoch, best_acc = load_checkpoint(model, optimizer, args.resume)

    # ── Tensorboard ──────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=LOG_DIR)

    # ── Early stopping ───────────────────────────────────────────────────────
    early_stop = EarlyStopping(patience=EARLY_STOP_PAT)

    # ── History ──────────────────────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Starting Training Loop")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        print(f"\n📘 Epoch {epoch}/{args.epochs}  |  LR: {scheduler.get_last_lr()[0]:.6f}")

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer)
        val_metrics   = validate(
            model, val_loader,   criterion, device, epoch, writer, "Val")

        scheduler.step()

        # Log
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])

        elapsed = time.time() - t0
        print(f"  ⏱  {elapsed:.1f}s  |  "
              f"Train → Loss {train_metrics['loss']:.4f}  Acc {train_metrics['acc']:.2f}%  |  "
              f"Val   → Loss {val_metrics['loss']:.4f}  Acc {val_metrics['acc']:.2f}%")

        # Save best
        is_best = val_metrics["acc"] > best_acc
        if is_best:
            best_acc = val_metrics["acc"]
            print(f"  🏆 New best val accuracy: {best_acc:.2f}%")

        save_checkpoint(
            state    = {
                "epoch"     : epoch,
                "model"     : model.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "best_acc"  : best_acc,
            },
            is_best   = is_best,
            save_dir  = MODEL_DIR,
            filename  = f"checkpoint_epoch{epoch:03d}.pth",
        )

        # Early stopping
        if early_stop(val_metrics["loss"]):
            print(f"\n⛔ Early stopping triggered at epoch {epoch}.")
            break

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\n🔬 Evaluating on test set …")
    best_ckpt = os.path.join(MODEL_DIR, "best_model.pth")
    if os.path.exists(best_ckpt):
        load_checkpoint(model, optimizer, best_ckpt)
    test_metrics = validate(model, test_loader, criterion, device, args.epochs, writer, "Test")
    print(f"  Test Accuracy: {test_metrics['acc']:.2f}%  |  Loss: {test_metrics['loss']:.4f}")

    # ── Save history & plot ───────────────────────────────────────────────────
    hist_path = os.path.join(LOG_DIR, "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history, save_dir=LOG_DIR)

    writer.close()
    print(f"\n✅ Training complete!  Best Val Acc: {best_acc:.2f}%")
    print(f"   Best model saved → {best_ckpt}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AyuLeafNet Training")
    parser.add_argument("--backbone",    type=str,  default=BACKBONE,
                        choices=["mobilenet_v2", "efficientnet_b0", "resnet50"])
    parser.add_argument("--epochs",      type=int,  default=EPOCHS)
    parser.add_argument("--lr",          type=float,default=LEARNING_RATE)
    parser.add_argument("--batch_size",  type=int,  default=BATCH_SIZE)
    parser.add_argument("--resume",      type=str,  default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--synthetic",   action="store_true",
                        help="Use synthetic data for quick architecture testing")
    args = parser.parse_args()
    main(args)
