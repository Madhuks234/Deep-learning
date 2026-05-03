"""
AyuLeafNet Evaluation
======================
Evaluate the trained model on the test set and generate:
  - Classification report
  - Confusion matrix
  - Per-class accuracy bar chart

Usage:
    python evaluate.py
    python evaluate.py --synthetic      ← without real data
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from config import CLASSES, MODEL_DIR, RESULT_DIR
from dataset import get_synthetic_loaders, get_dataloaders
from model import AyuLeafNet
from utils import (
    load_checkpoint, plot_confusion_matrix,
    print_classification_report, set_seed
)

import matplotlib.pyplot as plt
import numpy as np


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits, _ = model(images)
            preds  = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    return all_labels, all_preds


def per_class_accuracy(y_true, y_pred, classes):
    """Return per-class accuracy dictionary."""
    from collections import defaultdict
    correct = defaultdict(int)
    total   = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        total[t] += 1
        if t == p:
            correct[t] += 1
    return {classes[i]: 100 * correct[i] / max(total[i], 1) for i in range(len(classes))}


def plot_per_class_accuracy(acc_dict: dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    names  = list(acc_dict.keys())
    vals   = list(acc_dict.values())
    colors = ["#2d6a4f" if v >= 80 else "#d4a017" if v >= 60 else "#e63946"
              for v in vals]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(names, vals, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(y=80, color="#aaa", linestyle="--", linewidth=1, label="80% threshold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("AyuLeafNet — Per-Class Accuracy", fontsize=13, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.legend()
    plt.tight_layout()
    out = os.path.join(save_dir, "per_class_accuracy.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Per-class accuracy chart saved → {out}")


def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔬 AyuLeafNet Evaluation  |  Device: {device}\n")

    # Loaders
    if args.synthetic:
        _, _, test_loader = get_synthetic_loaders(batch_size=16)
    else:
        _, _, test_loader = get_dataloaders()

    # Model
    model = AyuLeafNet(pretrained=False).to(device)
    ckpt  = args.checkpoint or os.path.join(MODEL_DIR, "best_model.pth")
    load_checkpoint(model, None, ckpt)

    # Evaluate
    y_true, y_pred = evaluate(model, test_loader, device)

    overall_acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) * 100
    print(f"\n🎯 Overall Accuracy: {overall_acc:.2f}%\n")

    print_classification_report(y_true, y_pred, CLASSES)

    os.makedirs(RESULT_DIR, exist_ok=True)
    plot_confusion_matrix(y_true, y_pred, CLASSES, RESULT_DIR)
    per_cls = per_class_accuracy(y_true, y_pred, CLASSES)
    plot_per_class_accuracy(per_cls, RESULT_DIR)

    print(f"\n✅ Evaluation complete. Results saved to → {RESULT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--synthetic",  action="store_true")
    args = parser.parse_args()
    main(args)
