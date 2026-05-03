"""
AyuLeafNet Utilities
=====================
Helper functions: metrics, checkpointing, plotting, seeding, early stopping.
"""

import os
import random
import shutil
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from config import CLASSES, RESULT_DIR


# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# METRIC HELPERS
# ─────────────────────────────────────────────────────────────────────────────
class AverageMeter:
    """Tracks running average of a metric (e.g., loss, accuracy)."""
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy in percent."""
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        correct = pred.eq(labels).sum().item()
    return 100.0 * correct / labels.size(0)


def compute_topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 3) -> float:
    """Top-k accuracy in percent."""
    with torch.no_grad():
        topk_preds = logits.topk(k, dim=1).indices
        correct    = topk_preds.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
    return 100.0 * correct / labels.size(0)


# ─────────────────────────────────────────────────────────────────────────────
# EARLY STOPPING
# ─────────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    """Stops training if validation loss doesn't improve after `patience` epochs."""
    def __init__(self, patience: int = 7, delta: float = 1e-4):
        self.patience   = patience
        self.delta      = delta
        self.best_score = None
        self.counter    = 0

    def __call__(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True   # Stop!
        else:
            self.best_score = score
            self.counter    = 0
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINTING
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(
    state    : dict,
    is_best  : bool,
    save_dir : str,
    filename : str = "checkpoint.pth",
):
    """Save checkpoint; copy to best_model.pth if it's the best so far."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(save_dir, "best_model.pth")
        shutil.copyfile(filepath, best_path)


def load_checkpoint(
    model     : torch.nn.Module,
    optimizer : Optional[torch.optim.Optimizer],
    path      : str,
) -> tuple:
    """
    Load checkpoint.

    Returns
    -------
    (start_epoch, best_acc)
    """
    if not os.path.exists(path):
        print(f"⚠️  Checkpoint not found: {path}")
        return 1, 0.0

    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    epoch    = ckpt.get("epoch", 0) + 1
    best_acc = ckpt.get("best_acc", 0.0)
    print(f"✅ Loaded checkpoint from epoch {epoch - 1}  |  Best acc: {best_acc:.2f}%")
    return epoch, best_acc


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(history: Dict, save_dir: str = RESULT_DIR):
    """Plot and save training/validation loss & accuracy curves."""
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("AyuLeafNet — Training Curves", fontsize=15, fontweight="bold")

    # Loss
    axes[0].plot(epochs, history["train_loss"], "o-", color="#2d6a4f", label="Train")
    axes[0].plot(epochs, history["val_loss"],   "s--", color="#d4a017",  label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "o-", color="#2d6a4f", label="Train")
    axes[1].plot(epochs, history["val_acc"],   "s--", color="#d4a017",  label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Training curves saved → {out}")


def plot_confusion_matrix(
    y_true   : list,
    y_pred   : list,
    classes  : list = CLASSES,
    save_dir : str  = RESULT_DIR,
):
    """Plot and save a colour-coded confusion matrix."""
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlGn",
        xticklabels=classes, yticklabels=classes,
        linewidths=0.5, ax=ax,
    )
    ax.set_title("AyuLeafNet — Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Confusion matrix saved → {out}")


def print_classification_report(y_true, y_pred, classes=CLASSES):
    print("\n" + "=" * 60)
    print("  Classification Report")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))


# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM (simple hook-based implementation)
# ─────────────────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Class Activation Mapping via Gradient hooks.
    Highlights the image regions most responsible for the predicted class.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model       = model
        self.gradients   = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(
        self,
        image     : torch.Tensor,     # (1, 3, H, W)
        class_idx : Optional[int] = None,
    ) -> np.ndarray:
        """Return a (H, W) heatmap array (0-1)."""
        self.model.eval()
        logits, _ = self.model(image)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Pool gradients over channels
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1).squeeze()
        cam     = torch.clamp(cam, min=0)
        cam     = cam.cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
