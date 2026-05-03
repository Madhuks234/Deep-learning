"""
AyuLeafNet Quick Test
======================
Verifies the full pipeline (model, dataset, training loop) works
without needing a real leaf dataset.

Run:  python quick_test.py
"""

import torch
import json
import os

print("\n" + "=" * 60)
print("  AyuLeafNet — Quick Architecture & Pipeline Test")
print("=" * 60)

# ── 1. Config ─────────────────────────────────────────────────────────────
print("\n[1/6] Loading config …", end=" ")
from config import CLASSES, MEDICINAL_DB, NUM_CLASSES, IMAGE_SIZE
assert len(CLASSES) == NUM_CLASSES, "Class count mismatch!"
assert len(MEDICINAL_DB) == NUM_CLASSES, "Medicinal DB count mismatch!"
print("✅")

# ── 2. Model ─────────────────────────────────────────────────────────────
print("[2/6] Building AyuLeafNet (pretrained=False) …", end=" ")
from model import AyuLeafNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = AyuLeafNet(pretrained=False).to(device)
total  = sum(p.numel() for p in model.parameters())
print(f"✅  ({total:,} params | device: {device})")

# ── 3. Forward pass ───────────────────────────────────────────────────────
print("[3/6] Running forward pass (batch=4) …", end=" ")
dummy  = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
logits, conf = model(dummy)
assert logits.shape == (4, NUM_CLASSES), f"Unexpected logits shape: {logits.shape}"
assert conf.shape   == (4, 1),            f"Unexpected conf shape: {conf.shape}"
print(f"✅  logits={tuple(logits.shape)}, conf={tuple(conf.shape)}")

# ── 4. Prediction helper ──────────────────────────────────────────────────
print("[4/6] Testing model.predict() …", end=" ")
pred_idx, probs, conf_score = model.predict(dummy)
assert probs.shape == (4, NUM_CLASSES)
print(f"✅  predicted classes: {pred_idx.tolist()}")

# ── 5. Synthetic DataLoader ───────────────────────────────────────────────
print("[5/6] Creating synthetic DataLoaders …", end=" ")
from dataset import get_synthetic_loaders
train_loader, val_loader, test_loader = get_synthetic_loaders(batch_size=8)
imgs, labels = next(iter(train_loader))
assert imgs.shape[1:] == (3, IMAGE_SIZE, IMAGE_SIZE)
print(f"✅  batch shape: {tuple(imgs.shape)}")

# ── 6. Single training step ───────────────────────────────────────────────
print("[6/6] Running one training step …", end=" ")
import torch.nn as nn
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

imgs, labels = imgs.to(device), labels.to(device)
logits, _ = model(imgs)
loss = criterion(logits, labels)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print(f"✅  loss = {loss.item():.4f}")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ✅  ALL TESTS PASSED — AyuLeafNet is ready!")
print("=" * 60)
print(f"""
  Next steps:
  ──────────
  1. Prepare your data:
       data/
       ├── Tulsi/     (add .jpg images)
       ├── Neem/
       └── ... (one folder per class)

  2. Train the model:
       python train.py                   ← real data
       python train.py --synthetic       ← quick test

  3. Evaluate:
       python evaluate.py --synthetic

  4. Predict a single image:
       python predict.py --image path/to/leaf.jpg

  5. Launch the web app:
       streamlit run app.py
""")
