# 🌿 AyuLeafNet
### Hybrid CNN-Based Ayurvedic Leaf Classification for Medicinal Intelligence

> Identify 10 sacred Ayurvedic medicinal plants from leaf images and get instant medicinal intelligence — Dosha effects, Rasa, Virya, properties, and therapeutic uses.

---

## 📐 Architecture Overview

```
Input Image (224×224×3)
        │
        ├────────────────────────────────────────┐
        ▼                                        ▼
 ┌─────────────────┐                  ┌───────────────────┐
 │  Global Branch  │                  │   Local Branch    │
 │  (Backbone)     │                  │  (Custom CNN)     │
 │  MobileNetV2 /  │                  │  4 Conv Blocks    │
 │  EfficientNet   │                  │  Depthwise Sep.   │
 └────────┬────────┘                  └────────┬──────────┘
          │ CBAM Attention                      │ CBAM
          │ (Channel + Spatial)                 │
          ▼                                     ▼
        GAP → FC(512)                    GAP → FC(256)
          │                                     │
          └──────────── Concat (768) ───────────┘
                              │
                        FC(512) → BN → ReLU → Dropout
                              │
                          FC(256) → ReLU
                         ┌────┴────┐
                         ▼         ▼
                   Classifier   Confidence
                 (10 classes)    Score (1)
```

**Key innovations:**
- **Dual-branch hybrid** — global semantics + local texture
- **CBAM attention** — channel & spatial feature recalibration
- **Medicinal confidence score** — separate output for prediction reliability
- **Label smoothing** + **GradCAM** visualisation support

---

## 🌱 Supported Leaf Classes

| # | Leaf | Scientific Name | Key Uses |
|---|------|----------------|----------|
| 1 | Tulsi | *Ocimum tenuiflorum* | Respiratory, stress, fever |
| 2 | Neem | *Azadirachta indica* | Skin disorders, blood purification |
| 3 | Aloe Vera | *Aloe barbadensis* | Burns, digestion, wound healing |
| 4 | Turmeric | *Curcuma longa* | Arthritis, liver health, anti-inflammatory |
| 5 | Ashwagandha | *Withania somnifera* | Stress, anxiety, vitality |
| 6 | Brahmi | *Bacopa monnieri* | Memory, neurological support |
| 7 | Amla | *Phyllanthus emblica* | Immunity, anti-aging, digestion |
| 8 | Giloy | *Tinospora cordifolia* | Fever, diabetes, immunity |
| 9 | Mint | *Mentha* | Digestion, headaches, nausea |
| 10 | Curry Leaf | *Murraya koenigii* | Diabetes, hair growth |

---

## 📁 Project Structure

```
AyuLeafNet/
├── app.py              ← Streamlit web application
├── model.py            ← Hybrid CNN architecture (AyuLeafNet)
├── train.py            ← Training loop with early stopping
├── evaluate.py         ← Test-set evaluation + confusion matrix
├── predict.py          ← Single image / folder inference + GradCAM
├── dataset.py          ← Dataset, augmentation, DataLoaders
├── utils.py            ← Metrics, checkpointing, plotting
├── config.py           ← All hyperparameters & medicinal DB
├── quick_test.py       ← Verify setup without real data
├── requirements.txt    ← Python dependencies
│
├── data/               ← PUT YOUR DATASET HERE
│   ├── Tulsi/
│   │   ├── 001.jpg
│   │   └── ...
│   ├── Neem/
│   └── ...
│
├── models/             ← Saved checkpoints (auto-created)
├── logs/               ← Tensorboard logs (auto-created)
└── results/            ← Plots, confusion matrices (auto-created)
```

---

## 🚀 Quick Start

### 1. Clone / open in VS Code

```bash
# Open the AyuLeafNet folder in VS Code
code AyuLeafNet
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> 💡 **GPU users (CUDA 11.8):**
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> pip install -r requirements.txt
> ```

### 4. Verify setup (no dataset needed)

```bash
python quick_test.py
```

Expected output:
```
[1/6] Loading config …          ✅
[2/6] Building AyuLeafNet …     ✅  (3,271,498 params)
[3/6] Running forward pass …    ✅  logits=(4, 10)
[4/6] Testing model.predict()…  ✅
[5/6] Creating DataLoaders …    ✅
[6/6] Running training step …   ✅  loss = 2.3041
✅  ALL TESTS PASSED
```

---

## 📂 Dataset Setup

Download any Ayurvedic plant leaf dataset (e.g., from Kaggle — search *"medicinal leaf dataset"* or *"Ayurvedic plant dataset"*) and organise it:

```
data/
├── Tulsi/
│   ├── tulsi_001.jpg
│   ├── tulsi_002.jpg
│   └── ...
├── Neem/
│   └── ...
└── (one folder per class, matching config.py CLASSES list)
```

Minimum recommended: **50+ images per class** (200+ for good accuracy).

---

## 🏋️ Training

```bash
# Train with real data (MobileNetV2 backbone, default settings)
python train.py

# Quick test with synthetic data (no real images needed)
python train.py --synthetic

# Use EfficientNet backbone, 50 epochs
python train.py --backbone efficientnet_b0 --epochs 50

# Resume from a checkpoint
python train.py --resume models/checkpoint_epoch010.pth
```

Monitor training in TensorBoard:
```bash
tensorboard --logdir logs/
# Open http://localhost:6006
```

---

## 📊 Evaluation

```bash
python evaluate.py                   # real data
python evaluate.py --synthetic       # quick test
```

Generates in `results/`:
- `confusion_matrix.png`
- `per_class_accuracy.png`
- Classification report (console)

---

## 🔍 Predict

```bash
# Single image
python predict.py --image data/Tulsi/sample.jpg

# With GradCAM heatmap
python predict.py --image data/Tulsi/sample.jpg --gradcam

# Entire folder
python predict.py --folder data/test_images/

# Top-5 predictions
python predict.py --image leaf.jpg --top_k 5
```

---

## 🌐 Web Application

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

**Features:**
- 📷 Upload leaf images
- 🔬 Full medicinal intelligence output
- 🏅 Top-K predictions with confidence bars
- ⚡ Real-time inference

---

## ⚙️ Configuration

Edit `config.py` to customise:

```python
BACKBONE     = "mobilenet_v2"   # or "efficientnet_b0" | "resnet50"
IMAGE_SIZE   = 224
BATCH_SIZE   = 32
EPOCHS       = 30
LEARNING_RATE = 1e-4
EARLY_STOP_PAT = 7
```

---

## 📈 Expected Performance

| Backbone | Params | ImageNet Pretrained | Expected Val Acc |
|----------|--------|-------------------|-----------------|
| MobileNetV2 | ~3.4M | ✅ | 85–92% |
| EfficientNet-B0 | ~5.3M | ✅ | 88–94% |
| ResNet-50 | ~25M | ✅ | 87–93% |

*With 100+ images per class and 30 epochs.*

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**.
Always consult a qualified Ayurvedic practitioner before using medicinal herbs therapeutically.

---

*Built with PyTorch · Streamlit · OpenCV · scikit-learn*
