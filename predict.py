"""
AyuLeafNet Predict
===================
Run inference on a single image or a folder of images.

Usage:
    python predict.py --image path/to/leaf.jpg
    python predict.py --folder path/to/folder/
    python predict.py --image path/to/leaf.jpg --gradcam
"""

import argparse
import os
import time
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from PIL import Image
from torchvision import transforms

from config import CLASSES, MEDICINAL_DB, MODEL_DIR, RESULT_DIR, IMAGE_SIZE, MEAN, STD
from model import AyuLeafNet
from utils import GradCAM, load_checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORM FOR INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
infer_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized tensor back to uint8 numpy image for display."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD).view(3, 1, 1)
    img  = tensor.cpu() * std + mean
    img  = img.permute(1, 2, 0).numpy()
    img  = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE IMAGE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def predict_image(
    image_path : str,
    model      : AyuLeafNet,
    device     : torch.device,
    top_k      : int  = 3,
    show_gradcam: bool = False,
) -> dict:
    """
    Run inference on a single image.

    Returns
    -------
    dict with keys: class, confidence, top_k_preds, medicinal_info
    """
    # Load & transform
    image    = Image.open(image_path).convert("RGB")
    tensor   = infer_transform(image).unsqueeze(0).to(device)  # (1,3,H,W)

    # Inference
    t0       = time.time()
    pred_idx, probs, conf = model.predict(tensor)
    elapsed  = (time.time() - t0) * 1000

    pred_class  = CLASSES[pred_idx.item()]
    confidence  = probs[0, pred_idx.item()].item() * 100
    med_conf    = conf[0, 0].item() * 100

    # Top-k
    top_probs, top_idxs = probs[0].topk(top_k)
    top_k_preds = [
        {"class": CLASSES[i.item()], "prob": p.item() * 100}
        for p, i in zip(top_probs, top_idxs)
    ]

    # Medicinal info
    med_info = MEDICINAL_DB.get(pred_class, {})

    result = {
        "class"          : pred_class,
        "confidence"     : round(confidence, 2),
        "med_confidence" : round(med_conf, 2),
        "inference_ms"   : round(elapsed, 2),
        "top_k_preds"    : top_k_preds,
        "medicinal_info" : med_info,
        "image_path"     : str(image_path),
    }

    # Print result
    _print_result(result)

    # GradCAM
    if show_gradcam:
        _visualize_gradcam(model, tensor, pred_idx.item(), image_path)

    return result


def _print_result(result: dict):
    med = result["medicinal_info"]
    print("\n" + "🌿" * 30)
    print(f"  AyuLeafNet Prediction")
    print("🌿" * 30)
    print(f"  📍 Image        : {Path(result['image_path']).name}")
    print(f"  🌱 Predicted    : {result['class']}")
    print(f"  💯 Confidence   : {result['confidence']:.1f}%")
    print(f"  🔬 Med. Score   : {result['med_confidence']:.1f}%")
    print(f"  ⚡ Inference    : {result['inference_ms']} ms")
    print()
    print(f"  🏅 Top Predictions:")
    for i, p in enumerate(result["top_k_preds"]):
        bar = "█" * int(p["prob"] / 5)
        print(f"     {i+1}. {p['class']:<15} {p['prob']:5.1f}% {bar}")

    if med:
        print()
        print(f"  ── Medicinal Intelligence ────────────────────────")
        print(f"  📚 Scientific  : {med.get('scientific_name', '-')}")
        print(f"  👨‍⚕️ Rasa       : {med.get('rasa', '-')}")
        print(f"  🌡  Virya      : {med.get('virya', '-')}")
        print(f"  ⚖️  Dosha      : {med.get('dosha_effect', '-')}")
        print(f"  💊 Properties : {', '.join(med.get('properties', []))}")
        print(f"  🏥 Uses       : {med.get('uses', '-')}")
        print(f"  ⚠️  Caution   : {med.get('caution', '-')}")
    print("🌿" * 30 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
def _visualize_gradcam(
    model      : AyuLeafNet,
    tensor     : torch.Tensor,
    class_idx  : int,
    image_path : str,
):
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Hook onto last conv layer of backbone (works for MobileNetV2)
    try:
        target_layer = list(model.backbone.children())[-1][-1]
    except Exception:
        print("⚠️  GradCAM target layer not found; skipping.")
        return

    grad_cam = GradCAM(model, target_layer)
    heatmap  = grad_cam.generate(tensor, class_idx)

    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))
    heatmap_colored = (cm.jet(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)

    # Overlay on original image
    orig_img = denormalize(tensor[0])
    overlay  = cv2.addWeighted(orig_img, 0.6, heatmap_colored, 0.4, 0)

    # Save
    stem    = Path(image_path).stem
    out_path = os.path.join(RESULT_DIR, f"gradcam_{stem}.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"GradCAM — {CLASSES[class_idx]}", fontsize=13, fontweight="bold")
    axes[0].imshow(orig_img);       axes[0].set_title("Original");  axes[0].axis("off")
    axes[1].imshow(heatmap_colored);axes[1].set_title("GradCAM");   axes[1].axis("off")
    axes[2].imshow(overlay);        axes[2].set_title("Overlay");   axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📸 GradCAM saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PREDICTION ON A FOLDER
# ─────────────────────────────────────────────────────────────────────────────
def predict_folder(folder_path: str, model: AyuLeafNet, device: torch.device):
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images    = [f for f in Path(folder_path).iterdir()
                 if f.suffix.lower() in valid_ext]

    if not images:
        print(f"⚠️  No images found in {folder_path}")
        return

    results = []
    print(f"\n🔍 Predicting {len(images)} images in {folder_path}\n")
    for img_path in images:
        result = predict_image(str(img_path), model, device, top_k=1)
        results.append(result)

    # Summary
    print("\n" + "=" * 50)
    print("  BATCH SUMMARY")
    print("=" * 50)
    for r in results:
        print(f"  {Path(r['image_path']).name:<30} → {r['class']:<15} ({r['confidence']:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🌿 AyuLeafNet Inference  |  Device: {device}")

    # Load model
    model = AyuLeafNet(pretrained=False).to(device)
    ckpt  = args.checkpoint or os.path.join(MODEL_DIR, "best_model.pth")
    load_checkpoint(model, None, ckpt)
    model.eval()

    if args.image:
        predict_image(args.image, model, device,
                      top_k=args.top_k, show_gradcam=args.gradcam)
    elif args.folder:
        predict_folder(args.folder, model, device)
    else:
        print("❓ Provide --image <path> or --folder <path>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AyuLeafNet Prediction")
    parser.add_argument("--image",      type=str, default=None)
    parser.add_argument("--folder",     type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (default: models/best_model.pth)")
    parser.add_argument("--top_k",      type=int, default=3)
    parser.add_argument("--gradcam",    action="store_true",
                        help="Generate GradCAM visualisation")
    args = parser.parse_args()
    main(args)
