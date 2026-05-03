"""
AyuLeafNet — Streamlit Web Application
=======================================
Run:  streamlit run app.py
"""

import os
import io
import time
from pathlib import Path

import torch
import numpy as np
import streamlit as st
from PIL import Image
from torchvision import transforms

from config import (
    CLASSES, MEDICINAL_DB, MODEL_DIR, IMAGE_SIZE, MEAN, STD, PRETRAINED
)
from model import AyuLeafNet


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be FIRST streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "AyuLeafNet",
    page_icon  = "🌿",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #1b4332 0%, #40916c 50%, #d4a017 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}

.hero-sub {
    font-size: 1.05rem;
    color: #52796f;
    font-weight: 300;
    letter-spacing: 0.05em;
}

.leaf-card {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border-left: 5px solid #2d6a4f;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin: 0.8rem 0;
    box-shadow: 0 2px 8px rgba(45,106,79,0.1);
}

.result-card {
    background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%);
    border: 2px solid #40916c;
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(45,106,79,0.15);
}

.pred-class {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #1b4332;
}

.conf-badge {
    background: #d4a017;
    color: white;
    font-weight: 600;
    font-size: 1.1rem;
    border-radius: 30px;
    padding: 0.3rem 1.2rem;
    display: inline-block;
    margin-top: 0.5rem;
}

.med-section {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-top: 0.8rem;
    border: 1px solid #e9f5ec;
}

.med-pill {
    display: inline-block;
    background: #e9f5ec;
    color: #1b4332;
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    margin: 0.2rem;
    font-size: 0.85rem;
    font-weight: 500;
}

.sidebar-head {
    font-family: 'Playfair Display', serif;
    color: #1b4332;
    font-size: 1.2rem;
    font-weight: 700;
    border-bottom: 2px solid #40916c;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}

.bar-outer {
    background: #e9f5ec;
    border-radius: 8px;
    height: 10px;
    margin-top: 4px;
}

.bar-inner {
    background: linear-gradient(90deg, #40916c, #d4a017);
    border-radius: 8px;
    height: 10px;
}

.caution-box {
    background: #fff7e6;
    border-left: 4px solid #d4a017;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-top: 0.8rem;
    font-size: 0.9rem;
}

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = AyuLeafNet(pretrained=False).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model, device, True   # loaded real weights

    # Demo mode — random weights (architecture test)
    model.eval()
    return model, device, False


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
infer_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def run_inference(pil_image, model, device, top_k=3):
    tensor = infer_tf(pil_image).unsqueeze(0).to(device)
    t0     = time.time()
    with torch.no_grad():
        logits, conf = model(tensor)
    elapsed = (time.time() - t0) * 1000

    import torch.nn.functional as F
    probs    = F.softmax(logits, dim=1)[0]
    top_p, top_i = probs.topk(top_k)

    return {
        "class"       : CLASSES[top_i[0].item()],
        "confidence"  : float(top_p[0]) * 100,
        "med_conf"    : float(conf[0, 0]) * 100,
        "top_k"       : [
            {"class": CLASSES[i.item()], "prob": float(p) * 100}
            for p, i in zip(top_p, top_i)
        ],
        "inference_ms": round(elapsed, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-head">🌿 AyuLeafNet Settings</div>',
                    unsafe_allow_html=True)

        st.markdown("**Model Checkpoint**")
        default_ckpt = os.path.join(MODEL_DIR, "best_model.pth")
        ckpt_path = st.text_input(
            "Checkpoint path", value=default_ckpt, label_visibility="collapsed")

        top_k = st.slider("Top-K predictions", 1, 5, 3)

        st.markdown("---")
        st.markdown("**Supported Leaf Classes**")
        for cls in CLASSES:
            info = MEDICINAL_DB.get(cls, {})
            color = info.get("color", "#40916c")
            st.markdown(
                f'<span style="color:{color}">●</span> {cls}',
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.78rem;color:#999;">'
            '🔬 AyuLeafNet v1.0<br>'
            'Hybrid CNN | Medicinal Intelligence<br>'
            '</div>',
            unsafe_allow_html=True
        )

    return ckpt_path, top_k


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DISPLAY
# ─────────────────────────────────────────────────────────────────────────────
def display_result(result, trained):
    cls      = result["class"]
    conf     = result["confidence"]
    med_conf = result["med_conf"]
    med_info = MEDICINAL_DB.get(cls, {})

    if not trained:
        st.warning("⚠️ Running in DEMO mode (no trained weights). Predictions are random. Train the model first!")

    # ── Top prediction card ──────────────────────────────────────────────────
    st.markdown(f"""
    <div class="result-card">
        <div style="font-size:0.9rem;color:#52796f;letter-spacing:0.1em;text-transform:uppercase;">Identified Leaf</div>
        <div class="pred-class">{cls}</div>
        <div style="font-size:0.85rem;color:#52796f;font-style:italic;margin-top:0.2rem;">
            {med_info.get('scientific_name', '')}
        </div>
        <div class="conf-badge">Confidence: {conf:.1f}%</div>
        <div style="font-size:0.8rem;color:#777;margin-top:0.5rem;">
            Medicinal Score: {med_conf:.1f}%  &nbsp;|&nbsp;  ⚡ {result['inference_ms']} ms
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Top-K predictions ────────────────────────────────────────────────────
    st.markdown("#### 🏅 Top Predictions")
    for pred in result["top_k"]:
        color = MEDICINAL_DB.get(pred["class"], {}).get("color", "#40916c")
        width = int(pred["prob"])
        st.markdown(f"""
        <div style="margin:0.5rem 0;">
            <div style="display:flex;justify-content:space-between;font-size:0.9rem;">
                <span><b>{pred['class']}</b></span>
                <span>{pred['prob']:.1f}%</span>
            </div>
            <div class="bar-outer">
                <div class="bar-inner" style="width:{width}%;background:{color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Medicinal Intelligence ───────────────────────────────────────────────
    st.markdown("#### 🔬 Medicinal Intelligence")
    if med_info:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="med-section">
                <b>📚 Family</b><br>{med_info.get('family','—')}
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="med-section">
                <b>👅 Rasa (Taste)</b><br>{med_info.get('rasa','—')}
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="med-section">
                <b>🌡 Virya (Potency)</b><br>{med_info.get('virya','—')}
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="med-section">
                <b>⚖️ Dosha Effect</b><br>{med_info.get('dosha_effect','—')}
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="med-section">
                <b>🌿 Parts Used</b><br>{med_info.get('parts_used','—')}
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="med-section">
                <b>🏥 Therapeutic Uses</b><br>{med_info.get('uses','—')}
            </div>
            """, unsafe_allow_html=True)

        # Properties pills
        props = med_info.get("properties", [])
        if props:
            pills = " ".join(f'<span class="med-pill">{p}</span>' for p in props)
            st.markdown(f"**💊 Properties:**<br>{pills}", unsafe_allow_html=True)

        # Caution
        caution = med_info.get("caution", "")
        if caution:
            st.markdown(f"""
            <div class="caution-box">
                ⚠️ <b>Caution:</b> {caution}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No medicinal data found for this prediction.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ckpt_path, top_k = sidebar()

    # Hero header
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 2rem;">
        <div class="hero-title">🌿 AyuLeafNet</div>
        <div class="hero-sub">
            Hybrid CNN-Based Ayurvedic Leaf Classification for Medicinal Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    with st.spinner("Loading AyuLeafNet model…"):
        model, device, trained = load_model(ckpt_path)

    # Upload section
    st.markdown("---")
    col_upload, col_result = st.columns([1, 1.5], gap="large")

    with col_upload:
        st.markdown("### 📷 Upload Leaf Image")
        uploaded = st.file_uploader(
            "Upload a clear leaf image (JPG, PNG, WebP)",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
        )

        # Sample images hint
        st.markdown(
            '<div class="leaf-card">'
            '<b>💡 Tip:</b> For best results, use a clear, well-lit photo of a '
            'single leaf against a neutral background. Supported classes: '
            + ", ".join(CLASSES) +
            '</div>',
            unsafe_allow_html=True
        )

        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            st.image(pil_img, caption="Uploaded Image", use_column_width=True)

            if st.button("🔍 Classify Leaf", use_container_width=True, type="primary"):
                with st.spinner("Analysing leaf…"):
                    st.session_state["result"] = run_inference(
                        pil_img, model, device, top_k
                    )

    with col_result:
        st.markdown("### 🌿 Classification Result")
        if "result" in st.session_state and st.session_state["result"]:
            display_result(st.session_state["result"], trained)
        else:
            st.markdown("""
            <div style="text-align:center;color:#aaa;padding:4rem 0;">
                <div style="font-size:3rem;">🌱</div>
                <div style="margin-top:1rem;">Upload a leaf image and click <b>Classify</b></div>
            </div>
            """, unsafe_allow_html=True)

    # ── Model Architecture Info ───────────────────────────────────────────────
    with st.expander("🧠 Model Architecture"):
        st.markdown("""
        **AyuLeafNet** is a **Hybrid CNN** architecture consisting of:

        | Component | Description |
        |-----------|-------------|
        | **Global Branch** | Pre-trained MobileNetV2 backbone with CBAM attention |
        | **Local Branch** | Custom 4-block CNN for fine-grained leaf texture features |
        | **CBAM** | Convolutional Block Attention Module (Channel + Spatial) |
        | **Fusion** | Concatenation → FC(512) → FC(256) |
        | **Output** | Class logits + Medicinal confidence score |

        The dual-branch design captures both **global structure** (leaf shape, venation)
        and **local texture** (surface details, edges) for more robust classification.
        """)

    # ── About Doshas ────────────────────────────────────────────────────────
    with st.expander("📖 Ayurvedic Concepts — Dosha, Rasa, Virya"):
        st.markdown("""
        **Doshas** (constitutional energies):
        - 🌬️ **Vata** — governs movement, nervous system
        - 🔥 **Pitta** — governs digestion, metabolism
        - 🌊 **Kapha** — governs structure, lubrication

        **Rasa** (taste) — 6 tastes: Sweet, Sour, Salty, Pungent, Bitter, Astringent

        **Virya** (potency) — either Heating (Ushna) or Cooling (Sheeta)

        **Vipaka** (post-digestive effect) — the long-term effect after metabolism

        *Disclaimer: This tool is for educational purposes only.
        Always consult a qualified Ayurvedic practitioner before using medicinal herbs.*
        """)


if __name__ == "__main__":
    main()
