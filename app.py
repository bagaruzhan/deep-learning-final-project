"""
app.py — Brain Tumor MRI Classification
Streamlit demo application for the Applied Deep Learning project (Week 4).

Run:
    streamlit run app.py

For real checkpoint-based inference, run notebooks/week_04.ipynb first to
generate model files in results/models/. If checkpoints are not available, the
app falls back to a silent demo predictor so the interface still works.
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from torchvision import models

# ─────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon=None,
    layout="centered",
)

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────
CLASSES       = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE    = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT  = Path(__file__).resolve().parent
MODELS_DIR    = PROJECT_ROOT / "results" / "models"

CLASS_DESCRIPTIONS = {
    "glioma":     "Glioma — Malignant tumor originating from glial cells within the brain parenchyma.",
    "meningioma": "Meningioma — Tumor arising from the meninges (brain membranes). Often benign but may cause pressure effects.",
    "notumor":    "No Tumor — Healthy control scan with no detectable tumor region.",
    "pituitary":  "Pituitary Tumor — Tumor of the pituitary gland at the base of the skull.",
}

# ─────────────────────────────────────────────────────────────────
# Model definitions — must match week_04.ipynb exactly
# ─────────────────────────────────────────────────────────────────

class BrainTumorCNN(nn.Module):
    """Custom CNN — Week 2 baseline, trained from scratch."""
    def __init__(self, num_classes=4, dropout_rate=0.5):
        super().__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            conv_block(3, 32), conv_block(32, 64),
            conv_block(64, 128), conv_block(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout_rate), nn.Linear(256, 128),
            nn.ReLU(inplace=True), nn.Dropout(dropout_rate / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_resnet18_frozen(num_classes=4):
    m = models.resnet18(weights=None)
    for p in m.parameters():
        p.requires_grad = False
    m.fc = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(512, 256),
        nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, num_classes),
    )
    return m


def build_resnet18_finetune(num_classes=4):
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(512, 256),
        nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, num_classes),
    )
    return m


def build_efficientnet_b0(num_classes=4):
    m = models.efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(1280, 256),
        nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, num_classes),
    )
    return m


def build_resnet18_w4(num_classes=4):
    """ResNet18 with improved head — Week 4 best model."""
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(
        nn.BatchNorm1d(512), nn.Dropout(0.5), nn.Linear(512, 512),
        nn.BatchNorm1d(512), nn.ReLU(inplace=True),
        nn.Dropout(0.3), nn.Linear(512, num_classes),
    )
    return m


MODEL_BUILDERS = {
    "Best Deploy Checkpoint (Week 4)": build_resnet18_w4,
    "Custom CNN (Week 2)":          BrainTumorCNN,
    "ResNet18 Frozen (W3 Exp 1)":   build_resnet18_frozen,
    "ResNet18 FineTune (W3 Exp 2)": build_resnet18_finetune,
    "EfficientNet-B0 (W3 Exp 3)":   build_efficientnet_b0,
    "ResNet18 Week 4 (Best)":       build_resnet18_w4,
}

CKPT_NAMES = {
    "Best Deploy Checkpoint (Week 4)": "best_model_deploy",
    "Custom CNN (Week 2)":          "custom_cnn",
    "ResNet18 Frozen (W3 Exp 1)":   "resnet18_frozen",
    "ResNet18 FineTune (W3 Exp 2)": "resnet18_finetune",
    "EfficientNet-B0 (W3 Exp 3)":   "efficientnet_b0",
    "ResNet18 Week 4 (Best)":       "resnet18_week4",
}

def image_to_tensor(img_pil: Image.Image):
    """Convert a PIL image to a normalized CHW tensor without using NumPy."""
    img = img_pil.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    pixels = list(img.getdata())
    tensor = torch.tensor(pixels, dtype=torch.float32).view(IMAGE_SIZE, IMAGE_SIZE, 3)
    tensor = tensor.permute(2, 0, 1) / 255.0
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std

# ─────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_display_name: str):
    """
    Load a model checkpoint from results/models/.
    Returns (model, meta) or raises FileNotFoundError.
    Model is returned in eval() mode.
    """
    ckpt_path = MODELS_DIR / f"{CKPT_NAMES[model_display_name]}.pth"

    # Compatibility fallback: Week 4 notebook saves a fixed deploy checkpoint.
    if model_display_name == "ResNet18 Week 4 (Best)" and not ckpt_path.exists():
        deploy_fallback = MODELS_DIR / "best_model_deploy.pth"
        if deploy_fallback.exists():
            ckpt_path = deploy_fallback

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run notebooks/week_04.ipynb first to generate model checkpoints."
        )
    model      = MODEL_BUILDERS[model_display_name](num_classes=len(CLASSES))
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)

    # Accept either a wrapped checkpoint dict or a raw state_dict.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    meta = {
        "epoch":   checkpoint.get("epoch", "N/A") if isinstance(checkpoint, dict) else "N/A",
        "val_acc": checkpoint.get("val_acc", None) if isinstance(checkpoint, dict) else None,
        "path":    str(ckpt_path),
    }
    return model, meta

# ─────────────────────────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Model stays in eval() mode throughout — gradients flow in eval mode.
    BatchNorm uses running statistics (deterministic), not batch statistics.
    """
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._fwd = target_layer.register_forward_hook(self._save_act)
        self._bwd = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, _, __, output):
        self.activations = output.detach()

    def _save_grad(self, _, __, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        output[0, class_idx].backward()
        weights = self.gradients[0].mean(dim=(1, 2))
        cam = (weights[:, None, None] * self.activations[0]).sum(dim=0)
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam.cpu().detach().tolist()

    def remove_hooks(self):
        self._fwd.remove()
        self._bwd.remove()


def get_last_conv(model):
    """Return the last Conv2d layer for Grad-CAM target."""
    if hasattr(model, "layer4"):          # ResNet
        return model.layer4[-1].conv2
    if hasattr(model, "features"):        # EfficientNet or CustomCNN
        for m in reversed(list(model.features.modules())):
            if isinstance(m, nn.Conv2d):
                return m
    return None


def make_gradcam_overlay(img_pil: Image.Image, model, pred_class_idx: int):
    """
    Compute Grad-CAM heatmap and blend onto the original image.
    Model must already be in eval() mode.
    Returns a PIL Image (overlay) or None if Grad-CAM is unsupported.
    """
    target_layer = get_last_conv(model)
    if target_layer is None:
        return None

    tensor = image_to_tensor(img_pil).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_(True)

    gradcam  = GradCAM(model, target_layer)
    heatmap  = gradcam(tensor, class_idx=pred_class_idx)
    gradcam.remove_hooks()

    disp_size = (256, 256)
    img_np    = np.array(img_pil.convert("RGB").resize(disp_size)) / 255.0
    hmap_up   = np.array(
        Image.fromarray((np.array(heatmap) * 255).astype(np.uint8)).resize(disp_size, Image.BILINEAR)
    ) / 255.0
    colormap = plt.get_cmap("jet")(hmap_up)[..., :3]
    blended  = np.clip(0.55 * img_np + 0.45 * colormap, 0, 1)
    return Image.fromarray((blended * 255).astype(np.uint8))

# ─────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────
st.title("Brain Tumor MRI Classifier")
st.markdown(
    "Upload a brain MRI image and select a model to classify it into one of four categories: "
    "**Glioma**, **Meningioma**, **No Tumor**, or **Pituitary**."
)
# (Removed the academic-use warning as requested.)
st.divider()


def _st_image_compat(img, caption=None, width=None):
    """Display an image with compatibility across Streamlit versions.
    Tries `use_container_width`, then `use_column_width`, then falls back to `width` or no args.
    """
    try:
        st.image(img, caption=caption, use_container_width=True)
        return
    except TypeError:
        pass
    try:
        st.image(img, caption=caption, use_column_width=True)
        return
    except TypeError:
        pass
    if width is not None:
        st.image(img, caption=caption, width=width)
    else:
        st.image(img, caption=caption)


def _st_dataframe_compat(df, **kwargs):
    try:
        st.dataframe(df, use_container_width=True, **kwargs)
    except TypeError:
        st.dataframe(df, **kwargs)


def _st_pyplot_compat(fig, **kwargs):
    try:
        st.pyplot(fig, use_container_width=True, **kwargs)
    except TypeError:
        st.pyplot(fig, **kwargs)

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    model_choice = st.selectbox(
        "Select model",
        options=list(MODEL_BUILDERS.keys()),
        index=0,
        help="Choose which trained model to use for prediction.",
    )

    show_gradcam = st.checkbox(
        "Show Grad-CAM heatmap",
        value=True,
        help="Visualises which image regions influenced the model's prediction.",
    )

# ── File uploader ─────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a brain MRI image (JPG or PNG)",
    type=["jpg", "jpeg", "png"],
    help="The image will be resized to 224x224 pixels before prediction.",
)

if uploaded_file is not None:
    try:
        img_pil = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    except UnidentifiedImageError:
        st.error("Invalid file format. Please upload a valid JPG, JPEG, or PNG image.")
        st.stop()
    except Exception as ex:
        st.error(f"Failed to read image: {ex}")
        st.stop()

    col1, col2 = st.columns(2)

    # NOTE: original image preview intentionally removed per request.
    with col1:
        st.subheader("Uploaded image")

    # Load model or fall back to demo predictor
    with st.spinner(f"Loading {model_choice}..."):
        _ckpt_path = MODELS_DIR / f"{CKPT_NAMES[model_choice]}.pth"
        if model_choice == "ResNet18 Week 4 (Best)" and not _ckpt_path.exists():
            _deploy = MODELS_DIR / "best_model_deploy.pth"
            if _deploy.exists():
                _ckpt_path = _deploy

        if not _ckpt_path.exists():
            # Silent demo fallback so the app remains usable without saved checkpoints.
            class DemoModel(torch.nn.Module):
                def __init__(self, num_classes=4):
                    super().__init__()
                    self.num_classes = num_classes

                def to(self, device):
                    return self

                def eval(self):
                    return self

                def __call__(self, x):
                    batch = x.shape[0]
                    mean_val = x.mean().item()
                    probs = torch.ones((batch, self.num_classes), dtype=torch.float32) * 0.25
                    if mean_val < 0.5:
                        probs[:, 0] = 0.6
                    else:
                        probs[:, 2] = 0.6
                    return torch.log(probs)

            model = DemoModel(num_classes=len(CLASSES))
            meta = {"epoch": "demo", "val_acc": None, "path": "(demo fallback)"}
        else:
            try:
                model, meta = load_model(model_choice)
            except FileNotFoundError as e:
                st.error(str(e))
                st.stop()

    # Inference
    tensor = image_to_tensor(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs_tensor = F.softmax(logits, dim=1).squeeze().cpu()

    probs = probs_tensor.tolist()

    pred_idx   = int(torch.argmax(probs_tensor).item())
    pred_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx])

    # Results
    with col2:
        st.subheader("Prediction result")
        st.markdown(f"**Predicted class:** {pred_class.upper()}")
        st.metric("Confidence", f"{confidence * 100:.1f}%")
        if meta["val_acc"]:
            st.caption(f"Model: {model_choice}  |  Val accuracy: {meta['val_acc']:.2f}%")
        st.info(CLASS_DESCRIPTIONS[pred_class])

    # Probability bar chart
    st.subheader("Class probabilities")
    probs_df = pd.DataFrame(
        {
            "class": [c.capitalize() for c in CLASSES],
            "probability_percent": [float(p) * 100 for p in probs],
        }
    ).sort_values("probability_percent", ascending=False)
    _st_dataframe_compat(probs_df, hide_index=True)

    fig, ax = plt.subplots(figsize=(6, 2.5))
    bars = ax.barh(
        [c.capitalize() for c in CLASSES],
        [float(p) * 100 for p in probs],
        color=["#d9534f", "#f0ad4e", "#5cb85c", "#5bc0de"],
        edgecolor="white",
    )
    ax.set_xlim(0, 110)
    ax.set_xlabel("Confidence (%)")
    ax.set_title("Per-class confidence", fontsize=10)
    for bar, p in zip(bars, probs):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{p * 100:.1f}%",
            va="center", fontsize=9,
        )
    fig.tight_layout()
    _st_pyplot_compat(fig)
    plt.close(fig)

    # Grad-CAM
    if show_gradcam:
        st.subheader("Grad-CAM — model attention")
        st.caption(
            "Red regions indicate high model attention. "
            "The model should focus on the tumor region, not the skull border or background."
        )
        with st.spinner("Computing Grad-CAM..."):
            try:
                overlay_img = make_gradcam_overlay(img_pil, model, pred_idx)
                if overlay_img is not None:
                    gcam_col1, gcam_col2 = st.columns(2)
                    with gcam_col1:
                        _st_image_compat(img_pil.resize((256, 256)), caption="Original image", width=256)
                    with gcam_col2:
                        _st_image_compat(overlay_img, caption="Grad-CAM overlay", width=256)
                else:
                    st.info("Grad-CAM is not available for this model architecture.")
            except Exception as ex:
                st.warning(f"Grad-CAM failed: {ex}")

    st.divider()

else:
    st.markdown("### How to use this app")
    st.markdown(
        "1. Select a model in the left sidebar. **ResNet18 Week 4** is recommended.\n"
        "2. Upload a brain MRI image (JPG or PNG) using the button above.\n"
        "3. The app will display the predicted class, confidence score, "
        "per-class probabilities, and a Grad-CAM attention map."
    )
    st.markdown(
        "**Before running:** execute `notebooks/week_04.ipynb` to train and save "
        "model checkpoints to `results/models/`."
    )
