import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import tempfile
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from llm_integration import get_emergency_measures

st.set_page_config(
    page_title="DamageScope AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

  :root {
    --bg:       #0b0f14;
    --surface:  #111720;
    --border:   #1e2a38;
    --accent:   #00c8ff;
    --muted:    #4a5e72;
    --text:     #dce8f0;
    --text-dim: #7a96aa;
    --green:    #12c26b;
    --yellow:   #f5c518;
    --orange:   #f07c23;
    --red:      #e8304a;
  }

  html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
  }

  [data-testid="stHeader"]  { background: transparent !important; }
  [data-testid="stSidebar"] { display: none; }
  footer    { visibility: hidden; }
  #MainMenu { visibility: hidden; }

  .hero {
    padding: 3.5rem 0 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
  }
  .hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 0.6rem;
  }
  .hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.2rem, 5vw, 3.8rem);
    font-weight: 800;
    line-height: 1.05;
    margin: 0 0 0.8rem;
    color: #fff;
    letter-spacing: -0.02em;
  }
  .hero-title span { color: var(--accent); }
  .hero-sub {
    font-size: 1rem;
    color: var(--text-dim);
    max-width: 560px;
    line-height: 1.6;
  }

  /* ── Upload panels — fixed to contain everything inside ── */
  .upload-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.2rem 1.2rem;
    transition: border-color 0.2s;
  }
  .upload-panel:hover { border-color: var(--accent); }
  .panel-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .panel-label::before {
    content: '';
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent);
  }

  [data-testid="stFileUploader"] > div {
    background: transparent !important;
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-dim) !important;
  }
  [data-testid="stFileUploader"] > div:hover { border-color: var(--accent) !important; }
  [data-testid="stFileDropzoneInstructions"]  { color: var(--text-dim) !important; }

  .stButton > button {
    width: 100%;
    background: var(--accent) !important;
    color: #000 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.06em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.9rem 2rem !important;
    transition: opacity 0.2s !important;
    margin-top: 0.5rem;
  }
  .stButton > button:hover    { opacity: 0.85 !important; }
  .stButton > button:disabled { opacity: 0.4  !important; cursor: not-allowed !important; }

  .result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 2rem;
    margin-top: 1.5rem;
  }
  .result-badge {
    display: inline-block;
    padding: 0.4rem 1.1rem;
    border-radius: 100px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
  }
  .result-class {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 0.3rem;
  }
  .result-confidence {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: var(--text-dim);
    margin-bottom: 1.6rem;
  }

  .prob-row   { margin-bottom: 0.65rem; }
  .prob-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-dim);
    letter-spacing: 0.08em;
    margin-bottom: 0.28rem;
  }
  .prob-track {
    background: var(--border);
    border-radius: 4px;
    height: 6px;
  }
  .prob-fill {
    height: 6px;
    border-radius: 4px;
    transition: width 0.8s ease;
  }
  .prob-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-dim);
    text-align: right;
    margin-top: 2px;
  }

  .interpretation {
    background: rgba(0,200,255,0.05);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin-top: 1.4rem;
    font-size: 0.88rem;
    color: var(--text-dim);
    line-height: 1.6;
  }
  .interp-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.4rem;
  }

  .disclaimer {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.76rem;
    color: var(--muted);
    margin-top: 1.5rem;
    line-height: 1.5;
  }
  .disclaimer strong { color: var(--text-dim); }

  .section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    color: var(--muted);
    text-transform: uppercase;
    margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
  }

  [data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    margin-top: 1rem;
  }
  [data-testid="stExpander"] summary {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--text-dim) !important;
  }

  [data-testid="stSpinner"] { color: var(--accent) !important; }
  .stImage figcaption { color: var(--text-dim) !important; font-size: 0.72rem !important; }

  [data-testid="stAlert"] {
    background: rgba(232,48,74,0.1) !important;
    border: 1px solid var(--red) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
  }

  .stDownloadButton > button {
    background: transparent !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    padding: 0.5rem 1rem !important;
  }
  .stDownloadButton > button:hover {
    background: rgba(0,200,255,0.08) !important;
  }

  /* image caption label inside panel */
  .img-caption {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #4a5e72;
    text-align: center;
    margin-top: 0.4rem;
    letter-spacing: 0.1em;
  }

  /* overlay label for heatmap image */
  .overlay-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: var(--accent);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
  }
  /* ── Upload panel: target the native st.container wrapper ── */
  [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"]:has(.panel-inner) {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    transition: border-color 0.2s;
  }
  [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"]:has(.panel-inner):hover {
    border-color: var(--accent);
  }

  /* ── Preview image: constrain inside panel, rounded ── */
  .preview-img-wrap {
    border-radius: 8px;
    overflow: hidden;
    margin-top: 0.6rem;
    border: 1px solid var(--border);
    background: #0b0f14;
  }
  .preview-img-wrap img { display: block; width: 100%; height: auto; }

  /* ── File name chip shown after upload ── */
  .file-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(0,200,255,0.07);
    border: 1px solid rgba(0,200,255,0.2);
    border-radius: 100px;
    padding: 0.25rem 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: var(--accent);
    letter-spacing: 0.08em;
    margin: 0.5rem 0 0.2rem;
  }

  /* ── Placeholder shown before upload ── */
  .img-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 140px;
    background: rgba(255,255,255,0.02);
    border: 1px dashed var(--border);
    border-radius: 8px;
    margin-top: 0.8rem;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    gap: 0.5rem;
  }

  /* ── Tab styling to match dark theme ── */
  [data-testid="stTabs"] [role="tablist"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
  }
  [data-testid="stTabs"] button[role="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    color: var(--text-dim) !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 0.45rem 0.9rem !important;
  }
  [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: rgba(0,200,255,0.05) !important;
  }
  [data-testid="stTabs"] [data-testid="stTabContent"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    padding: 1rem !important;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
CLASSES = ["No Damage", "Minor Damage", "Major Damage", "Destroyed"]
COLORS  = ["#12c26b",   "#f5c518",      "#f07c23",      "#e8304a"]

INTERPRETATIONS = {
    0: "No significant structural impact detected. The building appears intact "
       "with no visible damage from the disaster event.",
    1: "Small, localized damage observed. Structural integrity is mostly "
       "maintained; repairs may be needed.",
    2: "Significant structural damage present. The building has sustained major "
       "harm and may be unsafe for occupancy.",
    3: "Severe or near-total destruction detected. The structure has collapsed "
       "or is critically compromised.",
}

MEAN = (78.67216964243309, 86.80138104794152, 64.97934000711008)
STD  = (41.37580197971469, 36.72895827341534, 34.51997208576517)


# ─────────────────────────────────────────────────────────────────────────────
# Preview helper
# ─────────────────────────────────────────────────────────────────────────────

def preview_geotiff(uploaded_file) -> Image.Image:
    import rasterio
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    uploaded_file.seek(0)
    try:
        with rasterio.open(tmp_path) as src:
            r = src.read(1).astype(np.float32)
            g = src.read(2).astype(np.float32)
            b = src.read(3).astype(np.float32)
    finally:
        os.unlink(tmp_path)

    def to_uint8(band):
        band = np.clip(band, 0, None)
        if band.max() > 0:
            band = band / band.max() * 255.0
        return band.astype(np.uint8)

    rgb = np.dstack([to_uint8(r), to_uint8(g), to_uint8(b)])
    return Image.fromarray(rgb, mode="RGB")


def save_upload_to_tempfile(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap overlay on image
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers  (3-panel upgrade)
# ─────────────────────────────────────────────────────────────────────────────
from scipy.ndimage import gaussian_filter

def damage_logits_to_intensity_map(damage_np: np.ndarray) -> np.ndarray:
    """
    Derive a [0,1] float intensity map from raw model output.

    • Works for (C, H, W) logit maps (softmax across channels) or
      pre-computed (H, W) label maps.
    • Suppresses weak / background signal via thresholding.
    • Lightly smooths to reduce pixel noise.
    """
    if damage_np.ndim == 3:
        # Softmax across channel axis → take damage-weighted sum
        exp = np.exp(damage_np - damage_np.max(axis=0, keepdims=True))
        softmax = exp / (exp.sum(axis=0, keepdims=True) + 1e-8)   # (C,H,W)

        n_ch = softmax.shape[0]
        if n_ch >= 4:
            # channels assumed: [no-dmg, minor, major, destroyed]  (or bg first)
            # weight by severity index so minor < major < destroyed
            weights = np.array([0.0, 0.15, 0.55, 1.0]) if n_ch == 4 \
                 else np.array([0.0, 0.0, 0.15, 0.55, 1.0])[:n_ch]
            intensity = (softmax * weights[:, None, None]).sum(axis=0)
        else:
            # Fallback: use max channel activation
            intensity = softmax.max(axis=0)

        # Suppress background / very-low-confidence pixels
        threshold = np.percentile(intensity, 55)
        intensity = np.where(intensity > threshold, intensity, 0.0)

        # Renormalise to [0, 1]
        if intensity.max() > 0:
            intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)

    else:
        # Already a label map (H, W)  →  normalise directly
        intensity = damage_np.astype(np.float32)
        if intensity.max() > 0:
            intensity = intensity / intensity.max()

    # Light Gaussian blur to clean up noise (sigma tunable)
    intensity = gaussian_filter(intensity.astype(np.float64), sigma=1.2)
    intensity = np.clip(intensity, 0.0, 1.0)
    return intensity.astype(np.float32)


def render_change_heatmap(damage_np: np.ndarray) -> bytes:
    """
    Panel 1 — Change Heatmap (Intensity).
    Hot-cold colormap, high-contrast, focused on meaningful regions.
    """
    intensity = damage_logits_to_intensity_map(damage_np)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=130)
    fig.patch.set_facecolor("#0b0f14")
    ax.set_facecolor("#0b0f14")

    im = ax.imshow(
        intensity,
        cmap="inferno",
        vmin=0.0, vmax=1.0,
        interpolation="bilinear",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.ax.tick_params(colors="#7a96aa", labelsize=7)
    cbar.set_label("Change Intensity", color="#7a96aa", fontsize=8)
    cbar.outline.set_edgecolor("#1e2a38")

    ax.set_title("Change Heatmap · Intensity", color="#dce8f0",
                 fontsize=9.5, pad=8,
                 fontfamily="monospace")
    ax.axis("off")
    plt.tight_layout(pad=0.4)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, facecolor="#0b0f14")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def render_damage_class_map(damage_np: np.ndarray) -> bytes:
    """
    Panel 2 — Damage Class Map.
    Per-pixel argmax, clean discrete colours, legend.
    """
    import matplotlib.patches as mpatches

    if damage_np.ndim == 3:
        label_map = np.argmax(damage_np, axis=0).astype(np.uint8)
        # If 5-channel (bg + 4 classes) shift so class 0 = no-damage
        n_ch = damage_np.shape[0]
        if n_ch == 5:
            label_map = np.where(label_map == 0, 255, label_map - 1).astype(np.int16)
            label_map = np.clip(label_map, 0, 3).astype(np.uint8)
    else:
        label_map = np.clip(damage_np.astype(np.uint8), 0, 3)

    # RGBA lookup  (no-dmg, minor, major, destroyed)
    PALETTE = np.array([
        [ 18, 194, 107, 220],   # green  — no damage
        [245, 197,  24, 220],   # yellow — minor
        [240, 124,  35, 225],   # orange — major
        [232,  48,  74, 235],   # red    — destroyed
    ], dtype=np.uint8)

    rgba = PALETTE[label_map]   # (H, W, 4)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=130)
    fig.patch.set_facecolor("#0b0f14")
    ax.set_facecolor("#111720")

    ax.imshow(rgba, interpolation="nearest")

    # Contour borders between zones for crispness
    for cls_idx, hex_col in enumerate(["#12c26b", "#f5c518", "#f07c23", "#e8304a"]):
        mask = (label_map == cls_idx).astype(np.float32)
        if mask.max() > 0:
            ax.contour(mask, levels=[0.5], colors=[hex_col],
                       linewidths=0.8, alpha=0.7)

    legend_handles = [
        mpatches.Patch(facecolor="#12c26b", edgecolor="#0b0f14", label="No Damage"),
        mpatches.Patch(facecolor="#f5c518", edgecolor="#0b0f14", label="Minor"),
        mpatches.Patch(facecolor="#f07c23", edgecolor="#0b0f14", label="Major"),
        mpatches.Patch(facecolor="#e8304a", edgecolor="#0b0f14", label="Destroyed"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=7.5,
        framealpha=0.80,
        facecolor="#0b0f14",
        edgecolor="#1e2a38",
        labelcolor="#dce8f0",
    )

    ax.set_title("Damage Class Map", color="#dce8f0",
                 fontsize=9.5, pad=8, fontfamily="monospace")
    ax.axis("off")
    plt.tight_layout(pad=0.4)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, facecolor="#0b0f14")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def render_overlay_on_post_image(post_img: Image.Image,
                                  damage_np: np.ndarray) -> bytes:
    """
    Panel 3 — Overlay on Post-Disaster Image.
    Intensity-aware alpha: only meaningful hotspots get colour,
    the underlying satellite image stays visible everywhere else.
    """
    import matplotlib.patches as mpatches

    intensity = damage_logits_to_intensity_map(damage_np)   # (H, W) in [0,1]

    if damage_np.ndim == 3:
        label_map = np.argmax(damage_np, axis=0).astype(np.uint8)
        n_ch = damage_np.shape[0]
        if n_ch == 5:
            label_map = np.where(label_map == 0, 255, label_map - 1)
            label_map = np.clip(label_map, 0, 3).astype(np.uint8)
    else:
        label_map = np.clip(damage_np.astype(np.uint8), 0, 3)

    h, w = intensity.shape

    # ── Resize post image to match output map ─────────────────────────────
    base = post_img.resize((w, h), Image.BILINEAR).convert("RGB")
    base_arr = np.array(base, dtype=np.float32) / 255.0   # (H,W,3) in [0,1]

    # ── Build colour overlay (H, W, 3) per class ──────────────────────────
    CLASS_RGB = np.array([
        [0.07, 0.76, 0.42],   # green
        [0.96, 0.77, 0.09],   # yellow
        [0.94, 0.49, 0.14],   # orange
        [0.91, 0.19, 0.29],   # red
    ], dtype=np.float32)

    colour_layer = CLASS_RGB[label_map]   # (H, W, 3)

    # ── Intensity-aware alpha: hotspots pop, background stays clear ───────
    # alpha ramps up smoothly with intensity; below threshold → 0
    alpha = np.power(intensity, 0.65)          # gamma < 1 → wider spread
    alpha = np.where(intensity > 0.05, alpha, 0.0)   # hard-zero background
    alpha = alpha[..., np.newaxis]             # (H, W, 1) for broadcasting

    # Slightly dim base where there IS damage so colour pops
    dim_factor = 1.0 - 0.35 * alpha[..., 0:1]
    composited = base_arr * dim_factor + colour_layer * alpha
    composited = np.clip(composited, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=130)
    fig.patch.set_facecolor("#0b0f14")
    ax.set_facecolor("#0b0f14")

    ax.imshow(composited, interpolation="bilinear")

    # Thin contour outlines around damage zones
    for cls_idx, hex_col in enumerate(["#12c26b", "#f5c518", "#f07c23", "#e8304a"]):
        mask = ((label_map == cls_idx) & (intensity > 0.1)).astype(np.float32)
        if mask.max() > 0:
            ax.contour(mask, levels=[0.5], colors=[hex_col],
                       linewidths=1.0, alpha=0.85)

    legend_handles = [
        mpatches.Patch(facecolor="#12c26b", edgecolor="#0b0f14", label="No Damage"),
        mpatches.Patch(facecolor="#f5c518", edgecolor="#0b0f14", label="Minor"),
        mpatches.Patch(facecolor="#f07c23", edgecolor="#0b0f14", label="Major"),
        mpatches.Patch(facecolor="#e8304a", edgecolor="#0b0f14", label="Destroyed"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=7.5,
        framealpha=0.80,
        facecolor="#0b0f14",
        edgecolor="#1e2a38",
        labelcolor="#dce8f0",
    )

    ax.set_title("Post-Disaster · Damage Overlay", color="#dce8f0",
                 fontsize=9.5, pad=8, fontfamily="monospace")
    ax.axis("off")
    plt.tight_layout(pad=0.4)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, facecolor="#0b0f14")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str, device: torch.device):
    try:
        from model.major_project import SiameseMultiFusion
    except ImportError as exc:
        st.error(
            "Could not import `SiameseMultiFusion` from `major_project.py`.\n"
            f"Error: {exc}"
        )
        return None, None

    model = SiameseMultiFusion().to(device)
    classifier = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 4),
    ).to(device)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        classifier.load_state_dict(checkpoint["classifier"])
    except FileNotFoundError:
        st.error(f"Checkpoint not found at `{checkpoint_path}`.")
        return None, None
    except KeyError as exc:
        st.error(f"Checkpoint missing key {exc}.")
        return None, None

    model.eval()
    classifier.eval()
    return model, classifier


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(pre_path, post_path, model, classifier, device):
    try:
        from preprocessing.preprocessor import preprocess_geotiff, get_transforms
    except ImportError as exc:
        raise ImportError(f"Could not import preprocessor: {exc}")

    inference_transform = get_transforms(MEAN, STD, image_size=256, train=False)
    pre_tensor  = preprocess_geotiff(pre_path,  MEAN, STD, inference_transform)
    post_tensor = preprocess_geotiff(post_path, MEAN, STD, inference_transform)

    pre_tensor  = pre_tensor.unsqueeze(0).to(device)
    post_tensor = post_tensor.unsqueeze(0).to(device)

    with torch.inference_mode():
        emb_pre, emb_post, damage_map = model(pre_tensor, post_tensor)
        diff   = torch.abs(emb_pre - emb_post)
        logits = classifier(diff * 2)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx  = int(np.argmax(probs))
    damage_np = damage_map[0].cpu().numpy()
    return pred_idx, probs, damage_np


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def render_prob_bars(probs) -> str:
    html = ""
    for cls, col, p in zip(CLASSES, COLORS, probs):
        pct = p * 100
        html += f"""
        <div class="prob-row">
          <div class="prob-label">{cls}</div>
          <div class="prob-track">
            <div class="prob-fill" style="width:{pct:.1f}%; background:{col};"></div>
          </div>
          <div class="prob-val">{pct:.1f}%</div>
        </div>"""
    return html


def build_summary_text(pred_idx, probs) -> str:
    lines = [
        "═══════════════════════════════════════",
        "    DAMAGESCOPE AI — ASSESSMENT REPORT",
        "═══════════════════════════════════════",
        f"Predicted Class : {CLASSES[pred_idx]}",
        f"Confidence      : {probs[pred_idx]*100:.1f}%",
        "",
        "Class Probabilities:",
    ]
    for cls, p in zip(CLASSES, probs):
        lines.append(f"  {cls:<16} {p*100:5.1f}%")
    lines += [
        "",
        "Interpretation:",
        INTERPRETATIONS[pred_idx],
        "",
        "─────────────────────────────────────",
        "DISCLAIMER: AI-assisted — validate with expert judgment.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# App Layout
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">🛰 Satellite Imagery Analysis</div>
  <div class="hero-title">Damage<span>Scope</span> AI</div>
  <p class="hero-sub">
    Upload pre- and post-disaster satellite imagery to assess structural
    damage severity using AI-powered change detection.
  </p>
</div>
""", unsafe_allow_html=True)

with st.expander("⚙️  Model Configuration", expanded=False):
    checkpoint_path = st.text_input(
        "Checkpoint path",
        value=os.path.join(PROJECT_ROOT, "model", "best_model.pth"),
        help="Path to best_model.pth relative to app.py",
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.markdown(
    f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;'
    f'color:#4a5e72;margin-bottom:1.5rem;">DEVICE: {str(device).upper()}</div>',
    unsafe_allow_html=True,
)

# ── Upload panels ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Input Imagery</div>', unsafe_allow_html=True)

col_pre, col_post = st.columns(2, gap="large")

# Store previews in session state so they survive reruns
if "pre_preview" not in st.session_state:
    st.session_state.pre_preview = None
if "post_preview" not in st.session_state:
    st.session_state.post_preview = None
if "inference_result" not in st.session_state:
    st.session_state["inference_result"] = None
if "pre_name" not in st.session_state:
    st.session_state["pre_name"] = None
if "post_name" not in st.session_state:
    st.session_state["post_name"] = None
if "llm_response" not in st.session_state:
    st.session_state["llm_response"] = None


with col_pre:
    st.markdown('<div class="upload-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Pre-Disaster Image</div>', unsafe_allow_html=True)
    pre_file = st.file_uploader(
        "Upload pre-disaster image",
        type=["tif", "tiff"],
        key="pre",
        label_visibility="collapsed",
    )
    if pre_file:
        try:
            img = preview_geotiff(pre_file)
            st.session_state.pre_preview = img
        except Exception as exc:
            st.warning(f"Preview unavailable: {exc}")
    if st.session_state.pre_preview is not None:
        st.image(st.session_state.pre_preview, use_container_width=True)
        st.markdown('<div class="img-caption">PRE-DISASTER</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_post:
    st.markdown('<div class="upload-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Post-Disaster Image</div>', unsafe_allow_html=True)
    post_file = st.file_uploader(
        "Upload post-disaster image",
        type=["tif", "tiff"],
        key="post",
        label_visibility="collapsed",
    )
    if post_file:
        try:
            img = preview_geotiff(post_file)
            st.session_state.post_preview = img
        except Exception as exc:
            st.warning(f"Preview unavailable: {exc}")
    if st.session_state.post_preview is not None:
        st.image(st.session_state.post_preview, use_container_width=True)
        st.markdown('<div class="img-caption">POST-DISASTER</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Run button ─────────────────────────────────────────────────────────────────
st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)

run_col, _ = st.columns([1, 2])
with run_col:
    run_btn = st.button(
        "⚡  Run Assessment",
        disabled=not (pre_file and post_file),
        use_container_width=True,
    )

# ── Inference & results ────────────────────────────────────────────────────────
# ── Inference: only run and store results ─────────────────────────────────────
if run_btn and pre_file and post_file:

    pre_tmp_path = None
    post_tmp_path = None

    with st.spinner("Running AI damage assessment …"):

        model, classifier = load_model(checkpoint_path, device)
        if model is None:
            st.stop()

        try:
            pre_file.seek(0)
            post_file.seek(0)
            pre_tmp_path = save_upload_to_tempfile(pre_file)
            post_tmp_path = save_upload_to_tempfile(post_file)

            pred_idx, probs, damage_np = run_inference(
                pre_tmp_path, post_tmp_path, model, classifier, device
            )

            # Store assessment results for later display
            st.session_state["inference_result"] = {
                "pred_idx": pred_idx,
                "probs": probs,
                "damage_np": damage_np,
            }
            st.session_state["pre_name"] = pre_file.name
            st.session_state["post_name"] = post_file.name

            # Reset old LLM response when a new assessment is run
            st.session_state["llm_response"] = None

        except Exception as exc:
            st.error(f"Inference failed: {exc}")
            st.stop()

        finally:
            for p in (pre_tmp_path, post_tmp_path):
                if p and os.path.exists(p):
                    os.unlink(p)


# ── Section 1: Assessment Results ─────────────────────────────────────────────
if st.session_state["inference_result"] is not None:

    pred_idx = st.session_state["inference_result"]["pred_idx"]
    probs = st.session_state["inference_result"]["probs"]
    damage_np = st.session_state["inference_result"]["damage_np"]

    st.markdown('<div class="section-title">Assessment Results</div>', unsafe_allow_html=True)

    res_col, map_col = st.columns([3, 2], gap="large")

    with res_col:
        color = COLORS[pred_idx]
        badge_style = f"background:{color}22; color:{color}; border:1px solid {color}88;"

        st.markdown(f"""
        <div class="result-card">
          <div class="result-badge" style="{badge_style}">
            {'&#9632; ' * (pred_idx + 1)}Severity Level {pred_idx}
          </div>
          <div class="result-class" style="color:{color};">{CLASSES[pred_idx]}</div>
          <div class="result-confidence">Confidence: {probs[pred_idx]*100:.1f}%</div>

          {render_prob_bars(probs)}

          <div class="interpretation">
            <div class="interp-title">Interpretation</div>
            {INTERPRETATIONS[pred_idx]}
          </div>

          <div class="disclaimer">
            <strong>Disclaimer:</strong> This tool is AI-assisted and supports,
            not replaces, expert judgment. Results should be validated by
            trained field personnel before operational decisions are made.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with map_col:
        tab1, tab2, tab3 = st.tabs([
            "🔥 Change Heatmap",
            "🗺️ Class Map",
            "🛰️ Overlay",
        ])

        with tab1:
            st.markdown(
                '<div class="overlay-label">Change Intensity · Inferno Scale</div>',
                unsafe_allow_html=True,
            )
            st.image(render_change_heatmap(damage_np), use_container_width=True)

        with tab2:
            st.markdown(
                '<div class="overlay-label">Predicted Damage Classes</div>',
                unsafe_allow_html=True,
            )
            st.image(render_damage_class_map(damage_np), use_container_width=True)

        with tab3:
            if st.session_state.post_preview is not None:
                st.markdown(
                    '<div class="overlay-label">Post-Disaster + Damage Overlay</div>',
                    unsafe_allow_html=True,
                )
                st.image(
                    render_overlay_on_post_image(
                        st.session_state.post_preview, damage_np
                    ),
                    use_container_width=True,
                )
            else:
                st.info("Post-disaster preview not available for overlay.")

    with st.expander("🔬  Technical Details"):
        st.markdown(f"**Raw probabilities:** `{[f'{p:.4f}' for p in probs]}`")
        st.markdown(f"**Predicted index:** `{pred_idx}` → `{CLASSES[pred_idx]}`")
        st.markdown(f"**Damage map shape:** `{damage_np.shape}`  (channels × H × W)")
        st.markdown(f"**Device:** `{device}`")
        st.markdown(
            "**Pipeline:** `preprocess_geotiff()` → rasterio band reads → z-score "
            "→ `get_transforms(train=False)` → resize 256×256 → backbone "
            "→ cross-attention fusion → `|emb_pre − emb_post| × 2` "
            "→ classifier → softmax"
        )

    st.download_button(
        label="⬇  Download Report (.txt)",
        data=build_summary_text(pred_idx, probs),
        file_name="damagescope_report.txt",
        mime="text/plain",
    )

    # ── Section 2: Safety Precautions ────────────────────────────────────────
    st.markdown('<div class="section-title">Safety Precautions</div>', unsafe_allow_html=True)

    st.info("Assessment is complete. Click the button below to generate safety precautions.")

    if st.button("Give Safety Precautions", use_container_width=True):
        with st.spinner("Generating safety precautions..."):
            success, llm_response = get_emergency_measures(
                pred_idx=pred_idx,
                probs=probs,
                pre_filename=st.session_state["pre_name"],
                post_filename=st.session_state["post_name"],
            )

            if success:
                st.session_state["llm_response"] = llm_response
            else:
                st.session_state["llm_response"] = f"Error: {llm_response}"

    if st.session_state["llm_response"]:
        st.markdown("### Generated Safety Precautions")
        st.markdown(st.session_state["llm_response"])