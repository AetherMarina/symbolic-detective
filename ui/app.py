"""
Streamlit Exploration Dashboard
Allows users to upload a query motif, adjust the hybrid weighting,
and visually inspect Top-K retrieved matches.
"""
import base64
import json
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image

# Import the centralized search engine
from symbolic_detective.pipeline.hybrid_search import MotifIndex
from symbolic_detective.pipeline.preprocess import preprocess_for_opencv
from symbolic_detective.pipeline.shape_features import compute_shape_features

# --- Configuration & Paths ---
st.set_page_config(
    page_title="Symbolic Detective",
    page_icon="assets/rosette.jpg" if Path("assets/rosette.jpg").exists() else "✴️",
    layout="wide",
)

DATA_DIR = Path("data")
CROPS_DIR = DATA_DIR / "crops"
METADATA_FILE = DATA_DIR / "metadata.json"
INDEX_FILE = DATA_DIR / "index.npz"
ROSETTE_PATH = Path("assets/rosette.jpg")


def get_base64_image(image_path: Path) -> str:
    """Reads a local image and encodes it to base64 for HTML embedding."""
    if image_path.exists():
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
            return f"data:image/jpeg;base64,{encoded}"
    return ""


rosette_b64 = get_base64_image(ROSETTE_PATH)

# --- Custom Styling (Cyan Neon & Amber Palette) ---
st.markdown(
    """
    <style>
    .stApp { background-color: #0b131f; color: #e2e8f0; }
    section[data-testid="stSidebar"] { background-color: #080d15; border-right: 1px solid rgba(6, 182, 212, 0.2); }
    .brand-header { display: flex; align-items: center; gap: 16px; margin-bottom: 0.5rem; }
    .brand-logo { width: 52px; height: 52px; border-radius: 10px; border: 1px solid #06b6d4; box-shadow: 0 0 14px rgba(6, 182, 212, 0.45); object-fit: cover; }
    .brand-title { font-family: monospace, system-ui, sans-serif; font-size: 2.2rem; font-weight: 700; letter-spacing: 0.05em; color: #38bdf8; text-shadow: 0 0 16px rgba(56, 189, 248, 0.35); margin: 0; }
    .sub-brand { color: #94a3b8; font-size: 0.95rem; margin-bottom: 1.5rem; letter-spacing: 0.02em; }
    .accent-amber { color: #06b6d4; }
    div[data-testid="stColumn"]:has(.element-container) { background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(56, 189, 248, 0.15); border-radius: 8px; padding: 12px; transition: border-color 0.2s ease, box-shadow 0.2s ease; }
    div[data-testid="stColumn"]:has(.element-container):hover { border-color: rgba(251, 146, 60, 0.45); box-shadow: 0 0 14px rgba(251, 146, 60, 0.25); }
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span { color: #94a3b8 !important; font-weight: 600 !important; letter-spacing: 0.02em; }
    .stSlider > div [data-baseweb="slider"] { accent-color: #06b6d4; }
    code { color: #fb923c !important; background-color: rgba(251, 146, 60, 0.08) !important; border: 1px solid rgba(251, 146, 60, 0.2) !important; }
    hr { border-color: rgba(6, 182, 212, 0.2) !important; }
    div[data-testid="stFileUploadDropzone"] { background-color: rgba(56, 189, 248, 0.08) !important; border: 2px dashed rgba(56, 189, 248, 0.6) !important; border-radius: 8px; }
    div[data-testid="stFileUploadDropzone"]:hover { background-color: rgba(56, 189, 248, 0.18) !important; border-color: #fb923c !important; }
    div[data-testid="stFileUploadDropzone"] div, div[data-testid="stFileUploadDropzone"] span, div[data-testid="stFileUploadDropzone"] small { color: #f8fafc !important; }
    /* Visible boxes */
    div[data-testid="stColumn"]:has(.element-container) { 
        background: rgba(15, 23, 42, 0.65); 
        border: 1px solid rgba(56, 189, 248, 0.15); 
        border-radius: 8px; 
        padding: 12px; 
        transition: border-color 0.2s ease, box-shadow 0.2s ease; 
    }
    div[data-testid="stColumn"]:has(.element-container):hover { 
        border-color: rgba(251, 146, 60, 0.45); 
        box-shadow: 0 0 14px rgba(251, 146, 60, 0.25); 
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Data Loading & Caching ---
@st.cache_resource(show_spinner="Delegating to MotifIndex engine...")
def load_search_engine():
    """Builds or loads the centralized MotifIndex and loads API metadata."""
    index = MotifIndex()

    if CROPS_DIR.exists():
        image_paths = []
        for ext in [".jpg", ".jpeg", ".png"]:
            image_paths.extend(list(CROPS_DIR.rglob(f"*{ext}")))

        if image_paths:
            # Try loading the persisted cache first
            if INDEX_FILE.exists():
                try:
                    index.load(INDEX_FILE)
                except RuntimeError as e:
                    # Triggered if FEATURE_VERSION or DINO_MODEL changes
                    st.toast(f"Cache invalidated: {e}. Rebuilding index...")
                    index.build_index(image_paths)
                    index.save(INDEX_FILE)
            else:
                # Build and save if no cache exists
                index.build_index(image_paths)
                index.save(INDEX_FILE)

    metadata = {}
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return index, metadata


# --- UI Sidebar ---
if rosette_b64:
    st.sidebar.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
            <img src="{rosette_b64}" width="34" height="34" style="border-radius: 6px; border: 1px solid #06b6d4; box-shadow: 0 0 8px rgba(6,182,212,0.4);" />
            <span style="font-family: monospace; font-size: 1.25rem; font-weight: bold; color: #06b6d4;">
                SYMBOLIC DETECTIVE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.sidebar.title("✴️ Symbolic Detective")

st.sidebar.markdown(
    "<span class='accent-amber'>●</span> <span class='sub-brand'>**v0.1 — Unknown Motif Finder**</span>",
    unsafe_allow_html=True,
)

motif_index, api_metadata = load_search_engine()

if not getattr(motif_index, "corpus", None):
    st.sidebar.error("⚠️ Corpus is empty.")
    st.warning("No motifs found in `data/crops/`. Add images to begin indexing.")
    st.stop()

st.sidebar.success(f"✦ Index Active: {len(motif_index.corpus)} motifs")

alpha = st.sidebar.slider(
    "Shape contribution",
    min_value=0.0,
    max_value=1.0,
    value=0.20,
    step=0.05,
    format="%.2f",
    help=(
        "Higher values emphasize edge-based geometric similarity. "
        "Lower values emphasize DINOv2 visual embedding similarity."
    )
)

st.sidebar.caption(f"{alpha:.0%} shape · {1 - alpha:.0%} visual embedding")
top_k = st.sidebar.slider("Number of results", min_value=5, max_value=50, value=12)

# --- Main Interface Header ---
if rosette_b64:
    st.markdown(
        f"""
        <div class="brand-header">
            <img src="{rosette_b64}" class="brand-logo" alt="Rosette Viewfinder" />
            <h1 class="brand-title">Visual Motif Retrieval</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.title("✴️ SYMBOLIC DETECTIVE")

st.markdown(
    "<p class='sub-brand'>Upload a cropped motif to find visually related artifacts across the collection.</p>",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("Upload Query Motif", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_dir = DATA_DIR / "debug"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / "temp_query.jpg"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Extracting geometric features and visual embeddings..."):
        # We manually extract the OpenCV geometry here solely to display the metrics and edge map in the UI.
        # MotifIndex handles its own internal extraction and scaling for the actual search.
        q_cv = preprocess_for_opencv(temp_path)
        q_shape = compute_shape_features(q_cv) if q_cv is not None else None

        try:
            top_results = motif_index.search(temp_path, alpha=alpha, top_k=top_k + 1)
        except ValueError as e:
            st.error(f"Search Engine Error: {e}")
            top_results = None

    if top_results is not None:
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.image(Image.open(temp_path), caption="Query Motif", use_container_width=True)
        with col2:
            if q_shape:
                vis = cv2.cvtColor(q_cv, cv2.COLOR_GRAY2BGR)
                vis[q_shape["edge_mask"] > 0] = (60, 146, 251)
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Extracted Edge Field", use_container_width=True)
            else:
                st.warning("No edge structure detected.")
        with col3:
            st.markdown("#### ✴️ Geometric signature")
            if q_shape:
                angular_spectrum = q_shape["angular_spectrum"]
                dominant_harmonic = int(angular_spectrum.argmax()) + 1
                dominant_magnitude = float(angular_spectrum.max())

                st.markdown(f"- Edge Density: `{q_shape['edge_density']:.4f}`")
                st.markdown(f"- Log Hu [0]: `{q_shape['log_hu'][0]:.3f}`")
                st.markdown(f"- Dominant Angular Harmonic: `{dominant_harmonic}`")
                st.markdown(f"- Harmonic Magnitude: `{dominant_magnitude:.3f}`")
            else:
                st.markdown("*(Shape extraction failed)*")

        # Filter out the exact uploaded image if it was pulled directly from the corpus
        valid_results = [res for res in top_results if res["path"].name != uploaded_file.name]
        # Enforce the strict slider count
        valid_results = valid_results[:top_k]

        st.markdown(f"### ✦ {len(valid_results)} closest visual matches")
        st.markdown("<hr>", unsafe_allow_html=True)

        for i in range(0, len(valid_results), 4):
            cols = st.columns(4)
            chunk = valid_results[i:i + 4]

            for j, res in enumerate(chunk):
                with cols[j]:
                    st.image(Image.open(res["path"]), use_container_width=True)

                    st.markdown(f"**Score:** `{res['s_hybrid']:.3f}`")
                    st.caption(f"Shape: {res['s_shape']:.2f} | Embed: {res['s_embed']:.2f}")

                    obj_id = res["path"].stem.split("_")[0]
                    prov = api_metadata.get(obj_id)

                    if prov:
                        period = prov.get("period") or prov.get("objectDate") or "Unknown Period"
                        culture = prov.get("culture") or "Unknown Culture"
                        st.caption(f"*{culture}, {period}*")
                    else:
                        st.caption(f"Artifact ID: {obj_id}")