import json
from pathlib import Path
import base64
import tempfile

import cv2
import streamlit as st
from PIL import Image

# Import the centralized search engine
from symbolic_detective.pipeline.hybrid_search import MotifIndex
from symbolic_detective.pipeline.preprocess import preprocess_for_opencv
from symbolic_detective.pipeline.shape_features import compute_shape_features
from symbolic_detective.utils.image_files import SUPPORTED_UPLOAD_TYPES, collect_images
from symbolic_detective.evaluation.configs import CLASSIFICATION_FROZEN_CONFIG, RETRIEVAL_FROZEN_CONFIG, MAX_SIM_THRESHOLD
from symbolic_detective.pipeline.motif_classifier import classify_motif

# --- Configuration & Paths ---
st.set_page_config(
    page_title="Symbolic Detective",
    page_icon="assets/rosette.jpg" if Path("assets/rosette.jpg").exists() else "✴️",
    layout="wide",
)

DATA_DIR = Path("data")
CROPS_DIR = DATA_DIR / "crops"
METADATA_FILE = DATA_DIR / "metadata.json"
INDEX_RET_FILE = DATA_DIR / "index_ret.npz"
INDEX_CLS_FILE = DATA_DIR / "index_cls.npz"
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

    .status-known {
        padding: 12px;
        background-color: rgba(16, 185, 129, 0.15);
        border-left: 4px solid #10b981;
        border-radius: 4px;
        margin-bottom: 15px;
    }
    .status-unknown {
        padding: 12px;
        background-color: rgba(245, 158, 11, 0.15);
        border-left: 4px solid #f59e0b;
        border-radius: 4px;
        margin-bottom: 15px;
    }
    .metric-value {
        font-family: monospace;
        font-size: 1.1em;
        font-weight: bold;
        color: #e2e8f0;
    }
    .disclaimer-text {
        font-size: 0.85em;
        color: #94a3b8;
        font-style: italic;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Data Loading & Caching ---
@st.cache_resource(show_spinner="Loading motif indices...")
def load_search_engine(corpus_fingerprint: str):
    """Builds or loads the centralized MotifIndex and loads API metadata."""
    index_ret = MotifIndex(feature_config=RETRIEVAL_FROZEN_CONFIG["feature_config"])
    index_cls = MotifIndex(feature_config=CLASSIFICATION_FROZEN_CONFIG["feature_config"])
    was_rebuilt = False

    if CROPS_DIR.exists():
        image_paths = collect_images(CROPS_DIR, recursive=True)

        if image_paths:
            # Classification index MUST strictly exclude negative controls
            classification_paths = [p for p in image_paths if p.parent.name != "negative_control"]

            was_rebuilt_ret = index_ret.load_or_build(INDEX_RET_FILE, image_paths)
            was_rebuilt_cls = index_cls.load_or_build(INDEX_CLS_FILE, classification_paths)
            was_rebuilt = was_rebuilt_ret or was_rebuilt_cls
            if was_rebuilt:
                print("⚙️ MotifIndex: Corpus or features modified. Index successfully rebuilt.")

    metadata = {}
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    thresholds = {'max_sim': MAX_SIM_THRESHOLD}
    return index_ret, index_cls, metadata, was_rebuilt, thresholds

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
    "<span class='accent-amber'>●</span> <span class='sub-brand'>**v0.2 — Hybrid Symbolic Search**</span>",
    unsafe_allow_html=True,
)

image_paths = (
    collect_images(CROPS_DIR, recursive=True)
    if CROPS_DIR.exists()
    else []
)
corpus_fingerprint = MotifIndex.compute_fingerprint(image_paths)
motif_index_ret, motif_index_cls, api_metadata, was_rebuilt, thresholds = load_search_engine(corpus_fingerprint)

# Safely trigger the UI toast exactly once per session if a rebuild occurred
if was_rebuilt and "notified_rebuild" not in st.session_state:
    st.toast("⚙️ Corpus or features modified. Index rebuilt.")
    st.session_state.notified_rebuild = True

if not getattr(motif_index_ret, "corpus", None):
    st.sidebar.error("⚠️ Corpus is empty.")
    st.warning("No motifs found in `data/crops/`. Add images to begin indexing.")
    st.stop()

st.sidebar.success(f"✦ Retrieval Index: {len(motif_index_ret.corpus)} motifs")
st.sidebar.success(f"✦ Classification Index: {len(motif_index_cls.corpus)} known motifs")

top_k = st.sidebar.slider("Retrieval Results (Top-N)", min_value=5, max_value=50, value=12)

# --- Main Interface Header ---
if rosette_b64:
    st.markdown(
        f"""
        <div class="brand-header">
            <img src="{rosette_b64}" class="brand-logo" alt="Rosette Viewfinder" />
            <h1 class="brand-title">Motif Identification & Retrieval</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.title("✴️ SYMBOLIC DETECTIVE")

st.markdown(
    "<p class='sub-brand'>Upload a cropped motif to identify its class and explore visually related artifacts.</p>",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("Upload Query Motif", type=SUPPORTED_UPLOAD_TYPES)

if uploaded_file is not None:
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
                suffix=Path(uploaded_file.name).suffix.lower(),
                delete=False,
        ) as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = Path(tmp.name)

        with st.spinner("Extracting geometry, performing classification & searching..."):
            # We manually extract the OpenCV geometry here solely to display the metrics and edge map in the UI.
            q_cv = preprocess_for_opencv(temp_path)
            q_shape_cls = compute_shape_features(q_cv, config=CLASSIFICATION_FROZEN_CONFIG["feature_config"]) if q_cv is not None else None

            try:
                # 1. Classification Pipeline
                classification_result = classify_motif(temp_path, motif_index_cls)
                predicted_class = classification_result["predicted_class"]
                max_sim = classification_result["max_sim"]
                weight_conf = classification_result["weight_conf"]
                threshold_used = classification_result["threshold_used"]

                # 2. Retrieval Pipeline
                top_results_ret = motif_index_ret.search(
                    temp_path,
                    alpha=RETRIEVAL_FROZEN_CONFIG["alpha"],
                    top_k=top_k
                )

            except ValueError as e:
                st.error(f"Search Engine Error: {e}")
                top_results_ret = None
                classification_result = None

        if top_results_ret is not None and classification_result is not None:

            # --- Classification Verdict Panel ---
            st.markdown("### 🔍 Classification Verdict")
            if predicted_class == "unknown":
                st.markdown(f"""
                <div class="status-unknown">
                    <h4 style="color: #f59e0b; margin:0 0 8px 0;">⚠ Unknown visual motif class</h4>
                    <p style="margin:0; color:#cbd5e1;">The system could not confidently identify this symbol among known historical classes.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                 st.markdown(f"""
                <div class="status-known">
                    <h4 style="color: #10b981; margin:0 0 8px 0;">✓ Recognized Motif: <span style="color:#ffffff;">{predicted_class.replace('_', ' ').title()}</span></h4>
                    <p style="margin:0; color:#cbd5e1;">Identified using a hybrid geometric and visual embedding strategy (K={CLASSIFICATION_FROZEN_CONFIG['k']} voting).</p>
                </div>
                """, unsafe_allow_html=True)

            # Extraction Column View
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                st.image(Image.open(temp_path), caption="Query Motif", width="stretch")
            with col2:
                if q_shape_cls:
                    vis = cv2.cvtColor(q_cv, cv2.COLOR_GRAY2BGR)
                    vis[q_shape_cls["edge_mask"] > 0] = (60, 146, 251)
                    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Extracted Edge Field (Class Config)", width="stretch")
                else:
                    st.warning("No edge structure detected.")
            with col3:
                st.markdown("#### ✴️ Classifier Confidence")
                st.markdown(f"Maximum Similarity (`max_sim`): <span class='metric-value'>{max_sim:.4f}</span> (Threshold: {thresholds['max_sim']:.4f})", unsafe_allow_html=True)
                st.markdown(f"Voting Agreement (`weight_conf`): <span class='metric-value'>{weight_conf:.4f}</span>", unsafe_allow_html=True)
                st.markdown("<p class='disclaimer-text'>*Note: These scores measure visual similarity and weighted class agreement, not calibrated probabilities. Classification does not establish historical or symbolic identity.*</p>", unsafe_allow_html=True)

            valid_results = top_results_ret

            st.markdown(
                f"""
                <h3 style="margin-top: 24px; margin-bottom: 4px;">✦ {len(valid_results)} Visually Similar Artifacts</h3>
                <p class="disclaimer-text" style="margin-top:0; margin-bottom:12px;">Visual similarity searches the dataset regardless of the classification verdict.</p>
                <hr style="margin-top: 0; margin-bottom: 12px; border: none; border-top: 1px solid #e0e0e0;">
                """,
                unsafe_allow_html=True,
            )

            for i in range(0, len(valid_results), 4):
                cols = st.columns(4)
                chunk = valid_results[i:i + 4]

                for j, res in enumerate(chunk):
                    with cols[j]:
                        st.image(Image.open(res["path"]), width="stretch")
                        st.markdown(f"<p style='color:#38bdf8; font-weight:bold; margin-bottom:0;'>{res['class'].replace('_', ' ').title()}</p>", unsafe_allow_html=True)
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
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
