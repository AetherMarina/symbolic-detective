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

from symbolic_detective.pipeline.embeddings import (
    compute_embedding_similarity,
    extract_visual_embedding,
)
from symbolic_detective.pipeline.preprocess import (
    preprocess_for_dinov2,
    preprocess_for_opencv,
)
from symbolic_detective.pipeline.shape_features import (
    compute_shape_features,
    compute_shape_similarity,
)

# --- Configuration & Paths ---
st.set_page_config(
    page_title="Symbolic Detective",
    page_icon="assets/rosette.jpg" if Path("assets/rosette.jpg").exists() else "✴️",
    layout="wide",
)

DATA_DIR = Path("data")
CROPS_DIR = DATA_DIR / "crops"
METADATA_FILE = DATA_DIR / "metadata.json"
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
    /* Dark Slate Canvas */
    .stApp {
        background-color: #0b131f;
        color: #e2e8f0;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #080d15;
        border-right: 1px solid rgba(6, 182, 212, 0.2);
    }

    /* Header & Brand Title */
    .brand-header {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 0.5rem;
    }
    .brand-logo {
        width: 52px;
        height: 52px;
        border-radius: 10px;
        border: 1px solid #06b6d4;
        box-shadow: 0 0 14px rgba(6, 182, 212, 0.45);
        object-fit: cover;
    }
    .brand-title {
        font-family: monospace, system-ui, sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        color: #38bdf8;
        text-shadow: 0 0 16px rgba(56, 189, 248, 0.35);
        margin: 0;
    }

    /* Subtle subtitle and section badges */
    .sub-brand {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
        letter-spacing: 0.02em;
    }
    .accent-cyan {
        color: #06b6d4;
    }
    .accent-amber {
        color: #02c2c2;
    }

    /* Style only individual result cards */
    div[class*="st-key-motif_card_"] {
        background: #111c2b;
        border: 1px solid rgba(6, 182, 212, 0.18);
        border-radius: 10px;
        padding: 14px;
        transition: border-color 0.2s ease;
    }
    
    div[class*="st-key-motif_card_"]:hover {
        border-color: rgba(6, 182, 212, 0.55);
    }

    /* Widget Labels (Sliders, Uploader) - Crisp #94a3b8 */
    div[data-testid="stWidgetLabel"],
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] span,
    div[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] label p,
    div[data-testid="stFileUploader"] label span {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em;
    }

    /* Small helper limit text below the upload label */
    div[data-testid="stFileUploader"] small {
        color: #64748b !important;
    }

    /* Sliders and Metrics */
    .stSlider > div [data-baseweb="slider"] {
        accent-color: #06b6d4;
    }
    code {
        color: #fb923c !important;
        background-color: rgba(251, 146, 60, 0.1) !important;
        border: 1px solid rgba(251, 146, 60, 0.25) !important;
    }
    hr {
        border-color: rgba(6, 182, 212, 0.2) !important;
    }

    /* Dark Elevated Dropzone matching the #0b131f canvas */
    div[data-testid="stFileUploadDropzone"] {
        background-color: rgba(15, 23, 42, 0.65) !important;
        border: 1px dashed rgba(148, 163, 184, 0.28) !important;
        border-radius: 8px !important;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-testid="stFileUploadDropzone"]:hover {
        background-color: rgba(15, 23, 42, 0.95) !important;
        border-color: rgba(6, 182, 212, 0.5) !important;
        box-shadow: 0 0 12px rgba(6, 182, 212, 0.15) !important;
    }

    /* Dropzone inner text & icons */
    div[data-testid="stFileUploadDropzone"] div, 
    div[data-testid="stFileUploadDropzone"] span, 
    div[data-testid="stFileUploadDropzone"] svg {
        color: #94a3b8 !important;
        fill: #94a3b8 !important;
    }

    /* Dropzone browse files button */
    div[data-testid="stFileUploadDropzone"] button {
        background-color: rgba(6, 182, 212, 0.1) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(6, 182, 212, 0.3) !important;
        border-radius: 6px !important;
    }
    div[data-testid="stFileUploadDropzone"] button:hover {
        background-color: rgba(6, 182, 212, 0.2) !important;
        border-color: #06b6d4 !important;
        color: #ffffff !important;
    }
    
    /* Move the main content closer to the top */
    .stMainBlockContainer {
        padding-top: 1rem !important;
    }
    
    /* Prevent the remaining header background from covering the title */
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    
    /* Reduce the space above the sidebar branding */
    [data-testid="stSidebarHeader"] {
        height: 2.5rem !important;
        margin-bottom: 0.25rem !important;
    }
    
    [data-testid="stSidebarUserContent"] {
        padding-top: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Data Loading & Caching ---
@st.cache_resource(show_spinner="Indexing corpus features (this takes a moment on startup)...")
def load_corpus():
    """Scans the crops directory, extracts features, and caches them."""
    corpus = []

    if not CROPS_DIR.exists():
        return corpus

    metadata = {}
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    for img_path in CROPS_DIR.rglob("*.jpg"):
        cv_img = preprocess_for_opencv(img_path)
        dino_img = preprocess_for_dinov2(img_path)

        if cv_img is None or dino_img is None:
            continue

        shape_feats = compute_shape_features(cv_img)
        dino_feats = extract_visual_embedding(dino_img)

        if shape_feats and dino_feats is not None:
            obj_id = img_path.stem.removeprefix("crop_").split("_", 1)[0]
            prov = metadata.get(obj_id, {})

            corpus.append({
                "path": img_path,
                "shape_vector": shape_feats["feature_vector"],
                "embed_vector": dino_feats,
                "provenance": prov,
            })

    return corpus


# --- UI Sidebar ---
if rosette_b64:
    st.sidebar.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
            <img src="{rosette_b64}" width="34" height="34" style="border-radius: 6px; border: 1px solid #06b6d4; box-shadow: 0 0 8px rgba(6,182,212,0.4);" />
            <span style="font-family: monospace; font-size: 1.25rem; font-weight: bold; color: #06b6d4;">
                SYMBOLIC DETECTIVE</span>
            </span>
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

corpus_data = load_corpus()
st.sidebar.success(f"✦ Corpus Indexed: {len(corpus_data)} motifs")

alpha = st.sidebar.slider(
    "Shape contribution",
    min_value=0.0,
    max_value=1.0,
    value=0.20,
    step=0.05,
    format="%.2f",
    help=(
        "Higher values emphasize contour geometry. "
        "Lower values emphasize visual embedding similarity."
    ),
)

st.sidebar.caption(
    f"{alpha:.0%} shape · {1 - alpha:.0%} visual embedding"
)

top_k = st.sidebar.slider(
    "Number of results",
    min_value=5,
    max_value=50,
    value=12,
)

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

uploaded_file = st.file_uploader(
    "Upload Query Motif (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    temp_dir = DATA_DIR / "debug"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / "temp_query.jpg"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Tracing constellation geometry & latent representations..."):
        q_cv = preprocess_for_opencv(temp_path)
        q_dino = preprocess_for_dinov2(temp_path)

        q_shape = compute_shape_features(q_cv) if q_cv is not None else None
        q_embed = extract_visual_embedding(q_dino) if q_dino is not None else None

    if not q_shape or q_embed is None:
        st.error("Failed to extract features from the uploaded query. Ensure it is a clear crop.")
    else:
        query_col, info_col = st.columns([1, 3])

        with query_col:
            st.image(
                Image.open(temp_path),
                use_container_width=True,
            )

        with info_col:
            st.markdown("### Your query motif")
            st.write(
                "Results are ranked by a combination of shape "
                "and visual embedding similarity."
            )
            st.caption(
                f"Current weighting: {alpha:.0%} shape · "
                f"{1 - alpha:.0%} visual embedding"
            )

        with st.expander("Query diagnostics"):
            contour_col, signature_col = st.columns(2)

            with contour_col:
                vis = cv2.cvtColor(q_cv, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(
                    vis,
                    [q_shape["contour"]],
                    -1,
                    (60, 146, 251),
                    2,
                )
                st.image(
                    cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                    caption="Detected contour",
                    use_container_width=True,
                )

            with signature_col:
                st.markdown("**Geometric signature**")
                st.write(f"Solidity: {q_shape['solidity']:.3f}")
                st.write(f"Compactness: {q_shape['compactness']:.3f}")
                st.write(f"Convexity defects: {q_shape['defects']}")
        # col1, col2, col3 = st.columns([1, 1, 2])
        # with col1:
        #     st.image(Image.open(temp_path), caption="Query Motif", use_container_width=True)
        # with col2:
        #     vis = cv2.cvtColor(q_cv, cv2.COLOR_GRAY2BGR)
        #     cv2.drawContours(vis, [q_shape["contour"]], -1, (60, 146, 251), 2)
        #     st.image(
        #         cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
        #         caption="Detected Contour",
        #         use_container_width=True,
        #     )
        # with col3:
        #     st.markdown("### ✴️ Geometric Signature")
        #     st.markdown(f"- **Solidity:** `{q_shape['solidity']:.3f}`")
        #     st.markdown(f"- **Compactness:** `{q_shape['compactness']:.3f}`")
        #     st.markdown(f"- **Defects (Points):** `{q_shape['defects']}`")

        st.markdown("<hr>", unsafe_allow_html=True)

        results = []
        for item in corpus_data:
            if item["path"].name == uploaded_file.name:
                continue

            s_shape = compute_shape_similarity(q_shape["feature_vector"], item["shape_vector"])
            s_embed = compute_embedding_similarity(q_embed, item["embed_vector"])
            s_hybrid = (alpha * s_shape) + ((1.0 - alpha) * s_embed)

            results.append({
                "path": item["path"],
                "s_shape": s_shape,
                "s_embed": s_embed,
                "s_hybrid": s_hybrid,
                "provenance": item["provenance"],
            })

        results.sort(key=lambda x: x["s_hybrid"], reverse=True)
        top_results = results[:top_k]

        st.markdown(f"### ✦ {len(top_results)} closest visual matches")

        if not top_results:
            st.info("No matches available. Check that the corpus contains crops.")

        # Create a new row for every four results.
        for row_start in range(0, len(top_results), 4):
            cols = st.columns(4)

            for offset, res in enumerate(
                    top_results[row_start:row_start + 4]
            ):
                rank = row_start + offset + 1
                prov = res["provenance"] or {}

                title = prov.get("title") or "Untitled artifact"
                date = (
                        prov.get("objectDate")
                        or prov.get("period")
                        or "Date unknown"
                )
                culture = prov.get("culture")
                place = (
                        prov.get("city")
                        or prov.get("region")
                        or prov.get("country")
                )
                source_url = prov.get("objectURL")

                with cols[offset]:
                    with st.container(
                            key=f"motif_card_{rank}",
                            border=False,
                    ):
                        st.caption(f"Match #{rank}")
                        st.write(f"**{title}**")
                        st.caption(date)

                        context = [
                            value for value in (culture, place) if value
                        ]
                        st.caption(
                            " · ".join(context)
                            or "Culture/place not recorded"
                        )

                        st.image(
                            Image.open(res["path"]),
                            use_container_width=True,
                        )

                        st.markdown(
                            f"**Similarity score:** "
                            f"`{res['s_hybrid']:.3f}`"
                        )

                        if source_url:
                            st.link_button("View source artifact", source_url)

                        with st.expander("Match details"):
                            st.write(
                                f"Shape similarity: {res['s_shape']:.3f}"
                            )
                            st.write(
                                "Visual embedding similarity: "
                                f"{res['s_embed']:.3f}"
                            )
                            st.caption(
                                "Similarity scores rank visual matches; "
                                "they are not identification confidence."
                            )
