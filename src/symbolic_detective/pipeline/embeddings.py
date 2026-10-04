"""
Pretrained Visual Embedding Extraction for Symbolic Detective v0.1.
Uses self-supervised DINOv2 (ViT-S/14) to extract dense latent representations
from preprocessed 224x224 RGB motif crops.
"""
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Global model cache to avoid reloading weights per query
_DINO_MODEL = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Standard ImageNet normalization used by DINOv2
DINO_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def get_dinov2_model(model_name: str = "dinov2_vits14"):
    """
    Loads pretrained DINOv2 via torch.hub in eval mode.
    """
    global _DINO_MODEL
    if _DINO_MODEL is None:
        # Downloads from torch hub cache (~85MB for ViT-S/14)
        _DINO_MODEL = torch.hub.load("facebookresearch/dinov2", model_name)
        _DINO_MODEL.to(_DEVICE)
        _DINO_MODEL.eval()
    return _DINO_MODEL


def extract_visual_embedding(
        rgb_224_img: np.ndarray,
        model_name: str = "dinov2_vits14"
) -> Optional[np.ndarray]:
    """
    Extracts an L2-normalized dense latent representation from a (224, 224, 3) RGB crop.

    Returns:
      1D numpy array of shape (384,) for ViT-S/14.
    """
    if rgb_224_img is None or rgb_224_img.shape != (224, 224, 3):
        return None

    model = get_dinov2_model(model_name)

    # Convert numpy RGB to normalized Torch Tensor: shape (1, 3, 224, 224)
    tensor = DINO_TRANSFORMS(rgb_224_img).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        # Forward pass generates CLS token embedding
        features = model(tensor)
        # L2-normalize feature vector
        features = torch.nn.functional.normalize(features, p=2, dim=1)

    return features.squeeze(0).cpu().numpy()


def compute_embedding_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Computes cosine similarity between two L2-normalized latent vectors.
    Since ||a|| = ||b|| = 1, cosine similarity is simply the dot product.
    Rescales from [-1, 1] to [0, 1] for linear hybridization:
      S_embed = (cos_sim + 1) / 2
    """
    cos_sim = float(np.dot(vec_a, vec_b))
    # Map [-1.0, 1.0] to [0.0, 1.0]
    normalized_sim = 0.5 * (cos_sim + 1.0)
    return max(0.0, min(1.0, normalized_sim))


if __name__ == "__main__":
    import sys
    from symbolic_detective.pipeline.preprocess import preprocess_for_dinov2

    if len(sys.argv) > 1:
        crop_path = Path(sys.argv[1])
        preprocessed = preprocess_for_dinov2(crop_path)

        if preprocessed is not None:
            emb = extract_visual_embedding(preprocessed)
            print(f"Extracted DINOv2 embedding for: {crop_path.name}")
            print(f"Embedding shape: {emb.shape} | L2 Norm: {np.linalg.norm(emb):.4f}")
            print(f"First 5 dimensions: {np.round(emb[:5], 4)}")
        else:
            print(f"Failed to load or preprocess {crop_path.name}")
