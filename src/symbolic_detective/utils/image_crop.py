import json
import re
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

from symbolic_detective.utils.image_files import artifact_key_from_path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
CROPS_MANIFEST = DEFAULT_DATA_DIR / "crops.json"


def record_crop(crop_id: str, artifact_key: str, motif_class: str, filename: str, bbox: list) -> None:
    manifest = {}
    if CROPS_MANIFEST.exists():
        with open(CROPS_MANIFEST, "r", encoding="utf-8") as f:
            try:
                manifest = json.load(f)
            except json.JSONDecodeError:
                manifest = {}

    manifest[crop_id] = {
        "crop_id": crop_id,
        "artifact_key": artifact_key,
        "motif_class": motif_class,
        "filename": filename,
        "crop_box": bbox,
    }

    CROPS_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(CROPS_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def get_unprocessed_images(raw_dir, base_crop_dir):
    """
    Scans the crops directory to find already processed artifacts using their base keys,
    then returns files from raw that have not been cropped yet.
    """
    raw_path = Path(raw_dir)
    crop_path = Path(base_crop_dir)

    processed_keys = set()

    if crop_path.exists():
        for crop_file in crop_path.rglob("*.*"):
            if crop_file.name.startswith("."):
                continue

            # Use the robust centralized parser instead of a local regex
            artifact_key = artifact_key_from_path(crop_file)
            processed_keys.add(artifact_key)

    unprocessed = []
    if raw_path.exists():
        for img_file in raw_path.glob("*.*"):
            if img_file.name.startswith("."):
                continue

            # Compare using the exact same key logic
            raw_key = artifact_key_from_path(img_file)
            if raw_key not in processed_keys:
                unprocessed.append(img_file)

    return unprocessed


def rapid_crop(
        raw_dir=DEFAULT_DATA_DIR / "raw",
        crop_dir=DEFAULT_DATA_DIR / "crops" / "eight_pointed_star",
        base_crop_dir=DEFAULT_DATA_DIR / "crops",
):
    crop_path = Path(crop_dir)
    crop_path.mkdir(parents=True, exist_ok=True)
    motif_class = crop_path.name

    images_to_process = get_unprocessed_images(raw_dir, base_crop_dir)
    print(f"Found {len(images_to_process)} images left to process.")

    for img_file in images_to_process:
        img = cv2.imread(str(img_file))
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        artifact_key = artifact_key_from_path(img_file)
        occurrence = 1

        while True:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(img_rgb)

            title = (
                f"Motif {occurrence} on {img_file.name}\n"
                f"1. Click Magnifying Glass (bottom) to zoom. Click it AGAIN to turn it off.\n"
                f"2. Draw your crop box.\n"
                f"3. Close the window to save the crop.\n"
                f"(Close window WITHOUT drawing to move to the next image)"
            )
            ax.set_title(title)

            crop_coords = []

            def onselect(eclick, erelease):
                x1, y1 = int(eclick.xdata), int(eclick.ydata)
                x2, y2 = int(erelease.xdata), int(erelease.ydata)
                crop_coords.clear()
                crop_coords.extend([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])

            selector = RectangleSelector(
                ax,
                onselect,
                useblit=True,
                button=[1],
                interactive=True,
            )

            plt.show()

            if not crop_coords:
                print(f"Moving to next image. Total saved: {occurrence - 1}")
                break

            x1, y1, x2, y2 = crop_coords
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            cropped_motif = img[y1:y2, x1:x2]
            if cropped_motif.size == 0:
                print("Invalid crop, trying again...")
                continue

            crop_stem = f"{artifact_key}__occ_{occurrence:02d}"
            crop_filename = f"{crop_stem}{img_file.suffix}"
            save_name = crop_path / crop_filename
            cv2.imwrite(str(save_name), cropped_motif)

            record_crop(
                crop_id=crop_stem,
                artifact_key=artifact_key,
                motif_class=motif_class,
                filename=crop_filename,
                bbox=[x1, y1, x2, y2],
            )

            print(f"Saved: {save_name}")
            occurrence += 1


if __name__ == "__main__":
    rapid_crop()
