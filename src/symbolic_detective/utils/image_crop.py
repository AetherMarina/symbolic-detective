import json
import re
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

from symbolic_detective.utils.image_files import artifact_key_from_path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
CROPS_MANIFEST = DEFAULT_DATA_DIR / "crops" / "crops.json"


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
        crop_dir=DEFAULT_DATA_DIR / "crops" / "unclassified",
        base_crop_dir=DEFAULT_DATA_DIR / "crops",
        single_image_path=None
):
    crop_path = Path(crop_dir)
    crop_path.mkdir(parents=True, exist_ok=True)
    motif_class = crop_path.name

    if single_image_path:
        img_path = Path(single_image_path)
        if not img_path.exists():
            print(f"Error: Could not find {img_path}")
            return
        images_to_process = [img_path]
        print(f"Processing single image: {img_path.name}")
    else:
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
                f"1. Draw crop box \n"
                f"2. Press 'Enter' to save crop\n"
                f"3. Press 'Esc' or 'q' to skip/move to next image"
            )
            ax.set_title(title)

            crop_coords = []

            # --- KEYBOARD SHORTCUTS ---
            def on_key(event):
                if event.key == 'enter':
                    # Close window to process the drawn coordinates
                    plt.close()
                elif event.key in ['escape', 'q']:
                    # Clear coordinates to signal "skip" and close window
                    crop_coords.clear()
                    plt.close()

            fig.canvas.mpl_connect('key_press_event', on_key)
            # --------------------------

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

            manager = plt.get_current_fig_manager()
            try:
                manager.window.showMaximized()
            except AttributeError:
                try:
                    manager.resize(*manager.window.maxsize())
                except AttributeError:
                    manager.full_screen_toggle()

            plt.show()

            # If Esc was pressed or window closed without drawing
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

            crop_stem = f"{img_file.stem}_occ_{occurrence:02d}"
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
    for p in ["europeana", "met"]:
        rapid_crop(
            raw_dir=DEFAULT_DATA_DIR / "raw" / p,
            crop_dir=DEFAULT_DATA_DIR / "crops" / "unclassified")
