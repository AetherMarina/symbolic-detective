import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def rapid_crop(raw_dir=DEFAULT_DATA_DIR/"raw", crop_dir=DEFAULT_DATA_DIR/"crops"/"eight_pointed"):
    raw_path = Path(raw_dir)
    crop_path = Path(crop_dir)
    crop_path.mkdir(parents=True, exist_ok=True)

    for img_file in raw_path.glob("*.*"):
        img = cv2.imread(str(img_file))
        if img is None:
            continue

        # Resize for display if the image is massive
        display_img = img.copy()
        scale = 1.0
        if display_img.shape[1] > 1000:
            scale = 1000 / display_img.shape[1]
            display_img = cv2.resize(display_img, None, fx=scale, fy=scale)

        occurrence = 1

        while True:
            # Update the window title so you know which occurrence you are cropping
            window_title = f"Motif {occurrence} on {img_file.name} (Drag & Space to save, C to next image)"

            # Opens a window. Drag a box, press Space/Enter to confirm.
            r = cv2.selectROI(window_title, display_img, fromCenter=False, showCrosshair=True)

            # If no box was drawn (user pressed 'c' or just hit Space without dragging)
            if r == (0, 0, 0, 0):
                print(f"Moving to next artifact. Total occurrences saved: {occurrence - 1}")
                break

            x, y, w, h = r

            # Map coordinates back to original resolution
            orig_x, orig_y = int(x / scale), int(y / scale)
            orig_w, orig_h = int(w / scale), int(h / scale)

            cropped_motif = img[orig_y:orig_y + orig_h, orig_x:orig_x + orig_w]

            # Construct the new filename with the occurrence counter
            # Example: 12345_primary_occ_01.jpg
            save_name = crop_path / f"{img_file.stem}_occ_{occurrence:02d}{img_file.suffix}"

            cv2.imwrite(str(save_name), cropped_motif)
            print(f"Saved: {save_name}")

            # Increment the counter so the next crop gets the next number
            occurrence += 1

        cv2.destroyAllWindows()


if __name__ == "__main__":
    # rapid_crop()
    rapid_crop(raw_dir=DEFAULT_DATA_DIR / "raw" / "rosette_dep3",
               crop_dir=DEFAULT_DATA_DIR / "crops" / "rosette")
    rapid_crop(raw_dir=DEFAULT_DATA_DIR / "raw" / "star_dep10",
               crop_dir=DEFAULT_DATA_DIR / "crops" / "five_pointed")
