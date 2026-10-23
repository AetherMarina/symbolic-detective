import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CROP_DIR = PROJECT_ROOT / "data" / "crops"
CROPS_METADATA_FILE = CROP_DIR / 'crops.json'


def update_crops_json(json_filename=CROPS_METADATA_FILE, search_directory=CROP_DIR):
    json_path = Path(json_filename)
    base_dir = Path(search_directory)

    if not json_path.exists():
        print(f"Error: Could not find {json_filename}")
        return

    print("Scanning folders to build an index of all files...")

    # Scan disk once and map file names to their parent folder names
    file_index = {}
    for filepath in base_dir.rglob("*"):
        if filepath.is_file():
            # Store exact match (e.g., 'image.jpg' or 'image')
            file_index[filepath.name] = filepath.parent.name
            # Store 'stem' match (drops extension if it exists)
            file_index[filepath.stem] = filepath.parent.name

    # Load the JSON data
    with open(json_path, "r", encoding="utf-8") as f:
        crops_data = json.load(f)

    # Handle both lists and dictionaries
    items = crops_data if isinstance(crops_data, list) else crops_data.values()
    updated_count = 0

    for item in items:
        if isinstance(item, dict) and item.get("motif_class") == "unclassified":
            # Target the 'filename' field instead of 'crop_id'
            target_filename = item.get("filename")

            if not target_filename:
                continue

            # Instant O(1) lookup against our index
            if target_filename in file_index:
                subfolder_name = file_index[target_filename]
                # Skip if the file is still in the "unclassified" folder
                if subfolder_name != "unclassified":
                    item["motif_class"] = subfolder_name
                    updated_count += 1
                    print(f"Updated: {target_filename} -> {subfolder_name}")
            else:
                # Fallback check just in case the JSON has extensions but the disk doesn't
                target_stem = Path(target_filename).stem
                if target_stem in file_index:
                    subfolder_name = file_index[target_stem]
                    # Skip if the file is still in the "unclassified" folder
                    if subfolder_name != "unclassified":
                        item["motif_class"] = subfolder_name
                        updated_count += 1
                        print(f"Updated (via stem match): {target_filename} -> {subfolder_name}")
                else:
                    print(f"Warning: No file found anywhere for filename '{target_filename}'")

    # Only save to a new file if we actually updated anything
    if updated_count > 0:
        output_filename = CROP_DIR / json_path.with_name(f"{json_path.stem}_updated{json_path.suffix}")
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(crops_data, f, indent=4)

        print(f"\nProcess complete. {updated_count} items updated.")
        print(f"Saved to: {output_filename}")
    else:
        print("\nProcess complete. 0 items updated. No new file created.")


if __name__ == "__main__":
    update_crops_json()
