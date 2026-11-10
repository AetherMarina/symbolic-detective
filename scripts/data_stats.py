import re
import json
import os
from pathlib import Path
from collections import defaultdict

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
METADATA_FILE = DEFAULT_DATA_DIR / 'metadata.json'
RAW_DIR = DEFAULT_DATA_DIR / "raw"
CROP_DIR = DEFAULT_DATA_DIR / "crops"
CROPS_JSON_FILE = CROP_DIR / "crops.json"


def check_metadata(metadata_path, raw_path):
    stats = {
        "expected_count": 0,
        "actual_count": 0,
        "missing_images_count": 0,
        "extra_images_count": 0,
        "warnings": []
    }

    if not metadata_path.exists():
        stats["warnings"].append(f"CRITICAL: Could not find '{metadata_path}'.")
        return stats

    with open(metadata_path, 'r', encoding='utf-8') as file:
        try:
            metadata = json.load(file)
        except json.JSONDecodeError:
            stats["warnings"].append(f"CRITICAL: '{metadata_path}' is not valid JSON.")
            return stats

    # 1. Index actual files
    actual_files = {}
    for root, dirs, files in os.walk(raw_path):
        for file_name in files:
            if file_name.startswith('.'):
                continue
            actual_files[file_name] = os.path.join(root, file_name)

    # 2. Extract expected files
    expected_files = {}
    for object_id, data in metadata.items():
        local_path = data.get("local_image_path", "")
        if local_path:
            file_name = os.path.basename(local_path)
            expected_files[file_name] = object_id

    stats["expected_count"] = len(expected_files)
    stats["actual_count"] = len(actual_files)

    # 3. Find Discrepancies
    for expected_filename, object_id in expected_files.items():
        if expected_filename not in actual_files:
            stats["warnings"].append(f"Metadata missing image [ID {object_id}]: '{expected_filename}'")
            stats["missing_images_count"] += 1

    for actual_filename, full_path in actual_files.items():
        if actual_filename not in expected_files:
            stats["warnings"].append(f"Extra image not in JSON: '{full_path}'")
            stats["extra_images_count"] += 1

    return stats


def check_crops(raw_path, crop_path, crops_json_path=None):
    stats = {
        "total_raw": 0,
        "cropped_raw": 0,
        "uncropped_raw": 0,
        "total_crops": 0,
        "distinct_crops": 0,
        "total_crops_no_unclassified": 0,
        "distinct_crops_no_unclassified": 0,
        "successful_matches": 0,
        "orphaned_crops": 0,
        "unregistered_crops": 0,
        "missing_crop_files": 0,
        "folder_counts": defaultdict(int),
        "json_folder_counts": defaultdict(int),
        "warnings": []
    }

    if not raw_path.exists() or not crop_path.exists():
        stats["warnings"].append("CRITICAL: 'raw' or 'crops' folder does not exist.")
        return stats

    # 1. Index raw folder
    raw_index = {}
    for raw_file in raw_path.rglob("*.*"):
        if raw_file.name.startswith('.'):
            continue
        stats["total_raw"] += 1
        raw_index[raw_file.stem] = raw_file

    # 2. Parse crops.json dynamically mapping motif_class to expected folder
    registered_crops = set()
    has_crops_json = False

    if crops_json_path and crops_json_path.exists():
        try:
            with open(crops_json_path, 'r', encoding='utf-8') as f:
                crops_data = json.load(f)
            has_crops_json = True

            iterable_items = []
            if isinstance(crops_data, dict):
                for k, v in crops_data.items():
                    if isinstance(v, dict):
                        v.setdefault("_key", k)
                        iterable_items.append(v)
                    else:
                        iterable_items.append({"_key": k, "value": v})
            elif isinstance(crops_data, list):
                iterable_items = crops_data

            # Store mapping as (expected_folder, filename)
            for entry in iterable_items:
                if not isinstance(entry, dict):
                    continue

                # The expected folder is the motif_class (default to empty string if missing)
                expected_folder = entry.get("motif_class", "")

                file_ref = (
                        entry.get("filename")
                        or entry.get("file_name")
                        or entry.get("crop_file")
                        or entry.get("image_path")
                        or entry.get("local_crop_path")
                        or entry.get("_key")
                )
                if file_ref:
                    filename = Path(file_ref).name
                    registered_crops.add((expected_folder, filename))
                    folder_display = expected_folder if expected_folder else "root (no subfolder)"
                    stats["json_folder_counts"][folder_display] += 1

        except json.JSONDecodeError:
            stats["warnings"].append(f"CRITICAL: '{crops_json_path}' is not valid JSON.")
    elif crops_json_path:
        stats["warnings"].append(f"WARNING: 'crops.json' not found at '{crops_json_path}'.")

    # 3. Scan ALL crops on disk and compare their physical folder to JSON
    found_crops = set()
    processed_stems = set()
    processed_stems_no_unclassified = set()

    for crop_file in crop_path.rglob("*.*"):
        if crop_file.name.startswith('.') or crop_file.name == "crops.json":
            continue

        stats["total_crops"] += 1

        # Determine relative folder structure (e.g., "rosette", "eight_petal_rossete")
        rel_folder_path = crop_file.parent.relative_to(crop_path)
        rel_folder = rel_folder_path.as_posix() if str(rel_folder_path) != "." else ""

        folder_display_name = rel_folder if rel_folder else "root (no subfolder)"
        stats["folder_counts"][folder_display_name] += 1

        original_stem = re.sub(r'_occ_\d+$', '', crop_file.stem)
        processed_stems.add(original_stem)

        # Track counts excluding "unclassified" and any of its subdirectories
        is_unclassified = (rel_folder == "unclassified") or rel_folder.startswith("unclassified/")
        if not is_unclassified:
            stats["total_crops_no_unclassified"] += 1
            processed_stems_no_unclassified.add(original_stem)

        # Match against raw files
        if original_stem in raw_index:
            stats["successful_matches"] += 1
        else:
            stats["warnings"].append(f"Orphaned crop (raw missing): '{original_stem}' from '{crop_file}'")
            stats["orphaned_crops"] += 1

        # Track what we actually found on disk as (actual_folder, filename)
        filename = crop_file.name
        found_crops.add((rel_folder, filename))

        # If it's on disk but JSON doesn't know about it being in THIS folder
        if has_crops_json and (rel_folder, filename) not in registered_crops:
            stats["warnings"].append(f"Disk file unexpected in folder '{rel_folder}/': '{filename}'")
            stats["unregistered_crops"] += 1

    # 4. Check for JSON entries missing from their expected directories
    if has_crops_json:
        for expected_folder, expected_file in registered_crops:
            if (expected_folder, expected_file) not in found_crops:
                stats["warnings"].append(
                    f"JSON entry missing on disk: Expected '{expected_file}' inside folder '{expected_folder}/'"
                )
                stats["missing_crop_files"] += 1

    stats["distinct_crops"] = len(processed_stems)
    stats["distinct_crops_no_unclassified"] = len(processed_stems_no_unclassified)

    # 5. Calculate cropped vs uncropped raw
    for raw_stem in raw_index.keys():
        if raw_stem in processed_stems:
            stats["cropped_raw"] += 1
        else:
            stats["uncropped_raw"] += 1

    return stats


def print_dashboard(meta_stats, crop_stats):
    print("\n" + "=" * 60)
    print("                 PROJECT STATISTICS DASHBOARD")
    print("=" * 60)

    print("\n--- 1. Metadata vs. Raw Images ---")
    print(f"Total entries in metadata:                     {meta_stats['expected_count']}")
    print(f"Total actual images on disk:                   {meta_stats['actual_count']}")
    print(f"Metadata missing images:                       {meta_stats['missing_images_count']}")
    print(f"Extra images (missing meta):                   {meta_stats['extra_images_count']}")

    print("\n--- 2. Raw vs. Cropped Images ---")
    print(f"Total raw images scanned:                      {crop_stats['total_raw']}")
    print(f"Images already cropped:                        {crop_stats['cropped_raw']}")
    print(f"Images NOT YET cropped:                        {crop_stats['uncropped_raw']}")
    print(f"Total crop files generated:                    {crop_stats['total_crops']}")
    print(f"Distinct images with crops:                    {crop_stats['distinct_crops']}")
    print(f"Total crop files generated w/o unclassified:   {crop_stats['total_crops_no_unclassified']}")
    print(f"Distinct images with crops w/o unclassified:   {crop_stats['distinct_crops_no_unclassified']}")
    print(f"Orphaned crop files:                           {crop_stats['orphaned_crops']}")

    print("\n--- 3. crops.json Integrity ---")
    print(f"Crops missing from crops.json:                 {crop_stats['unregistered_crops']}")
    print(f"JSON entries missing on disk:                  {crop_stats['missing_crop_files']}")

    print("\n--- 4. Crop Breakdown by Subfolder ---")
    # Combine folders found on disk and folders declared in JSON
    all_folders = set(crop_stats["folder_counts"].keys()) | set(crop_stats.get("json_folder_counts", {}).keys())

    if all_folders:
        for folder in sorted(all_folders):
            disk_count = crop_stats["folder_counts"].get(folder, 0)
            json_count = crop_stats["json_folder_counts"].get(folder, 0)

            # Format cleanly into columns
            folder_str = f"- {folder}:".ljust(30)
            print(f"{folder_str} {str(disk_count).rjust(4)} on disk  |  {str(json_count).rjust(4)} in JSON")
    else:
        print("No crop subfolders found.")

    all_warnings = meta_stats["warnings"] + crop_stats["warnings"]

    print("\n" + "=" * 60)
    if not all_warnings:
        print(" ACTION REQUIRED: None. All files match perfectly.")
    else:
        print(f" ACTION REQUIRED: {len(all_warnings)} Issues Found")
        print("=" * 60)
        for warning in all_warnings:
            print(f" [!] {warning}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print("Scanning directories and metadata...")
    m_stats = check_metadata(METADATA_FILE, RAW_DIR)
    c_stats = check_crops(RAW_DIR, CROP_DIR, CROPS_JSON_FILE)
    print_dashboard(m_stats, c_stats)