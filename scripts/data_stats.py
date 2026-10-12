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


def check_crops(raw_path, crop_path):
    stats = {
        "total_raw": 0,
        "cropped_raw": 0,
        "uncropped_raw": 0,
        "total_crops": 0,
        "distinct_crops": 0,
        "successful_matches": 0,
        "orphaned_crops": 0,
        "folder_counts": defaultdict(int),
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

    # 2. Traverse crops
    processed_stems = set()
    for crop_file in crop_path.rglob("*.*"):
        if crop_file.name.startswith('.'):
            continue

        stats["total_crops"] += 1

        # Track counts per subfolder
        # Calculates the folder path relative to the main 'crops' directory
        rel_folder = crop_file.parent.relative_to(crop_path)
        folder_name = str(rel_folder) if str(rel_folder) != "." else "root (no subfolder)"
        stats["folder_counts"][folder_name] += 1

        if crop_file.name == "crops.json":
            continue

        original_stem = re.sub(r'_occ_\d+$', '', crop_file.stem)
        processed_stems.add(original_stem)

        if original_stem in raw_index:
            stats["successful_matches"] += 1
        else:
            stats["warnings"].append(f"Orphaned crop (raw missing): '{original_stem}' from '{crop_file}'")
            stats["orphaned_crops"] += 1

    stats["distinct_crops"] = len(processed_stems)

    # 3. Calculate cropped vs uncropped
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
    print(f"Total entries in metadata:    {meta_stats['expected_count']}")
    print(f"Total actual images on disk:  {meta_stats['actual_count']}")
    print(f"Metadata missing images:      {meta_stats['missing_images_count']}")
    print(f"Extra images (missing meta):  {meta_stats['extra_images_count']}")

    print("\n--- 2. Raw vs. Cropped Images ---")
    print(f"Total raw images scanned:     {crop_stats['total_raw']}")
    print(f"Images already cropped:       {crop_stats['cropped_raw']}")
    print(f"Images NOT YET cropped:       {crop_stats['uncropped_raw']}")
    print(f"Total crop files generated:   {crop_stats['total_crops']}")
    print(f"Distinct images with crops:   {crop_stats['distinct_crops']}")
    print(f"Orphaned crop files:          {crop_stats['orphaned_crops']}")

    print("\n--- 3. Crop Breakdown by Subfolder ---")
    if crop_stats["folder_counts"]:
        # Sort folders alphabetically for consistent reading
        for folder, count in sorted(crop_stats["folder_counts"].items()):
            # ljust(30) aligns the counts neatly into a column
            print(f"- {folder}:".ljust(30) + f"{count} images")
    else:
        print("No crop subfolders found.")

    # Combine warnings
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
    c_stats = check_crops(RAW_DIR, CROP_DIR)
    print_dashboard(m_stats, c_stats)
