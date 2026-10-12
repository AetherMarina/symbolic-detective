"""
Metropolitan Museum of Art API Harvester
"""
import argparse
import requests
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

SEARCH_ENDPOINT = "https://collectionapi.metmuseum.org/public/collection/v1.1/search"
OBJECT_ENDPOINT = "https://collectionapi.metmuseum.org/public/collection/v1/objects"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def get_paginated_object_ids(query: str, department_id: Optional[int], target_limit: int) -> List[int]:
    """
    Queries the v1.1 paginated search API to retrieve a list of object IDs.
    """
    object_ids = []
    offset = 0
    page_size = 100  # v1.1 default max per page

    logging.info(f"Initiating search for '{query}' (Target limit: {target_limit})")

    while len(object_ids) < target_limit:
        params = {
            "q": query,
            "hasImages": "true",
            "limit": page_size,
            "offset": offset
        }
        if department_id:
            params["departmentId"] = department_id

        response = requests.get(SEARCH_ENDPOINT, params=params)

        if response.status_code != 200:
            logging.error(f"Search API failed with status {response.status_code}: {response.text}")
            break

        data = response.json()
        batch_ids = data.get("objectIDs") or []

        if not batch_ids:
            logging.info("No more results found.")
            break

        object_ids.extend(batch_ids)
        logging.info(f"Retrieved {len(batch_ids)} IDs (Total so far: {len(object_ids)} / {data.get('total')})")

        offset += page_size
        time.sleep(0.5)  # Scraping delay

    return object_ids[:target_limit]


def fetch_object_metadata(object_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetches detailed metadata and image URLs for a specific object ID.
    """
    response = requests.get(f"{OBJECT_ENDPOINT}/{object_id}")

    if response.status_code == 200:
        return response.json()
    else:
        logging.warning(f"Failed to fetch metadata for Object ID {object_id}")
        return None


def download_image(image_url: str, object_id: int, output_dir: Path) -> Optional[str]:
    """
    Downloads the primary image and returns the local file path.
    """
    if not image_url:
        return None

    # Handle URL extensions and fallback to .jpg
    ext = image_url.split('.')[-1].lower()
    if ext not in ['jpg', 'jpeg', 'png']:
        ext = 'jpg'

    file_name = f"{object_id}_primary.{ext}"
    file_path = output_dir / file_name

    if file_path.exists():
        logging.info(f"Image for {object_id} already exists. Skipping download.")
        return str(file_path.name)

    try:
        response = requests.get(image_url, stream=True, timeout=10)
        response.raise_for_status()

        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return str(file_path.name)

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download image for {object_id}: {e}")
        return None


def harvest_met_artifacts(query: str, department_id: Optional[int] = None,
                          limit: int = 50, data_dir: Path = DEFAULT_DATA_DIR):
    """
    Core function to harvest metadata and images from The Met API.
    """
    # Directory setup
    data_path = Path(data_dir)
    # Format a safe folder name based on the query and department
    safe_query = query.replace(" ", "_")
    if department_id:
        folder_name = f"{safe_query}_dep{department_id}"
    else:
        folder_name = safe_query

    raw_img_dir = data_path / "raw" / "met" / folder_name
    raw_img_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = data_path / "metadata.json"

    # Load existing metadata to prevent duplicate processing
    existing_metadata = {}
    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            existing_metadata = json.load(f)

    # 1. Get Target IDs
    object_ids = get_paginated_object_ids(query, department_id, limit)

    if not object_ids:
        logging.warning("No objects found matching the criteria.")
        return

    # 2. Process Each Object
    successful_harvests = 0
    for obj_id in object_ids:
        str_obj_id = str(obj_id)

        if str_obj_id in existing_metadata:
            logging.info(f"Object {obj_id} already in metadata. Skipping.")
            continue

        logging.info(f"Fetching data for Object ID: {obj_id}")
        metadata = fetch_object_metadata(obj_id)

        if not metadata:
            continue

        # # Prefer original resolution for accurate motif extraction, fallback to primaryImageSmall
        image_url = metadata.get("primaryImage") or metadata.get("primaryImageSmall")

        if not image_url:
            logging.warning(f"No valid public domain image found for Object {obj_id}. Skipping.")
            continue

        # Download the image
        local_filename = download_image(image_url, obj_id, raw_img_dir)

        if local_filename:
            # Store lean, relevant metadata
            existing_metadata[str_obj_id] = {
                "objectID": obj_id,
                "title": metadata.get("title", "Unknown"),
                "culture": metadata.get("culture", ""),
                "period": metadata.get("period", ""),
                "objectDate": metadata.get("objectDate", ""),
                "medium": metadata.get("medium", ""),
                "country": metadata.get("country", ""),
                "repository": "The Metropolitan Museum of Art",
                "objectURL": metadata.get("objectURL", ""),
                "local_image_path": f"raw/{local_filename}",
                "isPublicDomain": metadata.get("isPublicDomain", False)
            }
            successful_harvests += 1

            # Save incrementally to prevent data loss on script interruption
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(existing_metadata, f, indent=4, ensure_ascii=False)

        # API Rate limiting
        time.sleep(0.1)

    logging.info(f"Harvest complete. Successfully processed {successful_harvests} new artifacts.")
    logging.info(f"Metadata saved to {metadata_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harvest artifacts from The Met Open Access API")
    parser.add_argument("--query", type=str, required=True, help="Search query (e.g., 'star')")
    parser.add_argument("--department", type=int, help="Department ID (e.g., 3 for Ancient Near Eastern Art)")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of artifacts to harvest")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    args = parser.parse_args()

    harvest_met_artifacts(
        query=args.query,
        department_id=args.department,
        limit=args.limit,
        data_dir=Path(args.data_dir)
    )
