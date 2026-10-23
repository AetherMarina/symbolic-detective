"""
Cleveland Museum of Art (CMA) Open Access Harvester.
Queries the CMA API for motifs, downloads raw images, and updates metadata.json.
"""
import json
import time
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
METADATA_FILE = DATA_DIR / "metadata.json"
RAW_DIR = DATA_DIR / "raw"

CMA_API_BASE = "https://openaccess-api.clevelandart.org/api/artworks"


def load_metadata() -> dict:
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def save_metadata(metadata: dict):
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def download_image(url: str, save_path: Path) -> bool:
    if save_path.exists() and save_path.stat().st_size > 0:
        return True

    temp_path = save_path.with_suffix(save_path.suffix + ".part")

    try:
        with requests.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()

            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        if temp_path.stat().st_size == 0:
            raise ValueError("Downloaded image is empty")

        temp_path.replace(save_path)
        return True

    except (requests.RequestException, OSError, ValueError) as e:
        temp_path.unlink(missing_ok=True)
        print(f"Failed to download {url}: {e}")
        return False


def harvest_cma(query: str, limit_per_query: int = 100):
    print(f"--- Harvesting CMA for: '{query}' ---")
    metadata = load_metadata()

    query_folder_name = f"cma_{query.replace(' ', '_').replace('-', '_')}"
    save_dir = RAW_DIR / query_folder_name
    save_dir.mkdir(parents=True, exist_ok=True)

    offset = 0
    downloaded_count = 0
    skipped_count = 0

    while True:
        params = {
            "q": query,
            "cc0": "",
            "has_image": 1,
            "limit": 50,
            "skip": offset
        }

        print(f"Fetching offset {offset}...")
        response = requests.get(CMA_API_BASE, params=params, timeout=15)
        if response.status_code != 200:
            print(f"API Error {response.status_code}: {response.text}")
            break

        data = response.json().get("data", [])
        if not data:
            print("No more results.")
            break

        for item in data:
            if item.get("share_license_status") != "CC0":
                continue
            if downloaded_count >= limit_per_query:
                break

            # Prefix with cma_ to ensure global uniqueness in metadata.json
            raw_id = item.get("id")
            if raw_id is None:
                continue

            artifact_key = f"cma_{raw_id}"

            if artifact_key in metadata:
                skipped_count += 1
                continue

            images = item.get("images") or {}
            web_image = images.get("web") or {}
            img_url = web_image.get("url")
            if not img_url:
                continue

            img_suffix = Path(img_url.split("?")[0]).suffix or ".jpg"
            save_path = save_dir / f"{artifact_key}{img_suffix}"

            print(f"Downloading: {artifact_key}...")
            if download_image(img_url, save_path):
                local_rel_path = f"raw/cleveland/{save_path.name}"

                metadata[artifact_key] = {
                    "title": item.get("title", "Unknown Title"),
                    "date": item.get("creation_date", "Unknown Date"),
                    "culture": item.get("culture") or [],
                    "data_provider": "Cleveland Museum of Art",
                    "source_url": item.get("url", ""),
                    "image_url": img_url,
                    "rights": item.get("share_license_status", "Unknown Rights"),
                    "local_image_path": local_rel_path
                }
                downloaded_count += 1

                # Save periodically to prevent data loss
                if downloaded_count % 10 == 0:
                    save_metadata(metadata)

            time.sleep(0.5)  # API Rate limiting

        if downloaded_count >= limit_per_query or len(data) < 50:
            break

        offset += 50

    save_metadata(metadata)
    print(f"Finished '{query}'. Downloaded: {downloaded_count}, Skipped (Already in DB): {skipped_count}")


if __name__ == "__main__":
    search_queries = [
        "five-pointed star",
        "six-pointed star"
    ]

    for q in search_queries:
        harvest_cma(q, limit_per_query=25)
