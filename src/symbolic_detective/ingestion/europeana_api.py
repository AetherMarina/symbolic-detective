"""
Europeana REST API Client.
Handles querying the Europeana database, extracting image URLs,
downloading raw artifacts, and saving provenance metadata.
"""
import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from urllib.parse import urlparse
import requests

from symbolic_detective.utils.image_files import SUPPORTED_IMAGE_EXTENSIONS

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
BASE_URL = "https://api.europeana.eu/record/v2/search.json"


def first_value(value: Any, default: Any = None) -> Any:
    """Safely extracts the first value from Europeana's mixed list/string fields."""
    if value is None:
        return default
    if isinstance(value, list):
        return value[0] if value else default
    return value


class EuropeanaClient:

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("EUROPEANA_API_KEY")
        if not self.api_key:
            raise ValueError("Europeana API key is required. Set EUROPEANA_API_KEY environment variable.")

        self.session = requests.Session()
        self.metadata_file = DEFAULT_DATA_DIR / "metadata.json"
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)

    def search(self, query: str, rows: int = 24, cursor: str = "*") -> Dict[str, Any]:
        """
        Executes a search query against Europeana, strictly filtering for images.
        """
        params = {
            "wskey": self.api_key,
            "query": query,
            "media": True,
            "qf": "TYPE:IMAGE",
            "rows": rows,
            "cursor": cursor,
            "profile": "rich"
        }
        try:
            response = self.session.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Europeana search failed for query '{query}': {e}") from e

    def download_image(self, url: str, save_path: Path) -> bool:
        """Streams a remote image directly to disk."""
        if save_path.exists() and save_path.stat().st_size > 0:
            return True

        temp_path = save_path.with_suffix(save_path.suffix + ".part")

        try:
            with self.session.get(url, stream=True, timeout=15) as res:
                res.raise_for_status()

                with open(temp_path, "wb") as f:
                    for chunk in res.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            if temp_path.stat().st_size == 0:
                raise ValueError("Downloaded image is empty")

            temp_path.replace(save_path)
            return True

        except (requests.RequestException, OSError, ValueError) as e:
            temp_path.unlink(missing_ok=True)
            print(f"Warning: Failed to download {url} - {e}")
            return False

    def update_metadata(self, artifact_key: str, provenance: Dict[str, Any]) -> None:
        """Safely appends new artifact provenance to the central metadata store."""
        metadata = {}
        if self.metadata_file.exists():
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                try:
                    metadata = json.load(f)
                except json.JSONDecodeError:
                    pass

        metadata[artifact_key] = provenance

        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

    def ingest_collection(self, query: str, max_items: int = 50) -> None:
        """
        Main pipeline: Searches Europeana, downloads images to data/raw,
        and saves cultural metadata to data/metadata.json.
        """
        existing_ids = set()
        if self.metadata_file.exists():
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                existing_ids = set(metadata.keys())

        safe_query = re.sub(r"[^a-zA-Z0-9_\-]+", "_", query).strip("_")
        raw_img_dir = DEFAULT_DATA_DIR / "raw" / f"europeana_{safe_query}"
        raw_img_dir.mkdir(parents=True, exist_ok=True)

        print(f"Ingesting up to {max_items} artifacts for query: '{query}'")

        items_downloaded = 0
        cursor = "*"

        while items_downloaded < max_items:
            data = self.search(query, rows=min(50, max_items - items_downloaded), cursor=cursor)
            items = data.get("items", [])

            if not items:
                print("No more items found.")
                break

            for item in items:
                if items_downloaded >= max_items:
                    break

                raw_id = item.get("id")
                if not raw_id:
                    continue
                artifact_id = raw_id.replace("/", "_").strip("_")
                artifact_key = f"europeana_{artifact_id}"

                # Skip artifacts already registered in metadata.json
                if artifact_key in existing_ids:
                    print(f"Image for {artifact_key} already exists. Skipping download.")
                    continue

                image_urls = item.get("edmIsShownBy") or item.get("edmPreview")
                if not image_urls:
                    continue

                image_url = image_urls[0] if isinstance(image_urls, list) else image_urls
                title = first_value(item.get("title"), "Unknown Title")

                suffix = Path(urlparse(image_url).path).suffix.lower()
                if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
                    suffix = ".jpg"

                filename = f"{artifact_key}{suffix}"
                save_path = raw_img_dir / filename

                provenance = {
                    "source": "Europeana",
                    "source_id": raw_id,
                    "title": title,
                    "creator": first_value(item.get("dcCreator")),
                    "date": first_value(item.get("year")) or first_value(item.get("dcDate")),
                    "data_provider": first_value(item.get("dataProvider")),
                    "provider": first_value(item.get("provider")),
                    "country": first_value(item.get("country")),
                    "type": first_value(item.get("dcType")),
                    "source_url": item.get("guid", ""),
                    "image_url": image_url,
                    "rights": first_value(item.get("rights")),
                    "local_image_path": f"raw/europeana/{filename}"
                }

                if self.download_image(image_url, save_path):
                    self.update_metadata(artifact_key, provenance)
                    existing_ids.add(artifact_key)
                    print(f"[{items_downloaded + 1}/{max_items}] Saved: {filename} ({title[:30]}...)")
                    items_downloaded += 1
                    time.sleep(0.5)

            next_cursor = data.get("nextCursor")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor

        print(f"\nIngestion complete. Downloaded {items_downloaded} artifacts to {raw_img_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harvest artifacts from The Met Open Access API")
    parser.add_argument("--query", type=str, required=True, help="Search query (e.g., 'star')")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of artifacts to harvest")
    args = parser.parse_args()

    client = EuropeanaClient()
    client.ingest_collection(query=args.query, max_items=args.limit)
