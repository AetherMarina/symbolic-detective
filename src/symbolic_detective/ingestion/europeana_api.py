"""
Europeana REST API Client.
Handles querying the Europeana database, extracting image URLs,
downloading raw artifacts, and saving provenance metadata.
"""
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
BASE_URL = "https://api.europeana.eu/record/v2/search.json"


class EuropeanaClient:

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("EUROPEANA_API_KEY")
        if not self.api_key:
            raise ValueError("Europeana API key is required. Set EUROPEANA_API_KEY environment variable.")

        self.session = requests.Session()
        self.metadata_file = DEFAULT_DATA_DIR / "metadata.json"

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
            "profile": "rich"  # Requests expanded metadata
        }

        response = self.session.get(BASE_URL, params=params)
        response.raise_for_status()
        return response.json()

    def download_image(self, url: str, save_path: Path) -> bool:
        """Streams a remote image directly to disk."""
        if save_path.exists():
            return True

        try:
            res = self.session.get(url, stream=True, timeout=15)
            res.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in res.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except requests.RequestException as e:
            print(f"Warning: Failed to download {url} - {e}")
            return False

    def update_metadata(self, artifact_id: str, provenance: Dict[str, Any]) -> None:
        """Safely appends new artifact provenance to the central metadata store."""
        metadata = {}
        if self.metadata_file.exists():
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                try:
                    metadata = json.load(f)
                except json.JSONDecodeError:
                    pass

        metadata[artifact_id] = provenance

        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

    def ingest_collection(self, query: str, max_items: int = 50) -> None:
        """
        Main pipeline: Searches Europeana, downloads images to data/raw,
        and saves cultural metadata to data/metadata.json.
        """

        # Setup data directories
        data_dir = DEFAULT_DATA_DIR / "raw" / f"europeana_{query}"

        # Format a safe folder name based on the query
        raw_img_dir = DEFAULT_DATA_DIR / "raw" / f"europeana_{query.replace(" ", "_")}"
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

                # Europeana IDs look like '/2022302/id_string'
                raw_id = item.get("id", "")
                artifact_id = raw_id.replace("/", "_").strip("_")

                # edmIsShownBy is usually the high-res original; edmPreview is the thumbnail
                image_urls = item.get("edmIsShownBy") or item.get("edmPreview")
                if not image_urls:
                    continue

                image_url = image_urls[0] if isinstance(image_urls, list) else image_urls

                # Parse basic provenance
                title = item.get("title", ["Unknown Title"])[0]
                creators = item.get("dcCreator", ["Unknown Creator"])
                dates = item.get("year", item.get("dcDate", ["Unknown Period"]))
                institution = item.get("dataProvider", ["Unknown Institution"])[0]

                provenance = {
                    "title": title,
                    "culture": creators[0] if creators else "Unknown Culture",
                    "period": str(dates[0]) if dates else "Unknown Period",
                    "institution": institution,
                    "source_url": item.get("guid", ""),
                }

                # Download and save
                ext = image_url.split(".")[-1][:4].lower()
                if ext not in ["jpg", "jpeg", "png"]:
                    ext = "jpg"

                filename = f"{artifact_id}.{ext}"
                save_path = raw_img_dir / filename

                if self.download_image(image_url, save_path):
                    self.update_metadata(artifact_id, provenance)
                    print(f"[{items_downloaded + 1}/{max_items}] Saved: {filename} ({title[:30]}...)")
                    items_downloaded += 1

                    # Be polite to their API
                    time.sleep(0.5)

            # Pagination
            next_cursor = data.get("nextCursor")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor

        print(f"\nIngestion complete. Downloaded {items_downloaded} artifacts to {raw_img_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m symbolic_detective.data.europeana <search_query> [max_items]")
        sys.exit(1)

    query_str = sys.argv[1]
    max_limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    # Example queries: "Minoan seal", "Roman mosaic rosette", "Sumerian cylinder"
    client = EuropeanaClient()
    client.ingest_collection(query_str, max_items=max_limit)
