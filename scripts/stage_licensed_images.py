import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
METADATA_FILE = DEFAULT_DATA_DIR / 'metadata.json'
CROPS_DIR = DEFAULT_DATA_DIR / "crops"

ALLOWED_RIGHTS = {
    "http://creativecommons.org/publicdomain/zero/1.0/",
    "https://creativecommons.org/publicdomain/zero/1.0/",
    "http://creativecommons.org/publicdomain/mark/1.0/",
    "https://creativecommons.org/publicdomain/mark/1.0/",
    "http://creativecommons.org/licenses/by/4.0/",
    "https://creativecommons.org/licenses/by/4.0/",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def is_allowed(artifact):
    if artifact.get("repository") == "The Metropolitan Museum of Art":
        return artifact.get("isPublicDomain") is True

    rights = artifact.get("rights", "")
    return rights in ALLOWED_RIGHTS


def main():
    print("Current directory:", Path.cwd())
    print("Crops directory:", CROPS_DIR.resolve())
    print("Directory exists:", CROPS_DIR.exists())
    print("Total files:", len(list(CROPS_DIR.rglob("*"))))
    with METADATA_FILE.open(encoding="utf-8") as f:
        metadata = json.load(f)

    # Match longer IDs first to avoid partial matches
    artifact_ids = sorted(metadata, key=len, reverse=True)

    staged = []
    skipped = []
    unmatched = []

    for crop in CROPS_DIR.rglob("*"):
        if not crop.is_file() or crop.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        artifact_id = next(
            (
                aid
                for aid in artifact_ids
                if crop.stem == aid
                or crop.stem.startswith(aid + "_")
            ),
            None,
        )

        if artifact_id is None:
            unmatched.append(crop)
            continue

        artifact = metadata[artifact_id]

        if not is_allowed(artifact):
            skipped.append(crop)
            continue

        # -f allows staging explicitly approved files
        # even when crops/ is in .gitignore.
        subprocess.run(
            ["git", "add", "-f", "--", str(crop)],
            check=True,
        )
        staged.append(crop)

    print(f"\nStaged: {len(staged)}")
    print(f"Skipped (restricted or unverified): {len(skipped)}")
    print(f"Unmatched: {len(unmatched)}")

    if unmatched:
        print("\nUnmatched images:")
        for path in unmatched:
            print(f"  {path}")

    print("\nReview staged images using: git diff --cached --name-only")


if __name__ == "__main__":
    main()
