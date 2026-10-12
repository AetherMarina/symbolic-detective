import re
from pathlib import Path

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

SUPPORTED_UPLOAD_TYPES = tuple(
    ext.lstrip(".")
    for ext in SUPPORTED_IMAGE_EXTENSIONS
)


def collect_images(
    directory: Path,
    recursive: bool = False,
) -> list[Path]:
    """Collects supported image files in deterministic order."""

    files = directory.rglob("*") if recursive else directory.glob("*")

    return sorted(
        path
        for path in files
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def artifact_key_from_path(path: Path) -> str:
    """
    Extracts the source-qualified artifact key from a crop or raw image path.
    Splits safely across crop occurrence and primary image delimiters.
    """
    parts = re.split(r"__occ_|_occ_|_primary", path.stem)
    return parts[0]
