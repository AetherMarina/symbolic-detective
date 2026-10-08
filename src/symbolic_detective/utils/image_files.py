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
