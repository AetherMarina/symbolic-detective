from pathlib import Path
from symbolic_detective.utils.image_files import artifact_key_from_path, collect_images


def test_artifact_key_extraction():
    """Test: Multiple crops from the same artifact produce the same artifact ID."""
    paths = [
        Path("12345_occ_01.jpg"),
        Path("12345_occ_02.png"),
        Path("12345__occ_03.jpg"),
        Path("12345_primary.jpeg"),
    ]
    for p in paths:
        assert artifact_key_from_path(p) == "12345"


def test_museum_prefixes():
    """Test: Met, Europeana, and CMA artifact identifiers are preserved correctly."""
    assert artifact_key_from_path(Path("cma_98765_occ_01.jpg")) == "cma_98765"
    assert artifact_key_from_path(Path("met_12345_primary.jpg")) == "met_12345"
    assert artifact_key_from_path(Path("europeana_abc123__occ_02.jpg")) == "europeana_abc123"


def test_collect_supported_images(tmp_path: Path):
    """Test: Only supported image extensions are collected."""
    (tmp_path / "valid1.jpg").touch()
    (tmp_path / "valid2.png").touch()
    (tmp_path / "valid3.jpeg").touch()
    (tmp_path / "valid4.JPG").touch()  # Verify case-insensitivity
    (tmp_path / "invalid.txt").touch()
    (tmp_path / "invalid.pdf").touch()

    collected = collect_images(tmp_path)

    assert len(collected) == 4
    for p in collected:
        assert p.suffix.lower() in {".jpg", ".jpeg", ".png"}


def test_collect_images_sorted(tmp_path: Path):
    """Test: Image collection returns deterministic ordering."""
    # Create files in non-alphabetical order
    (tmp_path / "z_image.jpg").touch()
    (tmp_path / "a_image.jpg").touch()
    (tmp_path / "m_image.png").touch()

    collected = collect_images(tmp_path)
    names = [p.name for p in collected]

    assert names == ["a_image.jpg", "m_image.png", "z_image.jpg"]


def test_non_recursive_collection(tmp_path: Path):
    """Test: Images in nested folders aren't collected by default."""
    (tmp_path / "root_image.jpg").touch()

    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    (nested_dir / "nested_image.jpg").touch()

    # Default behavior (recursive=False)
    collected_flat = collect_images(tmp_path)
    assert len(collected_flat) == 1
    assert collected_flat[0].name == "root_image.jpg"

    # Explicit recursive behavior (recursive=True)
    collected_recursive = collect_images(tmp_path, recursive=True)
    assert len(collected_recursive) == 2
    assert set(p.name for p in collected_recursive) == {"root_image.jpg", "nested_image.jpg"}


def test_complex_europeana_artifact_id():
    path = Path("europeana_15502_MK_R__55462_occ_01.jpg")
    assert artifact_key_from_path(path) == "europeana_15502_MK_R__55462"
