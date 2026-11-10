import json
from pathlib import Path
import pytest

from scripts.create_split import create_split


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """
    Creates a synthetic crops directory structure with mock images.
    - Class A: 5 artifacts, 6 crops total (art1 has 2 crops)
    - Class B: 5 artifacts, 7 crops total (art7 has 2 crops, art1 overlaps with Class A)
    - Unclassified: 1 artifact, 1 crop (should be excluded)
    Total valid crops = 13
    Total valid artifacts = 10
    """
    data_dir = tmp_path / "crops"

    # Class A
    class_a = data_dir / "class_a"
    class_a.mkdir(parents=True)
    (class_a / "art1_occ_01.jpg").touch()
    (class_a / "art1_occ_02.jpg").touch()  # Multi-crop artifact
    (class_a / "art2_occ_01.jpg").touch()
    (class_a / "art3_occ_01.jpg").touch()
    (class_a / "art4_occ_01.jpg").touch()
    (class_a / "art5_occ_01.jpg").touch()

    # Class B
    class_b = data_dir / "class_b"
    class_b.mkdir(parents=True)
    (class_b / "art1_occ_03.jpg").touch()  # Multi-class overlap for art1
    (class_b / "art6_occ_01.jpg").touch()
    (class_b / "art7_occ_01.jpg").touch()
    (class_b / "art7_occ_02.jpg").touch()  # Multi-crop artifact
    (class_b / "art8_occ_01.jpg").touch()
    (class_b / "art9_occ_01.jpg").touch()
    (class_b / "art10_occ_01.jpg").touch()

    # Unclassified (should be ignored)
    unclassified = data_dir / "unclassified"
    unclassified.mkdir(parents=True)
    (unclassified / "art99_occ_01.jpg").touch()

    return data_dir


def test_no_artifact_leakage(synthetic_dataset: Path, tmp_path: Path):
    """Test: No artifact belongs to both dev and test partitions."""
    out_path = tmp_path / "split.json"
    create_split(synthetic_dataset, out_path, test_size=0.2, seed=42)

    with open(out_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    dev_aids = set(manifest["partitions"]["dev"])
    test_aids = set(manifest["partitions"]["test"])

    # Assert there is no intersection between the two sets
    assert dev_aids.isdisjoint(test_aids), "Artifact leakage detected between dev and test sets!"


def test_all_crops_assigned(synthetic_dataset: Path, tmp_path: Path):
    """Test: Every classified crop belongs to exactly one partition and counts match."""
    out_path = tmp_path / "split.json"
    create_split(synthetic_dataset, out_path, test_size=0.2, seed=42)

    with open(out_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    dev_crops = sum(manifest["crop_class_distribution"]["dev"].values())
    test_crops = sum(manifest["crop_class_distribution"]["test"].values())

    # We generated exactly 13 valid crops across classes A and B
    assert dev_crops + test_crops == 13, "Not all valid crops were accounted for in the split distributions."

    # Verify the artifact counts strictly match the 10 valid inputs
    assert manifest["metadata"]["total_artifacts"] == 10
    assert (manifest["metadata"]["dev_artifacts"] + manifest["metadata"]["test_artifacts"]) == 10

    # For a 10-artifact dataset, an 80/20 split should put exactly 2 artifacts in the test partition
    assert len(manifest["partitions"]["test"]) == 2


def test_multi_class_artifact_grouping(synthetic_dataset: Path, tmp_path: Path):
    """Test: Artifacts with crops in multiple classes stay unified and don't duplicate."""
    out_path = tmp_path / "split.json"
    create_split(synthetic_dataset, out_path, test_size=0.2, seed=42)

    manifest = json.loads(out_path.read_text(encoding="utf-8"))
    dev = set(manifest["partitions"]["dev"])
    test = set(manifest["partitions"]["test"])

    # art1 must exist in exactly one partition, never both
    assert ("art1" in dev) != ("art1" in test)

    # Ensure total crop calculation still equals 13 with the overlap included
    assert (
                   sum(manifest["crop_class_distribution"]["dev"].values())
                   + sum(manifest["crop_class_distribution"]["test"].values())
           ) == 13


def test_reproducible_split(synthetic_dataset: Path, tmp_path: Path):
    """Test: Same dataset and seed produce identical assignments."""
    out_path_1 = tmp_path / "split_1.json"
    out_path_2 = tmp_path / "split_2.json"

    create_split(synthetic_dataset, out_path_1, test_size=0.2, seed=42)
    create_split(synthetic_dataset, out_path_2, test_size=0.2, seed=42)

    with open(out_path_1, "r", encoding="utf-8") as f1, open(out_path_2, "r", encoding="utf-8") as f2:
        manifest_1 = json.load(f1)
        manifest_2 = json.load(f2)

    assert manifest_1["partitions"]["dev"] == manifest_2["partitions"]["dev"], "Dev partitions differ between runs."
    assert manifest_1["partitions"]["test"] == manifest_2["partitions"]["test"], "Test partitions differ between runs."


def test_unclassified_excluded(synthetic_dataset: Path, tmp_path: Path):
    """Test: Unclassified crops never appear in the split."""
    out_path = tmp_path / "split.json"
    create_split(synthetic_dataset, out_path, test_size=0.2, seed=42)

    with open(out_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    dev_aids = set(manifest["partitions"]["dev"])
    test_aids = set(manifest["partitions"]["test"])

    # Ensure the unclassified artifact ID does not appear
    assert "art99" not in dev_aids, "Unclassified artifact found in dev set."
    assert "art99" not in test_aids, "Unclassified artifact found in test set."

    # Ensure the unclassified class label is absent from the distributions
    assert "unclassified" not in manifest["artifact_class_distribution"]["dev"]
    assert "unclassified" not in manifest["crop_class_distribution"]["dev"]
