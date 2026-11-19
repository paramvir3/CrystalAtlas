from __future__ import annotations

import csv
import importlib.util
import io
import json
from hashlib import sha1, sha256
from pathlib import Path
from typing import Any

import pytest

from crystalatlas.benchmark_data import read_benchmark_cif_csv, select_training_pilot
from crystalatlas.errors import DomainError

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_mp20_pilot.py"
_SPEC = importlib.util.spec_from_file_location("mp20_pilot_script", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
pilot = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pilot)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    overlap: bool = False,
    repeated_payload: bool = False,
) -> tuple[Path, str, dict[str, Any]]:
    """Synthetic transport fixtures only; no real source rights are asserted."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_bytes(b"Synthetic fixture, no third-party content.\n")
    counts = {"train": 7, "val": 2, "test": 1}
    for split, count in counts.items():
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        writer.writerow(["", "material_id", "cif", "energy", "elements"])
        for index in range(count):
            material_id = f"synthetic-{split}-{index}"
            if overlap and split == "val" and index == 0:
                material_id = "synthetic-train-0"
            cif = "data_shared\r\n" if repeated_payload else f"data_{material_id}\r\n"
            cif += "# quotes 'and commas,' retained\r\n_cell_length_a 4\r\n"
            writer.writerow(
                [str(index + 100), material_id, cif, "=do_not_evaluate()", "['C', 'N']"]
            )
        (source / f"{split}.csv").write_bytes(buffer.getvalue().encode())
    frozen = {}
    artifacts = []
    for path in sorted(source.iterdir()):
        raw = path.read_bytes()
        blob = sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        digest = sha256(raw).hexdigest()
        frozen[path.name] = (len(raw), blob, digest)
        artifacts.append(
            {
                "path": path.name,
                "uri": f"{pilot.BASE_URI}/{path.name}",
                "byte_count": len(raw),
                "git_blob_sha1": blob,
                "sha256": digest,
            }
        )
    # These replace the frozen real-data constants only inside this test process.
    monkeypatch.setattr(pilot, "FROZEN_FILES", frozen)
    monkeypatch.setattr(pilot, "SPLIT_COUNTS", counts)
    receipt = {
        "schema": "crystalatlas.mp20-byte-acquisition.v1",
        "revision": pilot.REVISION,
        "retrieved_utc": "2026-09-14T00:00:00Z",
        "artifacts": artifacts,
        "training_authorized_by_this_receipt": False,
        "notice": "Synthetic test transport; no source licence grant.",
    }
    raw = _json_bytes(receipt)
    (source / "acquisition.json").write_bytes(raw)
    return source, sha256(raw).hexdigest(), receipt


def _tree(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in path.rglob("*")
        if item.is_file()
    }


def test_extraction_is_deterministic_pinned_train_only_and_byte_preserving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, pin, _ = _source(tmp_path, monkeypatch)
    source_before = _tree(source)
    first, second = tmp_path / "first", tmp_path / "second"
    options = {"expected_acquisition_sha256": pin, "count": 4, "seed": 17}
    receipt = pilot.prepare_mp20_pilot(source, first, **options)
    replay = pilot.prepare_mp20_pilot(source, second, **options)
    assert receipt == replay
    assert _tree(first) == _tree(second)
    assert _tree(source) == source_before
    assert receipt["parsed_source_counts"] == {"train": 7, "val": 2, "test": 1}
    assert receipt["indexed_source_rows"] == 10
    assert receipt["selected_count"] == 4
    assert receipt["training_authorized_by_this_receipt"] is False
    assert receipt["source_rights_status"] == "unresolved"
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_fingerprint"}
    assert sha256(_json_bytes(unsigned)).hexdigest() == receipt["receipt_fingerprint"]
    actual_files = set(_tree(first)) - {"extraction-receipt.json"}
    assert actual_files == {entry["path"] for entry in receipt["files"]}
    for artifact in receipt["files"]:
        raw = (first / artifact["path"]).read_bytes()
        assert sha256(raw).hexdigest() == artifact["sha256"]
        assert len(raw) == artifact["byte_count"]
    all_rows = [
        json.loads(line) for line in (first / "source-rows.jsonl").read_bytes().splitlines()
    ]
    assert len(all_rows) == 10
    assert sum(row["selected_for_pilot"] for row in all_rows) == 4
    assert all(
        not row["selected_for_pilot"] for row in all_rows if row["row"]["split"] != "train"
    )
    for row in all_rows:
        assert sha256(_json_bytes(row["row"])).hexdigest() == row["source_metadata_sha256"]
        assert (
            row["source_container_sha256"]
            == pilot.FROZEN_FILES[row["source_container_path"]][2]
        )
    training = read_benchmark_cif_csv(
        source / "train.csv",
        split="train",
        expected_count=7,
        expected_sha256=pilot.FROZEN_FILES["train.csv"][2],
    )
    selected = select_training_pilot(training, count=4, seed=17)
    bindings = json.loads((first / "selected-bindings.json").read_bytes())["bindings"]
    assert [item["source_record_id"] for item in bindings] == [
        row.material_id for row in selected
    ]
    record_index = json.loads((first / "record-index.json").read_bytes())
    payload_index = json.loads((first / "record-payload-index.json").read_bytes())
    assert record_index["record_ids"] == [row.material_id for row in selected]
    assert payload_index["records"] == [[row.material_id, row.cif_sha256] for row in selected]
    for rank, (binding, row) in enumerate(zip(bindings, selected, strict=True)):
        assert binding["selection_rank"] == rank
        assert binding["source_partition"] == "train"
        assert binding["source_row_index"] == row.row_index
        assert (first / binding["cif_path"]).read_bytes() == row.cif.encode()
        metadata_raw = (first / binding["metadata_path"]).read_bytes()
        assert sha256(metadata_raw).hexdigest() == row.metadata_sha256
        assert json.loads(metadata_raw)["metadata"]["energy"] == "=do_not_evaluate()"
    selection = json.loads((first / "selection.json").read_bytes())
    assert selection["refill_after_preprocessing_rejection"] is False
    assert selection["published_holdouts_available_to_training"] is False


@pytest.mark.parametrize("bad_pin", ["0" * 64, "NOT-A-DIGEST", True])
def test_external_pin_failure_produces_no_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_pin: object,
) -> None:
    source, _, _ = _source(tmp_path, monkeypatch)
    output = tmp_path / "output"
    with pytest.raises(DomainError, match="acquisition"):
        pilot.prepare_mp20_pilot(source, output, expected_acquisition_sha256=bad_pin, count=2)
    assert not output.exists()


@pytest.mark.parametrize(
    "mutation", ["revision", "rights", "digest", "path", "duplicate", "extra"]
)
def test_rehashed_acquisition_cannot_change_the_frozen_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    source, _, receipt = _source(tmp_path, monkeypatch)
    if mutation == "revision":
        receipt["revision"] = "0" * 40
    elif mutation == "rights":
        receipt["training_authorized_by_this_receipt"] = True
    elif mutation == "digest":
        receipt["artifacts"][0]["sha256"] = "0" * 64
    elif mutation == "path":
        receipt["artifacts"][0]["path"] = "../README.md"
    elif mutation == "duplicate":
        receipt["artifacts"][1] = receipt["artifacts"][0]
    else:
        receipt["unexpected"] = True
    raw = _json_bytes(receipt)
    (source / "acquisition.json").write_bytes(raw)
    with pytest.raises(DomainError, match="acquisition"):
        pilot.prepare_mp20_pilot(
            source,
            tmp_path / "output",
            expected_acquisition_sha256=sha256(raw).hexdigest(),
            count=2,
        )
    assert not (tmp_path / "output").exists()


def test_duplicate_acquisition_keys_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _, _ = _source(tmp_path, monkeypatch)
    raw = b'{"schema":"one","schema":"two"}'
    (source / "acquisition.json").write_bytes(raw)
    with pytest.raises(DomainError, match="repeats a key"):
        pilot.prepare_mp20_pilot(
            source,
            tmp_path / "output",
            expected_acquisition_sha256=sha256(raw).hexdigest(),
            count=2,
        )


def test_all_holdout_bytes_and_counts_are_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, pin, _ = _source(tmp_path, monkeypatch)
    test = source / "test.csv"
    original = test.read_bytes()
    test.write_bytes(original.replace(b"synthetic-test-0", b"synthetic-test-1"))
    with pytest.raises(DomainError, match="source artifact bytes changed"):
        pilot.prepare_mp20_pilot(
            source, tmp_path / "output", expected_acquisition_sha256=pin, count=2
        )
    assert not (tmp_path / "output").exists()
    test.write_bytes(original)
    monkeypatch.setattr(pilot, "SPLIT_COUNTS", {"train": 7, "val": 3, "test": 1})
    with pytest.raises(DomainError, match="count"):
        pilot.prepare_mp20_pilot(
            source, tmp_path / "output", expected_acquisition_sha256=pin, count=2
        )
    assert not (tmp_path / "output").exists()


def test_cross_split_identifier_overlap_is_not_silently_resolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, pin, _ = _source(tmp_path, monkeypatch, overlap=True)
    with pytest.raises(DomainError, match="overlap across published splits"):
        pilot.prepare_mp20_pilot(
            source, tmp_path / "output", expected_acquisition_sha256=pin, count=2
        )
    assert not (tmp_path / "output").exists()


def test_shared_cif_bytes_do_not_collapse_record_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, pin, _ = _source(tmp_path, monkeypatch, repeated_payload=True)
    output = tmp_path / "output"
    receipt = pilot.prepare_mp20_pilot(source, output, expected_acquisition_sha256=pin, count=4)
    assert receipt["selected_count"] == 4
    assert len(tuple((output / "cifs").iterdir())) == 1
    assert len(tuple((output / "metadata").iterdir())) == 4
    assert len(json.loads((output / "selected-bindings.json").read_bytes())["bindings"]) == 4


def test_existing_output_is_preserved_and_symlink_paths_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, pin, _ = _source(tmp_path, monkeypatch)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "user-owned.txt"
    marker.write_bytes(b"preserve this")
    with pytest.raises(DomainError, match="already exists"):
        pilot.prepare_mp20_pilot(source, output, expected_acquisition_sha256=pin, count=2)
    assert marker.read_bytes() == b"preserve this"
    assert tuple(output.iterdir()) == (marker,)
    link = tmp_path / "linked-source"
    link.symlink_to(source, target_is_directory=True)
    with pytest.raises(DomainError, match="symlinks"):
        pilot.prepare_mp20_pilot(
            link, tmp_path / "new", expected_acquisition_sha256=pin, count=2
        )
    parent_link = tmp_path / "linked-output-parent"
    parent_link.symlink_to(output, target_is_directory=True)
    with pytest.raises(DomainError, match="symlinks"):
        pilot.prepare_mp20_pilot(
            source, parent_link / "new", expected_acquisition_sha256=pin, count=2
        )


@pytest.mark.parametrize(
    "options", [{"count": 0}, {"count": 513}, {"count": True}, {"seed": -1}]
)
def test_invalid_selection_bounds_leave_no_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    options: dict[str, object],
) -> None:
    source, pin, _ = _source(tmp_path, monkeypatch)
    settings = {"count": 2, "seed": 17, "expected_acquisition_sha256": pin}
    settings.update(options)
    with pytest.raises(DomainError):
        pilot.prepare_mp20_pilot(source, tmp_path / "output", **settings)
    assert not (tmp_path / "output").exists()


def test_interrupted_publication_has_no_completion_receipt_and_is_not_erased(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, pin, _ = _source(tmp_path, monkeypatch)
    original_write = pilot._write_exclusive

    def fail_on_index(path: Path, payload: bytes) -> None:
        if path.name == "source-rows.jsonl":
            raise OSError("injected storage failure")
        original_write(path, payload)

    monkeypatch.setattr(pilot, "_write_exclusive", fail_on_index)
    output = tmp_path / "output"
    with pytest.raises(OSError, match="injected storage failure"):
        pilot.prepare_mp20_pilot(source, output, expected_acquisition_sha256=pin, count=2)
    assert (output / "selection.json").is_file()
    assert not (output / "extraction-receipt.json").exists()
    with pytest.raises(DomainError, match="already exists"):
        pilot.prepare_mp20_pilot(source, output, expected_acquisition_sha256=pin, count=2)
