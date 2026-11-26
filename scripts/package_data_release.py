"""Package the official JSONL release for transfer to shared project storage."""

import argparse
import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.project_config import (
    file_sha256, load_config, read_jsonl, require, resolve_base_path, rows_digest,
    write_json,
)


def _release_file(base_path, relative):
    """Resolve a recorded path while keeping every member inside 01_data."""
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts,
            f"Invalid release-relative path: {relative}")
    path = base_path / relative
    require(path.resolve().is_relative_to((base_path / "01_data").resolve()),
            f"Release path escapes 01_data: {relative}")
    require(path.is_file(), f"Missing release file: {relative}")
    return path


def _verify_rows(path, description):
    require(file_sha256(path) == description["file_sha256"], f"Release file hash differs: {path}")
    rows = read_jsonl(path)
    require(rows_digest(rows) == description["rows_sha256"], f"Release row hash differs: {path}")
    require(len(rows) == description["rows"], f"Release row count differs: {path}")
    require(len({row["id"] for row in rows}) == len(rows), f"Duplicate record IDs: {path}")
    counts = {str(label): count for label, count in sorted(Counter(row["label"] for row in rows).items())}
    require(counts == description["per_class"], f"Release per-class counts differ: {path}")
    if "ids" in description:
        require([row["id"] for row in rows] == description["ids"], f"Release ordered IDs differ: {path}")
    return rows


def verify_data_release(base_path):
    """Read-only verification of all official and B2/B3 artifacts before transfer.

    Verification uses portable manifest and preparation reports, without model
    imports, source-cache access, WordNet loading, or network requests.
    """
    base_path = Path(base_path).expanduser().resolve()
    data_root = base_path / "01_data"
    manifest_path = _release_file(base_path, "01_data/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("schema_version") == 1 and bool(manifest.get("protocol_id")), "Invalid official release manifest.")
    data_config = manifest["config"]
    settings, seeds = data_config["settings"], data_config["seeds"]
    expected_datasets = {"validation.jsonl", "test.jsonl"} | {
        f"{setting}/seed_{seed}.jsonl" for setting in settings for seed in seeds
    }
    require(set(manifest["datasets"]) == expected_datasets, "Official manifest has missing or extra datasets.")
    require(set(manifest["metadata_files"]) == {"label_mapping.json", "rare_classes.json", "dataset_statistics.json"},
            "Official manifest has missing or extra metadata files.")
    verified_files = {manifest_path}
    originals = {}
    for name, description in manifest["datasets"].items():
        path = _release_file(base_path, f"01_data/{name}")
        originals[name] = _verify_rows(path, description)
        verified_files.add(path)
    for name, expected_hash in manifest["metadata_files"].items():
        path = _release_file(base_path, f"01_data/{name}")
        require(file_sha256(path) == expected_hash, f"Release metadata hash differs: {path}")
        verified_files.add(path)

    # Every B2/B3 dataset needs its corresponding complete preparation report.
    for condition in ("B2", "B3"):
        for setting in settings:
            for seed in seeds:
                source_name = f"{setting}/seed_{seed}.jsonl"
                source_description = manifest["datasets"][source_name]
                source_relative = f"01_data/{source_name}"
                data_relative = f"01_data/baselines/{condition}/{setting}/seed_{seed}.jsonl"
                report_relative = f"01_data/baselines/reports/{condition}/{setting}/seed_{seed}.json"
                output_path = _release_file(base_path, data_relative)
                report_path = _release_file(base_path, report_relative)
                report = json.loads(report_path.read_text())
                require(report.get("schema_version") == 1 and report.get("status") == "prepared",
                        f"Incomplete baseline preparation report: {report_path}")
                specification = report["specification"]
                require(rows_digest([specification]) == report["config_sha256"],
                        f"Baseline configuration hash differs: {report_path}")
                for record in (report, specification):
                    require((record["condition"], record["setting"], record["seed"]) == (condition, setting, seed),
                            f"Baseline report identity differs: {report_path}")
                    require(record["source_file_sha256"] == source_description["file_sha256"]
                            and record["source_rows_sha256"] == source_description["rows_sha256"],
                            f"Baseline source hashes differ from the official release: {report_path}")
                require(specification["protocol_id"] == manifest["protocol_id"], f"Baseline protocol differs: {report_path}")
                require(specification["source_path"] == source_relative, f"Baseline source path differs: {report_path}")
                require(specification["num_classes"] == data_config["dataset"]["num_classes"],
                        f"Baseline label count differs: {report_path}")
                require(specification["dataset_source"] == {"id": manifest["source"]["dataset_id"],
                                                           "revision": manifest["source"]["revision"]},
                        f"Baseline pinned source differs: {report_path}")
                expected_method = "random_oversampling" if condition == "B2" else "easy_data_augmentation"
                require(specification["method"] == expected_method, f"Baseline method differs: {report_path}")
                require(report["data_path"] == data_relative and report["report_path"] == report_relative,
                        f"Baseline output/report path differs: {report_path}")
                rows = _verify_rows(output_path, report)
                original = originals[source_name]
                require(rows[:len(original)] == original, f"Baseline changed the real training prefix: {output_path}")
                parents = {row["id"]: row for row in original}
                for row in rows[len(original):]:
                    require(row.get("parent_id") in parents, f"Baseline introduced an unknown parent: {output_path}")
                    require(row["label"] == parents[row["parent_id"]]["label"], f"Baseline intended label changed: {output_path}")
                if report["b1_identical_inputs"]:
                    require(condition == "B2" and output_path.read_bytes() == (data_root / source_name).read_bytes(),
                            f"Balanced B2 identity differs from B1: {output_path}")
                require(report["statistics"]["training_rows"] == len(rows), f"Baseline reported training size differs: {report_path}")
                verified_files.update((output_path, report_path))
    actual_files = {path for path in data_root.rglob("*") if path.is_file()}
    require(actual_files == verified_files,
            "Release contains files not covered by its manifest or baseline reports; review the release before packaging.")
    return manifest


def _verify_archive(archive, records):
    try:
        with zipfile.ZipFile(archive) as bundle:
            require(len(bundle.infolist()) == len(records)
                    and set(bundle.namelist()) == {row["path"] for row in records},
                    "Handoff ZIP membership differs from the verified inventory.")
            for row in records:
                payload = bundle.read(row["path"])
                require(len(payload) == row["bytes"] and hashlib.sha256(payload).hexdigest() == row["sha256"],
                        f"Handoff ZIP payload differs from the verified inventory: {row['path']}")
    except (zipfile.BadZipFile, OSError) as error:
        raise ValueError(f"Cannot verify handoff ZIP: {archive}") from error


def package_data_release(config, base_path):
    base_path = Path(base_path).expanduser().resolve()
    manifest = verify_data_release(base_path)
    require(config["protocol_id"] == manifest["protocol_id"], "Packaging configuration uses a different protocol.")
    for key in ("dataset", "settings", "seeds"):
        require(config[key] == manifest["config"][key], f"Packaging configuration differs from the data release: {key}")
    data_root = base_path / "01_data"
    files = sorted(path for path in data_root.rglob("*") if path.is_file())
    records = [{"path": path.relative_to(base_path).as_posix(), "bytes": path.stat().st_size,
                "sha256": file_sha256(path)} for path in files]
    handoff = base_path / "handoff"
    archive = handoff / f"{config['protocol_id']}_data_release.zip"
    inventory = handoff / f"{config['protocol_id']}_data_release.json"
    if archive.exists() or inventory.exists():
        if not archive.is_file() or not inventory.is_file():
            raise ValueError("The existing handoff package is incomplete; review it before replacing it")
        previous = json.loads(inventory.read_text())
        if (previous["files"] != records or previous["archive_sha256"] != file_sha256(archive)
                or previous["archive_bytes"] != archive.stat().st_size
                or previous["archive_name"] != archive.name or previous["protocol_id"] != config["protocol_id"]):
            raise ValueError("Existing handoff package differs; use a new release version")
        _verify_archive(archive, records)
        print("Reusing verified handoff package:", archive)
        return previous
    handoff.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(archive.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, arcname=path.relative_to(base_path).as_posix())
    temporary.replace(archive)
    _verify_archive(archive, records)
    result = {"protocol_id": config["protocol_id"], "files": records,
              "archive_name": archive.name, "archive_bytes": archive.stat().st_size,
              "archive_sha256": file_sha256(archive),
              "contents": "Official data and baseline JSONL files; no model weights"}
    write_json(inventory, result)
    print("Data handoff package:", archive)
    print("Package SHA-256:", result["archive_sha256"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--base-path")
    parser.add_argument("--verify-only", action="store_true", help="Verify all release files without creating or changing a package")
    args = parser.parse_args()
    config = load_config(args.config)
    base_path = resolve_base_path(config, args.base_path)
    if args.verify_only:
        manifest = verify_data_release(base_path)
        print("Verified data release:", manifest["protocol_id"], base_path / "01_data")
    else:
        package_data_release(config, base_path)
