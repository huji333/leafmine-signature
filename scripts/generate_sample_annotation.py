"""Generate or update a sample annotation CSV from segmented PNGs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
from typing import Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
NAMING_MODULE_PATH = REPO_ROOT / "models" / "utils" / "naming.py"


def _load_canonical_sample_name() -> Callable[[str | Path], str]:
    spec = importlib.util.spec_from_file_location("leafmine_naming", NAMING_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load naming helpers from {NAMING_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[misc]
    return module.canonical_sample_name


canonical_sample_name = _load_canonical_sample_name()


def derive_sample_ids(directory: Path, pattern: str) -> list[str]:
    directory = directory.expanduser()
    if not directory.exists():
        raise FileNotFoundError(f"{directory} does not exist.")

    files = sorted(
        path for path in directory.glob(pattern) if path.is_file() and path.suffix.lower() == ".png"
    )
    if not files:
        raise ValueError(f"No PNG files matching '{pattern}' found under {directory}.")

    seen: set[str] = set()
    ordered: list[str] = []
    for path in files:
        name = canonical_sample_name(path)
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def read_existing_annotation(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        return [], ["sample_id"]
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or ["sample_id"]
        if "sample_id" not in headers:
            raise ValueError(f"{path} must contain a 'sample_id' column.")
        rows = [row for row in reader if row.get("sample_id")]
    return rows, headers


def merge_rows(
    desired_ids: Sequence[str],
    existing_rows: Sequence[dict[str, str]],
    headers: Sequence[str],
) -> list[dict[str, str]]:
    lookup = {row["sample_id"]: row for row in existing_rows if row.get("sample_id")}
    merged: list[dict[str, str]] = []

    for sample_id in desired_ids:
        row = lookup.get(sample_id)
        if row is None:
            row = {column: "" for column in headers}
            row["sample_id"] = sample_id
        merged.append(row)

    desired_set = set(desired_ids)
    merged.extend(row for row in existing_rows if row.get("sample_id") not in desired_set)
    return merged


def write_annotation_csv(
    rows: Sequence[dict[str, str]],
    destination: Path,
    headers: Sequence[str],
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(headers))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in headers})
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upsert sample annotations derived from segmented PNG filenames."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "data" / "segmented",
        help="Directory containing segmented PNGs (default: data/segmented).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "sample_annotation.csv",
        help="Destination CSV path (default: data/sample_annotation.csv).",
    )
    parser.add_argument(
        "--glob",
        default="segmented_*.png",
        help="Glob pattern for selecting files (default: segmented_*.png).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sample_ids = derive_sample_ids(args.input_dir, args.glob)
    existing_rows, headers = read_existing_annotation(args.output)
    merged_rows = merge_rows(sample_ids, existing_rows, headers)
    write_annotation_csv(merged_rows, args.output, headers)
    print(f"Upserted {len(sample_ids)} segmented sample(s) into {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
