#!/usr/bin/env python3
"""Verify metadata hashes and optionally reproduce all published splits on CPU."""

import argparse
import gzip
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def manifest_digest(path):
    """Compare decompressed contents independently of the installed zlib version."""
    if path.suffix != ".gz":
        return digest(path)
    value = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def compare_splits(reference, candidate, expected_splits):
    generated = sorted(p for p in candidate.rglob("*") if p.is_file())
    if {p.relative_to(candidate).as_posix() for p in generated} != expected_splits:
        raise ValueError("Regenerated split inventory differs from published metadata")
    for path in generated:
        relative = path.relative_to(candidate)
        if manifest_digest(path) != manifest_digest(reference / relative):
            raise ValueError("Regenerated split differs: " + str(relative))
    print("Reproduced {} split files; all contents match".format(len(generated)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-info", type=Path,
                        default=Path(__file__).resolve().parents[1] / "meta_info")
    parser.add_argument("--rebuild", action="store_true",
                        help="Regenerate splits from quality_scores.csv.gz and compare")
    parser.add_argument("--candidate", type=Path,
                        help="Compare an already regenerated metadata directory")
    parser.add_argument("--write-checksums", action="store_true",
                        help="Maintainer: create SHA256SUMS for the current metadata")
    args = parser.parse_args()
    root = args.meta_info.resolve()
    checksum_file = root / "SHA256SUMS"
    if args.write_checksums:
        files = sorted(path for path in root.rglob("*") if path.is_file() and path != checksum_file)
        if not files:
            parser.error("No metadata to hash")
        checksum_file.write_text("".join(
            "{}  {}\n".format(digest(path), path.relative_to(root).as_posix()) for path in files
        ), encoding="utf-8")
    entries = {}
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name in entries:
            raise ValueError("Invalid checksum entry: " + name)
        entries[name] = expected
        if digest(root / relative) != expected:
            raise ValueError("Checksum mismatch: " + name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p != checksum_file}
    if actual != set(entries):
        raise ValueError("Metadata inventory differs from SHA256SUMS")
    print("Verified {} metadata checksums".format(len(entries)), flush=True)
    expected_splits = actual - {"quality_scores.csv.gz", "provenance.json"}
    if args.candidate:
        compare_splits(root, args.candidate.resolve(), expected_splits)
    if args.rebuild:
        with tempfile.TemporaryDirectory(prefix="imdb_split_rebuild_") as temporary:
            target = Path(temporary) / "meta_info"
            subprocess.run([
                sys.executable, str(Path(__file__).with_name("build_splits.py")),
                "--scores", str(root / "quality_scores.csv.gz"),
                "--output-dir", str(target),
            ], check=True)
            compare_splits(root, target, expected_splits)


if __name__ == "__main__":
    main()
