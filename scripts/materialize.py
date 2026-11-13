#!/usr/bin/env python3
"""Expand portable manifests and optionally copy/link the fixed test images."""

import argparse
import gzip
from pathlib import Path, PurePosixPath
import shutil


def paths_from(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            key = line.rstrip("\n")
            relative = PurePosixPath(key)
            if (relative.is_absolute() or len(relative.parts) < 2
                    or any(part in (".", "..") for part in relative.parts)
                    or "\\" in key):
                raise ValueError("Invalid source key: " + repr(key))
            yield relative


def populate(source, destination, mode):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-info", type=Path,
                        default=Path(__file__).resolve().parents[1] / "meta_info")
    parser.add_argument("--images-root", type=Path, required=True,
                        help="all_faces directory containing the original relative source paths")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory; existing nonempty directories are rejected")
    parser.add_argument("--test-images", choices=("none", "symlink", "copy"), default="none",
                        help="Optionally create test/ and id/{lq,hq,others} image layouts")
    args = parser.parse_args()
    source_root = args.images_root.resolve()
    output = args.output_dir.resolve()
    metadata = args.meta_info.resolve()
    if not source_root.is_dir():
        parser.error("--images-root must be an existing source crop directory")
    if output == source_root or source_root in output.parents or output in source_root.parents:
        parser.error("Output must be separate from the source image tree")
    if output == metadata or metadata in output.parents or output in metadata.parents:
        parser.error("Output must be separate from the published metadata tree")
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty: " + str(output))

    manifest_files = sorted(metadata.rglob("*.txt.gz"))
    if not manifest_files:
        parser.error("No .txt.gz manifests in --meta-info")
    # Validate every source before writing anything. This catches incomplete downloads.
    all_sources = list(paths_from(metadata / "all.txt.gz"))
    missing = [str(path) for path in all_sources if not (source_root / str(path)).is_file()]
    if missing:
        parser.error("Missing {} source images; first missing: {}".format(len(missing), missing[0]))

    output.mkdir(parents=True, exist_ok=True)
    for manifest in manifest_files:
        target = output / "meta_info" / manifest.relative_to(metadata).with_suffix("")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            for key in paths_from(manifest):
                handle.write(str(source_root / str(key)) + "\n")
    identities = metadata / "test/identities.txt"
    shutil.copy2(identities, output / "meta_info/test/identities.txt")

    if args.test_images != "none":
        layouts = {
            "test/iqa.txt.gz": ("test", True),
            "test/queries.txt.gz": ("id/lq", False),
            "test/hq_pool.txt.gz": ("id/hq", False),
            "test/others.txt.gz": ("id/others", False),
        }
        for manifest, (directory, flatten) in layouts.items():
            seen = set()
            for key in paths_from(metadata / manifest):
                name = key.name if flatten else str(key)
                if name in seen:
                    raise ValueError("Duplicate destination: " + name)
                seen.add(name)
                populate(source_root / str(key), output / directory / name, args.test_images)
    print("Expanded {} source-path manifests in {}".format(len(manifest_files), output / "meta_info"))
    if args.test_images != "none":
        print("Created fixed test image layouts in {}".format(output))


if __name__ == "__main__":
    main()
