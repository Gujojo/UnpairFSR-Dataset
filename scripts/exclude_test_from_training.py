#!/usr/bin/env python3
"""Exclude reserved test sources and identities, then apply same-ID HQ filtering.

Run after ``build_splits.py --defer-training``::

    python exclude_test_from_training.py --meta-info generated_meta

The default output is ``generated_meta/train``. Only metadata is written;
source images are never moved or modified. This script needs only Python's
standard library and can also be rerun with an independent --output-dir.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import shutil
import tempfile
from pathlib import Path, PurePosixPath


def validate_source_key(key: str) -> None:
    parsed = PurePosixPath(key)
    if (
        parsed.is_absolute()
        or len(parsed.parts) < 2
        or any(part in (".", "..") for part in parsed.parts)
        or parsed.as_posix() != key
        or "\\" in key
        or any(character in key for character in "\x00\r\n")
        or key.strip() != key
        or parsed.suffix.lower() != ".png"
    ):
        raise ValueError(f"Unsafe or unsupported image key: {key!r}")


def identity(key: str) -> str:
    # Equivalent to PurePosixPath(key).parent.name for validated POSIX keys.
    return key.rsplit("/", 2)[-2]


def read_paths(path: Path) -> set[str]:
    if not path.exists() and path.suffix == ".gz":
        path = path.with_suffix("")
    opener = gzip.open if path.suffix == ".gz" else open
    records = set()
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        for number, line in enumerate(handle, start=1):
            key = line[:-1] if line.endswith("\n") else line
            validate_source_key(key)
            if key in records:
                raise ValueError(f"Duplicate source in {path.name}, line {number}: {key!r}")
            records.add(key)
    return records


def write_paths(path: Path, paths: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as handle:
                for key in sorted(paths):
                    handle.write(key + "\n")


def make_training_pools(
    all_paths: set[str],
    lq: set[str],
    raw_top: dict[int, set[str]],
    heldout_ids: set[str],
    iqa: set[str],
) -> dict[str, set[str]]:
    """Return train/* manifests, preserving the original effective population.

    Full test identities are removed from every HQ/WoLQ pool. Stage 2 LQ
    additionally requires an identity with at least one retained Top40 HQ
    reference. The two legacy intermediate manifests retain their historical
    definitions and are not final training pools.
    """
    if set(raw_top) != {20, 40, 60, 80}:
        raise ValueError("Expected raw HQ pools for Top20, Top40, Top60 and Top80")
    if not all_paths or not lq <= all_paths or not iqa <= lq:
        raise ValueError("All/LQ/IQA manifests have inconsistent source membership")
    if not all(paths <= all_paths for paths in raw_top.values()):
        raise ValueError("Raw HQ pools contain sources outside all.txt")
    if not raw_top[20] <= raw_top[40] <= raw_top[60] <= raw_top[80]:
        raise ValueError("Raw HQ pools are not nested Top20 <= Top40 <= Top60 <= Top80")
    if raw_top[80] & lq:
        raise ValueError("Raw HQ and LQ source pools overlap")
    all_ids = {identity(key) for key in all_paths}
    if not heldout_ids or not heldout_ids <= all_ids:
        raise ValueError("Reserved test identities are empty or absent from all.txt")
    hq_ids = {identity(key) for key in raw_top[40]}
    if {identity(key) for key in iqa} & hq_ids:
        raise ValueError("IQA test identities must have no selected raw Top40 HQ")
    heldout_all = {key for key in all_paths if identity(key) in heldout_ids}
    if iqa & heldout_all:
        raise ValueError("IQA and retrieval test sources overlap")

    identity_disjoint = all_paths - heldout_all
    train_top = {
        percentage: paths & identity_disjoint for percentage, paths in raw_top.items()
    }
    reference_ids = {identity(key) for key in train_top[40]}
    # Publish the filtering historically performed inside MetaIDRefDataset.
    # Both explicit exclusions are redundant with same-ID HQ filtering here,
    # but state the test removal directly and are checked below.
    train_lq = {
        key for key in lq - heldout_all - iqa if identity(key) in reference_ids
    }
    original_effective_lq = {key for key in lq if identity(key) in reference_ids}
    if train_lq != original_effective_lq:
        raise ValueError("Explicit test exclusion changed the original effective LQ population")
    without_lq = identity_disjoint - lq
    final_pools = [train_lq, without_lq, *train_top.values()]
    if any(paths & iqa or paths & heldout_all for paths in final_pools):
        raise ValueError("A final training pool contains reserved test source images")

    outputs = {
        "train/without_test_images.txt.gz": all_paths - iqa,
        "train/identity_disjoint.txt.gz": identity_disjoint,
        "train/without_lq.txt.gz": without_lq,
        "train/lq.txt.gz": train_lq,
    }
    outputs.update({
        f"train/hq_top{percentage}.txt.gz": paths
        for percentage, paths in train_top.items()
    })
    return outputs


def load_training_inputs(meta_info: Path) -> tuple:
    all_paths = read_paths(meta_info / "all.txt.gz")
    lq = read_paths(meta_info / "quality/lq.txt.gz")
    raw_top = {
        percentage: read_paths(meta_info / (
            "quality/hq.txt.gz" if percentage == 40 else f"quality/hq_top{percentage}.txt.gz"
        ))
        for percentage in (20, 40, 60, 80)
    }
    lines = (meta_info / "test/identities.txt").read_text(encoding="utf-8").splitlines()
    if any(
        not name or name.strip() != name or "/" in name or "\\" in name
        or name in (".", "..") or "\x00" in name for name in lines
    ) or len(set(lines)) != len(lines):
        raise ValueError("Invalid or duplicate reserved identity names")
    heldout_ids = set(lines)
    iqa = read_paths(meta_info / "test/iqa.txt.gz")
    recorded_all = read_paths(meta_info / "test/heldout_all.txt.gz")
    queries = read_paths(meta_info / "test/queries.txt.gz")
    hq_pool = read_paths(meta_info / "test/hq_pool.txt.gz")
    gallery = read_paths(meta_info / "test/gallery.txt.gz")
    others = read_paths(meta_info / "test/others.txt.gz")
    expected_all = {key for key in all_paths if identity(key) in heldout_ids}
    if recorded_all != expected_all:
        raise ValueError("Retrieval heldout_all does not match the reserved identities")
    if queries != expected_all & lq or hq_pool != expected_all & raw_top[40]:
        raise ValueError("Retrieval LQ/HQ categories do not match raw quality pools")
    if others != expected_all - queries - hq_pool:
        raise ValueError("Retrieval others does not complete the held-out partition")
    if not gallery <= hq_pool or len(gallery) != len(heldout_ids):
        raise ValueError("Gallery must contain one reserved HQ source per identity")
    if {identity(key) for key in gallery} != heldout_ids:
        raise ValueError("Gallery identity coverage differs from the reserved identities")
    if {identity(key) for key in queries} != heldout_ids:
        raise ValueError("Query identity coverage differs from the reserved identities")
    if (
        queries & gallery
        or {key.rsplit("/", 1)[-1] for key in queries}
        & {key.rsplit("/", 1)[-1] for key in gallery}
    ):
        raise ValueError("Query and gallery source images/basenames overlap")
    return all_paths, lq, raw_top, heldout_ids, iqa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-info", required=True, type=Path, help="Generated all/quality/test metadata root")
    parser.add_argument("--output-dir", type=Path, help="New or empty train directory; default META_INFO/train")
    args = parser.parse_args()
    output = args.output_dir if args.output_dir is not None else args.meta_info / "train"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite a nonempty output: {output}")
    pools = make_training_pools(*load_training_inputs(args.meta_info))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for relative, paths in sorted(pools.items()):
            write_paths(staging / PurePosixPath(relative).name, paths)
        if output.exists():
            output.rmdir()
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"counts": {name: len(paths) for name, paths in sorted(pools.items())}}, indent=2))


if __name__ == "__main__":
    main()
