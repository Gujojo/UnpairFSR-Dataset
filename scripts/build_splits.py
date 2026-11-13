#!/usr/bin/env python3
"""Rebuild the released IMDb-Face manifests from frozen per-image scores.

All image records use a source path relative to ``all_faces``. Identities are
the immediate parent directory names, matching the original split pipeline.
This script only writes metadata and never moves or modifies source images.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import numpy as np

from exclude_test_from_training import identity, make_training_pools, validate_source_key, write_paths


def read_scores(path: Path) -> list[tuple[str, float, float]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    seen = set()
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["path", "crfiqa", "musiq"]:
            raise ValueError("Scores must have exactly the header path,crfiqa,musiq")
        for number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed score row {number}")
            key = row["path"]
            validate_source_key(key)
            if key in seen:
                raise ValueError(f"Duplicate image key on row {number}: {key!r}")
            crfiqa, musiq = float(row["crfiqa"]), float(row["musiq"])
            if not math.isfinite(crfiqa) or not math.isfinite(musiq):
                raise ValueError(f"Non-finite score on row {number}: {key!r}")
            seen.add(key)
            rows.append((key, crfiqa, musiq))
    if not rows:
        raise ValueError("Scores contain no image records")
    return sorted(rows)


def group_paths(paths: set[str]) -> dict[str, list[str]]:
    grouped = defaultdict(list)
    for key in sorted(paths):
        grouped[identity(key)].append(key)
    return dict(grouped)


def exact_identity_subset(
    lq_by_identity: dict[str, list[str]],
    eligible_identities: list[str],
    target: int,
    seed: int,
) -> list[str]:
    """Match the original shuffled, first-hit, descending subset-sum selection."""
    identities = list(eligible_identities)
    random.Random(seed).shuffle(identities)
    predecessor = [None] * (target + 1)
    predecessor[0] = (-1, "")
    for name in identities:
        count = len(lq_by_identity[name])
        if count > target:
            continue
        for total in range(target, count - 1, -1):
            if predecessor[total] is None and predecessor[total - count] is not None:
                predecessor[total] = (total - count, name)
        if predecessor[target] is not None:
            break
    if predecessor[target] is None:
        raise ValueError(f"Could not form an exact LQ total of {target}")

    selected = []
    total = target
    while total:
        previous, name = predecessor[total]
        selected.append(name)
        total = previous
    return sorted(selected)


def build_splits(
    rows: list[tuple[str, float, float]],
    split_seed: int,
    gallery_seed: int,
    iqa_count: int,
    query_count: int,
) -> tuple[dict[str, set[str]], list[str], dict]:
    all_paths = {row[0] for row in rows}
    # The original pipeline converts score arrays to float32 for quantiles,
    # then compares the original SQLite Python floats to those thresholds.
    scores = np.asarray([(row[1], row[2]) for row in rows], dtype=np.float32)
    if not np.isfinite(scores).all():
        raise ValueError("Scores overflow when converted to float32")
    thresholds = {}
    for quantile in (20, 40, 60, 80):
        # NumPy's default quantile interpolation/method is linear, including
        # releases predating the method= keyword.
        values = np.quantile(scores, quantile / 100.0, axis=0)
        thresholds[f"crfiqa_q{quantile}"] = float(values[0])
        thresholds[f"musiq_q{quantile}"] = float(values[1])
    lq = {
        key for key, face, general in rows
        if face < thresholds["crfiqa_q20"] and general < thresholds["musiq_q20"]
    }
    top = {
        percentage: {
            key for key, face, general in rows
            if face > thresholds[f"crfiqa_q{100 - percentage}"]
            and general > thresholds[f"musiq_q{100 - percentage}"]
        }
        for percentage in (20, 40, 60, 80)
    }
    hq = top[40]
    hq_by_identity = group_paths(hq)
    lq_by_identity = group_paths(lq)

    candidates = sorted(key for key in lq if identity(key) not in hq_by_identity)
    if len(candidates) < iqa_count:
        raise ValueError(f"Only {len(candidates)} no-HQ candidates for {iqa_count} IQA images")
    if len({PurePosixPath(key).name for key in candidates}) != len(candidates):
        raise ValueError("No-HQ candidates have duplicate basenames; flat test layout is ambiguous")
    iqa = set(random.Random(split_seed).sample(candidates, iqa_count))

    eligible = sorted(set(hq_by_identity) & set(lq_by_identity))
    selected_identities = exact_identity_subset(
        lq_by_identity, eligible, query_count, split_seed
    )
    heldout_ids = set(selected_identities)
    heldout_all = {key for key in all_paths if identity(key) in heldout_ids}
    queries = heldout_all & lq
    hq_pool = heldout_all & hq
    others = heldout_all - queries - hq_pool
    rng = random.Random(gallery_seed)
    gallery = {rng.choice(hq_by_identity[name]) for name in selected_identities}

    training_pools = make_training_pools(all_paths, lq, top, heldout_ids, iqa)
    train_top = {
        percentage: training_pools[f"train/hq_top{percentage}.txt.gz"]
        for percentage in top
    }
    reference_ids = {identity(key) for key in train_top[40]}
    train_lq = training_pools["train/lq.txt.gz"]
    without_lq = training_pools["train/without_lq.txt.gz"]
    outputs = {
        "all.txt.gz": all_paths,
        "quality/lq.txt.gz": lq,
        "quality/hq.txt.gz": hq,
        "test/iqa.txt.gz": iqa,
        "test/queries.txt.gz": queries,
        "test/gallery.txt.gz": gallery,
        "test/hq_pool.txt.gz": hq_pool,
        "test/others.txt.gz": others,
        "test/heldout_all.txt.gz": heldout_all,
    }
    outputs.update(training_pools)
    for percentage in (20, 60, 80):
        outputs[f"quality/hq_top{percentage}.txt.gz"] = top[percentage]

    checks = {
        "query_count_matches_target": len(queries) == query_count,
        "one_gallery_image_per_query_identity": (
            len(gallery) == len(heldout_ids)
            and {identity(key) for key in gallery} == {identity(key) for key in queries}
        ),
        "queries_and_gallery_have_disjoint_sources": not (queries & gallery),
        "queries_and_gallery_have_disjoint_basenames": not (
            {PurePosixPath(key).name for key in queries}
            & {PurePosixPath(key).name for key in gallery}
        ),
        "query_basenames_are_unique": (
            len({PurePosixPath(key).name for key in queries}) == len(queries)
        ),
        "heldout_categories_partition_all": (
            not (queries & hq_pool) and queries | hq_pool | others == heldout_all
        ),
        "iqa_and_retrieval_sources_disjoint": not (iqa & heldout_all),
        "stage2_lq_excludes_iqa_sources": not (train_lq & iqa),
        "stage2_lq_excludes_heldout_identities": not (
            {identity(key) for key in train_lq} & heldout_ids
        ),
        "stage2_lq_has_same_identity_top40_reference": (
            {identity(key) for key in train_lq} <= reference_ids
        ),
        "without_lq_excludes_all_lq": not (without_lq & lq),
        "training_top_subsets_are_nested": (
            train_top[20] <= train_top[40] <= train_top[60] <= train_top[80]
        ),
        "training_top_subsets_exclude_lq_and_iqa": all(
            not (paths & lq) and not (paths & iqa) for paths in train_top.values()
        ),
        "all_training_top_subsets_exclude_heldout_identities": all(
            not ({identity(key) for key in paths} & heldout_ids)
            for paths in train_top.values()
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("Split invariants failed: " + ", ".join(failed))

    summary = {
        "format_version": 1,
        "source_key": "canonical source path relative to all_faces, preserving nested directories",
        "identity_key": "immediate parent directory name, matching the original pipeline",
        "split_seed": split_seed,
        "gallery_seed": gallery_seed,
        "quantile_dtype": "float32",
        "quantile_method": "linear",
        "threshold_population": "all scored accepted crops, before test exclusion",
        "threshold_comparison": "original float64 scores; strict < for LQ, > for HQ/Top",
        "thresholds": thresholds,
        "counts": {name: len(paths) for name, paths in sorted(outputs.items())},
        "identity_counts": {
            "all": len({identity(key) for key in all_paths}),
            "iqa_test": len({identity(key) for key in iqa}),
            "retrieval_test": len(heldout_ids),
            "eligible_retrieval": len(eligible),
            "stage2_lq": len({identity(key) for key in train_lq}),
            "stage2_hq_pool": len(reference_ids),
        },
        "no_hq_lq_candidates": len(candidates),
        "requested_iqa_images": iqa_count,
        "requested_query_images": query_count,
        "definitions": {
            "quality/lq.txt.gz": "Both scores strictly below their global 20th percentile.",
            "quality/hq.txt.gz": "Both scores strictly above their global 60th percentile.",
            "quality/hq_topP.txt.gz": (
                "Both scores strictly above global (100-P)th percentiles, before test exclusion; "
                "quality/hq.txt.gz is the raw Top40 pool."
            ),
            "train/hq_topP.txt.gz": (
                "Both scores strictly above global (100-P)th percentiles, "
                "excluding every retrieval-test identity; P is not a final retained percentage."
            ),
            "train/without_lq.txt.gz": "All crops minus retrieval-test identities minus raw LQ.",
            "train/lq.txt.gz": (
                "Raw LQ with an identity present in train/hq_top40; "
                "the exact effective original Stage 2 LQ population."
            ),
            "train/without_test_images.txt.gz": (
                "Legacy wotest intermediate: all crops minus only the IQA test images. "
                "Includes retrieval-test identities; do not use as a final training manifest."
            ),
            "train/identity_disjoint.txt.gz": (
                "Legacy woid intermediate: all crops minus retrieval-test identities. "
                "Still includes IQA test sources; do not use as a final training manifest."
            ),
            "test/iqa.txt.gz": "Seeded sample of LQ images from identities absent from raw HQ.",
            "test/queries.txt.gz": "Every LQ image of the selected retrieval-test identities.",
            "test/gallery.txt.gz": "One seeded raw HQ choice per selected identity, fixed for every method.",
            "test/heldout_all.txt.gz": "All HQ, LQ and other images of selected retrieval-test identities.",
        },
        "checks": checks,
    }
    return outputs, selected_identities, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, type=Path, help="Frozen path,crfiqa,musiq CSV or CSV.gz")
    parser.add_argument("--output-dir", required=True, type=Path, help="New or empty metadata directory")
    parser.add_argument("--seed", type=int, default=20260722, help="IQA and identity-group split seed")
    parser.add_argument("--gallery-seed", type=int, default=20260726)
    parser.add_argument("--iqa-count", type=int, default=2000)
    parser.add_argument("--query-count", type=int, default=2000)
    parser.add_argument(
        "--defer-training", action="store_true",
        help="Write all/quality/test only; run exclude_test_from_training.py next to create train/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iqa_count <= 0 or args.query_count <= 0:
        raise ValueError("IQA and query targets must be positive")
    output = args.output_dir
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite a nonempty output: {output}")
    rows = read_scores(args.scores)
    outputs, identities, summary = build_splits(
        rows, args.seed, args.gallery_seed, args.iqa_count, args.query_count
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for relative, paths in sorted(outputs.items()):
            if args.defer_training and relative.startswith("train/"):
                continue
            write_paths(staging / relative, paths)
        (staging / "test/identities.txt").write_text(
            "".join(name + "\n" for name in identities), encoding="utf-8"
        )
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            output.rmdir()  # Only the prechecked empty directory is accepted.
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"counts": summary["counts"], "checks": summary["checks"]}, indent=2))


if __name__ == "__main__":
    main()
