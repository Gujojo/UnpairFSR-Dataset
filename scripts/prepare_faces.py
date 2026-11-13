#!/usr/bin/env python3
"""Prepare 128-pixel IMDb faces with the production MTCNN crop rules.

Input is the downloaded ``IMDB_clean_face/<identity>/<image>.jpg`` tree;
nested identity directories are preserved as well.
The IMDb downloader has already expanded the dataset's annotated face rectangle
by a factor of two. This script takes those source crops, not the raw URL images.

Image dependencies are imported after argument parsing, so ``--help`` works
without a GPU or an installed image-processing environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
ARCFACE_POINTS = (
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
)
MANIFEST_FIELDS = (
    "source_path", "output_path", "decision", "bbox", "landmarks",
    "crop_box", "x_inside_ratio", "y_inside_ratio", "bbox_contained",
    "eyes_same_side", "mouth_same_side", "padding_left_top_right_bottom", "error",
)


def load_dependencies():
    """Load optional pixel-preparation dependencies only when doing the work."""
    global cv2, np, torch, MTCNN, Image, SimilarityTransform
    try:
        import cv2
        import numpy as np
        import torch
        from facenet_pytorch import MTCNN
        from PIL import Image
        from skimage.transform import SimilarityTransform
    except ImportError as error:
        raise RuntimeError(
            "Face preparation requires numpy, opencv-python, Pillow, "
            "scikit-image, torch, torchvision, and facenet-pytorch. "
            "Manifest regeneration from the released scores does not require "
            "these optional image dependencies."
        ) from error


def crop_geometry(landmarks, target_size):
    """Infer a canonical square's center and scale without rotating the image."""
    transform = SimilarityTransform()
    if not transform.estimate(
        np.asarray(landmarks, dtype=np.float32),
        np.asarray(ARCFACE_POINTS, dtype=np.float32),
    ):
        return None
    scale = float(transform.scale)
    if not np.isfinite(scale) or scale <= 0:
        return None
    try:
        center_x, center_y, _ = np.linalg.inv(transform.params) @ np.array(
            [56.0, 56.0, 1.0]
        )
    except np.linalg.LinAlgError:
        return None
    return float(center_x), float(center_y), target_size / scale / 2.0


def overlap_ratio(start, end, limit):
    overlap = max(0.0, min(float(limit), end) - max(0.0, start))
    return overlap / max(end - start, 1e-8)


def classify(image, bbox, landmarks, target_size, min_inside_ratio):
    geometry = crop_geometry(landmarks, target_size)
    if geometry is None:
        return {"decision": "fail_geometry"}
    center_x, center_y, radius = geometry
    crop_box = (
        center_x - radius, center_y - radius,
        center_x + radius, center_y + radius,
    )
    height, width = image.shape[:2]
    x_ratio = overlap_ratio(crop_box[0], crop_box[2], width)
    y_ratio = overlap_ratio(crop_box[1], crop_box[3], height)
    fully_inside = x_ratio >= 1.0 - 1e-6 and y_ratio >= 1.0 - 1e-6
    bbox_contained = (
        crop_box[0] <= bbox[0] and crop_box[1] <= bbox[1]
        and crop_box[2] >= bbox[2] and crop_box[3] >= bbox[3]
    )
    nose_x = float(landmarks[2, 0])
    eyes_same_side = (
        (float(landmarks[0, 0]) - nose_x)
        * (float(landmarks[1, 0]) - nose_x) > 0
    )
    mouth_same_side = (
        (float(landmarks[3, 0]) - nose_x)
        * (float(landmarks[4, 0]) - nose_x) > 0
    )

    if eyes_same_side and mouth_same_side:
        decision = "fail_profile_eyes_and_mouth_same_side"
    elif x_ratio < min_inside_ratio or y_ratio < min_inside_ratio:
        decision = "fail_outside_over_10pct"
    elif fully_inside and bbox_contained:
        decision = "pass_standard"
    elif fully_inside:
        decision = "pass_canonical_tight_inside"
    elif bbox_contained:
        decision = "pass_reflect_padding"
    else:
        decision = "pass_canonical_tight_reflect_padding"
    return {
        "decision": decision,
        "center_x": center_x,
        "center_y": center_y,
        "radius": radius,
        "crop_box": crop_box,
        "x_inside_ratio": x_ratio,
        "y_inside_ratio": y_ratio,
        "bbox_contained": bool(bbox_contained),
        "eyes_same_side": bool(eyes_same_side),
        "mouth_same_side": bool(mouth_same_side),
    }


def reflect_crop(image, center_x, center_y, radius, target_size):
    height, width = image.shape[:2]
    x1, y1 = math.floor(center_x - radius), math.floor(center_y - radius)
    x2, y2 = math.ceil(center_x + radius), math.ceil(center_y + radius)
    padding = (max(0, -x1), max(0, -y1), max(0, x2 - width), max(0, y2 - height))
    left, top, right, bottom = padding
    padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    crop = padded[y1 + top:y2 + top, x1 + left:x2 + left]
    if crop.size == 0:
        raise RuntimeError("Empty crop")
    # Preserve the production channel-wise Pillow resize of the BGR array.
    resized = np.asarray(
        Image.fromarray(crop).resize((target_size, target_size), Image.Resampling.BICUBIC)
    )
    return resized, padding


def prepare(args):
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")
    if input_root == output_root or input_root in output_root.parents or output_root in input_root.parents:
        raise ValueError("Input and output must be separate directory trees")
    if args.target_size < 1 or not 0 < args.min_inside_ratio <= 1:
        raise ValueError("Require target-size > 0 and 0 < min-inside-ratio <= 1")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard-index < num-shards")

    paths = sorted(
        path for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise RuntimeError(f"No jpg/jpeg/png images found under {input_root}")
    relative_paths = [path.relative_to(input_root) for path in paths]
    if any(len(path.parts) < 2 for path in relative_paths):
        raise ValueError("Expected images within identity directories under input-root")
    outputs = [path.with_suffix(".png") for path in relative_paths]
    if len(outputs) != len(set(outputs)):
        raise ValueError("Multiple source files would produce the same PNG path")
    selected = [
        path for index, path in enumerate(paths)
        if index % args.num_shards == args.shard_index
    ]
    report_root = output_root / "_metadata" / f"shard{args.shard_index:02d}"
    if report_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {report_root}")
    for source in selected:
        destination = output_root / source.relative_to(input_root).with_suffix(".png")
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {destination}")

    load_dependencies()
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
    detector = MTCNN(keep_all=True, device=device)
    report_root.mkdir(parents=True)
    counts = Counter()
    with (report_root / "processing_manifest.csv").open("w", newline="", encoding="utf-8") as manifest, \
         (report_root / "accepted.txt").open("w", encoding="utf-8") as accepted:
        writer = csv.DictWriter(manifest, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for index, source in enumerate(selected, 1):
            relative = source.relative_to(input_root)
            row = {"source_path": relative.as_posix(), "decision": ""}
            image = cv2.imread(str(source))
            if image is None:
                row["decision"] = "fail_read_error"
            else:
                try:
                    boxes, _, landmark_sets = detector.detect(
                        Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
                        landmarks=True,
                    )
                except Exception as error:
                    boxes, landmark_sets = None, None
                    row.update(decision="fail_detection_exception", error=str(error))
                if not row["decision"] and (boxes is None or landmark_sets is None):
                    row["decision"] = "fail_no_face"
                if not row["decision"]:
                    bbox = np.asarray(boxes[0], dtype=np.float32)
                    landmarks = np.asarray(landmark_sets[0], dtype=np.float32)
                    result = classify(image, bbox, landmarks, args.target_size, args.min_inside_ratio)
                    row.update({key: result[key] for key in MANIFEST_FIELDS if key in result})
                    row["bbox"] = ",".join(map(str, bbox.tolist()))
                    row["landmarks"] = ";".join(f"{float(x)},{float(y)}" for x, y in landmarks)
                    row["crop_box"] = ",".join(map(str, result.get("crop_box", ())))
                    if result["decision"].startswith("pass"):
                        try:
                            crop, padding = reflect_crop(
                                image, result["center_x"], result["center_y"],
                                result["radius"], args.target_size,
                            )
                        except Exception as error:
                            row.update(decision="fail_crop_error", error=str(error))
                        else:
                            destination = output_root / relative.with_suffix(".png")
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            if not cv2.imwrite(str(destination), crop):
                                raise OSError(f"Could not write crop: {destination}")
                            row["output_path"] = destination.relative_to(output_root).as_posix()
                            row["padding_left_top_right_bottom"] = ",".join(map(str, padding))
                            accepted.write(row["output_path"] + "\n")
            counts[row["decision"]] += 1
            writer.writerow(row)
            if index % args.progress_every == 0:
                print(f"Processed {index:,}/{len(selected):,} images", flush=True)

    summary = {
        "input_image_count": len(paths),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "processed_images": len(selected),
        "accepted_images": sum(n for decision, n in counts.items() if decision.startswith("pass")),
        "target_size": args.target_size,
        "min_inside_ratio_per_axis": args.min_inside_ratio,
        "resize_interpolation": "pillow_bicubic",
        "padding": "opencv_BORDER_REFLECT_101",
        "device": str(device),
        "counts": dict(sorted(counts.items())),
        "source_path_base": "input-root",
        "output_path_base": "output-root",
    }
    (report_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="Downloaded IMDB_clean_face directory")
    parser.add_argument("--output-root", type=Path, required=True, help="Destination all_faces directory")
    parser.add_argument("--device", help="Torch device, e.g. cpu or cuda:0; defaults to available CUDA, otherwise CPU")
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--min-inside-ratio", type=float, default=0.90)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
