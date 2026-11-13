# IMDb-Face dataset construction and split metadata

This directory contains the dataset preparation scripts and the canonical split
metadata for our IMDb-Face experiments. It includes the WoLQ and Top20/40/60/80
training subsets, both test sets, the explicit Stage 2 training exclusion step,
and the fixed face identification protocol. Images and model weights are obtained
separately. The dataset is **IMDb-Face**, rather than IMDB-WIKI.

## One-command preparation

Run the following commands from this directory. Split regeneration needs only
Python and NumPy; it does not require images, a GPU, or quality model weights.
The split scripts were verified with Python 3.8 and NumPy 1.24.4. The pinned
NumPy version supports Python 3.8–3.11.

```bash
python -m pip install -r scripts/requirements.txt
bash scripts/prepare_dataset.sh --output-dir /path/to/imdb_prepared
```

This runs quality/test selection, then the standalone training exclusion script,
then compares every generated split with the published metadata. Existing
nonempty output directories are rejected. Select a Python environment using
`--python /path/to/python` or the `PYTHON` environment variable.

To also prepare image directories and local training manifests from existing
128 × 128 crops:

```bash
bash scripts/prepare_dataset.sh \
  --images-root /path/to/mtcnn/all_faces \
  --output-dir /path/to/imdb_prepared \
  --test-images symlink
```

The output contains:

```text
imdb_prepared/
  meta_info/                   # portable, regenerated .txt.gz split lists
  prepared/
    meta_info/train/lq.txt     # explicit effective Stage 2 LQ list
    meta_info/train/hq_top40.txt
    meta_info/...              # all lists expanded to local absolute paths
    test/<filename>.png        # the separate 2,000-image IQA test set
    id/lq/<identity>/*.png     # 2,000 identification queries
    id/hq/<identity>/*.png     # 3,330 held-out HQ candidates
    id/others/<identity>/*.png
```

Use `--test-images copy` to copy test images, or `none` to expand manifests only.
Training images stay in the supplied `all_faces` tree. Local manifests point to
that tree, including the source paths of test images. For identification, use
the **476 entries in `prepared/meta_info/test/gallery.txt`**, rather than all
3,330 HQ candidates.

## Inputs and face preparation

The original downloaded input is `IMDB_clean_face`, produced from
`IMDb-Face.csv` by the dataset downloader: filenames retain
`<IMDb index>_<image number>.jpg`, grouped by the supplied identity name. The
download step expands the annotated face rectangle by a factor of two. Our
local download contained 1,026,187 source images; the crop pipeline accepted
794,629 of them.

If `all_faces` is not already available, the same entry point can prepare crops:

```bash
bash scripts/prepare_dataset.sh \
  --source-root /path/to/IMDB_clean_face \
  --device cuda:0 \
  --output-dir /path/to/imdb_prepared
```

`prepare_faces.py` preserves the production MTCNN detection and crop rules:
first detected face and five landmarks, an axis-aligned canonical square,
at least 90% overlap with the source on each axis, profile rejection, OpenCV
reflection padding, and Pillow bicubic resizing to 128 × 128 PNG. It preserves
source stems and relative directories. Its optional dependencies are PyTorch,
torchvision, facenet-pytorch, NumPy, OpenCV, Pillow, and scikit-image. It supports
CPU and GPU processing and `--num-shards` / `--shard-index` for parallel jobs.

New downloads or different detector dependencies can change image availability
or detections. The published source inventory and frozen quality scores define
the canonical split; materialization checks that its source images are present.
Crop preparation alone does not recompute quality scores. The released frozen
scores let readers reproduce every split without rerunning CR-FIQA or MUSIQ.

## What the scripts do

| Script | Input → output |
| --- | --- |
| `prepare_dataset.sh` | Runs split selection → explicit exclusion → comparison; optionally prepares crops and local image layouts |
| `prepare_faces.py` | Downloaded `IMDB_clean_face` → `all_faces/<relative source path>.png` and crop logs |
| `export_scores.py` | Completed IQA SQLite database(s) → portable frozen `path,crfiqa,musiq` CSV.gz |
| `build_splits.py` | Frozen scores → quality categories, test selections, and training lists; `--defer-training` leaves training list writing to the next step |
| `exclude_test_from_training.py` | Quality categories and fixed test metadata → identity-isolated training subsets and effective Stage 2 LQ list |
| `materialize.py` | Portable metadata + local crop root → absolute-path `.txt` lists and optional copied/linked test images |
| `verify.py` | Checks metadata SHA256; `--rebuild` regenerates splits and compares their contents |

The explicit exclusion step can also be run separately on an intermediate split:

```bash
python scripts/build_splits.py \
  --scores meta_info/quality_scores.csv.gz \
  --output-dir /path/to/rebuilt_meta --defer-training
python scripts/exclude_test_from_training.py --meta-info /path/to/rebuilt_meta
python scripts/verify.py --candidate /path/to/rebuilt_meta
```

To export frozen metadata from an existing completed IQA database:

```bash
python scripts/export_scores.py \
  --database /path/to/iqa_scores.sqlite3 \
  --images-root /path/to/mtcnn/all_faces \
  --output /path/to/quality_scores.csv.gz
```

Use either the merged database or its nonoverlapping shards, with the schema
`scores(crop_path, crfiqa, musiq)`. Do not combine a merged database with its
shards. The exporter preserves the original score values. CR-FIQA used aligned
112 × 112 RGB faces and MUSIQ used the 128 × 128 RGB crops. Quality inference is
not required for reproducing the supplied split.

## Selection and exclusion rules

Quantiles are computed globally over all 794,629 accepted images, before test
exclusion. Score arrays are converted to float32 for NumPy's linear quantiles;
strict comparisons use the original stored scores.

* LQ: **both** CR-FIQA and MUSIQ are below their respective 20th percentiles.
* HQ: **both** scores are above their respective 60th percentiles.
* TopP: **both** scores exceed their respective `(100-P)`th percentiles; then
  every image of each identification test identity is excluded. Thus Top20,
  Top40, Top60 and Top80 are nested intersections, not final retained fractions.
* WoLQ: all images excluding identification test identities, minus the full LQ
  category. All final HQ/WoLQ subsets also exclude the IQA test source images.

The two tests serve different purposes:

1. **IQA test:** sort the 20,916 LQ source paths whose identities have no HQ
   image; sample 2,000 using `random.Random(20260722).sample`. These are 1,366
   identities. The selected source images are excluded from training.
2. **Identification test:** start with the sorted 15,629 identities having both
   HQ and LQ images, shuffle with seed `20260722`, and use the first-reachable
   descending subset-sum procedure to select whole identities totaling exactly
   2,000 LQ images. Reserve **all 14,940 images of the resulting 476 identities**,
   including 3,330 HQ images and 9,610 other images, from training.

**Stage 2 explicitly retains only LQ images whose identity has a remaining
Top40 HQ reference.** The resulting `train/lq.txt.gz` contains 69,366 images
from 15,153 identities. It is exactly the effective population used by the
original identity-aware loader. The original raw LQ list had 92,282 rows and
included test records; those records were filtered by the loader. The release
makes that filtering explicit before training and checks both test exclusions.
The Stage 2 HQ reference manifest is `train/hq_top40.txt.gz`; all of its
200,886 eligible references are published, including identities with no LQ.
References are sampled from the same identity as the selected LQ image.

## Canonical metadata and formats

The standard reference files are in **[`meta_info/`](meta_info/)**. Each `.txt.gz`
is a UTF-8, newline-delimited list, sorted lexicographically, with one complete
relative source path per line. Paths are relative to `all_faces`, normally
`identity/filename.png`; nested names such as `AC/DC/filename.png` are preserved.
The identity label is the immediate parent directory name, matching the original
pipeline. There are no collisions between these labels in the released pool.

```text
50_Cent/nm3949531_10.png
50_Cent/nm3949531_102.png
```

`quality_scores.csv.gz` has columns `path,crfiqa,musiq`, one row for each accepted
image. `test/identities.txt` is an uncompressed list of identity labels.
`summary.json` records exact counts, thresholds, seeds, definitions and
isolation checks; `provenance.json` records comparisons with the original
manifests. `SHA256SUMS` covers every metadata file.

| Canonical path under `meta_info/` | Images | Purpose |
| --- | ---: | --- |
| `all.txt.gz` | 794,629 | Complete accepted crop inventory |
| `quality/lq.txt.gz` | 92,282 | Raw quality category, before training exclusion |
| `quality/hq.txt.gz` | 204,216 | Raw HQ category |
| `quality/hq_top20.txt.gz` / `hq_top60.txt.gz` / `hq_top80.txt.gz` | See summary | Global categories before identity exclusion |
| `train/without_lq.txt.gz` | 689,407 | WoLQ training subset |
| `train/hq_top20.txt.gz` | 60,185 | Top20 training subset |
| `train/hq_top40.txt.gz` | 200,886 | Top40 training subset / Stage 2 HQ references |
| `train/hq_top60.txt.gz` | 373,077 | Top60 training subset |
| `train/hq_top80.txt.gz` | 558,759 | Top80 training subset |
| `train/lq.txt.gz` | 69,366 | Explicit effective Stage 2 LQ training list |
| `test/iqa.txt.gz` | 2,000 | IQA test sources |
| `test/queries.txt.gz` | 2,000 | Identification query sources |
| `test/gallery.txt.gz` | 476 | Fixed single-shot gallery |
| `test/hq_pool.txt.gz` | 3,330 | All held-out HQ candidates |
| `test/others.txt.gz` | 9,610 | Other held-out images |
| `test/heldout_all.txt.gz` | 14,940 | Whole-identity exclusion inventory |

Two legacy intermediate lists are also included for auditing:
`train/without_test_images.txt.gz` (792,629; old `wotest`) removes only the IQA
test images, and `train/identity_disjoint.txt.gz` (779,689; old `woid`) removes
only the identification identities. **Neither intermediate is a final training
list.** Use the final WoLQ/TopP/LQ lists described above.

## Identification protocol

The full evaluation configuration is [`scripts/protocol.json`](scripts/protocol.json),
including recognizer checkpoint SHA256, resizing, color order, normalization,
aggregation and tie handling. The identification test is a fixed 476-way,
single-shot closed-set protocol. Seed `20260726` selects one native HQ image per
identity from sorted candidates; `test/gallery.txt.gz` freezes the exact choice.

Queries and gallery images have **disjoint source-image keys and disjoint
filenames**. A gallery image can share the query's identity, which supplies its
positive match, but cannot be the query's source image. An SR result keeps its
LQ source key, so it is subject to the same exclusion. These are source-record
exclusions; the pipeline does not perform pixel or perceptual duplicate removal.
Only queries are restored; the native HQ gallery is fixed across all methods.

ArcFace uses RGB 112 × 112 input, normalized to `[-1,1]`. SphereFace20a uses
**BGR** 96 × 112 input, normalized by `(x-127.5)/128`. Both use Pillow bicubic
resizing, evaluation mode, L2-normalized features, and no additional alignment
or test-time augmentation. Report mean same-identity cosine and Rank-1 accuracy
over all 2,000 queries, using this same gallery and configuration for the native
LQ baseline and every restored result.

For a complete metadata-only reproduction check:

```bash
python scripts/verify.py --rebuild
```
