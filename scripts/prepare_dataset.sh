#!/usr/bin/env bash
# Rebuild the published splits, explicitly exclude tests, and prepare local data.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
release_dir="$(dirname -- "$script_dir")"
python_bin="${PYTHON:-python3}"
output_dir="imdb_prepared"
images_root=""
source_root=""
device="cpu"
test_images="symlink"

usage() {
    cat <<'USAGE'
Usage: bash scripts/prepare_dataset.sh [options]

  --output-dir DIR    New output directory (default: ./imdb_prepared)
  --images-root DIR   Existing all_faces crops; expand local training manifests
  --source-root DIR   IMDB_clean_face source tree; also run MTCNN preparation
  --device DEVICE     Crop preparation device (default: cpu; e.g. cuda:0)
  --test-images MODE  none, symlink, or copy (default: symlink)
  --python PYTHON     Python executable with NumPy (or set PYTHON)

With no image input, rebuild and verify all metadata only. --images-root and
--source-root are mutually exclusive. Existing nonempty outputs are rejected.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --output-dir|--images-root|--source-root|--device|--test-images|--python)
            if [[ $# -lt 2 || -z "$2" ]]; then
                printf 'Missing value for %s\n' "$1" >&2
                exit 2
            fi
            case "$1" in
                --output-dir) output_dir="$2" ;;
                --images-root) images_root="$2" ;;
                --source-root) source_root="$2" ;;
                --device) device="$2" ;;
                --test-images) test_images="$2" ;;
                --python) python_bin="$2" ;;
            esac
            shift 2 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -n "$images_root" && -n "$source_root" ]]; then
    printf '%s\n' 'Use only one of --images-root and --source-root.' >&2
    exit 2
fi
case "$test_images" in
    none|symlink|copy) ;;
    *) printf '%s\n' '--test-images must be none, symlink, or copy.' >&2; exit 2 ;;
esac
if [[ -e "$output_dir" && ( ! -d "$output_dir" || -n "$(ls -A -- "$output_dir")" ) ]]; then
    printf 'Refusing to overwrite a nonempty output: %s\n' "$output_dir" >&2
    exit 2
fi
"$python_bin" -c 'import numpy' || {
    printf '%s\n' 'Install NumPy in the selected Python environment: python -m pip install numpy==1.24.4' >&2
    exit 1
}

# Check the published inputs before creating any working outputs.
"$python_bin" "$script_dir/verify.py" --meta-info "$release_dir/meta_info"
mkdir -p -- "$output_dir"
output_dir="$(cd -- "$output_dir" && pwd)"

# Step 1: select the quality partitions and both fixed test sets.
"$python_bin" "$script_dir/build_splits.py" \
    --scores "$release_dir/meta_info/quality_scores.csv.gz" \
    --output-dir "$output_dir/meta_info" --defer-training

# Step 2: explicit training exclusion, including same-identity HQ filtering.
"$python_bin" "$script_dir/exclude_test_from_training.py" \
    --meta-info "$output_dir/meta_info"

# Step 3: require every split to match the published canonical metadata.
"$python_bin" "$script_dir/verify.py" \
    --meta-info "$release_dir/meta_info" --candidate "$output_dir/meta_info"

# Optional pixel preparation. Frozen scores determine the published split;
# newly downloaded images must cover that fixed source inventory.
if [[ -n "$source_root" ]]; then
    images_root="$output_dir/images/all_faces"
    "$python_bin" "$script_dir/prepare_faces.py" \
        --input-root "$source_root" --output-root "$images_root" --device "$device"
fi
if [[ -n "$images_root" ]]; then
    "$python_bin" "$script_dir/materialize.py" \
        --meta-info "$output_dir/meta_info" --images-root "$images_root" \
        --output-dir "$output_dir/prepared" --test-images "$test_images"
fi
printf 'Dataset preparation complete: %s\n' "$output_dir"
