#!/usr/bin/env bash

set -euo pipefail

# -----------------------------
# CONFIG
# -----------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
cd "$PROJECT_ROOT"

# Always place benchmark assets under ./data/benchmark (relative to project root).
DATA_ROOT="data/benchmark"
POPE_DIR="$DATA_ROOT/pope"
COCO_DIR="$DATA_ROOT/coco_subset"

# Absolute paths for internal filesystem operations.
DATA_ROOT_ABS="$PROJECT_ROOT/$DATA_ROOT"
POPE_DIR_ABS="$PROJECT_ROOT/$POPE_DIR"
COCO_DIR_ABS="$PROJECT_ROOT/$COCO_DIR"

# Download parallelism can be configured with:
# - CLI flag: --jobs N (or -j N)
# - Environment variable: BENCHMARK_DOWNLOAD_JOBS
MAX_PARALLEL="${BENCHMARK_DOWNLOAD_JOBS:-4}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        -j|--jobs)
            if [ "$#" -lt 2 ]; then
                echo "Error: $1 requires a value."
                exit 1
            fi
            MAX_PARALLEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $(basename "$0") [--jobs N]"
            echo "  --jobs, -j N   Max parallel image downloads (default: 4)"
            exit 0
            ;;
        *)
            echo "Error: unknown argument '$1'."
            echo "Usage: $(basename "$0") [--jobs N]"
            exit 1
            ;;
    esac
done

if ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --jobs must be a positive integer, got '$MAX_PARALLEL'."
    exit 1
fi

BASE_URL="https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco"
COCO_IMG_BASE="http://images.cocodataset.org/val2014"

# -----------------------------
# REQUIREMENTS
# -----------------------------
if command -v curl >/dev/null 2>&1; then
    DOWNLOADER="curl"
elif command -v wget >/dev/null 2>&1; then
    DOWNLOADER="wget"
else
    echo "Error: neither 'curl' nor 'wget' is available in PATH."
    exit 1
fi

download_file() {
    local url="$1"
    local output="$2"

    if [ "$DOWNLOADER" = "curl" ]; then
        curl -fsSL "$url" -o "$output"
    else
        wget -q -O "$output" "$url"
    fi
}

# -----------------------------
# CREATE DIRS
# -----------------------------
mkdir -p "$POPE_DIR"
mkdir -p "$COCO_DIR"

# -----------------------------
# DOWNLOAD POPE FILES
# -----------------------------
echo "Downloading POPE annotation files..."
cd "$POPE_DIR_ABS"

pope_missing=0
for split in random popular adversarial; do
    FILE="coco_pope_${split}.json"
    URL="$BASE_URL/$FILE"

    if [ -f "$FILE" ] && [ -s "$FILE" ]; then
        echo "Annotation already present, skipping: $FILE"
    else
        pope_missing=$((pope_missing + 1))
        download_file "$URL" "$FILE"
    fi
done

if [ "$pope_missing" -eq 0 ]; then
    echo "All POPE annotation files already present."
fi

# -----------------------------
# EXTRACT UNIQUE IMAGE NAMES
# -----------------------------
echo "Extracting image list..."

cd "$POPE_DIR_ABS"

if [ -s image_list.txt ]; then
    echo "Image list already present, skipping regeneration."
else
    python -c "import glob,json,pathlib; imgs=set();
for p in glob.glob('coco_pope_*.json'):
    text=pathlib.Path(p).read_text(encoding='utf-8').strip()
    if not text:
        continue
    try:
        data=json.loads(text)
        if isinstance(data,list):
            for row in data:
                if isinstance(row,dict) and 'image' in row:
                    imgs.add(str(row['image']).strip())
        elif isinstance(data,dict) and 'image' in data:
            imgs.add(str(data['image']).strip())
    except json.JSONDecodeError:
        for line in text.splitlines():
            line=line.strip()
            if not line:
                continue
            row=json.loads(line)
            if isinstance(row,dict) and 'image' in row:
                imgs.add(str(row['image']).strip())
print('\n'.join(sorted(imgs)))" > image_list.txt
fi

NUM_IMAGES=$(wc -l < image_list.txt)
echo "Found $NUM_IMAGES unique images"

# -----------------------------
# DOWNLOAD ONLY REQUIRED IMAGES
# -----------------------------
echo "Downloading required COCO images..."
echo "Parallel workers: $MAX_PARALLEL"

cd "$COCO_DIR_ABS"

pids=()
failed=0
missing_images=0

while read -r img; do
    # Remove Windows CR if present and skip empty lines.
    img="${img%$'\r'}"
    [ -z "$img" ] && continue

    if [ ! -f "$img" ]; then
        missing_images=$((missing_images + 1))
        while [ "${#pids[@]}" -ge "$MAX_PARALLEL" ]; do
            still_running=()
            for pid in "${pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    still_running+=("$pid")
                else
                    if ! wait "$pid"; then
                        failed=$((failed + 1))
                    fi
                fi
            done
            pids=("${still_running[@]}")
            [ "${#pids[@]}" -ge "$MAX_PARALLEL" ] && sleep 0.1
        done

        (
            mkdir -p "$(dirname -- "$img")"
            download_file "$COCO_IMG_BASE/$img" "$img"
        ) &
        pids+=("$!")
    fi
done < "$POPE_DIR_ABS/image_list.txt"

if [ "$missing_images" -eq 0 ]; then
    echo "All required images already present."
fi

for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=$((failed + 1))
    fi
done

if [ "$failed" -gt 0 ]; then
    echo "Error: $failed image download(s) failed."
    exit 1
fi

echo ""
echo "Done!"
echo "Project root: $PROJECT_ROOT"
echo "Images stored in: $COCO_DIR"
echo "Annotations in: $POPE_DIR"
