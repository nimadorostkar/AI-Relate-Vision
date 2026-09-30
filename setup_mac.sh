#!/usr/bin/env bash
# One-time setup for AI-Relate-Vision on a Mac (Intel or Apple Silicon).
#
#   ./setup_mac.sh                                  # everything, skips what exists
#   ./setup_mac.sh --objects "person, dog, frisbee" # rebuild the detector with YOUR classes
#   ./setup_mac.sh --rebuild-detector               # rebuild with the default 497 classes
#   ./setup_mac.sh --add-predicates "high-fiving, repairing"  # teach NEW relation phrases
#
# 1. .venv with the torch-free runtime (onnxruntime, OpenVINO, OpenCV)
# 2. the RelateAnything ViT-S+ relation graph from Hugging Face (~208 MB)
# 3. the YOLO-World object detector, rebuilt locally in a throwaway venv
#    (ultralytics is AGPL and never redistributed; torch is only needed here)
# 4. the OpenVINO IR of the relation head (what runs fastest on a laptop CPU)
# 5. optional: new relation phrases, embedded by the model's own text encoder
# The throwaway venv (.build-env/, ~1.5 GB) is deleted afterwards unless
# KEEP_BUILD_ENV=1.
set -euo pipefail
cd "$(dirname "$0")"
REPO="$PWD"

OBJECTS=""
REBUILD_DET=0
NEW_PREDICATES=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --objects) OBJECTS="$2"; REBUILD_DET=1; shift 2 ;;
        --rebuild-detector) REBUILD_DET=1; shift ;;
        --add-predicates) NEW_PREDICATES="$2"; shift 2 ;;
        -h|--help) sed -n 2,18p "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

say() { printf "\n\033[1;32m==>\033[0m %s\n" "$*"; }

BUILD=.build-env
build_env() {      # torch + ultralytics + transformers, only for one-time builds
    if [[ ! -f "$BUILD/.ready" ]]; then
        say "Preparing the one-time build environment (torch, ultralytics; ~1.5 GB)"
        [[ -x "$BUILD/bin/python" ]] || "$PY" -m venv "$BUILD"
        "$BUILD/bin/python" -m pip install -q --upgrade pip
        "$BUILD/bin/python" -m pip install -q -r requirements-build.txt
        touch "$BUILD/.ready"
    fi
}
cleanup_build() {
    if [[ "${KEEP_BUILD_ENV:-0}" != 1 ]]; then
        rm -rf "$BUILD" checkpoints/detectors checkpoints/relsgg-vits16plus
    fi
}
trap cleanup_build EXIT

# --- python ------------------------------------------------------------------
PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
    for c in python3.12 python3.11 python3.13 python3; do
        if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)'; then
            PY="$c"; break
        fi
    done
fi
[[ -n "$PY" ]] || { echo "Python 3.10-3.13 is required (python.org or: brew install python@3.12)" >&2; exit 1; }
say "Using $($PY --version) ($(uname -m))"

# --- 1. runtime venv ----------------------------------------------------------
if [[ ! -x .venv/bin/python ]]; then
    say "Creating .venv"
    "$PY" -m venv .venv
fi
say "Installing runtime packages"
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements-mac.txt

# --- 2. relation model ------------------------------------------------------------
DIST=deploy/dist/relsgg-vits16plus
HF=https://huggingface.co/maelic/relsgg-vits16plus/resolve/main
if [[ ! -s "$DIST/relateanything.onnx" ]]; then
    say "Downloading the relation model (208 MB)"
    curl -L --fail --retry 3 -C - -o "$DIST/relateanything.onnx.part" "$HF/relateanything.onnx"
    mv "$DIST/relateanything.onnx.part" "$DIST/relateanything.onnx"
fi
for f in relateanything.json predicate_bank.npz calibration.json thresholds.json; do
    [[ -s "$DIST/$f" ]] || curl -sL --fail -o "$DIST/$f" "$HF/$f"
done

# --- 3. detector ---------------------------------------------------------------------
DET=deploy/dist/detector-local
if [[ $REBUILD_DET == 1 || ! -s "$DET/detector.onnx" || ! -s "$DET/detector_416.onnx" ]]; then
    build_env
    say "Building the object detector (YOLO-World + CLIP text encoder)"
    mkdir -p checkpoints/detectors "$DET"
    CLASSES_ARGS=()
    if [[ -n "$OBJECTS" ]]; then
        IFS=',' read -r -a arr <<< "$OBJECTS"
        for c in "${arr[@]}"; do
            c="$(echo "$c" | sed 's/^ *//; s/ *$//')"
            [[ -n "$c" ]] && CLASSES_ARGS+=("$c")
        done
        say "Custom object classes (${#CLASSES_ARGS[@]}): ${CLASSES_ARGS[*]}"
    fi
    ( cd checkpoints/detectors && \
      "$REPO/$BUILD/bin/python" "$REPO/deploy/reparam_detector.py" --arch yolo-world \
          --base yolov8s-worldv2.pt --out yolov8s-worldv2_megasg497.pt \
          ${CLASSES_ARGS[@]+--classes "${CLASSES_ARGS[@]}"} )
    for size in 640 416; do
        "$BUILD/bin/python" deploy/export_detector_onnx.py --size s --imgsz $size
        suffix=$([[ $size == 640 ]] && echo "" || echo "_$size")
        cp checkpoints/detectors/yolov8s-worldv2_megasg497.onnx "$DET/detector$suffix.onnx"
        cp checkpoints/detectors/yolov8s-worldv2_megasg497.json "$DET/detector$suffix.json"
    done
fi

# --- 4. OpenVINO IR of the relation head --------------------------------------------
if [[ ! -s "$DIST/relateanything_fp16.xml" ]]; then
    say "Converting the relation model to OpenVINO"
    .venv/bin/python -m relatevision.convert
fi

# --- 5. new relation phrases (optional) ------------------------------------------------
if [[ -n "$NEW_PREDICATES" ]]; then
    build_env
    say "Adding relation phrases: $NEW_PREDICATES"
    "$BUILD/bin/python" -m relatevision.add_predicates "$NEW_PREDICATES"
fi

mkdir -p videos outputs
say "Checking speed on this machine"
.venv/bin/python relate_video.py --bench

cat <<EOF

Done. Put your videos in:  $REPO/videos
  ./run.sh                      play them live with relations
  ./run.sh --export             write annotated MP4s + JSON to outputs/
  ./run.sh --webcam             live camera
  ./run.sh --help               all options
EOF
