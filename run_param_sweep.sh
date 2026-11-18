#!/usr/bin/env bash
# Sweep of z-voxel count (D) and label blob (sigma, radius): simulate once, label per config, train, evaluate.
# Resumable: a stage whose output already exists is skipped, so re-running continues where it stopped.
#
#   bash run_param_sweep.sh                 # everything
#   CONFIGS="121:0.65:2 161:0.65:2" bash run_param_sweep.sh   # only some configs
#
# Each config is D:blob_sigma:blob_r. Results: $ROOT/summary.csv, summary.png, rmse_z_by_depth.png
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python}
BASE_PARAMS=${BASE_PARAMS:-/home/omer/training_results/AutoDS3D/161126_mito/sample1/results/training_data/gui/param.pickle}
ROOT=${ROOT:-/home/omer/training_results/AutoDS3D/161126_mito/sample1/results/param_sweep}
DEVICE=${DEVICE:-cuda:0}
N_TRAIN=${N_TRAIN:-10000}
N_TEST=${N_TEST:-1000}
EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-16}
LR=${LR:-0.001}
NUM_WORKERS=${NUM_WORKERS:-8}

# D sweep at the current blob (0.65 voxel, r 2), then blob sweep at D=121.
# sigma >= 1 needs r 3, otherwise the blob is cut off at ~2 sigma.
CONFIGS=${CONFIGS:-"61:0.65:2 91:0.65:2 121:0.65:2 161:0.65:2 201:0.65:2 121:0.5:2 121:0.8:2 121:1.0:3 121:1.3:3"}

mkdir -p "$ROOT/logs" "$ROOT/eval"

simulate() {  # name n seed
    if [[ -f "$ROOT/images/$1/gt.pickle" ]]; then echo "[skip] images/$1"; return; fi
    echo "[images] $1 ($2)"
    "$PYTHON" param_sweep.py images --base_params "$BASE_PARAMS" --out "$ROOT/images/$1" \
        --n "$2" --seed "$3" --device "$DEVICE" 2>&1 | tee "$ROOT/logs/images_$1.log"
}

simulate train "$N_TRAIN" 1
simulate test "$N_TEST" 2

for cfg in $CONFIGS; do
    IFS=: read -r D SIGMA R <<< "$cfg"
    NAME="D${D}_s${SIGMA}_r${R}"
    CFG_DIR="$ROOT/configs/$NAME"
    RUN_DIR="$ROOT/runs/$NAME"
    echo "=================== $NAME ==================="

    if [[ ! -f "$CFG_DIR/y.pickle" ]]; then
        "$PYTHON" param_sweep.py labels --images "$ROOT/images/train" --out "$CFG_DIR" \
            --D "$D" --blob_sigma "$SIGMA" --blob_r "$R"
    fi

    if ! ls "$RUN_DIR"/fit_*.pickle >/dev/null 2>&1; then
        rm -rf "$RUN_DIR"  # an unfinished run is started over
        "$PYTHON" training.py --x_folder_path "$ROOT/images/train/x" --data_dir "$CFG_DIR" --save_path "$RUN_DIR" \
            --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --lr "$LR" --device "$DEVICE" \
            --num_workers "$NUM_WORKERS" 2>&1 | tee "$ROOT/logs/train_$NAME.log"
    fi

    if [[ ! -f "$ROOT/eval/$NAME.json" ]]; then
        "$PYTHON" param_sweep.py evaluate --config "$CFG_DIR" --run "$RUN_DIR" --test "$ROOT/images/test" \
            --out "$ROOT/eval/$NAME.json" --device "$DEVICE" 2>&1 | tee "$ROOT/logs/eval_$NAME.log"
    fi
done

"$PYTHON" param_sweep.py summarize --root "$ROOT"
