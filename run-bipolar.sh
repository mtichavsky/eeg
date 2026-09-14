#!/bin/bash -l
#
# Bipolar surrogate ablation: Fp2-Fp1 vs C4-C3 vs T8-T7 vs in-ear (real IDUN reference).
# See docs/plans/2026-09-14-bipolar-surrogate-ablation.md for the full design.
#
# Launches in waves of one run per GPU: EC wave, then EO wave, then (optionally) the
# ec+eo wave for the three headset montages. Waves run sequentially so 4 concurrent
# 10-fold CV jobs (one per montage) don't compete for the same GPUs.
#
# Usage: ./run-bipolar.sh          (required 8 runs: EC + EO waves)
#        ./run-bipolar.sh --with-ec+eo   (also launches the optional 3-run ec+eo wave)

set -uo pipefail

cd eeg
ml Python/3.12.3-GCCcore-13.3.0
ml CUDA/12.6.0
source activate
CHECKPOINT_DIR=../experiments
export EEG_DATA_DIR=/home/xticha09/

mkdir -p "$CHECKPOINT_DIR"

# Headline hyperparameters (unchanged, untuned) from the CNN-AttnS in-ear headline config.
COMMON_ARGS=(
    --model CNNAttnS --dataset all --class-mode 2 --n-folds 10
    --batch-size 64 --epochs 100 --lr 1e-4 --dropout 0.05 --weight-decay 5e-5
    --focal-loss --patience 20 --val-every 1
)

# Launch one training run in the background on a dedicated GPU.
# $1 = GPU index, $2 = experiment name (also the checkpoint subdir), rest = main.py args
launch() {
    local gpu="$1" name="$2"
    shift 2
    local out_dir="$CHECKPOINT_DIR/$name"
    mkdir -p "$out_dir"
    echo "[launch] GPU $gpu -> $out_dir"
    CUDA_VISIBLE_DEVICES="$gpu" nohup python main.py train \
        --checkpoint-dir="$out_dir" "${COMMON_ARGS[@]}" "$@" \
        >"$out_dir/stdout.log" 2>&1 &
}

wave_ec() {
    launch 0 all_040_fp2-fp1_ec --channel Fp2-Fp1 --condition ec
    launch 1 all_040_c4-c3_ec --channel C4-C3 --condition ec
    launch 2 all_040_t8-t7_ec --channel T8-T7 --condition ec
    launch 3 all_040_inear_ec --channel in-ear --condition ec
}

wave_eo() {
    launch 0 all_040_fp2-fp1_eo --channel Fp2-Fp1 --condition eo
    launch 1 all_040_c4-c3_eo --channel C4-C3 --condition eo
    launch 2 all_040_t8-t7_eo --channel T8-T7 --condition eo
    launch 3 all_040_inear_eo --channel in-ear --condition eo
}

# Optional: connects to the existing cnnattns-binary_wve run (the in-ear ec+eo row already
# exists and does not need rerunning) — only run this wave if GPU time allows.
wave_ec_eo_optional() {
    launch 0 all_040_fp2-fp1_ec+eo --channel Fp2-Fp1 --condition ec+eo
    launch 1 all_040_c4-c3_ec+eo --channel C4-C3 --condition ec+eo
    launch 2 all_040_t8-t7_ec+eo --channel T8-T7 --condition ec+eo
}

echo "[launch] EC wave"
wave_ec
wait
echo "[launch] EC wave finished"

echo "[launch] EO wave"
wave_eo
wait
echo "[launch] EO wave finished"

if [[ "${1:-}" == "--with-ec+eo" ]]; then
    echo "[launch] optional ec+eo wave"
    wave_ec_eo_optional
    wait
    echo "[launch] optional ec+eo wave finished"
fi

echo "[launch] all runs finished"
