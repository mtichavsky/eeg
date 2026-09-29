#!/bin/bash -l
#
# AllTransformerV4 Round 1: binary, 8-channel, ec+eo regularization sweep.
# Goal: 80% chunk accuracy with balanced sensitivity/specificity (not a 95/50 split).
# See experiments/_runs/STATE.md for the full history: this exact 4-config sweep was
# dispatched manually across 2 CPU-only laptops and never completed a single fold
# (dataloader OOM kills, laptop suspend, oversubscribed threads) — this script is the
# same sweep, unchanged, for whenever GPU access is available. Run it once, then send
# Claude the resulting cv_results.txt files (or results.txt) to design Round 2.
#
# Anchor to beat: aug03 10-fold run = 76.46% chunk, 81.6% sens, 68.4% spec
#   (adam, dropout=0.1, wd=1e-4, lr=1e-4, schedule below).
#
# Two independent regularization tracks, each at two intensities:
#   A (adam)  - kept for continuity with the adam aug03 anchor (coupled L2)
#   B (adamw) - decoupled weight decay; wd values are NOT comparable across tracks
#
# Usage: ./run-round1-atv4.sh

set -uo pipefail

cd eeg
ml Python/3.12.3-GCCcore-13.3.0
ml CUDA/12.6.0
source activate
CHECKPOINT_DIR=../experiments/binary/8channel
export EEG_DATA_DIR=/home/xticha09/

mkdir -p "$CHECKPOINT_DIR"

# Schedule identical to the aug03 anchor run, so results stay comparable.
COMMON_ARGS=(
    --model AllTransformerV4 --dataset all --class-mode 2 --channel all
    --condition ec+eo --n-folds 10 --epochs 200 --patience 50 --batch-size 64
    --val-every 1 --focal-loss --focal-gamma 2.0 --lr 1e-4
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

wave_round1() {
    launch 0 all_031a_all_ec+eo --optimizer adam  --dropout 0.3 --weight-decay 1e-3  # A1
    launch 1 all_031b_all_ec+eo --optimizer adam  --dropout 0.5 --weight-decay 3e-3  # A2
    launch 2 all_032a_all_ec+eo --optimizer adamw --dropout 0.3 --weight-decay 1e-2  # B1
    launch 3 all_032b_all_ec+eo --optimizer adamw --dropout 0.5 --weight-decay 5e-2  # B2
}

echo "[launch] Round 1: adam vs adamw x moderate vs heavy regularization"
wave_round1
wait
echo "[launch] Round 1 finished"
