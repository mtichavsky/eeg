#!/bin/bash -l
#
# Explainability study for Section VII, in the order the plan runs it.
#
#   ./run-explainability.sh posthoc    # E1-E4, E6: no GPU training, hours
#   ./run-explainability.sh retrain    # P0, E5, E8, E9: full CV runs, days
#   ./run-explainability.sh all
#
# The post-hoc stage needs only existing checkpoints, so run it first: it validates the whole
# harness before any GPU time is spent on retraining.

set -uo pipefail

cd eeg
ml Python/3.12.3-GCCcore-13.3.0
ml CUDA/12.6.0
source activate
export EEG_DATA_DIR=/home/xticha09/

# Runs whose checkpoints the post-hoc analyses read. Adjust if paths differ on this machine.
PAPER_RUNS=../thesis-text/paper/experiments
CHECKPOINT_DIR=../explain-retrain

# The eight-channel model the ablations target, and the in-ear model for the band comparison.
EIGHT_CHANNEL_RUN="$PAPER_RUNS/binary/8channel/alltransformer-binary"
IN_EAR_RUN="$PAPER_RUNS/binary/in-ear/cnnattns-binary_wve"

# Shared with the paper's AllTransformerV4 binary run, minus --channel and --checkpoint-dir.
ATV4_ARGS=(
    --model AllTransformerV4 --dataset all --condition ec+eo
    --batch-size 64 --n-folds 10 --val-every 1
    --lr 1e-4 --epochs 200 --patience 50 --dropout 0.1 --weight-decay 1e-4
    --class-mode 2 --focal-loss
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
        --checkpoint-dir="$out_dir" "$@" \
        >"$out_dir/stdout.log" 2>&1 &
}

# ── Post-hoc: no training, reads existing checkpoints ─────────────────────────
run_posthoc() {
    echo "[explain] eight-channel: channels, occlusion, bands, mirror, attribution, probes"
    python main.py explain "$EIGHT_CHANNEL_RUN" --experiment all

    # The in-ear model has a single channel, so only the band and attribution analyses apply.
    echo "[explain] in-ear: bands, attribution"
    python main.py explain "$IN_EAR_RUN" --experiment bands
    python main.py explain "$IN_EAR_RUN" --experiment attribution

    echo "[explain] results in <run>/explain/*.json"
}

# ── Retraining: P0 baseline, E5 reduced montages, E8 site audit, E9 control ───
run_retrain() {
    mkdir -p "$CHECKPOINT_DIR"

    # P0: the paper's headline config, this time keeping the checkpoints. Everything in E5 is
    # compared against this, so it must finish before the montage rows are interpreted.
    launch 1 atv4-binary-all8 "${ATV4_ARGS[@]}" --channel all

    # E5: reduced montages. T7+T8 is the direct test of the in-ear surrogate's premise.
    launch 2 atv4-binary-temporal "${ATV4_ARGS[@]}" --channel T7,T8
    launch 3 atv4-binary-frontal  "${ATV4_ARGS[@]}" --channel Fp1,Fp2
    wait

    # Fill in the top-k montage once the E1 ranking is known; edit the channel list, then run.
    launch 1 atv4-binary-top4 "${ATV4_ARGS[@]}" --channel Fp1,Fp2,T7,T8

    # E8: does anxiety separate from comorbid *within* a single site? CANE alone has healthy,
    # anxiety and comorbid subjects, so a result here cannot be a dataset fingerprint.
    launch 2 atv4-4class-cane-only \
        --model AllTransformerV4 --dataset cane --channel all --condition ec+eo \
        --batch-size 64 --n-folds 10 --val-every 1 \
        --lr 1e-4 --epochs 200 --patience 50 --dropout 0.1 --weight-decay 1e-4 \
        --class-mode 4 --focal-loss

    # E9: shuffled-label control. Accuracy must collapse to chance; feeds the E7 sanity checks.
    # NOTE: needs a --shuffle-labels flag, which does not exist yet.
    # launch 3 atv4-binary-shuffled "${ATV4_ARGS[@]}" --channel all --shuffle-labels
    wait
}

case "${1:-all}" in
    posthoc) run_posthoc ;;
    retrain) run_retrain ;;
    all)     run_posthoc && run_retrain ;;
    *)       echo "usage: $0 {posthoc|retrain|all}" >&2; exit 2 ;;
esac

echo "[explain] done"
