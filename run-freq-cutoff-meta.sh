#!/bin/bash
#
# Metacentrum (PBS) launcher for Experiment B (frequency-cutoff ablation): submits one
# qsub job per run through metacentrum/train_job.pbs, like run-bipolar-meta.sh.
#
# Retrains the two headline models with the spectrogram cropped at 30 Hz (shape (31, 41))
# and, on the same code, at the default 70 Hz (shape (72, 41)), so the comparison is paired
# (identical folds, identical code) instead of confounded with run-to-run noise. See
# docs/plans/2026-09-14-frequency-cutoff-ablation.md.
#
# Run this ON A METACENTRUM FRONTEND, from the repo root of the clone the jobs execute from
# (REPO_DIR in metacentrum/train_job.pbs), checked out on the branch that has --freq-cutoff.
#
# Usage: ./run-freq-cutoff-meta.sh          (exactly 4 runs, all 10-fold ec+eo binary)

set -uo pipefail

JOB_SCRIPT="$(dirname "$0")/metacentrum/train_job.pbs"

# Table II hyperparameters, unchanged and untuned. 10-fold CV is mandatory.
SHARED_ARGS="--dataset all --class-mode 2 --condition ec+eo --n-folds 10 --batch-size 64 --lr 1e-4 --focal-loss --val-every 1"
INEAR_ARGS="$SHARED_ARGS --model CNNAttnS --channel in-ear --epochs 100 --dropout 0.05 --weight-decay 5e-5 --patience 20"
ALL_ARGS="$SHARED_ARGS --model AllTransformerV4 --channel all --epochs 200 --dropout 0.1 --weight-decay 1e-4 --focal-gamma 2.0 --patience 50"

submit() {
    local name="$1"
    shift
    echo "[submit] $name"
    qsub -N "$name" -v "EXP_NAME=$name,MAIN_ARGS=$*" "$JOB_SCRIPT"
}

# --freq-cutoff 70 is passed explicitly (it equals the default) so both cutoffs go through
# the identical code path and log the flag.
echo "[submit] CNNAttnS in-ear"
submit all_041_inear_ec+eo_f70 "$INEAR_ARGS" --freq-cutoff 70
submit all_041_inear_ec+eo_f30 "$INEAR_ARGS" --freq-cutoff 30

echo "[submit] AllTransformerV4 8-channel"
submit all_041_all_ec+eo_f70 "$ALL_ARGS" --freq-cutoff 70
submit all_041_all_ec+eo_f30 "$ALL_ARGS" --freq-cutoff 30

echo "[submit] all jobs queued - check with: qstat -u \$USER"
