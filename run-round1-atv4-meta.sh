#!/bin/bash
#
# Metacentrum (PBS) version of run-round1-atv4.sh: submits one qsub job per
# config instead of backgrounding processes on fixed GPU indices.
#
# AllTransformerV4 Round 1: binary, 8-channel, ec+eo regularization sweep.
# Goal: 80% chunk accuracy with balanced sensitivity/specificity (not a 95/50 split).
# Anchor to beat: aug03 10-fold run = 76.46% chunk, 81.6% sens, 68.4% spec
#   (adam, dropout=0.1, wd=1e-4, lr=1e-4, schedule below).
#
# Run this ON A METACENTRUM FRONTEND, from the repo root, after the
# one-time setup (git clone + poetry install into $REPO_DIR/.venv,
# dataset rsync into $DATA_DIR).
#
# Usage: ./run-round1-atv4-meta.sh

set -uo pipefail

JOB_SCRIPT="$(dirname "$0")/metacentrum/train_job.pbs"

# Schedule identical to the aug03 anchor run, so results stay comparable.
COMMON_ARGS="--model AllTransformerV4 --dataset all --class-mode 2 --channel all --condition ec+eo --n-folds 10 --epochs 200 --patience 50 --batch-size 64 --val-every 1 --focal-loss --focal-gamma 2.0 --lr 1e-4"

submit() {
    local name="$1"
    shift
    echo "[submit] $name"
    qsub -N "$name" -v "EXP_NAME=binary/8channel/$name,MAIN_ARGS=$COMMON_ARGS $*" "$JOB_SCRIPT"
}

echo "[submit] Round 1: adam vs adamw x moderate vs heavy regularization"
submit all_031a_all_ec+eo --optimizer adam  --dropout 0.3 --weight-decay 1e-3  # A1
submit all_031b_all_ec+eo --optimizer adam  --dropout 0.5 --weight-decay 3e-3  # A2
submit all_032a_all_ec+eo --optimizer adamw --dropout 0.3 --weight-decay 1e-2  # B1
submit all_032b_all_ec+eo --optimizer adamw --dropout 0.5 --weight-decay 5e-2  # B2

echo "[submit] Round 1 queued - check with: qstat -u \$USER"
