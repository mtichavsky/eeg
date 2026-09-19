#!/bin/bash
#
# Metacentrum (PBS) launcher for the single-dataset CV runs (objection 1, see
# docs/plans/2026-09-19-explainability-research-notes.md section 7 item 2).
# Modelled on run-bipolar-meta.sh: one qsub job per experiment via
# metacentrum/train_job.pbs; the scheduler runs as many concurrently as free GPUs
# allow and queues the rest, so there is no need for sequential EC/EO "waves" here.
#
# Question: does CANE (or SAD) signal exist when the model is trained on that dataset
# alone? Same model and hyperparameters as Exp A (all_040_*), but ONE dataset per run.
# MDD is the positive control (should stay well above its base rate). The channel is
# the headset T8-T7 bipolar montage, NOT 'in-ear' (which would swap CANE for IDUN).
# 10-fold CV always; never reduce folds.
#
# Run this ON A METACENTRUM FRONTEND, from the repo root, after the one-time
# setup (git clone + poetry install into $REPO_DIR/.venv, dataset rsync into
# $DATA_DIR - same setup as run-bipolar-meta.sh).
#
# Usage: ./run-single-dataset-meta.sh        (6 runs: {cane,sad,mdd} x {ec,eo})
#
# Afterwards: poetry run python helpers-print/single_dataset_summary.py --root <experiments dir>

set -uo pipefail

JOB_SCRIPT="$(dirname "$0")/metacentrum/train_job.pbs"

# Hyperparameters identical to Exp A (run-bipolar-meta.sh); --dataset is set per run below.
COMMON_ARGS="--model CNNAttnS --class-mode 2 --n-folds 10 --batch-size 64 --epochs 100 --lr 1e-4 --dropout 0.05 --weight-decay 5e-5 --focal-loss --patience 20 --val-every 1"

submit() {
    local name="$1"
    shift
    echo "[submit] $name"
    qsub -N "$name" -v "EXP_NAME=$name,MAIN_ARGS=$COMMON_ARGS $*" "$JOB_SCRIPT"
}

echo "[submit] EC runs"
submit cane_041_t8-t7_ec --dataset cane --channel T8-T7 --condition ec
submit sad_041_t8-t7_ec  --dataset sad  --channel T8-T7 --condition ec
submit mdd_041_t8-t7_ec  --dataset mdd  --channel T8-T7 --condition ec

echo "[submit] EO runs"
submit cane_041_t8-t7_eo --dataset cane --channel T8-T7 --condition eo
submit sad_041_t8-t7_eo  --dataset sad  --channel T8-T7 --condition eo
submit mdd_041_t8-t7_eo  --dataset mdd  --channel T8-T7 --condition eo

echo "[submit] all jobs queued - check with: qstat -u \$USER"
