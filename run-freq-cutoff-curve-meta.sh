#!/bin/bash
#
# Metacentrum (PBS) launcher extending Experiment B (frequency-cutoff ablation) from two points
# (70 Hz, 30 Hz) to a curve: CNN-AttnS in-ear at 50 Hz and 40 Hz, same code, hyperparameters and
# folds as all_041_inear_ec+eo_f70 / _f30 (run-freq-cutoff-meta.sh). One qsub per run through
# metacentrum/train_job.pbs. The 70/30 Hz points come from the all_041 runs, so nothing is
# re-run for them. See docs/plans/2026-09-19-explainability-research-notes.md section 7 item 3.
#
# The CNN cannot be built below about 21 Hz, so the curve stops at 30 Hz on the low side.
# 10-fold CV always; never reduce folds.
#
# Run this ON A METACENTRUM FRONTEND, from the repo root of the clone the jobs execute from
# (REPO_DIR in metacentrum/train_job.pbs).
#
# Usage: ./run-freq-cutoff-curve-meta.sh        (2 runs, 10-fold ec+eo binary)
#
# Afterwards: poetry run python helpers-print/freq_cutoff_curve_summary.py --root <experiments dir>

set -uo pipefail

JOB_SCRIPT="$(dirname "$0")/metacentrum/train_job.pbs"

# Identical to INEAR_ARGS in run-freq-cutoff-meta.sh (Table II hyperparameters, untuned).
INEAR_ARGS="--dataset all --class-mode 2 --condition ec+eo --n-folds 10 --batch-size 64 --lr 1e-4 --focal-loss --val-every 1 --model CNNAttnS --channel in-ear --epochs 100 --dropout 0.05 --weight-decay 5e-5 --patience 20"

submit() {
    local name="$1"
    shift
    echo "[submit] $name"
    qsub -N "$name" -v "EXP_NAME=$name,MAIN_ARGS=$*" "$JOB_SCRIPT"
}

submit all_042_inear_ec+eo_f50 "$INEAR_ARGS" --freq-cutoff 50
submit all_042_inear_ec+eo_f40 "$INEAR_ARGS" --freq-cutoff 40

echo "[submit] all jobs queued - check with: qstat -u \$USER"
