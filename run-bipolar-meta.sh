#!/bin/bash
#
# Metacentrum (PBS) version of run-bipolar.sh: submits one qsub job per
# experiment instead of backgrounding processes on fixed GPU indices.
# The scheduler runs as many concurrently as free GPUs allow and queues
# the rest, so there is no need for sequential EC/EO "waves" here.
#
# Run this ON A METACENTRUM FRONTEND, from the repo root, after the
# one-time setup (git clone + poetry install into $REPO_DIR/.venv,
# dataset rsync into $DATA_DIR — see docs/plans or ask Claude for the
# exact commands).
#
# Usage: ./run-bipolar-meta.sh          (required 8 runs: EC + EO)
#        ./run-bipolar-meta.sh --with-ec+eo   (also the optional 3-run ec+eo wave)

set -uo pipefail

JOB_SCRIPT="$(dirname "$0")/metacentrum/train_job.pbs"

# Headline hyperparameters (unchanged, untuned) from the CNN-AttnS in-ear headline config.
COMMON_ARGS="--model CNNAttnS --dataset all --class-mode 2 --n-folds 10 --batch-size 64 --epochs 100 --lr 1e-4 --dropout 0.05 --weight-decay 5e-5 --focal-loss --patience 20 --val-every 1"

submit() {
    local name="$1"
    shift
    echo "[submit] $name"
    qsub -N "$name" -v "EXP_NAME=$name,MAIN_ARGS=$COMMON_ARGS $*" "$JOB_SCRIPT"
}

echo "[submit] EC wave"
submit all_040_fp2-fp1_ec --channel Fp2-Fp1 --condition ec
submit all_040_c4-c3_ec   --channel C4-C3   --condition ec
submit all_040_t8-t7_ec   --channel T8-T7   --condition ec
submit all_040_inear_ec   --channel in-ear  --condition ec

echo "[submit] EO wave"
submit all_040_fp2-fp1_eo --channel Fp2-Fp1 --condition eo
submit all_040_c4-c3_eo   --channel C4-C3   --condition eo
submit all_040_t8-t7_eo   --channel T8-T7   --condition eo
submit all_040_inear_eo   --channel in-ear  --condition eo

# Optional: connects to the existing cnnattns-binary_wve run (the in-ear ec+eo row
# already exists and does not need rerunning) — only submit this wave if useful.
if [[ "${1:-}" == "--with-ec+eo" ]]; then
    echo "[submit] optional ec+eo wave"
    submit all_040_fp2-fp1_ec+eo --channel Fp2-Fp1 --condition ec+eo
    submit all_040_c4-c3_ec+eo   --channel C4-C3   --condition ec+eo
    submit all_040_t8-t7_ec+eo   --channel T8-T7   --condition ec+eo
fi

echo "[submit] all jobs queued - check with: qstat -u \$USER"
