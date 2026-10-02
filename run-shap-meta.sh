#!/bin/bash
#
# Metacentrum (PBS) launcher for the grouped-SHAP study (first phase, no optional runs):
# docs/plans/2026-09-26-shap-explainability.md, Phase 5.
#
# Explains the f70 reruns of the two headline models, whose checkpoints fit the current code:
#   CNN-AttnS in-ear      -> bands (sample + mean baseline + random-init control)
#   AllTransformerV4 8-ch -> bands, channels, grid (sample); bands + channels (mean);
#                            bands (random-init control)
#
# The mean-baseline and random-init jobs compare themselves with the sample-baseline /
# trained results, so they are queued with `depend=afterok` on the job that writes those.
#
# Run this ON A METACENTRUM FRONTEND from the root of a checkout that has `main.py explain`; the
# jobs run the code of that checkout (CODE_DIR), with the shared venv (a plain `poetry install`
# includes `shap`). Do not switch its branch while jobs are queued or running.
#
# qsub -v splits on commas, so explain arguments passed through MAIN_ARGS must not contain any
# (no `--folds 1,3`).
#
# Usage: ./run-shap-meta.sh              (8 jobs)
#        ./run-shap-meta.sh --smoke      (2 jobs: fold 1, 8 chunks; in-ear bands, 8-ch all games)

set -uo pipefail

CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
JOB_SCRIPT="$CODE_DIR/metacentrum/explain_job.pbs"
EXPERIMENTS="/storage/brno2/home/$USER/experiments"

# TODO: train_job.pbs pre-creates experiments/<name>/, which pushes main.py to a random-suffix
# sibling; fixing that there would make these suffixes unnecessary.
# main.py writes to a random-suffix sibling of the directory the job template creates; these
# are the directories that actually hold results.txt and fold_N_best.pth.
INEAR_RUN="$EXPERIMENTS/all_041_inear_ec+eo_f70_ubd"
ALL_RUN="$EXPERIMENTS/all_041_all_ec+eo_f70_xwn"

COMMON="--max-chunks-per-fold 400 --seed 42"

# $1 = job name, $2 = run dir, $3 = walltime, $4 = extra qsub args ("" for none), rest = explain args.
# Prints the job id; aborts the whole launch if qsub fails, so no job depends on a missing id.
submit() {
    local name="$1" run="$2" walltime="$3" extra="$4" id
    shift 4
    # shellcheck disable=SC2086
    id=$(qsub -N "$name" -l "walltime=$walltime" $extra \
        -v "CODE_DIR=$CODE_DIR,RUN_DIR=$run,LABEL=$name,MAIN_ARGS=$* $COMMON" "$JOB_SCRIPT") || true
    if [[ -z "$id" ]]; then
        echo "qsub failed for $name; aborting (earlier jobs stay queued)" >&2
        exit 1
    fi
    echo "$id"
}

after() {
    echo "-W depend=afterok:$1"
}

if [[ "${1:-}" == "--smoke" ]]; then
    # --folds makes this a *_partial run, so it never overwrites or feeds the full results.
    qsub -N shap_smoke -v "CODE_DIR=$CODE_DIR,RUN_DIR=$INEAR_RUN,LABEL=shap_smoke,MAIN_ARGS=--experiment shap-bands --folds 1 --max-chunks-per-fold 8" "$JOB_SCRIPT"
    # AllTransformerV4 has not run on real data yet: all three games, K=8 to keep the grid cheap.
    qsub -N shap_smoke_all -v "CODE_DIR=$CODE_DIR,RUN_DIR=$ALL_RUN,LABEL=shap_smoke_all,MAIN_ARGS=--experiment shap-all --folds 1 --max-chunks-per-fold 8 --background-size 8" "$JOB_SCRIPT"
    exit 0
fi

echo "[submit] CNN-AttnS in-ear"
inear_bands=$(submit shap_inear_bands "$INEAR_RUN" 02:00:00 "" --experiment shap-bands --background-size 32) || exit 1
echo "  $inear_bands"
submit shap_inear_bands_mean "$INEAR_RUN" 01:00:00 "$(after "$inear_bands")" \
    --experiment shap-bands --baseline mean
submit shap_inear_bands_random "$INEAR_RUN" 02:00:00 "$(after "$inear_bands")" \
    --experiment shap-bands --background-size 32 --random-init

echo "[submit] AllTransformerV4 8-channel"
all_bands=$(submit shap_all_bands "$ALL_RUN" 03:00:00 "" --experiment shap-bands --background-size 32) || exit 1
all_channels=$(submit shap_all_channels "$ALL_RUN" 05:00:00 "" --experiment shap-channels --background-size 32) || exit 1
echo "  $all_bands $all_channels"
# Grid: K=8 keeps the ~50k coalitions/chunk affordable; 3 seeds on 50 chunks/fold give the SE.
submit shap_all_grid "$ALL_RUN" 12:00:00 "" \
    --experiment shap-grid --background-size 8 --grid-permutations 64 \
    --grid-se-seeds 3 --grid-se-chunks 50
submit shap_all_bands_mean "$ALL_RUN" 01:00:00 "$(after "$all_bands")" \
    --experiment shap-bands --baseline mean
submit shap_all_channels_mean "$ALL_RUN" 01:00:00 "$(after "$all_channels")" \
    --experiment shap-channels --baseline mean
submit shap_all_bands_random "$ALL_RUN" 03:00:00 "$(after "$all_bands")" \
    --experiment shap-bands --background-size 32 --random-init

echo "[submit] all jobs queued - check with: qstat -u \$USER"
echo "Fetch: rsync -av metacentrum:$INEAR_RUN/explain/ experiments/all_041_inear_ec+eo_f70/explain/"
echo "       rsync -av metacentrum:$ALL_RUN/explain/ experiments/all_041_all_ec+eo_f70/explain/"
