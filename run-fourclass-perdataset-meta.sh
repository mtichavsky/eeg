#!/bin/bash
#
# Metacentrum (PBS) launcher that reruns the two four-class runs behind tab:fourclass of the paper
# so they store the per-dataset confusion matrices (compute_per_dataset_metrics now records a
# K x K matrix per dataset, and results.txt gets a "Per-Dataset Chunk Confusion Matrices"
# section). With those, helpers-print/dataset_prior_baseline.py can decompose the 4-class accuracy
# per dataset against the dataset-prior baseline (always predict a dataset's majority class), i.e.
# check how much of the 4-class result is dataset identity (MDD supplies almost all Depression
# labels, CANE/IDUN all Comorbid ones). See docs/plans/2026-09-19-explainability-research-notes.md
# section 7 item 2 and 4.
#
# Each run mirrors the config dump at the top of the results.txt of an existing run, so the rerun
# differs from the paper's numbers only by training noise (see the caveats below):
#
#   all_043_all_ec+eo_4class    <- AllTransformerV4, 8-channel; paper row "AllTransformerV4"
#                                  (64.6 chunk / 64.8 subj, recall A/D/C/H 47/94/88/46)
#                                  = /home/milan/eeg/experiments/4class/8channel/alltransformer-4class
#                                  (checkpoint_dir ../jul30/alltransformer-4class, 64.57 chunk,
#                                  64.76 subj, recall 47.11/93.52/87.66/45.99)
#   all_043_inear_ec+eo_4class  <- CNN-AttnS, in-ear; paper row "CNN-AttnS"
#                                  (67.3 chunk / 67.3 subj, recall 64/90/84/52)
#                                  = /home/milan/eeg/experiments/4class/in-ear/cnnattns-4class-inear
#                                  (checkpoint_dir ../jul30/cnnattns-4class-inear, 67.30 chunk,
#                                  67.34 subj, recall 64.37/90.23/83.64/51.60)
#
# Both original runs: git_commit 454c49e-dirty (the spectrogram-frequency-axis commit); current
# main only adds flags whose defaults leave training unchanged (--optimizer adam, --freq-cutoff 70).
#
# Caveats for comparing with the paper:
# * Training is unseeded (RANDOM_SEED = 42 seeds only the fold assignment; there is no
#   torch.manual_seed), so expect run-to-run noise of a point or two in chunk accuracy, more in
#   single folds and in the per-class recalls. Folds are identical, so per-fold comparisons with
#   the original results.txt are meaningful.
# * 10-fold CV is mandatory; never reduce --n-folds.
# * The GPU model/driver may differ from the original runs, a further (small) source of noise.
#
# Run this ON A METACENTRUM FRONTEND, from the repo root of the clone the jobs execute from
# (REPO_DIR in metacentrum/train_job.pbs), checked out on a branch that has this change
# (`git checkout <branch> && git pull --ff-only` first; do not switch branches while jobs are
# queued or running - jobs read the code when they start).
#
# Usage: ./run-fourclass-perdataset-meta.sh        (2 runs, 10-fold ec+eo, --class-mode 4)
#
# Fetching: the job template creates experiments/<name>/ (for stdout.log) before main.py starts,
# so main.py writes checkpoints/results.txt to a random-suffix sibling (<name>_abc); fetch both:
#   rsync -a metacentrum:/storage/brno2/home/tichavskym/experiments/<name><suffix>/ experiments/<name>/
#   rsync -a metacentrum:/storage/brno2/home/tichavskym/experiments/<name>/stdout.log experiments/<name>/
# Afterwards:
#   poetry run python helpers-print/dataset_prior_baseline.py --confusion-matrices \
#       --root experiments 'all_043_*'

set -uo pipefail

JOB_SCRIPT="$(dirname "$0")/metacentrum/train_job.pbs"

# Shared by both four-class runs (values from the two results.txt config dumps): 10-fold CV,
# ec+eo, batch 64, 100 epochs, patience 20, focal loss gamma 2, validate every epoch, Adam with
# cosine LR (the defaults), 4-class labels on the combined MDD + CANE/IDUN + SAD data.
SHARED_ARGS="--dataset all --class-mode 4 --condition ec+eo --n-folds 10 --batch-size 64 --epochs 100 --patience 20 --focal-loss --focal-gamma 2.0 --val-every 1"
ALL_ARGS="$SHARED_ARGS --model AllTransformerV4 --channel all --lr 5e-4 --dropout 0.1 --weight-decay 1e-4"
INEAR_ARGS="$SHARED_ARGS --model CNNAttnS --channel in-ear --lr 1e-4 --dropout 0.05 --weight-decay 5e-5"

submit() {
    local name="$1"
    shift
    echo "[submit] $name"
    qsub -N "$name" -v "EXP_NAME=$name,MAIN_ARGS=$*" "$JOB_SCRIPT"
}

echo "[submit] four-class AllTransformerV4 (8-channel) and CNN-AttnS (in-ear)"
submit all_043_all_ec+eo_4class "$ALL_ARGS"
submit all_043_inear_ec+eo_4class "$INEAR_ARGS"

# ---------------------------------------------------------------------------------------------
# OPTIONAL, DISABLED BY DEFAULT: binary AllTransformerV4 headline run (Table II, 76.5 chunk /
# 78.6 subj) with per-dataset matrices. Mirrors /home/milan/eeg/experiments/results.txt
# (checkpoint_dir ../aug03/atv4-lowlr_oex: lr 1e-4, epochs 200, patience 50, dropout 0.1,
# wd 1e-4) and equals ALL_ARGS of run-freq-cutoff-meta.sh (all_041_all_ec+eo_f70 reran it on
# current code: 75.4 chunk). Uncomment to enable.
#
# BINARY_ALL_ARGS="--dataset all --class-mode 2 --condition ec+eo --n-folds 10 --batch-size 64 --lr 1e-4 --focal-loss --focal-gamma 2.0 --val-every 1 --model AllTransformerV4 --channel all --epochs 200 --dropout 0.1 --weight-decay 1e-4 --patience 50"
# submit all_043_all_ec+eo_binary "$BINARY_ALL_ARGS"
# ---------------------------------------------------------------------------------------------

echo "[submit] all jobs queued - check with: qstat -u \$USER"
