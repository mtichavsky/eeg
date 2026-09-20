#!/bin/bash
#
# AllTransformerV4 Round 2 (branch exp/atv4-round2): binary, 8-channel, ec+eo, 10-fold.
# Goal: 80% chunk accuracy (10-fold aggregate as reported in the paper) with balanced
# sensitivity/specificity. Round 1 (all_031*/032*) found no config above the 76.46% aug03 anchor.
#
# Findings driving this round:
#   * cosine T_max=200 but early stopping ends runs at epoch 51-170, so the LR never anneals
#     -> here --epochs is shortened so the cosine schedule actually completes.
#   * adam + heavy regularization collapses to a one-class predictor; adamw is stable.
#
# Runs from a separate git worktree of this branch so the shared clone's checkout (and any
# jobs running from it) is never switched. Usage: ./run-atv4-round2-meta.sh <batch>
#   batch 1: schedule-length / regularization / lr / loss / selection-metric probes (6 runs)
#   batch 2: higher lr + lighter regularization at the full 200-epoch schedule (5 runs)

set -uo pipefail

JOB_SCRIPT="$(dirname "$0")/metacentrum/train_job.pbs"
CODE_DIR="/storage/brno2/home/${USER}/eeg-atv4-r2"

COMMON_ARGS="--model AllTransformerV4 --dataset all --class-mode 2 --channel all --condition ec+eo --n-folds 10 --batch-size 64 --val-every 1 --focal-loss"

submit() {
    local name="$1"
    shift
    echo "[submit] $name :: $*"
    qsub -N "$name" -v "CODE_DIR=$CODE_DIR,EXP_NAME=binary/8channel/$name,MAIN_ARGS=$COMMON_ARGS $*" "$JOB_SCRIPT"
}

batch1() {
    # a: anchor hyperparameters, only the schedule shortened -> isolates the annealing effect
    submit all_050a_all_ec+eo --optimizer adam  --dropout 0.1 --weight-decay 1e-4 --lr 1e-4 --focal-gamma 2.0 --epochs 60 --patience 60
    # b: round-1 winner (032a) with the shortened schedule
    submit all_050b_all_ec+eo --optimizer adamw --dropout 0.3 --weight-decay 1e-2 --lr 1e-4 --focal-gamma 2.0 --epochs 60 --patience 60
    # c: as b, lighter dropout
    submit all_050c_all_ec+eo --optimizer adamw --dropout 0.1 --weight-decay 1e-2 --lr 1e-4 --focal-gamma 2.0 --epochs 60 --patience 60
    # d: as b, 3x learning rate
    submit all_050d_all_ec+eo --optimizer adamw --dropout 0.3 --weight-decay 1e-2 --lr 3e-4 --focal-gamma 2.0 --epochs 60 --patience 60
    # e: as b, focal gamma 0 (class-weighted cross-entropy)
    submit all_050e_all_ec+eo --optimizer adamw --dropout 0.3 --weight-decay 1e-2 --lr 1e-4 --focal-gamma 0 --epochs 60 --patience 60
    # f: as b, checkpoint chosen by balanced accuracy
    submit all_050f_all_ec+eo --optimizer adamw --dropout 0.3 --weight-decay 1e-2 --lr 1e-4 --focal-gamma 2.0 --epochs 60 --patience 60 --select-metric balanced
}

batch2() {
    # Batch 1: shorter schedule hurt (a<baseline, b<032a); lr 3e-4 (d) and lighter dropout (c)
    # helped; balanced selection (f) hurt. Now: those levers at the full 200-epoch schedule.
    local FULL="--epochs 200 --patience 50 --focal-gamma 2.0"
    submit all_051a_all_ec+eo --optimizer adamw --dropout 0.3 --weight-decay 1e-2 --lr 3e-4 $FULL
    submit all_051b_all_ec+eo --optimizer adamw --dropout 0.1 --weight-decay 1e-2 --lr 3e-4 $FULL
    submit all_051c_all_ec+eo --optimizer adamw --dropout 0.3 --weight-decay 1e-2 --lr 1e-3 $FULL
    submit all_051d_all_ec+eo --optimizer adamw --dropout 0.1 --weight-decay 1e-3 --lr 3e-4 $FULL
    submit all_051e_all_ec+eo --optimizer adamw --dropout 0.3 --weight-decay 1e-2 --lr 3e-4 --epochs 200 --patience 50 --focal-gamma 0
}

case "${1:-}" in
    2) batch2 ;;
    1) batch1 ;;
    *) echo "usage: $0 <batch: 1|2>"; exit 1 ;;
esac
echo "[submit] queued - check with: qstat -u \$USER"
