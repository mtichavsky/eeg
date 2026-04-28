# EXPERIMENTS

CUDA_VISIBLE_DEVICES=0 EEG_DATA_DIR=/home/xticha09 python main.py train  --model DeformerS --dataset all --channel in-ear  --condition ec+eo  --checkpoint-dir experiments/all-1.0-inear-binary-deformers-low-dropout --focal-loss --dropout 0.1 --weight-decay 1e-4
[1]  1647056 Running                 CUDA_VISIBLE_DEVICES=1 EEG_DATA_DIR=/home/xticha09 python main.py train --model DeformerS --dataset all --channel in-ear --condition ec+eo --checkpoint-dir experiments/all-1.0-inear-binary-deformers-sign-og --focal-loss --dropout 0.5 --weight-decay 1e-5 > /dev/null 2>&1 &
[3]- 1647387 Running                 CUDA_VISIBLE_DEVICES=3 EEG_DATA_DIR=/home/xticha09 python main.py train --model DeformerS --dataset all --channel in-ear --condition ec+eo --checkpoint-dir experiments/all-1.0-inear-binary-deformers-sign-10drop --focal-loss --dropout 0.1 --weight-decay 1e-5 > /dev/null 2>&1 &
[4]+ 1657000 Running                 CUDA_VISIBLE_DEVICES=1 EEG_DATA_DIR=/home/xticha09 python main.py train --channel all --batch-size 64 --dataset all --model DeformerS --condition ec+eo --n-folds 10 --lr 1e-3 --dropout 0.5 --weight-decay 1e-5 --val-every 1 --focal-loss --class-mode 4 --checkpoint-dir=experiments/all3-1.0-all-4class-deformers > /dev/null 2>&1 &
[5]+ 1659162 Running                 CUDA_VISIBLE_DEVICES=3 EEG_DATA_DIR=/home/xticha09 python main.py train --channel all --batch-size 64 --dataset all --model DeformerS --condition ec+eo --n-folds 10 --lr 1e-3 --dropout 0.2 --weight-decay 1e-4 --val-every 1 --focal-loss --checkpoint-dir=experiments/all3-1.0-all-deformers-weight-drop > /dev/null 2>&1 &


