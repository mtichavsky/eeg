# EXPERIMENTS

| model                      | accuracy                        | sensitivity (Recall) | specificity    | directory                                       |
|----------------------------|---------------------------------|----------------------|----------------|-------------------------------------------------|
| binary, in-ear, 3 dts.     | 76.86% ± 2.84% x 77.79% ± 2.74% | 86.23% ± 4.29%       | 63.90% ± 4.66% | all3-016-inear-binary-smaller                   |
| 4-class, in-ear, 3 dts.    | 63.62% ± 4.45%                  |                      |                | all3-016-inear-4class-smaller                   | 
| binary, 8-channel, 3 dts.  | 76.83% ± 2.12%                  | 88.72% ± 2.86%       | 58.25% ± 6.44% | all3-017-all-binary-smallerall-weighted-sampler |
| 4-class, 8-channel, 3 dts. | ---                             |                      |                |                                                 |


CUDA_VISIBLE_DEVICES=0 EEG_DATA_DIR=/home/xticha09 python main.py train   --channel in-ear   --batch-size 64   --dataset all   --model Smaller   --condition ec+eo   --n-folds 6   --dropout 0.1   --weight-decay 1e-4   --val-every 1   --checkpoint-dir=experiments/all3-017-inear-binary-smaller-weighted-sampler
CUDA_VISIBLE_DEVICES=2 EEG_DATA_DIR=/home/xticha09 python main.py train   --channel all   --batch-size 64   --dataset all   --model SmallerAll   --condition ec+eo   --n-folds 6   --dropout 0.1   --weight-decay 1e-4   --val-every 1   --checkpoint-dir=experiments/all3-017-all-binary-smallerall-weighted-sampler



### all3-014b-all-binary-smallerall

- Loss curve visualization doesn't work after switch to JSON logs
- Problem AX_MALIK missing healthy classes drives accuracy through the roof, also the specificity is still low
- Command: `systemd-run --user --scope -p CPUQuota=200% python main.py train   --channel all   --batch-size 64   --dataset all   --model SmallerAll   --condition ec+eo   --n-folds 6   --dropout 0.1   --weight-decay 1e-4   --val-every 1   --checkpoint-dir=experiments/both-014b-all-binary-smallerall`


### all3-014b-all-4class-smallerall

- The confusion matrix doesn't make sense, there's definitely more depression samples, why is that???
  - but MDD shows good performance, probably in anxiety+depression I'd be looking for a problem
- AX_MALIK 100% accuracy, CANE absolutely shitty
- Command `systemd-run --user --scope -p CPUQuota=200% python main.py train   --channel all   --batch-size 64   --dataset all   --model SmallerAll   --condition ec+eo   --n-folds 6   --dropout 0.1   --weight-decay 1e-4   --val-every 1   --checkpoint-dir=experiments/both-014b-all-4class-smallerall --class-mode=4`

### all3-014b-inear-binary-smaller

- Command: `systemd-run --user --scope -p CPUQuota=200% python main.py train   --channel in-ear   --batch-size 64   --dataset all   --model Smaller   --condition ec+eo   --n-folds 6   --dropout 0.1   --weight-decay 1e-4   --val-every 1   --checkpoint-dir=experiments/both-014-inear-binary-smaller`

---

```bash
python main.py train --skip-ica --channel all --batch-size 64 --dataset both --model SmallerAll --condition ec+eo --n-folds 6 --dropout 0.1 --weight-decay 1e-4 --val-every 1 --checkpoint-dir=experiments/both-testing01
```


```bash
systemd-run --user --scope -p CPUQuota=200% python main.py train \
  --channel all \
  --batch-size 64 \
  --dataset all \
  --model SmallerAll \
  --condition ec+eo \
  --n-folds 6 \
  --dropout 0.1 \
  --weight-decay 1e-4 \
  --val-every 1 \
  --checkpoint-dir=experiments/both-013-all-binary-smallerall
Running as unit: run-p10425-i10426.scope; invocation ID: 7a584f01f90e408e9ba659cffdae14c6
[2026-02-01 09:16:22,693 WARNING main.get_unique_checkpoint_dir] Checkpoint directory 'experiments/both-testing01' already exists. Using 'experiments/both-testing01_ckk' instead to avoid overwriting.



Running as unit: run-p14447-i14448.scope; invocation ID: 9fb19df3a11b428cbf28b135feee932e




Running as unit: run-p42262-i42263.scope; invocation ID: af1f662c605147f2854bf3bdaa2d25b1
```
systemctl --user set-property run-p10425-i10426.scope CPUQuota=300%
