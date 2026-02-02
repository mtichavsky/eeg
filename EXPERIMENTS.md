# EXPERIMENTS

| model              | accuracy                          | sensitivity (Recall)                   | specificity                      | directory                                  |
|--------------------|-----------------------------------|----------------------------------------|----------------------------------|--------------------------------------------|
| binary, in-ear     |                                   |                                        |                                  |                                            |
| 4-class, in-ear    |                                   |                                        |                                  |                                            |
| binary, 8-channel  | 78.88% ± 3.70%, (72.72% - 83.90%) | 95.69% ± 6.60% (Min: 81.82% - 100.00%) | 51.81% ± 6.95% (40.00% - 59.92%) | experiments/both-012-all-binary-smallerall |
| 4-class, 8-channel |                                   |                                        |                                  |                                            |


Make sure I get all the necessary stats.
Run something to figure out impact of adding weights to cross entropy.
Default is 2 class

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
