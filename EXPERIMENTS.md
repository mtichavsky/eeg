# EXPERIMENTS

| model                        | accuracy                        | sensitivity (Recall) | specificity    | directory                                             |
|------------------------------|---------------------------------|----------------------|----------------|-------------------------------------------------------|
| binary, in-ear, 3 dts.       | 76.86% ± 2.84% x 77.79% ± 2.74% | 86.23% ± 4.29%       | 63.90% ± 4.66% | all3-016-inear-binary-smaller; **6 fold only**        |
| binary, in-ear, LSTM         | 77.44% ± 4.64% × 75.87% ± 6.78% | 88.87% ± 7.11%       | 61.96% ± 8.59% | all3-018-inear-binary-smaller-weighted-sampler        |
| binary, in-ear, Attn         | 78.26% ± 4.92% × 77.39% ± 7.05% | 89.41% ± 7.02%       | 63.11% ± 8.76% | all3-018-inear-binary-smallerAttn                     |
| **binary, in-ear, Attn+FL**  | 77.76% ± 4.64% × 78.74% ± 7.50% | 84.21% ± 6.97%       | 68.78% ± 6.26% | all3-019-inear-binary-smallerAttn-focal               |
| 4-class, in-ear, 3 dts.      | 63.62% ± 4.45%                  |                      |                | all3-016-inear-4class-smaller, **6 fold only**        | 
| **4-class, in-ear, Attn+FL** | 66.51% ± 7.32% × 67.10% ± 9.13% |                      |                | all3-019-inear-4class-smallerAttn-focal               |
| binary, 8-ch, 3 dts.         | 76.83% ± 2.12%                  | 88.72% ± 2.86%       | 58.25% ± 6.44% | all3-017-all-binary-smallerall-weighted-sampler       |
| **binary, 8-ch, V2Attn**     | 79.10% ± 4.90% × 79.83% ± 5.98% | 89.43% ± 6.78%       | 63.42% ±11.22% | all3-020-all-binary-smallerAllV2Attn-weighted-sampler |
| binary, 8-ch, V2Attn+FL d2   | 76.99% ± 5.16% × 76.75% ± 6.61% | 80.84% ± 7.26%       | 70.80% ± 9.98% | all3-020-all-binary-smallerAllV2Attn-focal-d2         |
| 4-class, 8-ch, SmallerAll    | 63.46% ± 9.60% × 64.57% ±11.36% |                      |                | all3-018-all-4class-smallerAll-weighted-sampler       |
| 4-class, 8-ch, Attn          | 64.76% ± 8.05% × 66.09% ± 8.23% |                      |                | all3-018-all-4class-smallerAllAttn-weighted-sampler   |
| **4-class, 8-ch, V2Attn**    | 65.91% ± 6.63% × 68.49% ± 6.68% |                      |                | all3-020-all-4class-smallerAllV2Attn-weighted-sampler |

CUDA_VISIBLE_DEVICES=2 EEG_DATA_DIR=/home/xticha09 python main.py train   --channel all   --batch-size 64   --dataset all   --model SmallerAllV3 --condition ec+eo   --n-folds 10   --dropout 0.1  --weight-decay 1e-4   --val-every 1  --focal-loss --checkpoint-dir=experiments/all3-021-all-binary-smallerAllV3-focal

CUDA_VISIBLE_DEVICES=1 EEG_DATA_DIR=/home/xticha09 python main.py train   --channel in-ear   --batch-size 64   --dataset all   --model SmallerAttn   --condition ec+eo   --n-folds 10   --dropout 0.3   --weight-decay 1e-4   --val-every 1 --focal-loss   --checkpoint-dir=experiments/all3-020-inear-binary-smallerAttn-focal-d3
CUDA_VISIBLE_DEVICES=3 EEG_DATA_DIR=/home/xticha09 python main.py train   --channel in-ear   --batch-size 64   --dataset all   --model SmallerAttn   --condition ec+eo   --n-folds 10   --dropout 0.1  --class-mode 4 --weight-decay 1e-4   --val-every 1 --focal-loss  --checkpoint-dir=experiments/all3-019-inear-4class-smallerAttn-focal
CUDA_VISIBLE_DEVICES=2 EEG_DATA_DIR=/home/xticha09 python main.py train   --channel all   --batch-size 64   --dataset all   --model SmallerAllV2Attn --condition ec+eo   --n-folds 10   --dropout 0.3  --weight-decay 1e-4   --val-every 1  --focal-loss --checkpoint-dir=experiments/all3-020-all-binary-smallerAllV2Attn-focal-d3
CUDA_VISIBLE_DEVICES=2 EEG_DATA_DIR=/home/xticha09 python main.py train   --channel all   --batch-size 64   --dataset all   --model SmallerAllV2Attn --condition ec+eo   --n-folds 10   --dropout 0.1  --class-mode 4 --weight-decay 1e-4   --val-every 1   --checkpoint-dir=experiments/all3-020-all-4class-smallerAllV2Attn-weighted-sampler

Higher dropout doesn't make sense.

### all3-020-all-binary-smallerAllV2Attn-weighted-sampler

- Model: SmallerAllV2Attn (per-channel shared CNN + cross-channel self-attention + temporal attention), binary, 8-channel, ec+eo, 10-fold, weighted sampler
- Chunk accuracy: 79.10% ± 4.90% | Subject accuracy: 79.83% ± 5.98%
- Chunk sensitivity: 89.43% ± 6.78% | Chunk specificity: 63.42% ± 11.22%
- Per-dataset chunk accuracy: CANE 72.55%, MDD 90.20%, SAD 68.08%
- vs SmallerAllAttn (018, 8ch binary): no direct baseline but matches in-ear SmallerAttn (78.26% chunk) — 8 channels provide no gain over 1 channel
- Command: `CUDA_VISIBLE_DEVICES=1 EEG_DATA_DIR=/home/xticha09 python main.py train   --channel all   --batch-size 64   --dataset all   --model SmallerAllV2Attn --condition ec+eo   --n-folds 10   --dropout 0.1  --weight-decay 1e-4   --val-every 1   --checkpoint-dir=experiments/all3-020-all-binary-smallerAllV2Attn-weighted-sampler`

### all3-020-all-4class-smallerAllV2Attn-weighted-sampler

- Model: SmallerAllV2Attn (per-channel shared CNN + cross-channel self-attention + temporal attention), 4-class, 8-channel, ec+eo, 10-fold, weighted sampler
- Chunk accuracy: 65.91% ± 6.63% | Subject accuracy: 68.49% ± 6.68%
- Chunk recall: Healthy 55.87%, Anxiety 42.18%, Depression 90.35%, Comorbid 79.63%
- Per-dataset chunk accuracy: CANE 46.30%, MDD 86.59%, SAD 69.14%
- vs SmallerAllAttn (018): +1.15pp chunk, +2.40pp subject — marginal improvement; CANE still at ~46%
- Architecture: per-channel CNN (shared weights) preserves channel identity longer than Conv3D, but cross-channel attention still provides no measurable gain over single-channel models
- Command: `CUDA_VISIBLE_DEVICES=2 EEG_DATA_DIR=/home/xticha09 python main.py train   --channel all   --batch-size 64   --dataset all   --model SmallerAllV2Attn --condition ec+eo   --n-folds 10   --dropout 0.1  --class-mode 4 --weight-decay 1e-4   --val-every 1   --checkpoint-dir=experiments/all3-020-all-4class-smallerAllV2Attn-weighted-sampler`

### all3-019-inear-binary-smallerAttn-focal

- Model: SmallerAttn, binary, in-ear, ec+eo, 10-fold, focal loss (gamma=2.0) + weighted sampler
- Chunk accuracy: 77.76% ± 4.64% | Subject accuracy: 78.74% ± 7.50%
- Chunk sensitivity: 84.21% ± 6.97% | Chunk specificity: 68.78% ± 6.26%
- Per-dataset chunk accuracy: CANE 69.56%, MDD 89.03%, SAD 62.62%
- vs 018-Attn (no focal): specificity +5.67pp, sensitivity -5.20pp — better balanced, balanced accuracy 76.26% → 76.50%
- Command: `CUDA_VISIBLE_DEVICES=3 EEG_DATA_DIR=/home/xticha09 python main.py train   --channel in-ear   --batch-size 64   --dataset all   --model SmallerAttn   --condition ec+eo   --n-folds 10   --dropout 0.1   --weight-decay 1e-4   --val-every 1 --focal-loss   --checkpoint-dir=experiments/all3-019-inear-binary-smallerAttn-focal`

### all3-019-inear-4class-smallerAttn-focal

- Model: SmallerAttn, 4-class, in-ear, ec+eo, 10-fold, focal loss (gamma=2.0) + weighted sampler
- Chunk accuracy: 66.51% ± 7.32% | Subject accuracy: 67.10% ± 9.13%
- Chunk recall: Healthy 50.39%, Anxiety 60.90%, Depression 89.44%, Comorbid 89.31%
- Per-dataset chunk accuracy: CANE 42.60%, MDD 85.92%, SAD 62.17%
- vs 018-Attn (no focal): overall +1.75pp chunk; Anxiety +14.92pp, Comorbid +18.97pp, but Healthy -9.97pp
- Command: `CUDA_VISIBLE_DEVICES=3 EEG_DATA_DIR=/home/xticha09 python main.py train   --channel in-ear   --batch-size 64   --dataset all   --model SmallerAttn   --condition ec+eo   --n-folds 10   --dropout 0.1   --weight-decay 1e-4   --val-every 1 --focal-loss --class-mode 4  --checkpoint-dir=experiments/all3-019-inear-4class-smallerAttn-focal`

### all3-018-all-4class-smallerAll-weighted-sampler

- Model: SmallerAll (LSTM), 4-class, 8-channel, ec+eo, 10-fold, weighted sampler
- Chunk accuracy: 63.46% ± 9.60% | Subject accuracy: 64.57% ± 11.36%
- Chunk recall: Healthy 54.64%, Anxiety 57.69%, Depression 86.06%, Comorbid 61.78%
- Per-dataset chunk accuracy: CANE 44.40%, MDD 84.07%, SAD 65.87%
- Overall chunk accuracy (aggregated): 63.30%
- Command: `CUDA_VISIBLE_DEVICES=3 EEG_DATA_DIR=/home/xticha09 python main.py train   --channel all   --batch-size 64   --dataset all   --model SmallerAll   --condition ec+eo   --n-folds 10   --dropout 0.1  --class-mode 4 --weight-decay 1e-4   --val-every 1   --checkpoint-dir=experiments/all3-018-all-4class-smallerAll-weighted-sampler`

### all3-018-all-4class-smallerAllAttn-weighted-sampler

- Model: SmallerAllAttn (self-attention replacing LSTM), 4-class, 8-channel, ec+eo, 10-fold, weighted sampler
- Chunk accuracy: 64.76% ± 8.05% | Subject accuracy: 66.09% ± 8.23%
- Chunk recall: Healthy 60.36%, Anxiety 45.98%, Depression 85.04%, Comorbid 70.34%
- Per-dataset chunk accuracy: CANE 46.42%, MDD 86.48%, SAD 62.26%
- Overall chunk accuracy (aggregated): 64.72%
- Attention slightly improves overall accuracy (+1.3pp chunk, +1.5pp subject) and Comorbid recall, but hurts Anxiety recall vs LSTM baseline
- Command: `CUDA_VISIBLE_DEVICES=3 EEG_DATA_DIR=/home/xticha09 python main.py train   --channel all   --batch-size 64   --dataset all   --model SmallerAllAttn   --condition ec+eo   --n-folds 10   --dropout 0.1  --class-mode 4 --weight-decay 1e-4   --val-every 1   --checkpoint-dir=experiments/all3-018-all-4class-smallerAllAttn-weighted-sampler`

### all3-018-inear-binary-smaller-weighted-sampler

- Model: Smaller (LSTM), binary, in-ear, ec+eo, 10-fold, weighted sampler
- Chunk accuracy: 77.44% ± 4.64% | Subject accuracy: 75.87% ± 6.78%
- Chunk sensitivity: 88.87% ± 7.11% | Chunk specificity: 61.96% ± 8.59%
- Per-dataset chunk accuracy: CANE 70.30%, MDD 86.50%, SAD 66.15%
- Overall chunk accuracy (aggregated): 77.49%
- Command: `CUDA_VISIBLE_DEVICES=2 EEG_DATA_DIR=/home/xticha09 python main.py train   --channel in-ear   --batch-size 64   --dataset all   --model Smaller   --condition ec+eo   --n-folds 10   --dropout 0.1   --weight-decay 1e-4   --val-every 1   --checkpoint-dir=experiments/all3-018-inear-binary-smaller-weighted-sampler`

### all3-018-inear-binary-smallerAttn

- Model: SmallerAttn (self-attention replacing LSTM), binary, in-ear, ec+eo, 10-fold
- Chunk accuracy: 78.26% ± 4.92% | Subject accuracy: 77.39% ± 7.05%
- Chunk sensitivity: 89.41% ± 7.02% | Chunk specificity: 63.11% ± 8.76%
- Per-dataset chunk accuracy: CANE 70.73%, MDD 88.16%, SAD 65.79%
- Overall chunk accuracy (aggregated): 78.39%
- Attention marginally outperforms LSTM (+0.82pp chunk, +1.52pp subject), consistent with 4-class findings
- Command: `CUDA_VISIBLE_DEVICES=3 EEG_DATA_DIR=/home/xticha09 python main.py train   --channel in-ear   --batch-size 64   --dataset all   --model SmallerAttn   --condition ec+eo   --n-folds 10   --dropout 0.1   --weight-decay 1e-4   --val-every 1   --checkpoint-dir=experiments/all3-018-inear-binary-smallerAttn`

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
