# EXPERIMENTS

| model                              | accuracy (chunk × subject)        | sensitivity          | specificity    | directory                                             |
|------------------------------------|-----------------------------------|----------------------|----------------|-------------------------------------------------------|
| binary, in-ear, 3 dts.             | 76.86% ± 2.84% × 77.79% ± 2.74% | 86.23% ± 4.29%       | 63.90% ± 4.66% | all3-016-inear-binary-smaller; **6 fold only**        |
| binary, in-ear, LSTM               | 77.44% ± 4.64% × 75.87% ± 6.78% | 88.87% ± 7.11%       | 61.96% ± 8.59% | all3-018-inear-binary-smaller-weighted-sampler        |
| binary, in-ear, Attn               | 78.26% ± 4.92% × 77.39% ± 7.05% | 89.41% ± 7.02%       | 63.11% ± 8.76% | all3-018-inear-binary-smallerAttn                     |
| **binary, in-ear, Attn+FL**        | 77.76% ± 4.64% × 78.74% ± 7.50% | 84.21% ± 6.97%       | 68.78% ± 6.26% | all3-019-inear-binary-smallerAttn-focal               |
| binary, in-ear, Attn+FL, lr=1e-3   | 76.66% ± 2.36% × 76.61% ± 5.78% | 86.91% ± 6.63%       | 62.71% ±10.22% | all3-024-inear-binary-smallerAttn-focal; **6 fold**   |
| 4-class, in-ear, 3 dts.            | 63.62% ± 4.45%                   |                      |                | all3-016-inear-4class-smaller; **6 fold only**        |
| **4-class, in-ear, Attn+FL**       | 66.51% ± 7.32% × 67.10% ± 9.13% |                      |                | all3-019-inear-4class-smallerAttn-focal               |
| binary, 8-ch, 3 dts.               | 76.83% ± 2.12%                   | 88.72% ± 2.86%       | 58.25% ± 6.44% | all3-017-all-binary-smallerall-weighted-sampler       |
| **binary, 8-ch, V2Attn**           | 79.10% ± 4.90% × 79.83% ± 5.98% | 89.43% ± 6.78%       | 63.42% ±11.22% | all3-020-all-binary-smallerAllV2Attn-weighted-sampler |
| binary, 8-ch, V2Attn+FL d2         | 76.99% ± 5.16% × 76.75% ± 6.61% | 80.84% ± 7.26%       | 70.80% ± 9.98% | all3-020-all-binary-smallerAllV2Attn-focal-d2         |
| **binary, 8-ch, V3+FL**            | 78.84% ± 6.39% × 80.22% ± 6.94% | 83.74% ±11.02%       | 70.18% ±13.38% | all3-021-all-binary-smallerAllV3-focal                |
| binary, 8-ch, V3+FL (run 2)        | 78.56% ± 3.54% × 79.39% ± 6.35% | 86.68% ± 6.10%       | 66.56% ± 9.59% | all3-022-all-binary-smallerAllV3-focal                |
| binary, 8-ch, AllTransformerV4     | 76.53% ± 2.48% × 74.27% ± 3.00% | 87.22% ± 7.80%       | 59.53% ±10.55% | all3-024-all-2class-V4-weighted-sampler; **6 fold**   |
| binary, 8-ch, V4, lr=5e-4          | 77.58% ± 3.41% × 77.54% ± 6.69% | 87.92% ± 4.58%       | 61.50% ± 7.10% | all3-024-all-2class-V4-LR; **6 fold**                 |
| binary, 8-ch, DeformerS, d=0.3     | 64.85% ± 8.27% × 65.10% ± 8.78% | 66.87% ±24.36%       | 63.15% ±17.13% | all3-023-all-2class-DeformerS-weighted-sampler-30drop |
| binary, 8-ch, DeformerS-head       | 75.71% ± 4.30% × 74.13% ± 7.91% | 85.79% ± 6.29%       | 59.59% ±14.57% | all3-024-all-2class-DeformerS-head; **6 fold**        |
| 4-class, 8-ch, SmallerAll          | 63.46% ± 9.60% × 64.57% ±11.36% |                      |                | all3-018-all-4class-smallerAll-weighted-sampler       |
| 4-class, 8-ch, Attn                | 64.76% ± 8.05% × 66.09% ± 8.23% |                      |                | all3-018-all-4class-smallerAllAttn-weighted-sampler   |
| **4-class, 8-ch, V2Attn**          | 65.91% ± 6.63% × 68.49% ± 6.68% |                      |                | all3-020-all-4class-smallerAllV2Attn-weighted-sampler |

CUDA_VISIBLE_DEVICES=2 EEG_DATA_DIR=/home/xticha09 python main.py train   --channel all   --batch-size 64   --dataset all   --model SmallerAllV3 --condition ec+eo   --n-folds 10   --dropout 0.1  --weight-decay 1e-4   --val-every 1  --focal-loss --checkpoint-dir=experiments/all3-021-all-binary-smallerAllV3-focal
CUDA_VISIBLE_DEVICES=3 EEG_DATA_DIR=/home/xticha09 python main.py train   --channel all   --batch-size 64   --dataset all   --model SmallerAllV3 --condition ec+eo   --n-folds 10   --dropout 0.1  --weight-decay 1e-4   --val-every 1  --focal-loss --checkpoint-dir=experiments/all3-022-all-binary-smallerAllV3-focal

CUDA_VISIBLE_DEVICES=1 EEG_DATA_DIR=/home/xticha09 python main.py train   --channel in-ear   --batch-size 64   --dataset all   --model SmallerAttn   --condition ec+eo   --n-folds 10   --dropout 0.3   --weight-decay 1e-4   --val-every 1 --focal-loss   --checkpoint-dir=experiments/all3-020-inear-binary-smallerAttn-focal-d3
CUDA_VISIBLE_DEVICES=3 EEG_DATA_DIR=/home/xticha09 python main.py train   --channel in-ear   --batch-size 64   --dataset all   --model SmallerAttn   --condition ec+eo   --n-folds 10   --dropout 0.1  --class-mode 4 --weight-decay 1e-4   --val-every 1 --focal-loss  --checkpoint-dir=experiments/all3-019-inear-4class-smallerAttn-focal
CUDA_VISIBLE_DEVICES=2 EEG_DATA_DIR=/home/xticha09 python main.py train   --channel all   --batch-size 64   --dataset all   --model SmallerAllV2Attn --condition ec+eo   --n-folds 10   --dropout 0.3  --weight-decay 1e-4   --val-every 1  --focal-loss --checkpoint-dir=experiments/all3-020-all-binary-smallerAllV2Attn-focal-d3
CUDA_VISIBLE_DEVICES=2 EEG_DATA_DIR=/home/xticha09 python main.py train   --channel all   --batch-size 64   --dataset all   --model SmallerAllV2Attn --condition ec+eo   --n-folds 10   --dropout 0.1  --class-mode 4 --weight-decay 1e-4   --val-every 1   --checkpoint-dir=experiments/all3-020-all-4class-smallerAllV2Attn-weighted-sampler

CUDA_VISIBLE_DEVICES=2 EEG_DATA_DIR=/home/xticha09 python main.py train   --channel all   --batch-size 64   --dataset all   --model AllTransformerV4 --condition ec+eo   --n-folds 10   --dropout 0.1  --class-mode 2 --weight-decay 1e-4   --val-every 1   --checkpoint-dir=experiments/all3-022-all-2class-V3All-weighted-sampler

CUDA_VISIBLE_DEVICES=3 EEG_DATA_DIR=/home/xticha09 python main.py train   --channel all   --batch-size 64   --dataset all   --model Deformer --condition ec+eo   --n-folds 10   --dropout 0.1  --class-mode 2 --weight-decay 1e-4   --val-every 1   --checkpoint-dir=experiments/all3-023-all-2class-Deformer-weighted-sampler

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

### all3-024-inear-binary-smallerAttn-focal

- Model: SmallerAttn, binary, in-ear (IDUN+MDD+AX_MALIK), ec+eo, **6-fold**, focal loss, lr=1e-3, dropout=0.3, weight_decay=1e-3
- Chunk accuracy: 76.66% ± 2.36% | Subject accuracy: 76.61% ± 5.78%
- Chunk sensitivity: 86.91% ± 6.63% | Chunk specificity: 62.71% ± 10.22%
- Per-dataset chunk accuracy: CANE/IDUN 69.29%, MDD 88.84%, SAD 59.59%
- Train acc at best val: 73.87% ± 5.80%  — train/val close, suggesting LR=1e-3 + WD=1e-3 is too aggressive for exploration
- Best epochs: fold 1=44, folds 2/4=9/16, folds 3/5/6=1/2/1 — model dies in 3 of 6 folds within first 2 epochs
- vs 019 (best: 77.76%, lr=1e-4, wd=1e-4): −1.10pp chunk. Higher LR does not help; SAD accuracy collapsed (59.6% vs 62.6%)
- **Conclusion**: lr=1e-3 with wd=1e-3 is too aggressive for SmallerAttn; 019 remains best in-ear config
- Command: `CUDA_VISIBLE_DEVICES=0 EEG_DATA_DIR=/home/xticha09 python main.py train --channel in-ear --batch-size 64 --dataset all --model SmallerAttn --condition ec+eo --n-folds 6 --dropout 0.3 --weight-decay 1e-3 --lr 0.001 --val-every 1 --focal-loss --checkpoint-dir=experiments/all3-024-inear-binary-smallerAttn-focal`

### all3-024-all-2class-V4-weighted-sampler

- Model: AllTransformerV4 ((channel×time) token Transformer), binary, 8-channel, ec+eo, **6-fold**, lr=1e-4, dropout=0.1, wd=1e-4
- Chunk accuracy: 76.53% ± 2.48% | Subject accuracy: 74.27% ± 3.00%
- Chunk sensitivity: 87.22% ± 7.80% | Chunk specificity: 59.53% ± 10.55%
- Per-dataset chunk accuracy: CANE 70.78%, MDD 87.27%, SAD 63.67%
- Train acc at best val: 73.77% ± 10.51% — high variance fold-to-fold (range 55-88%)
- Best epochs: very inconsistent (1, 10, 42, 45, 2, 12) — fold 3 stops at epoch 1, fold 4 at epoch 45; architecture is sensitive to fold composition
- vs SmallerAllV2Attn 020 (79.10%): −2.57pp. V4 (token transformer) underperforms the CNN+LSTM baseline; likely under-regularized
- Command: `CUDA_VISIBLE_DEVICES=2 EEG_DATA_DIR=/home/xticha09 python main.py train --channel all --batch-size 64 --dataset all --model AllTransformerV4 --condition ec+eo --n-folds 6 --dropout 0.1 --class-mode 2 --lr 1e-4 --weight-decay 1e-4 --val-every 1 --checkpoint-dir=experiments/all3-024-all-2class-V4-weighted-sampler`

### all3-024-all-2class-V4-5sec

- Model: AllTransformerV4, binary, 8-channel, ec+eo, **6-fold**, lr=1e-4, dropout=0.1, wd=1e-4, `--chunk-duration 5`
- **Note: `--chunk-duration 5` has no effect on AllTransformerV4** — it is a spectrogram model (not in RAW_EEG_MODELS), so it always uses 10-second chunks. Chunk counts identical to standard 10s runs.
- Chunk accuracy: 76.84% ± 2.88% | Subject accuracy: 74.37% ± 4.86%
- Chunk sensitivity: 89.53% ± 7.10% | Chunk specificity: 56.78% ± 8.71%
- Per-dataset chunk accuracy: CANE 72.98%, MDD 87.77%, SAD 56.97%
- Train acc at best val: 75.61% ± 3.29% — unusually consistent vs previous V4 run
- Best epochs: moderate (3–24), slightly more stable than V4-weighted-sampler
- vs V4-weighted-sampler (same architecture): +0.31pp. Effectively a replicate with slightly different RNG state; difference within noise
- Command: `CUDA_VISIBLE_DEVICES=2 EEG_DATA_DIR=/home/xticha09 python main.py train --channel all --batch-size 64 --dataset all --model AllTransformerV4 --condition ec+eo --n-folds 6 --dropout 0.1 --class-mode 2 --lr 1e-4 --weight-decay 1e-4 --val-every 1 --epochs 200 --chunk-duration 5 --checkpoint-dir=experiments/all3-024-all-2class-V4-5sec`

### all3-024-all-2class-V4-LR

- Model: AllTransformerV4, binary, 8-channel, ec+eo, **6-fold**, lr=5e-4, dropout=0.1, wd=1e-4, `--chunk-duration 5` (no effect)
- Chunk accuracy: 77.58% ± 3.41% | Subject accuracy: 77.54% ± 6.69%
- Chunk sensitivity: 87.92% ± 4.58% | Chunk specificity: 61.50% ± 7.10%
- Per-dataset chunk accuracy: CANE 72.93%, MDD 88.35%, SAD 61.38%
- Train acc at best val: 82.12% ± 14.21% — **very high variance** (range 62-98%); folds 3/4/5 heavily overfit (train 95-98%)
- Best epochs: highly bimodal — folds 1/2/6 stop at epoch 1/1/2 (diverge immediately); folds 3/4/5 run 22-47 epochs
- vs V4 lr=1e-4 (024-V4-weighted-sampler): +1.05pp chunk, +3.27pp subject. Higher LR helps on average but adds instability
- Loss curves: dramatic fold-to-fold variance; LR=5e-4 is on the edge of stability for V4
- **Best V4 result so far** but high fold variance makes it unreliable
- Command: `CUDA_VISIBLE_DEVICES=1 EEG_DATA_DIR=/home/xticha09 python main.py train --channel all --batch-size 64 --dataset all --model AllTransformerV4 --condition ec+eo --n-folds 6 --dropout 0.1 --class-mode 2 --lr 5e-4 --weight-decay 1e-4 --val-every 1 --epochs 200 --chunk-duration 5 --checkpoint-dir=experiments/all3-024-all-2class-V4-LR`

### all3-023-all-2class-Deformer-weighted-sampler

- Model: Deformer (full, ~5.4M params, 21MB checkpoints), binary, 8-channel, ec+eo, **10-fold**, lr=1e-4, dropout=0.1
- No results.txt generated (training likely completed without writing summary)
- Per-fold best chunk acc from log: fold 1≈0.77, fold 2≈0.76, fold 3≈0.77, fold 4≈0.66, fold 5≈0.75, fold 6≈0.73, fold 7≈0.73, fold 8=0.77, fold 9=0.65, fold 10=0.80 → estimated mean ~74%
- Massive overfitting: train acc reaches 97-99% by epoch 3-5; best val occurs at epoch 2-18 then degrades sharply
- Val loss spiky and increasing from epoch 5+; model memorizes training set almost immediately
- Pattern: 10-fold folds with fewer pathological samples (fold 4, 9) collapse to ~65%; lucky folds reach ~80%
- **Conclusion**: Full Deformer is far too large for this dataset (~330 subjects). Not competitive with SmallerAllV2Attn/V3 despite more parameters.

### all3-023-all-2class-Deformer-weighted-sampler-50-dropout

- Model: Deformer (full), binary, 8-channel, ec+eo, 6-fold, lr=1e-4, dropout=0.5, wd=1e-4
- Chunk accuracy: 63.83% ± 9.86% | Subject accuracy: 64.91% ± 7.93%
- Chunk sensitivity: 57.37% ± 31.28% | Chunk specificity: 71.07% ± 23.65%
- Per-dataset chunk accuracy: CANE 56.01%, MDD 78.53%, SAD 51.68%
- Train acc at best val: 93.29% ± 4.76% — dropout=0.5 reduces overfitting slightly but not enough
- High sensitivity variance (±31%) indicates model is threshold-unstable across folds
- vs d=0.1 (10-fold estimated ~74%): much worse. Excessive dropout hurts Deformer substantially.

### all3-023-all-2class-DeformerS-weighted-sampler-30drop

- Model: DeformerS (~588K params), binary, 8-channel, ec+eo, 6-fold, lr=1e-4, dropout=0.3, wd=1e-4
- Chunk accuracy: 64.85% ± 8.27% | Subject accuracy: 65.10% ± 8.78%
- Chunk sensitivity: 66.87% ± 24.36% | Chunk specificity: 63.15% ± 17.13%
- Per-dataset chunk accuracy: CANE 56.68%, MDD 75.21%, SAD 60.32%
- Train acc at best val: 96.21% ± 2.65% — **extreme overfitting** despite being smaller model
- Best epochs: wildly variable (6, 6, 93, 9, 8, 8) — fold 3 runs to epoch 93 suggesting lucky initialization
- DeformerS at standard lr=1e-4 fails dramatically; raw-EEG approach needs fundamentally different regularization

### all3-023-all-2class-DeformerS-weighted-sampler-50drop

- Model: DeformerS, binary, 8-channel, ec+eo, 6-fold, lr=1e-4, dropout=0.5, wd=1e-4
- Chunk accuracy: 60.78% ± 13.43% | Subject accuracy: 62.10% ± 14.38%
- Chunk sensitivity: 55.63% ± 34.53% | Chunk specificity: 71.25% ± 25.96%
- Train acc at best val: 89.69% ± 12.16% — still massively overfit even with heavy dropout
- Two folds completely fail (chunk ≈ 39-46%) — model collapses to majority-class prediction
- **Conclusion**: DeformerS cannot be fixed with dropout alone at standard LR. Architecture doesn't regularize well on ~8000 training chunks.

### all3-024-all-2class-DeformerS-head

- Model: DeformerS with modified head (likely different classifier layers), binary, 8-channel, ec+eo, **6-fold**, lr=1e-3, dropout=0.5, wd=1e-3
- Chunk accuracy: 75.71% ± 4.30% | Subject accuracy: 74.13% ± 7.91%
- Chunk sensitivity: 85.79% ± 6.29% | Chunk specificity: 59.59% ± 14.57%
- Per-dataset chunk accuracy: CANE 69.95%, MDD 87.58%, SAD 58.17%
- Train acc at best val: 91.35% ± 6.19% — still overfit but at higher LR the model finds usable solutions faster
- Best epochs: early (1-5 for 4 folds, 19 and 30 for 2 folds) — aggressive LR+WD pushes early convergence
- vs DeformerS-30drop (lr=1e-4): **+10.86pp chunk** — dramatically better. The key was lr=1e-3 not the head modification.
- vs SmallerAllV2Attn best (79.10%): −3.39pp. DeformerS still trails CNN-LSTM baselines.
- SAD still at 58%, same bottleneck as all other models

### all3-022-all-binary-smallerAllV3-focal

- Model: SmallerAllV3 (4b, concat) — **first run of the concat architecture**; 021 turned out to use the old 3a design
- Config: binary, 8-channel, ec+eo, 10-fold, focal loss (gamma=2.0), dropout=0.1, rnn_hidden=100, chan_d_model=128
- Chunk accuracy: 78.56% ± 3.54% | Subject accuracy: 79.39% ± 6.35%
- Chunk sensitivity: 86.68% ± 6.10% | Chunk specificity: 66.56% ± 9.59% | Balanced acc: 76.62%
- Per-dataset chunk accuracy: CANE 72.19%, MDD 89.62%, SAD 66.49%
- vs 021 (3a design): −0.28pp chunk, within noise. Different architecture so not a clean comparison.
- Loss curves: epoch-1/2 best in ~3 folds, train/val divergence from epoch 3+. Overfitting pattern consistent with 021 despite different architecture.
- Command: `CUDA_VISIBLE_DEVICES=2 EEG_DATA_DIR=/home/xticha09 python main.py train --channel all --batch-size 64 --dataset all --model SmallerAllV3 --condition ec+eo --n-folds 10 --dropout 0.1 --weight-decay 1e-4 --val-every 1 --focal-loss --checkpoint-dir=experiments/all3-022-all-binary-smallerAllV3-focal`

### all3-021-all-binary-smallerAllV3-focal_hsf

- Model: SmallerAllV3 (3a design — rnn_hidden=330, chan_d_model=128) — V3 with oversized LSTM
- Config: binary, 8-channel, EC only (4217 chunks vs 8444 for ec+eo), focal loss, dropout=0.2
- Chunk accuracy: 78.68% ± 4.98% | Subject accuracy: 79.24% ± 8.41%
- Chunk sensitivity: 85.15% ± 8.56% | Chunk specificity: 68.62% ± 13.35% | Balanced acc: 76.89%
- Note: trained on EC condition only (half the data). High variance in subject accuracy (±8.4%) reflects smaller val sets per fold.
- vs 021 (V3 3b, ec+eo): nearly identical despite 3a design and half the data. Confirms the concat approach (3b) offers no systematic advantage once overfitting dominates both.
- Command: `CUDA_VISIBLE_DEVICES=2 EEG_DATA_DIR=/home/xticha09 python main.py train --channel all --batch-size 64 --dataset all --model SmallerAllV3 --condition ec --n-folds 10 --dropout 0.2 --weight-decay 1e-4 --val-every 1 --focal-loss --checkpoint-dir=experiments/all3-021-all-binary-smallerAllV3-focal_hsf`

### all3-021-all-binary-smallerAllV3-focal

- Model: SmallerAllV3 (4a, per-frame attention) — confirmed via `nhead=8, rnn_hidden=330` in training log
- Config: binary, 8-channel, ec+eo, 10-fold, focal loss (gamma=2.0), dropout=0.1, rnn_hidden=330, chan_d_model=128
- Chunk accuracy: 78.84% ± 6.39% | Subject accuracy: 80.22% ± 6.94%
- Chunk sensitivity: 83.74% ± 11.02% | Chunk specificity: 70.18% ± 13.38% | Balanced acc: 76.96%
- Per-dataset chunk accuracy: CANE 72.21%, MDD 88.47%, SAD 73.02%
- Best balanced accuracy of all experiments (76.96%). SAD accuracy improved significantly vs 020 (+4.94pp).
- Loss curves: epoch-1/2 best in folds 1, 5, 10. Val loss spiky throughout. Train loss decreasing steadily. Classic overfitting from oversized chan_proj layer (~885K params of ~1M total).
- Key takeaway: V3 4a (per-frame attention) matches V2Attn in accuracy. The 4b concat design was not tested until 022.
- Command: `CUDA_VISIBLE_DEVICES=2 EEG_DATA_DIR=/home/xticha09 python main.py train --channel all --batch-size 64 --dataset all --model SmallerAllV3 --condition ec+eo --n-folds 10 --dropout 0.1 --weight-decay 1e-4 --val-every 1 --focal-loss --checkpoint-dir=experiments/all3-021-all-binary-smallerAllV3-focal`

### all3-020-all-binary-smallerAllV2Attn-focal-d3

- Model: SmallerAllV2Attn, binary, 8-channel, ec+eo, 10-fold, focal loss, dropout=0.3
- Incomplete run — no results.txt. Abandoned (higher dropout found not to help for V2Attn).

### all3-020-inear-binary-smallerAttn-focal-d3

- Model: SmallerAttn, binary, in-ear, ec+eo, 10-fold, focal loss, dropout=0.3
- Chunk accuracy: 77.30% ± 5.29% | Subject accuracy: 76.77% ± 5.94%
- Chunk sensitivity: 84.84% ± 9.51% | Chunk specificity: 66.54% ± 14.60%
- vs 019 (dropout=0.1): +0.54pp chunk, nearly identical. Dropout d=0.3 does not help SmallerAttn.

### all3-020-inear-binary-smallerAttn-focal-d5

- Model: SmallerAttn, binary, in-ear, ec+eo, 10-fold, focal loss, dropout=0.5 (typo: d5 = 0.5)
- Chunk accuracy: 76.32% ± 5.81% | Subject accuracy: 73.86% ± 6.93%
- Chunk sensitivity: 85.76% ± 9.58% | Chunk specificity: 64.00% ± 10.44%
- vs 019: −1.44pp chunk, −4.88pp subject. Too much dropout hurts SmallerAttn.

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
