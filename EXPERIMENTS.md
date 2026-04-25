# EXPERIMENTS

| model                                | accuracy (chunk × subject)          | sensitivity    | specificity        | directory                                                                        |
|--------------------------------------|-------------------------------------|----------------|--------------------|----------------------------------------------------------------------------------|
| binary, in-ear, 3 dts.               | 76.86% ± 2.84% × 77.79% ± 2.74%     | 86.23% ± 4.29% | 63.90% ± 4.66%     | all3-016-inear-binary-smaller; **6 fold only**                                   |
| binary, in-ear, LSTM                 | 77.44% ± 4.64% × 75.87% ± 6.78%     | 88.87% ± 7.11% | 61.96% ± 8.59%     | all3-018-inear-binary-smaller-weighted-sampler                                   |
| binary, in-ear, Attn                 | 78.26% ± 4.92% × 77.39% ± 7.05%     | 89.41% ± 7.02% | 63.11% ± 8.76%     | all3-018-inear-binary-smallerAttn                                                |
| **binary, in-ear, Attn+FL**          | 77.76% ± 4.64% × 78.74% ± 7.50%     | 84.21% ± 6.97% | 68.78% ± 6.26%     | all3-019-inear-binary-smallerAttn-focal                                          |
| binary, in-ear, Attn+FL, lr=1e-3     | 76.66% ± 2.36% × 76.61% ± 5.78%     | 86.91% ± 6.63% | 62.71% ±10.22%     | all3-024-inear-binary-smallerAttn-focal; **6 fold**                              |
| 4-class, in-ear, 3 dts.              | 63.62% ± 4.45%                      |                |                    | all3-016-inear-4class-smaller; **6 fold only**                                   |
| 4-class, 8-ch, SmallerAll            | 63.46% ± 9.60% × 64.57% ±11.36%     |                |                    | all3-018-all-4class-smallerAll-weighted-sampler                                  |
| 4-class, 8-ch, Attn                  | 64.76% ± 8.05% × 66.09% ± 8.23%     |                |                    | all3-018-all-4class-smallerAllAttn-weighted-sampler                              |
| **4-class, 8-ch, V2Attn**            | 65.91% ± 6.63% × 68.49% ± 6.68%     |                |                    | all3-020-all-4class-smallerAllV2Attn-weighted-sampler                            |
| binary, 8-ch, LGGNet (hem)           | —                                   | —              | —                  | all3-027-all-binary-lggnet; **planned**                                          |
| binary, 8-ch, LGGNetS (hem)          | 73.35% ± 4.66% × 74.42% ± 6.99%     | 85.25% ± 8.30% | 54.85% ± 8.47%     | all3-027-all-binary-lggnet-s; **10 fold**                                        |
| binary, 8-ch, LGGNetS out_graph=16   | 72.80% ± 3.96% × 74.38% ± 4.83%     | 81.92% ± 7.07% | 58.15% ±10.04%     | all3-027-all-binary-lggnet-s-out16_byh; **10 fold**                              |
| binary, 8-ch, LGGNet frontal         | —                                   | —              | —                  | all3-027-all-binary-lggnet-frontal; **planned**                                  |
| binary, 8-ch, LGGNet+FL              | 74.55% ± 4.02% × 76.51% ± 7.19%     | 85.79% ± 5.33% | 56.72% ± 8.43%     | all3-027-all-binary-lggnet-focal; **10 fold**                                    |
| binary, 8-ch, LGGNetS+FL wd=5e-4     | 72.40% ± 3.23% × 72.44% ± 3.76%     | 79.27% ± 6.08% | 61.81% ±10.18%     | all3-027-all-binary-lggnet-s-focal-5e-4; **8 fold**                              |
| 4-class, 8-ch, LGGNet+FL             | —                                   | —              | —                  | all3-027-all-4class-lggnet-focal; **planned**                                    |
| 4-class, 8-ch, TSception+FL          | —                                   | —              | —                  | all3-026-all-4class-tsception-focal; **6 fold, planned**                         |

> **Tip:** append `> /dev/null 2>&1 &` to any command to run it in the background and detach from the terminal.

---

### 028 series — CAR + linear detrend preprocessing ablation

**Change under test (commit b25d514, 2026-04-22):** MDD and SAD (AX_MALIK) datasets were missing two
preprocessing steps that CANE and IDUN already had, producing inconsistent inputs across datasets:
1. **Average reference (CAR)** — subtract per-time-point mean across all 8 EEG channels
2. **Linear detrend** — removes slow DC drift per channel before chunking
3. **Per-chunk z-score** — also added alongside the above

Previously only CANE/IDUN had CAR+detrend; after the change all four datasets share the same
normalisation sequence.  For `--channel in-ear`, CAR is skipped (meaningless with 2 electrodes);
detrend is still applied to T7/T8 before the bipolar derivation.

Two baseline models were re-run with the updated preprocessing:
- **CNNCatLSTM + focal loss** (028 _jjr, 10-fold) — replicate of 022
- **DeformerS-head** (028 _ucb 6-fold + 028 _dzf 10-fold) — replicate of 024-DeformerS-head

#### all3-028-all-binary-smallerAllV3-focal_jjr

- Model: CNNCatLSTM (~1M params, 4b concat), binary, 8-channel, **CAR+detrend preprocessing**, ec+eo, **10-fold**, focal loss (gamma=2.0)
- Hyperparams: lr=1e-4, dropout=0.1, wd=1e-4 (identical to 022)
- Chunk accuracy: 77.73% ± 3.97% | Subject accuracy: 77.61% ± 5.37%
- Chunk sensitivity: 84.86% ± 6.09% | Chunk specificity: 66.96% ± 7.70%
- Train acc at best val: 79.80% ± 15.37% — very high variance (folds 5, 8, 10 stop at epoch 1–2)
- Per-dataset chunk accuracy: CANE 72.93% (spec 53.29%), MDD 87.44% (spec 86.36%), SAD 64.64% (spec 63.88%)
- vs 022 baseline (no CAR): −0.83pp chunk, −1.78pp subject. Sens/spec nearly unchanged (+0.40pp spec).
- **Result: preprocessing change has a small negative effect on CNNCatLSTM+FL.** CAR/detrend does not help the
  spectrogram-based pipeline; the STFT log-magnitude representation already discards DC offsets, so the extra
  normalisation adds noise without benefit. Performance is within 1pp and within noise, but consistently lower.
- Command: `CUDA_VISIBLE_DEVICES=0 EEG_DATA_DIR=/home/xticha09 python main.py train --channel all --batch-size 64 --dataset all --model CNNCatLSTM --condition ec+eo --n-folds 10 --lr 1e-4 --dropout 0.1 --weight-decay 1e-4 --val-every 1 --focal-loss --checkpoint-dir=experiments/all3-028-all-binary-smallerAllV3-focal_jjr`

#### all3-028-all-2class-DeformerS-head_ucb

- Model: DeformerS (~588K params), binary, 8-channel, **CAR+detrend preprocessing**, ec+eo, **6-fold**
- Hyperparams: lr=1e-3, dropout=0.5, wd=1e-3 (identical to 024-DeformerS-head)
- Chunk accuracy: 74.96% ± 5.60% | Subject accuracy: 75.41% ± 6.28%
- Chunk sensitivity: 77.00% ± 11.46% | Chunk specificity: 71.18% ± 5.64%
- Per-dataset chunk accuracy: CANE 67.28% (spec 42.41%), MDD 85.93% (spec 83.66%), SAD 66.71% (spec 67.97%)
- vs 024-DeformerS-head baseline (no CAR): −0.75pp chunk, +1.28pp subject; **+11.59pp specificity, −8.79pp sensitivity**
- Command: `CUDA_VISIBLE_DEVICES=1 EEG_DATA_DIR=/home/xticha09 python main.py train --channel all --batch-size 64 --dataset all --model DeformerS --condition ec+eo --n-folds 6 --lr 1e-3 --dropout 0.5 --weight-decay 1e-3 --val-every 1 --checkpoint-dir=experiments/all3-028-all-2class-DeformerS-head_ucb`

#### all3-028-all-2class-DeformerS-head-10fold_dzf

- Model: DeformerS (~588K params), binary, 8-channel, **CAR+detrend preprocessing**, ec+eo, **10-fold**
- Hyperparams: identical to all3-028-all-2class-DeformerS-head_ucb
- Chunk accuracy: 76.02% ± 8.05% | Subject accuracy: 75.58% ± 10.63%
- Chunk sensitivity: 80.24% ± 11.07% | Chunk specificity: 69.60% ± 12.05%
- Per-dataset chunk accuracy: CANE 66.82% (spec 41.40%), MDD 88.33% (spec 88.01%), SAD 66.31% (spec 69.12%)
- vs 024-DeformerS-head (6-fold): +0.31pp chunk, +1.45pp subject; +10.01pp specificity vs baseline
- **Combined 028 DeformerS verdict:** Chunk accuracy is essentially unchanged (−0.75pp to +0.31pp), but CAR+detrend
  dramatically rebalances sens/spec for DeformerS — specificity rises ~10–12pp while sensitivity drops ~6–9pp.
  MDD specificity goes from ~60% to 83–88%, and SAD specificity improves by ~9–10pp. The raw-EEG pipeline benefits
  from explicit DC removal that the spectrogram pipeline gets "for free" via the log transform. The CANE specificity
  (41–42%) remains the main bottleneck — CAR helps MDD/SAD but not CANE bias.
- **Decision guidance:** Keep the preprocessing change. It does not hurt accuracy for CNNCatLSTM (−0.83pp, within
  noise) and produces a meaningfully more balanced model for DeformerS without losing accuracy. More importantly,
  harmonising the pipeline across all four datasets is scientifically correct: models trained on combined data
  should not receive differently-normalised inputs per dataset.
- Command: `CUDA_VISIBLE_DEVICES=2 EEG_DATA_DIR=/home/xticha09 python main.py train --channel all --batch-size 64 --dataset all --model DeformerS --condition ec+eo --n-folds 10 --lr 1e-3 --dropout 0.5 --weight-decay 1e-3 --val-every 1 --checkpoint-dir=experiments/all3-028-all-2class-DeformerS-head-10fold_dzf`

---

### LGGNet experiments (027 series)

LGGNet (TNNLS 2023) introduces a local-global graph approach: multi-scale temporal CNN → local
graph filter (mean-pools channels into anatomical brain regions) → global GCN with learnable
adjacency mask. Two variants: LGGNet (~1.17M params) and LGGNetS (~585K, matches DeformerS).

Two bugs were found and fixed in `thesis/lggnet.py` (2026-04-21):
1. `Aggregator._get_idx` returned end indices instead of start indices; `forward()` rewritten
   with `x.narrow()` matching the reference implementation exactly.
2. `PowerLayer.forward` added `.clamp(min=1e-6)` before `torch.log` to prevent `-inf` from
   near-zero pooled power values.

**Rationale for lr=1e-3, dropout=0.5:**
Paper defaults. Validated by the DeformerS-head experiment: raw-EEG models fail at lr=1e-4 but
recover at lr=1e-3. The local filter weight (230K params) and GCN (922K params) are prone to
overfit, so dropout=0.5 matches the paper's regularization strategy.

**Graph topologies for 8 channels (Fp1,Fp2,T7,T8,C3,C4,Cz,Oz):**
- `hemisphere` (default, 3 regions): Left {Fp1,T7,C3} / Right {Fp2,T8,C4} / Midline {Cz,Oz}
- `frontal` (4 regions): {Fp1,Fp2} / {T7,T8} / {C3,C4,Cz} / {Oz}

#### all3-027-all-binary-lggnet

- Model: LGGNet (~1.17M params), binary, 8-channel, hemisphere graph, ec+eo, **6-fold**
- Hyperparams: lr=1e-3, dropout=0.5, wd=1e-4, batch=64, pool=32 (250 Hz scaled from paper's 16@128 Hz)
- Purpose: core LGGNet baseline with paper-recommended settings (post-bugfix re-run of 026);
  establishes whether graph-based EEG modelling outperforms CNN-LSTM baselines (V2Attn 79.10%)
- Command: `CUDA_VISIBLE_DEVICES=0 EEG_DATA_DIR=/home/xticha09 python main.py train --channel all --batch-size 64 --dataset all --model LGGNet --condition ec+eo --n-folds 6 --lr 1e-3 --dropout 0.5 --weight-decay 1e-4 --val-every 1 --checkpoint-dir=experiments/all3-027-all-binary-lggnet`

#### all3-027-all-binary-lggnet-s

- Model: LGGNetS (~585K params), binary, 8-channel, hemisphere graph, ec+eo, **10-fold**
- Hyperparams: lr=1e-3, dropout=0.5, wd=1e-4, batch=64, out_graph=32, pool=32, pool_step_rate=0.25
- Chunk accuracy: 73.35% ± 4.66% | Subject accuracy: 74.42% ± 6.99%
- Chunk sensitivity: 85.25% ± 8.30% | Chunk specificity: 54.85% ± 8.47%
- Train acc at best val: 89.68% ± 10.32% — overfit gap ~16pp
- Per-dataset chunk accuracy: CANE 65.47% (sens 72.37%, spec 30.14%), MDD 86.95% (sens 85.14%, spec 89.45%), SAD 58.91% (sens 53.04%, spec 66.14%)
- **Result: training works post-bugfix; 73.35% chunk accuracy, competitive with TSceptionS (70.80%) but
  significantly below V2Attn/V3+FL (~79%).** High sensitivity (85%) at the cost of very low specificity
  (55%), driven by CANE collapsing to near-random specificity (30%). MDD alone performs excellently
  (87%); the model struggles to generalize to CANE and SAD patterns.
- **Key finding:** The overfit gap (train 90% vs val 73%) is large, and early stopping fires very
  early (epochs 1–9 in most folds), suggesting the model memorizes MDD fast then overfits. Strong
  MDD bias: the GCN adjacency matrix may learn MDD-specific connectivity patterns that don't transfer.
- Command: `CUDA_VISIBLE_DEVICES=1 EEG_DATA_DIR=/home/xticha09 python main.py train --channel all --batch-size 64 --dataset all --model LGGNetS --condition ec+eo --n-folds 10 --lr 1e-3 --dropout 0.5 --weight-decay 1e-4 --val-every 1 --checkpoint-dir=experiments/all3-027-all-binary-lggnet-s`

#### all3-027-all-binary-lggnet-s-out16_byh

- Model: LGGNetS, binary, 8-channel, hemisphere graph, ec+eo, **10-fold**, out_graph=16
- Hyperparams: identical to all3-027-all-binary-lggnet-s except out_graph=16
- Chunk accuracy: 72.80% ± 3.96% | Subject accuracy: 74.38% ± 4.83%
- Chunk sensitivity: 81.92% ± 7.07% | Chunk specificity: 58.15% ± 10.04%
- Train acc at best val: 95.19% ± 5.99% — overfit gap ~22pp (worse than out_graph=32!)
- Per-dataset chunk accuracy: CANE 65.37% (sens 73.41%, spec 33.75%), MDD 86.45% (sens 85.26%, spec 88.03%), SAD 57.58% (sens 51.97%, spec 63.57%)
- **Result: nearly identical performance to out_graph=32.** Halving GCN output dim from 32→16 doesn't
  hurt accuracy materially (72.80% vs 73.35%) but slightly improves specificity (58.15% vs 54.85%) at
  the cost of sensitivity (81.92% vs 85.25%). The overfitting gap is *larger* despite smaller capacity
  (train 95% vs val 73%); the model still memorizes fast, just from a narrower GCN output.
- **Key finding:** out_graph is not the bottleneck — the GCN learns to discriminate in fewer dimensions
  equally well. The real problem is generalization across datasets, not GCN capacity.
- Command: `CUDA_VISIBLE_DEVICES=0 EEG_DATA_DIR=/home/xticha09 python main.py train --channel all --batch-size 64 --dataset all --model LGGNetS --condition ec+eo --n-folds 10 --lr 1e-3 --dropout 0.5 --weight-decay 1e-4 --val-every 1 --lggnet-out-graph 16 --checkpoint-dir=experiments/all3-027-all-binary-lggnet-s-out16_byh`

#### all3-027-all-binary-lggnet-frontal

- Model: LGGNet (~1.17M params), binary, 8-channel, **frontal graph**, ec+eo, **6-fold**
- Hyperparams: identical to all3-027-all-binary-lggnet
- Purpose: graph topology ablation — 4-region anatomical grouping (frontal/temporal/central/occipital)
  vs 3-region hemisphere grouping. Clinically motivated: frontal asymmetry (Fp1−Fp2) and
  temporal asymmetry (T7−T8) are established biomarkers for depression and anxiety.
- Command: `CUDA_VISIBLE_DEVICES=2 EEG_DATA_DIR=/home/xticha09 python main.py train --channel all --batch-size 64 --dataset all --model LGGNet --condition ec+eo --n-folds 6 --lr 1e-3 --dropout 0.5 --weight-decay 1e-4 --val-every 1 --lggnet-graph-type frontal --checkpoint-dir=experiments/all3-027-all-binary-lggnet-frontal`

#### all3-027-all-binary-lggnet-focal

- Model: LGGNet (~1.17M params), binary, 8-channel, hemisphere graph, ec+eo, **10-fold**, focal loss (gamma=2.0)
- Hyperparams: lr=1e-3, dropout=0.5, wd=1e-4, focal gamma=2.0
- Chunk accuracy: 74.55% ± 4.02% | Subject accuracy: 76.51% ± 7.19%
- Chunk sensitivity: 85.79% ± 5.33% | Chunk specificity: 56.72% ± 8.43%
- Subject sensitivity: 90.63% ± 8.76% | Subject specificity: 55.27% ± 11.89%
- Train acc at best val: 88.28% ± 13.16% — high variance, some folds stop at epoch 1-3
- Per-dataset chunk accuracy: CANE 68.65%, MDD 86.83%, SAD 58.20%
- CANE specificity: 38.94% (low but best for LGGNet series so far)
- **Best LGGNet result to date.** Full LGGNet (previously collapsed in 026 without FL) now trains
  successfully with focal loss. +1.20pp chunk vs LGGNetS without FL; subject accuracy +2.09pp.
- Focal loss pattern observed previously (sens↓, spec↑) does NOT appear here — sensitivity stays
  high (85.79%) while specificity remains low (56.72%). Model still heavily biased toward pathological.
- High training variance (train acc 62–99%): folds 6,7 stop at epoch 1 with train acc ~62–65%,
  while folds 2,4,9 overfit to 99%+ before stopping. Points to LR sensitivity.
- vs LGGNetS (no focal, 027-s): +1.20pp chunk (74.55% vs 73.35%), +2.09pp subject (76.51% vs 74.42%)
- vs best 8-ch binary (CNNAttn 027, 79.10%): −4.55pp chunk — still behind CNN-LSTM baseline
- Command: `CUDA_VISIBLE_DEVICES=3 EEG_DATA_DIR=/home/xticha09 python main.py train --channel all --batch-size 64 --dataset all --model LGGNet --condition ec+eo --n-folds 10 --lr 1e-3 --dropout 0.5 --weight-decay 1e-4 --val-every 1 --focal-loss --checkpoint-dir=experiments/all3-027-all-binary-lggnet-focal`

#### all3-027-all-binary-lggnet-s-focal-5e-4

- Model: LGGNetS (~585K params), binary, 8-channel, hemisphere graph, ec+eo, **8-fold**, focal loss (gamma=2.0)
- Hyperparams: lr=1e-3, dropout=0.5, wd=**5e-4** (5× higher than baseline), focal gamma=2.0
- Chunk accuracy: 72.40% ± 3.23% | Subject accuracy: 72.44% ± 3.76%
- Chunk sensitivity: 79.27% ± 6.08% | Chunk specificity: 61.81% ± 10.18%
- Subject sensitivity: 80.45% ± 6.03% | Subject specificity: 60.43% ± 11.41%
- Train acc at best val: 94.08% ± 12.53% — still overfit but lower than lggnet-focal's 88%
- Per-dataset chunk accuracy: CANE 64.09%, MDD 85.74%, SAD 58.64%
- CANE specificity: 33.99% — worse than lggnet-focal's 38.94%
- Higher wd (5e-4) vs baseline (1e-4): **−0.95pp chunk, −2.0pp subject** vs LGGNetS without FL (027-s).
  The increased regularisation suppresses sensitivity (79.27% vs 85.25%) while raising specificity
  (61.81% vs 54.85%), producing a more balanced but ultimately lower accuracy.
- 8-fold instead of 10-fold makes direct comparison noisier; result should be interpreted cautiously.
- vs LGGNetS wd=1e-4 (no focal): −0.95pp chunk, but +6.96pp specificity — the tradeoff is real.
- vs lggnet-focal (full, 10-fold): −2.15pp chunk, −4.07pp subject — larger model + 1e-4 wd wins.
- **Conclusion:** wd=5e-4 is too strong for LGGNetS with FL. The baseline wd=1e-4 should be paired
  with FL instead to test whether FL alone can fix specificity without over-penalising the weights.
- Command: `EEG_DATA_DIR=/home/xticha09 python main.py train --channel all --batch-size 64 --dataset all --model LGGNetS --condition ec+eo --n-folds 8 --lr 1e-3 --dropout 0.5 --weight-decay 5e-4 --val-every 1 --focal-loss --checkpoint-dir=experiments/all3-027-all-binary-lggnet-s-focal-5e-4`

#### all3-027-all-4class-lggnet-focal

- Model: LGGNet (~1.17M params), **4-class**, 8-channel, hemisphere graph, ec+eo, **6-fold**, focal loss
- Hyperparams: lr=1e-3, dropout=0.5, wd=1e-4, focal gamma=2.0
- Purpose: core 4-class LGGNet experiment. Key scientific question: can learning inter-region
  functional connectivity (via the learnable global adjacency) help distinguish the four classes
  where CNNs plateau at ~66%? The global adjacency is potentially the most expressive part of
  the architecture for class-specific connectivity patterns.
- Command: `CUDA_VISIBLE_DEVICES=0 EEG_DATA_DIR=/home/xticha09 python main.py train --channel all --batch-size 64 --dataset all --model LGGNet --condition ec+eo --n-folds 6 --lr 1e-3 --dropout 0.5 --weight-decay 1e-4 --val-every 1 --class-mode 4 --focal-loss --checkpoint-dir=experiments/all3-027-all-4class-lggnet-focal`

### all3-026-all-binary-tsceptions-ch4_rpr

- Model: TSceptionS (~102 K params with 4s input), binary, 8-channel, ec+eo, **10-fold**
- Hyperparams: identical to all3-026-all-binary-tsception-s, except `--chunk-duration 4` and `--n-folds 10`
- Chunk accuracy: 71.52% ± 4.35% | Subject accuracy: 72.61% ± 8.00%
- Chunk sensitivity: 77.88% ± 5.95% | Chunk specificity: 61.30% ± 9.42%
- Per-dataset chunk accuracy: CANE 62.75%, MDD 84.72%, SAD 61.91%
- Train acc at best val: 95.50% ± 6.73% — overfit gap reduced to ~24pp (from ~27pp at 10s)
- **Purpose:** closest-to-paper baseline. The original TSception paper split DEAP trials into 4s segments
  downsampled to 128 Hz (512 samples). This run uses 4s × 250 Hz = **1000 samples** — not resampled, but
  comparable to the paper's second dataset (MAHNOB-HCI, 256 Hz). FC input shrinks from 8172 → 3114 features,
  model from ~264 K → ~102 K params.
- **Result: marginal improvement.** +0.7pp chunk vs 10s TSceptionS; overfit gap 24pp vs 27pp. Cutting the
  flat feature 2.6× moved the needle only 0.7pp — the architecture plateaus regardless of input length.
  CANE specificity remains at ~30%, confirming this is a task-fit problem, not an input-size problem.
- vs CNNAttn (79.10%): −7.6pp. Conclusion: TSception does not transfer competitively to this task.
- Command: `CUDA_VISIBLE_DEVICES=1 EEG_DATA_DIR=/home/xticha09 python main.py train --channel all --batch-size 128 --dataset all --model TSceptionS --condition ec+eo --n-folds 10 --lr 1e-3 --dropout 0.3 --weight-decay 0 --l1-lambda 1e-6 --val-every 1 --epochs 200 --chunk-duration 4 --checkpoint-dir=experiments/all3-026-all-binary-tsceptions-ch4_rpr`

### all3-026-all-binary-tsception-s

- Model: TSceptionS (~264K params, hidden=32), binary, 8-channel, ec+eo, **6-fold**
- Hyperparams: identical to all3-026-all-binary-tsception
- Chunk accuracy: 70.80% ± 4.60% | Subject accuracy: 72.93% ± 5.69%
- Chunk sensitivity: 78.08% ± 6.95% | Chunk specificity: 59.06% ± 5.18%
- Per-dataset chunk accuracy: CANE 62.65%, MDD 83.46%, SAD 60.93%
- CANE specificity: 31.57% — still low but +7.78pp over TSception
- Train acc at best val: 97.71% ± 4.07% — still heavily overfit but less so than 1.05M model
- **TSceptionS beats TSception** (+1.96pp chunk, +1.93pp subject): forcing hidden=32 prevents FC from memorising
  the training set as effectively; the inception+spatial feature extractor is the same in both models
- 99% of parameters still in FC (261,504 of 264,281); same structural problem, just smaller
- vs CNNAttn (79.10%): −8.30pp. Better than TSception but still well below CNN-LSTM baseline
- Command: `CUDA_VISIBLE_DEVICES=1 EEG_DATA_DIR=/home/xticha09 python main.py train --channel all --batch-size 128 --dataset all --model TSceptionS --condition ec+eo --n-folds 6 --lr 1e-3 --dropout 0.3 --weight-decay 0 --l1-lambda 1e-6 --val-every 1 --epochs 200 --checkpoint-dir=experiments/all3-026-all-binary-tsception-s`

### all3-026-all-binary-tsception

- Model: TSception (~1.05M params, hidden=128), binary, 8-channel, ec+eo, **6-fold**, paper-exact hyperparams
- Hyperparams: lr=1e-3, dropout=0.3, wd=0, l1-lambda=1e-6, batch=128, epochs=200
- Chunk accuracy: 68.84% ± 5.65% | Subject accuracy: 71.00% ± 9.11%
- Chunk sensitivity: 77.82% ± 7.77% | Chunk specificity: 55.01% ± 4.67%
- Per-dataset chunk accuracy: CANE 59.70%, MDD 82.08%, SAD 60.93%
- **CANE specificity: 23.79%** — model predicts nearly all healthy CANE subjects as pathological
- Train acc at best val: 99.13% ± 1.32% — catastrophic overfitting, ~30pp gap with val
- Root cause: **99.7% of model parameters are in the FC layer** (8172×128 = 1,046,016 of 1,049,081 total).
  Original paper used 4ch × 1024-sample inputs → FC input ~3200. Our 8ch × 2500-sample input grows it to 8172
  (2.5×), making the FC layer the entire model rather than a classifier on learned features
- vs CNNAttn (79.10%): −10.26pp. FC-dominated architecture cannot generalise
- Command: `CUDA_VISIBLE_DEVICES=0 EEG_DATA_DIR=/home/xticha09 python main.py train --channel all --batch-size 128 --dataset all --model TSception --condition ec+eo --n-folds 6 --lr 1e-3 --dropout 0.3 --weight-decay 0 --l1-lambda 1e-6 --val-every 1 --epochs 200 --checkpoint-dir=experiments/all3-026-all-binary-tsception`

### all3-026-all-binary-lggnet-s

- Model: LGGNetS (~585K params), binary, 8-channel, hemisphere graph, ec+eo, **6-fold**
- **Result: identical failure to all3-026-all-binary-lggnet** — same NaN loss pattern, same bugs
- Exact same per-fold accuracies as LGGNet (39.21% chunk, 40.61% subject, all predict healthy):
  both models fail before learning anything, so num_T 64 vs 32 makes no observable difference
- Fix applied: same as all3-026-all-binary-lggnet.
- Command: `CUDA_VISIBLE_DEVICES=1 EEG_DATA_DIR=/home/xticha09 python main.py train --channel all --batch-size 64 --dataset all --model LGGNetS --condition ec+eo --n-folds 6 --lr 1e-3 --dropout 0.5 --weight-decay 1e-4 --val-every 1 --checkpoint-dir=experiments/all3-026-all-binary-lggnet-s`

### all3-026-all-binary-lggnet

- Model: LGGNet (~1.17M params), binary, 8-channel, hemisphere graph, ec+eo, **6-fold**
- Hyperparams: lr=1e-3, dropout=0.5, wd=1e-4, batch=64
- **Result: complete failure — loss=NaN every epoch, sensitivity=0%, specificity=100% all folds**
- Root cause 1: **Aggregator `_get_idx` bug** — returned `idx_[1:]` = `[3,6,8]` (end indices) but
  loop used them as start indices, so the Left hemisphere region (channels 0-2) was never aggregated,
  and the last slice `x[:, 8:, :]` was an empty tensor. Mean of empty tensor → NaN → propagates
  to all subsequent ops and loss.
- Root cause 2: **PowerLayer `log(0)` instability** — `torch.log(avg_pool(x²))` without a clamp
  produces `-inf` for near-zero power windows (common after bandpass/notch filtering in CANE data).
- Fix applied (2026-04-21): `thesis/lggnet.py` updated — `_get_idx` corrected to return start
  indices `[0, 3, 6]` and `forward` rewrites to use `x.narrow()`; PowerLayer adds `.clamp(min=1e-6)`.
- Re-run as all3-027-all-binary-lggnet with fixed code.
- Command: `CUDA_VISIBLE_DEVICES=0 EEG_DATA_DIR=/home/xticha09 python main.py train --channel all --batch-size 64 --dataset all --model LGGNet --condition ec+eo --n-folds 6 --lr 1e-3 --dropout 0.5 --weight-decay 1e-4 --val-every 1 --checkpoint-dir=experiments/all3-026-all-binary-lggnet`

### TSception experiments (026 series) — planned (remaining)

TSception (IEEE Trans. Affective Computing 2022) introduces multi-scale temporal inception
convolutions (kernels at 0.5/0.25/0.125 × fs) concatenated along the time axis, followed by
asymmetric spatial convolutions (global channel kernel + hemisphere-stride kernel) designed to
capture left–right EEG asymmetry. Unlike CNN-LSTM models, it uses a flat→FC head (no recurrence),
making it particularly fast and easy to regularise.

Two variants:
- **TSception** (~1.05M params, hidden=128): paper's original configuration; comparable to CNNCatLSTM
- **TSceptionS** (~264K params, hidden=32): paper's own cross-dataset recommendation; comparable to SmallerAll

**Rationale for lr=1e-3, dropout=0.3, wd=0, l1=1e-6, batch=128:**
Paper's exact Train.py defaults. L1 regularisation (lambda=1e-6) replaces the project's standard
L2 weight decay — the paper adds `lambda × sum(|w|)` directly to the CE loss. Batch=128 follows
the paper; larger batches may help the flat FC head converge. The raw-EEG models that failed at
lr=1e-4 (DeformerS) recovered at lr=1e-3, consistent with the paper's choice.

**Key questions:**
1. Does focal loss improve specificity for TSception as it did for all CNN-LSTM models?
2. Can TSception handle 4-class beyond the ~66% CNN-LSTM plateau?

#### all3-026-all-binary-tsception-focal

- Model: TSception (~1.05M params), binary, 8-channel, ec+eo, **6-fold**, focal loss (gamma=2.0)
- Hyperparams: lr=1e-3, dropout=0.3, wd=0, l1-lambda=1e-6, batch=128, epochs=200
- Purpose: focal loss ablation — FL consistently shifted sensitivity/specificity balance in
  CNN-LSTM experiments (+5–7pp specificity at cost of ~1pp sensitivity). Tests if the same
  pattern holds for TSception's flat-head architecture
- Command: `CUDA_VISIBLE_DEVICES=2 EEG_DATA_DIR=/home/xticha09 python main.py train --channel all --batch-size 128 --dataset all --model TSception --condition ec+eo --n-folds 6 --lr 1e-3 --dropout 0.3 --weight-decay 0 --l1-lambda 1e-6 --val-every 1 --epochs 200 --focal-loss --checkpoint-dir=experiments/all3-026-all-binary-tsception-focal`

#### all3-026-all-binary-tsception-s-focal

- Model: TSceptionS (~264K params), binary, 8-channel, ec+eo, **6-fold**, focal loss (gamma=2.0)
- Hyperparams: identical to all3-026-all-binary-tsception-focal
- Purpose: small model + FL — can the 264K model with focal loss close the gap to the 1.05M model?
  If TSceptionS+FL ≈ TSception+FL it suggests the hidden-layer bottleneck (32 vs 128) is not the
  limiting factor, and the inception+spatial feature extractor alone carries the discriminative power
- Command: `CUDA_VISIBLE_DEVICES=3 EEG_DATA_DIR=/home/xticha09 python main.py train --channel all --batch-size 128 --dataset all --model TSceptionS --condition ec+eo --n-folds 6 --lr 1e-3 --dropout 0.3 --weight-decay 0 --l1-lambda 1e-6 --val-every 1 --epochs 200 --focal-loss --checkpoint-dir=experiments/all3-026-all-binary-tsception-s-focal`

#### all3-026-all-4class-tsception-focal

- Model: TSception (~1.05M params), **4-class**, 8-channel, ec+eo, **6-fold**, focal loss (gamma=2.0)
- Hyperparams: lr=1e-3, dropout=0.3, wd=0, l1-lambda=1e-6, batch=128, epochs=200
- Purpose: 4-class TSception. The hemisphere spatial kernel has a clinically motivated hypothesis:
  left–right asymmetry patterns differ across anxiety and depression, so the asymmetric spatial
  branch may better separate those two classes than symmetric Conv3D/attention approaches.
  Prior 4-class best is V2Attn at 65.91%; comorbid class recall is the main bottleneck
- Command: `CUDA_VISIBLE_DEVICES=0 EEG_DATA_DIR=/home/xticha09 python main.py train --channel all --batch-size 128 --dataset all --model TSception --condition ec+eo --n-folds 6 --lr 1e-3 --dropout 0.3 --weight-decay 0 --l1-lambda 1e-6 --val-every 1 --epochs 200 --focal-loss --class-mode 4 --checkpoint-dir=experiments/all3-026-all-4class-tsception-focal`

### all3-024-all-2class-DeformerS-head

- Model: DeformerS with modified head (likely different classifier layers), binary, 8-channel, ec+eo, **6-fold**, lr=1e-3, dropout=0.5, wd=1e-3
- Chunk accuracy: 75.71% ± 4.30% | Subject accuracy: 74.13% ± 7.91%
- Chunk sensitivity: 85.79% ± 6.29% | Chunk specificity: 59.59% ± 14.57%
- Per-dataset chunk accuracy: CANE 69.95%, MDD 87.58%, SAD 58.17%
- Train acc at best val: 91.35% ± 6.19% — still overfit but at higher LR the model finds usable solutions faster
- Best epochs: early (1-5 for 4 folds, 19 and 30 for 2 folds) — aggressive LR+WD pushes early convergence
- vs DeformerS-30drop (lr=1e-4): **+10.86pp chunk** — dramatically better. The key was lr=1e-3 not the head modification.
- vs CNNAttn best (79.10%): −3.39pp. DeformerS still trails CNN-LSTM baselines.
- SAD still at 58%, same bottleneck as all other models

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

### all3-024-all-2class-V4-weighted-sampler

- Model: AllTransformerV4 ((channel×time) token Transformer), binary, 8-channel, ec+eo, **6-fold**, lr=1e-4, dropout=0.1, wd=1e-4
- Chunk accuracy: 76.53% ± 2.48% | Subject accuracy: 74.27% ± 3.00%
- Chunk sensitivity: 87.22% ± 7.80% | Chunk specificity: 59.53% ± 10.55%
- Per-dataset chunk accuracy: CANE 70.78%, MDD 87.27%, SAD 63.67%
- Train acc at best val: 73.77% ± 10.51% — high variance fold-to-fold (range 55-88%)
- Best epochs: very inconsistent (1, 10, 42, 45, 2, 12) — fold 3 stops at epoch 1, fold 4 at epoch 45; architecture is sensitive to fold composition
- vs CNNAttn 020 (79.10%): −2.57pp. V4 (token transformer) underperforms the CNN+LSTM baseline; likely under-regularized
- Command: `CUDA_VISIBLE_DEVICES=2 EEG_DATA_DIR=/home/xticha09 python main.py train --channel all --batch-size 64 --dataset all --model AllTransformerV4 --condition ec+eo --n-folds 6 --dropout 0.1 --class-mode 2 --lr 1e-4 --weight-decay 1e-4 --val-every 1 --checkpoint-dir=experiments/all3-024-all-2class-V4-weighted-sampler`

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

### all3-023-all-2class-DeformerS-weighted-sampler-50drop

- Model: DeformerS, binary, 8-channel, ec+eo, 6-fold, lr=1e-4, dropout=0.5, wd=1e-4
- Chunk accuracy: 60.78% ± 13.43% | Subject accuracy: 62.10% ± 14.38%
- Chunk sensitivity: 55.63% ± 34.53% | Chunk specificity: 71.25% ± 25.96%
- Train acc at best val: 89.69% ± 12.16% — still massively overfit even with heavy dropout
- Two folds completely fail (chunk ≈ 39-46%) — model collapses to majority-class prediction
- **Conclusion**: DeformerS cannot be fixed with dropout alone at standard LR. Architecture doesn't regularize well on ~8000 training chunks.

### all3-023-all-2class-DeformerS-weighted-sampler-30drop

- Model: DeformerS (~588K params), binary, 8-channel, ec+eo, 6-fold, lr=1e-4, dropout=0.3, wd=1e-4
- Chunk accuracy: 64.85% ± 8.27% | Subject accuracy: 65.10% ± 8.78%
- Chunk sensitivity: 66.87% ± 24.36% | Chunk specificity: 63.15% ± 17.13%
- Per-dataset chunk accuracy: CANE 56.68%, MDD 75.21%, SAD 60.32%
- Train acc at best val: 96.21% ± 2.65% — **extreme overfitting** despite being smaller model
- Best epochs: wildly variable (6, 6, 93, 9, 8, 8) — fold 3 runs to epoch 93 suggesting lucky initialization
- DeformerS at standard lr=1e-4 fails dramatically; raw-EEG approach needs fundamentally different regularization

### all3-023-all-2class-Deformer-weighted-sampler-50-dropout

- Model: Deformer (full), binary, 8-channel, ec+eo, 6-fold, lr=1e-4, dropout=0.5, wd=1e-4
- Chunk accuracy: 63.83% ± 9.86% | Subject accuracy: 64.91% ± 7.93%
- Chunk sensitivity: 57.37% ± 31.28% | Chunk specificity: 71.07% ± 23.65%
- Per-dataset chunk accuracy: CANE 56.01%, MDD 78.53%, SAD 51.68%
- Train acc at best val: 93.29% ± 4.76% — dropout=0.5 reduces overfitting slightly but not enough
- High sensitivity variance (±31%) indicates model is threshold-unstable across folds
- vs d=0.1 (10-fold estimated ~74%): much worse. Excessive dropout hurts Deformer substantially.

### all3-023-all-2class-Deformer-weighted-sampler

- Model: Deformer (full, ~5.4M params, 21MB checkpoints), binary, 8-channel, ec+eo, **10-fold**, lr=1e-4, dropout=0.1
- No results.txt generated (training likely completed without writing summary)
- Per-fold best chunk acc from log: fold 1≈0.77, fold 2≈0.76, fold 3≈0.77, fold 4≈0.66, fold 5≈0.75, fold 6≈0.73, fold 7≈0.73, fold 8=0.77, fold 9=0.65, fold 10=0.80 → estimated mean ~74%
- Massive overfitting: train acc reaches 97-99% by epoch 3-5; best val occurs at epoch 2-18 then degrades sharply
- Val loss spiky and increasing from epoch 5+; model memorizes training set almost immediately
- Pattern: 10-fold folds with fewer pathological samples (fold 4, 9) collapse to ~65%; lucky folds reach ~80%
- **Conclusion**: Full Deformer is far too large for this dataset (~330 subjects). Not competitive with CNNAttn/V3 despite more parameters.

### all3-022-all-binary-smallerAllV3-focal

- Model: CNNCatLSTM (4b, concat) — **first run of the concat architecture**; 021 turned out to use the old 3a design
- Config: binary, 8-channel, ec+eo, 10-fold, focal loss (gamma=2.0), dropout=0.1, rnn_hidden=100, chan_d_model=128
- Chunk accuracy: 78.56% ± 3.54% | Subject accuracy: 79.39% ± 6.35%
- Chunk sensitivity: 86.68% ± 6.10% | Chunk specificity: 66.56% ± 9.59% | Balanced acc: 76.62%
- Per-dataset chunk accuracy: CANE 72.19%, MDD 89.62%, SAD 66.49%
- vs 021 (3a design): −0.28pp chunk, within noise. Different architecture so not a clean comparison.
- Loss curves: epoch-1/2 best in ~3 folds, train/val divergence from epoch 3+. Overfitting pattern consistent with 021 despite different architecture.
- Command: `CUDA_VISIBLE_DEVICES=2 EEG_DATA_DIR=/home/xticha09 python main.py train --channel all --batch-size 64 --dataset all --model CNNCatLSTM --condition ec+eo --n-folds 10 --dropout 0.1 --weight-decay 1e-4 --val-every 1 --focal-loss --checkpoint-dir=experiments/all3-022-all-binary-smallerAllV3-focal`

### all3-021-all-binary-smallerAllV3-focal

- Model: CNNCatLSTM (4a, per-frame attention) — confirmed via `nhead=8, rnn_hidden=330` in training log
- Config: binary, 8-channel, ec+eo, 10-fold, focal loss (gamma=2.0), dropout=0.1, rnn_hidden=330, chan_d_model=128
- Chunk accuracy: 78.84% ± 6.39% | Subject accuracy: 80.22% ± 6.94%
- Chunk sensitivity: 83.74% ± 11.02% | Chunk specificity: 70.18% ± 13.38% | Balanced acc: 76.96%
- Per-dataset chunk accuracy: CANE 72.21%, MDD 88.47%, SAD 73.02%
- Best balanced accuracy of all experiments (76.96%). SAD accuracy improved significantly vs 020 (+4.94pp).
- Loss curves: epoch-1/2 best in folds 1, 5, 10. Val loss spiky throughout. Train loss decreasing steadily. Classic overfitting from oversized chan_proj layer (~885K params of ~1M total).
- Key takeaway: V3 4a (per-frame attention) matches V2Attn in accuracy. The 4b concat design was not tested until 022.
- Command: `CUDA_VISIBLE_DEVICES=2 EEG_DATA_DIR=/home/xticha09 python main.py train --channel all --batch-size 64 --dataset all --model CNNCatLSTM --condition ec+eo --n-folds 10 --dropout 0.1 --weight-decay 1e-4 --val-every 1 --focal-loss --checkpoint-dir=experiments/all3-021-all-binary-smallerAllV3-focal`

### all3-021-all-binary-smallerAllV3-focal_hsf

- Model: CNNCatLSTM (3a design — rnn_hidden=330, chan_d_model=128) — V3 with oversized LSTM
- Config: binary, 8-channel, EC only (4217 chunks vs 8444 for ec+eo), focal loss, dropout=0.2
- Chunk accuracy: 78.68% ± 4.98% | Subject accuracy: 79.24% ± 8.41%
- Chunk sensitivity: 85.15% ± 8.56% | Chunk specificity: 68.62% ± 13.35% | Balanced acc: 76.89%
- Note: trained on EC condition only (half the data). High variance in subject accuracy (±8.4%) reflects smaller val sets per fold.
- vs 021 (V3 3b, ec+eo): nearly identical despite 3a design and half the data. Confirms the concat approach (3b) offers no systematic advantage once overfitting dominates both.
- Command: `CUDA_VISIBLE_DEVICES=2 EEG_DATA_DIR=/home/xticha09 python main.py train --channel all --batch-size 64 --dataset all --model CNNCatLSTM --condition ec --n-folds 10 --dropout 0.2 --weight-decay 1e-4 --val-every 1 --focal-loss --checkpoint-dir=experiments/all3-021-all-binary-smallerAllV3-focal_hsf`

### all3-020-all-binary-smallerAllV2Attn-weighted-sampler

- Model: CNNAttn (per-channel shared CNN + cross-channel self-attention + temporal attention), binary, 8-channel, ec+eo, 10-fold, weighted sampler
- Chunk accuracy: 79.10% ± 4.90% | Subject accuracy: 79.83% ± 5.98%
- Chunk sensitivity: 89.43% ± 6.78% | Chunk specificity: 63.42% ± 11.22%
- Per-dataset chunk accuracy: CANE 72.55%, MDD 90.20%, SAD 68.08%
- vs SmallerAllAttn (018, 8ch binary): no direct baseline but matches in-ear SmallerAttn (78.26% chunk) — 8 channels provide no gain over 1 channel
- Command: `CUDA_VISIBLE_DEVICES=1 EEG_DATA_DIR=/home/xticha09 python main.py train   --channel all   --batch-size 64   --dataset all   --model CNNAttn --condition ec+eo   --n-folds 10   --dropout 0.1  --weight-decay 1e-4   --val-every 1   --checkpoint-dir=experiments/all3-020-all-binary-smallerAllV2Attn-weighted-sampler`

### all3-020-all-4class-smallerAllV2Attn-weighted-sampler

- Model: CNNAttn (per-channel shared CNN + cross-channel self-attention + temporal attention), 4-class, 8-channel, ec+eo, 10-fold, weighted sampler
- Chunk accuracy: 65.91% ± 6.63% | Subject accuracy: 68.49% ± 6.68%
- Chunk recall: Healthy 55.87%, Anxiety 42.18%, Depression 90.35%, Comorbid 79.63%
- Per-dataset chunk accuracy: CANE 46.30%, MDD 86.59%, SAD 69.14%
- vs SmallerAllAttn (018): +1.15pp chunk, +2.40pp subject — marginal improvement; CANE still at ~46%
- Architecture: per-channel CNN (shared weights) preserves channel identity longer than Conv3D, but cross-channel attention still provides no measurable gain over single-channel models
- Command: `CUDA_VISIBLE_DEVICES=2 EEG_DATA_DIR=/home/xticha09 python main.py train   --channel all   --batch-size 64   --dataset all   --model CNNAttn --condition ec+eo   --n-folds 10   --dropout 0.1  --class-mode 4 --weight-decay 1e-4   --val-every 1   --checkpoint-dir=experiments/all3-020-all-4class-smallerAllV2Attn-weighted-sampler`

### all3-020-all-binary-smallerAllV2Attn-focal-d3

- Model: CNNAttn, binary, 8-channel, ec+eo, 10-fold, focal loss, dropout=0.3
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
- Command: `systemd-run --user --scope -p CPUQuota=200% python main.py train   --channel all   --batch-size 64   --dataset all   --model SmallerAll   --condition ec+eo   --n-folds 6   --dropout 0.1   --weight-decay 1e-4   --val-every 1   --checkpoint-dir=experiments/both-014b-all-4class-smallerall --class-mode=4`
### all3-014b-inear-binary-smaller

- Command: `systemd-run --user --scope -p CPUQuota=200% python main.py train   --channel in-ear   --batch-size 64   --dataset all   --model Smaller   --condition ec+eo   --n-folds 6   --dropout 0.1   --weight-decay 1e-4   --val-every 1   --checkpoint-dir=experiments/both-014-inear-binary-smaller`
