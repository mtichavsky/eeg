# Experiments

## Summary Table: Comparing Experiments

| Experiment                                             | Dataset | Channel | Condition | Batch | Model    | Dropout | Chunk Acc (Min-Max)            | Subject Acc (Min-Max)      | Key Observation                                            |
|--------------------------------------------------------|---------|---------|-----------|-------|----------|---------|--------------------------------|----------------------------|------------------------------------------------------------|
| **mdd_006_fp1_ec**                                     | MDD     | Fp1     | EC        | 64    | Original | 0.5     | 91.3% ± 7.7% (75.6-98.6%)      | 94.0% ± 9.4% (75.0-100%)   |                                                            |
| mdd_006_t4_ec                                          | MDD     | T4      | EC        | 64    | Original | 0.5     | 86.5% ± 5.2% (75.6-91.0%)      | 88.6% ± 7.3% (75.0-100%)   |                                                            |
| **mdd_006_t3_ec+eo**                                   | MDD     | T3      | EC+EO     | 64    | Original | 0.4     | 88.3% ± 7.9% (79.1-100%)       | 89.1% ± 8.2% (79.0-100%)   |                                                            |
| **mdd_008_t3_ec+eo_smaller**                           | MDD     | T3      | EC+EO     | 64    | Smaller  | 0.1     | 86.4% ± 7.5% (77.8-98.5%)      | 88.2% ± 7.4% (78.9-100%)   | Behaves very similar considering it's ~50% smaller         |
| **mdd_006_t4_ec+eo**                                   | MDD     | T4      | EC+EO     | 64    | Original | 0.4     | 87.3% ± 8.1% (75.9-100%)       | 89.2% ± 6.3% (83.3-100%)   |                                                            |
| cane_006_fp1_ec (baseline)                             | CANE    | Fp1     | EC        | 64    | Original | 0.5     | 57.6%                          | 61.9%                      |                                                            |
| cane_008_t7_ec_og                                      | CANE    | T7      | EC        | 32    | Original | 0.8     | 48.0%                          | 48.8%                      | All 6 folds peaked at epoch 1 - complete learning failure  |
| cane_008_t7_ec_og_rnd                                  | CANE    | T7      | EC        | 32    | Original | 0.1     | 61.2%                          | 62.7%                      | 3/6 folds at epoch 1, high variance                        |
| cane_008_t7_ec_smaller                                 | CANE    | T7      | EC        | 32    | Smaller  | 0.5     | 61.0%                          | 62.7%                      | 2/6 folds peaked at epoch 1, some learning                 |
| **cane_008_t7_ec_smaller_lwa**                         | CANE    | T7      | EC        | 32    | Smaller  | 0.1     | **62.2% (53.6%-81.8%)**        | 65.1%                      | Better learning, but still 3/6 folds at epoch 1            |
| **Transfer Learning (009 series - freeze CNN only)**   |         |         |           |       |          |         |                                |                            |                                                            |
| cane_009_fp1_ec_transfer                               | CANE    | Fp1     | EC        | 64    | Transfer | 0.5     | 61.1% ± 12.7% (43.2-78.4%)     | 65.5% ± 18.0% (42.9-83.3%) | Best fold 78.4% chunk; high variance; 1/6 folds at epoch 1 |
| cane_009_t8_ec_transfer                                | CANE    | T8      | EC        | 64    | Transfer | 0.5     | 49.1% ± 6.8% (37.7-56.3%)      | 48.8% ± 10.0% (33.3-66.7%) | 3/6 folds at epoch 1 - T8 transfer worse than baseline     |
| cane_009_t8_ec_transfer_lowdrop                        | CANE    | T8      | EC        | 64    | Transfer | 0.1     | 51.7% ± 10.5% (35.5-65.9%)     | 54.4% ± 13.2% (33.3-66.7%) | 3/6 folds at epoch 1; low dropout didn't help T8           |
| **Transfer Learning (010 series - freeze CNN + LSTM)** |         |         |           |       |          |         |                                |                            |                                                            |
| cane_010_fp1_ec_transfer                               | CANE    | Fp1     | EC        | 64    | Transfer | 0.5     | 60.0% ± 11.8% (36.4-72.1%)     | 59.9% ± 9.8% (42.9-66.7%)  | 2/6 folds at epoch 1; only classifier trainable            |
| cane_010_fp1_ec_transfer_lowdrop                       | CANE    | Fp1     | EC        | 64    | Transfer | 0.1     | **61.0% ± 13.7% (43.2-81.8%)** | 68.3% ± 19.2% (42.9-100%)  | **Best result**: Fold 3 achieved 81.8% chunk, 100% subject |

## 009+010: Transfer learning

Transfer learning provides marginal but consistent improvement (~3-4% mean chunk accuracy) over baseline CANE training,
with the best configuration being Fp1 transfer with frozen CNN+LSTM and low dropout (0.1), which achieved one fold with
81.8% chunk accuracy. However, the fundamental challenge of high cross-subject variability in the CANE dataset remains
unsolved.

T8 completely failed on the transfer.

Suggestions:
- Data augmentation for problematic subjects - Folds 1 and 5 consistently underperform. Consider time-domain
   augmentation (jittering, scaling) to increase effective training data.
- Try different pretrained folds - All experiments use fold_1_best.pth. Try transfer from fold 3 or fold 4 which may
   have learned more generalizable features.
- Increase classifier capacity - With frozen CNN+LSTM (only 8,610 trainable params), consider adding another dense layer
  or increasing hidden dimensions in the classifier head.

Parameter breakdown:
| Layer                        | Params  |
|------------------------------|---------|
| CNN (conv1 + conv2)          | 57,696  |
| LSTM                         | 732,000 |
| Classifier (fc1 + fc2 + out) | 8,610   |
| Total                        | 798,306 |

Freezing options:
- --freeze-cnn: 740,610 trainable (93%) - LSTM + classifier
- --freeze-lstm: 8,610 trainable (1%) - classifier only (implies --freeze-cnn)
- [ ] TODO experiment with only freezing lstm without freezing CNN

## Key Insights from 008 Experiments

- Model size reduction made minimal difference for CANE
  - I think I can try tweak some params for this setup: **cane_008_t7_ec_smaller_lwa**, but the loss curves still look bad
  - [ ] **TODO**: try what Fp1, T7 does with 64 batch, maybe no dropout and ec+eo?
- For MDD, making the model smaller and lowering dropout leads to very similar results for a single channel.
  - meaning, I can easily save 50% on compute without much difference in the service provided

 
## Key Insights from 006 Experiments

### 1. Batch Size Impact on MDD

- **fp1_006_mdd_ec (batch 64)**: 94.0% vs **fp1_004_mdd_ec (batch 32)**: 90.5%
- ~3.5% improvement, though variance increased (±9.4% vs ±8.4%)
- **Conclusion**: Larger batches provide more stable gradients and better learning on limited data

### 2. Fp1 Channel Confirmed as Best for Depression

- **Fp1**: 94.0% vs **T4**: 88.6%
- **Reason**: Frontal asymmetry is the key biomarker for depression — T4 (right temporal) misses this critical signal

### 3. CANE Dataset Performance Issues

- **Fp1 channel performance**: 61.9% (essentially random for 2-class classification)
- **Critical issue**: In `cane_006_t8_ec+eo`, 5 out of 6 folds peaked at epoch 1:
  - Fold 1: best at epoch 1
  - Fold 2: best at epoch 1
  - Fold 3: best at epoch 1
  - Fold 4: best at epoch 1
  - Fold 5: best at epoch 1
  - Fold 6: best at epoch 12 ← only fold that showed learning
- **Conclusion**: Model cannot learn from this data with current architecture

### 4. The "Fold 6 Problem" in MDD

Both MDD experiments show Fold 6 performing significantly worse:
- **mdd_006_fp1_ec Fold 6**: 75.0% subject accuracy (vs 100% in other folds)
- **mdd_006_t4_ec Fold 6**: 75.0% subject accuracy
- **Hypothesis**: 1-2 subjects in that fold are fundamentally hard to classify

### Priorities Moving Forward

**Priority 1: MDD results are satisfactory**
- 94% subject accuracy with Fp1/EC/batch64 is solid for ~60 subjects
- High variance is inherent to small-sample EEG studies

**Priority 2: Re-evaluate CANE approach**
- Current model architecture cannot learn anxiety patterns with 39 subjects
- Consider alternative approaches or architectures 

## 009 + 010 

```bash
poetry run python main.py train --skip-ica \
--channel Fp1 \
--batch-size 64 \
--checkpoint-dir=experiments/cane_010_fp1_ec_transfer \
--dataset cane \
--condition ec \
--n-folds 6 \
--dropout 0.5 \
--weight-decay 1e-4 \
--val-every 1 \
--pretrained-checkpoint experiments/mdd_006_fp1_ec/fold_1_best.pth \
--freeze-cnn \
--freeze-lstm
```


```bash
poetry run python main.py train --skip-ica \
--channel Fp1 \
--batch-size 64 \
--checkpoint-dir=experiments/cane_010_fp1_ec_transfer_lowdrop \
--dataset cane \
--condition ec \
--n-folds 6 \
--dropout 0.1 \
--weight-decay 1e-4 \
--val-every 1 \
--pretrained-checkpoint experiments/mdd_006_fp1_ec/fold_1_best.pth \
--freeze-cnn \
--freeze-lstm
```

```bash
poetry run python main.py train --skip-ica \
--channel Fp1 \
--batch-size 64 \
--checkpoint-dir=experiments/cane_009_fp1_ec_transfer \
--dataset cane \
--condition ec \
--n-folds 6 \
--dropout 0.5 \
--weight-decay 1e-4 \
--val-every 1 \
--pretrained-checkpoint experiments/mdd_006_fp1_ec/fold_1_best.pth \
--freeze-cnn
```


```bash
poetry run python main.py train --skip-ica \
--channel T8 \
--batch-size 64 \
--checkpoint-dir=experiments/cane_009_t8_ec_transfer \
--dataset cane \
--condition ec \
--n-folds 6 \
--dropout 0.5 \
--weight-decay 1e-4 \
--val-every 1 \
--pretrained-checkpoint experiments/mdd_006_t4_ec/fold_1_best.pth \
--freeze-cnn
```


```bash
poetry run python main.py train --skip-ica \
--channel T8 \
--batch-size 64 \
--checkpoint-dir=experiments/cane_009_t8_ec_transfer_lowdrop \
--dataset cane \
--condition ec \
--n-folds 6 \
--dropout 0.1 \
--weight-decay 1e-4 \
--val-every 1 \
--pretrained-checkpoint experiments/mdd_006_t4_ec/fold_1_best.pth \
--freeze-cnn
```

## 006 for T4

```bash
python main.py train --skip-ica \
--channel T4 \
--batch-size 64 \
--checkpoint-dir=experiments/mdd_006_t4_ec+eo \
--dataset mdd \
--condition ec+eo \
--n-folds 6 \
--dropout 0.4 \
--weight-decay 1e-4 \
--val-every 1
```

## How I was running 008 batch (Smaller model)

```bash
python main.py train --skip-ica \
--channel T7 \
--batch-size 32 \
--checkpoint-dir=experiments/cane_008_t7_ec_smaller \
--dataset cane \
--condition ec \
--n-folds 6 \
--dropout 0.5 \
--weight-decay 1e-4 \
--val-every 1 \
--model Smaller
```

- Output dir: `experiments/cane_008_t7_ec_smaller`

```bash
python main.py train --skip-ica \
--channel T7 \
--batch-size 32 \
--checkpoint-dir=experiments/cane_008_t7_ec_smaller \
--dataset cane \
--condition ec \
--n-folds 6 \
--dropout 0.1 \
--weight-decay 1e-4 \
--val-every 1 \
--model Smaller
```

- Output dir: `experiments/cane_008_t7_ec_smaller_lwa`

```bash
python main.py train --skip-ica \
--channel T7 \
--batch-size 32 \
--checkpoint-dir=experiments/cane_008_t7_ec_og \
--dataset cane \
--condition ec \
--n-folds 6 \
--dropout 0.5 \
--weight-decay 1e-4 \
--val-every 1
```

- Output dir: `experiments/cane_008_t7_ec_og_rnd`

```bash
python main.py train --skip-ica \
--channel T7 \
--batch-size 32 \
--checkpoint-dir=experiments/cane_008_t7_ec_og \
--dataset cane experiments/cane_008_t7_ec_og_rnd\
--condition ec \
--n-folds 6 \
--dropout 0.1 \
--weight-decay 1e-4 \
--val-every 1
```

- Output dir: `experiments/cane_008_t7_ec_og_rnd`


```bash
python main.py train --skip-ica \
--channel T3 \
--batch-size 64 \
--checkpoint-dir=experiments/mdd_008_t3_ec+eo_smaller \
--dataset mdd \
--model Smaller \
--condition ec+eo \
--n-folds 6 \
--dropout 0.1 \
--weight-decay 1e-4 \
--val-every 1
```

- Output dir: ![experiments/mdd_008_t3_ec+eo_smaller](experiments/mdd_008_t3_ec+eo_smaller)

## Detailed Experiment Results (006 Series)

### MDD Fp1 EC 006

```bash
python main.py train --skip-ica \
--channel Fp1 \
--batch-size 64 \
--checkpoint-dir=experiments/mdd_006_fp1_ec \
--dataset mdd \
--condition ec \
--n-folds 6 \
--dropout 0.5 \
--weight-decay 1e-4 \
--val-every 1
```

**Results:**
- SUBJECT Accuracy Mean: 0.9398 ± 0.0941, Min: 0.7500, Max: 1.0000
- CHUNK Accuracy Mean: 0.9125 ± 0.0771, Min: 0.7562, Max: 0.9862

![training](experiments/mdd_006_fp1_ec/loss_curves/all_folds_combined_loss_curves.png)

### MDD T4 EC 006

```bash
python main.py train --skip-ica \
--channel T4 \
--batch-size 64 \
--checkpoint-dir=experiments/mdd_006_t4_ec \
--dataset mdd \
--condition ec \
--n-folds 6 \
--dropout 0.5 \
--weight-decay 1e-4 \
--val-every 1
```

**Results:**
- SUBJECT Accuracy Mean: 0.8856 ± 0.0730, Min: 0.7500, Max: 1.0000
- CHUNK Accuracy Mean: 0.8652 ± 0.0518, Min: 0.7562, Max: 0.9101

![training](experiments/mdd_006_t4_ec/loss_curves/all_folds_combined_loss_curves.png)

### MDD T3 EC+EO 006

```bash
python main.py train --skip-ica \
--channel T3 \
--batch-size 64 \
--checkpoint-dir=experiments/mdd_006_t3_ec+eo \
--dataset mdd \
--condition ec+eo \
--n-folds 6 \
--dropout 0.4 \
--weight-decay 1e-4 \
--val-every 1
```

**Results:**
- SUBJECT Accuracy Mean: 0.8912 ± 0.0815, Min: 0.7895, Max: 1.0000
- CHUNK Accuracy Mean: 0.8828 ± 0.0785, Min: 0.7906, Max: 1.0000

![training](experiments/mdd_006_t3_ec+eo/loss_curves/all_folds_combined_loss_curves.png)

### CANE Fp1 EC 006

```bash
python main.py train --skip-ica \
--channel Fp1 \
--batch-size 64 \
--checkpoint-dir=experiments/cane_006_fp1_ec \
--dataset cane \
--condition ec \
--n-folds 6 \
--dropout 0.5 \
--weight-decay 1e-4 \
--val-every 1
```

**Results:**
- SUBJECT Accuracy Mean: 0.6190 ± 0.0858, Min: 0.5000, Max: 0.7143
- CHUNK Accuracy Mean: 0.5758 ± 0.0722, Min: 0.4645, Max: 0.6648

![training](experiments/cane_006_fp1_ec/loss_curves/all_folds_combined_loss_curves.png)

---

## Earlier Experiments (005 Series)

### MDD+CANE EC+EO Fp1 005

```bash
python main.py train --skip-ica \
--channel Fp1 \
--batch-size 32 \
--checkpoint-dir=experiments/both_fp1_ec+eo_005 \
--dataset both \
--condition ec+eo \
--n-folds 6 \
--dropout 0.6 \
--weight-decay 1e-4 \
--val-every 1
```

**Results:**
- CHUNK Accuracy Mean: 0.7390 ± 0.0794, Min: 0.5881, Max: 0.8328

![training](experiments/both_fp1_ec+eo_005/training_ec+eo_Fp1_noica_20251130_163158.log)

### MDD T3 EC+EO 005 (Lower Batch Size)

**Note:** TODO - try this with batch 128?

```bash
python main.py train --skip-ica \
--channel T3 \
--batch-size 16 \
--checkpoint-dir=experiments/mdd_t3_ec+eo_005 \
--dataset mdd \
--condition ec+eo \
--n-folds 6 \
--dropout 0.6 \
--weight-decay 1e-4 \
--val-every 1
```

**Results:** See [cv_results.txt](experiments/mdd_t3_ec+eo_005/cv_results.txt)

![loss curves](experiments/mdd_t3_ec+eo_005/loss_curves/all_folds_combined_loss_curves.png)

### CANE T7 EC+EO 005 (Batch Size 8)

**Observations:**
- Performance was significantly worse (see loss curves)
- Likely needs larger batch size and smaller dropout
- Combined metric in results gives inconsistent values - need separate metrics
- Loss curve plots appear broken

```bash
python main.py train --skip-ica \
--channel T7 \
--batch-size 8 \
--checkpoint-dir=experiments/cane_t7_ec+eo_005 \
--dataset cane \
--condition ec+eo \
--n-folds 6 \
--dropout 0.6 \
--weight-decay 1e-4 \
--val-every 1
```

![losses](experiments/cane_t7_ec+eo_005/loss_curves/all_folds_combined_loss_curves.png)

**Second attempt with adjusted parameters:**
```bash
python main.py train --skip-ica \
--channel T7 \
--batch-size 64 \
--checkpoint-dir=experiments/cane_t7_ec+eo_005_2nd \
--dataset cane \
--condition ec+eo \
--n-folds 6 \
--dropout 0.3 \
--weight-decay 1e-4 \
--val-every 1
```

---

## Archived Experiments (004 Series and Earlier)

### CANE T7 EC 005 (Batch Size 32)

```bash
python main.py train --skip-ica \
  --channel T7 \
  --batch-size 32 \
  --checkpoint-dir=experiments/t7_005_beta_cane_ec \
  --dataset cane \
  --condition ec \
  --n-folds 6 \
  --dropout 0.5 \
  --weight-decay 1e-4 \
  --val-every 1
```

![loss functions](experiments/t7_005_beta_cane_ec/loss_curves/all_folds_combined_loss_curves.png)

### MDD Fp1 EC+EO 005 Beta (Nov 26)

**Notes:**
- Labeled as "beta" experiment
- Directory: `experiments/fp1_005_mdd_ec`
- TODO: Re-run with current logic and consult analysis

```bash
python main.py train --skip-ica \
  --channel Fp1 \
  --batch-size 32 \
  --checkpoint-dir=experiments/fp1_005_mdd_ec \
  --dataset mdd \
  --condition ec+eo \
  --n-folds 6 \
  --dropout 0.5 \
  --weight-decay 1e-4 \
  --val-every 1
```

### MDD Fp1 EC 004 (Dropout and L2 Regularization)

```bash
python main.py train --skip-ica \
  --channel Fp1 \
  --batch-size 32 \
  --checkpoint-dir=experiments/fp1_004_mdd_ec \
  --dataset mdd \
  --condition ec \
  --n-folds 6 \
  --dropout 0.5 \
  --weight-decay 1e-4 \
  --val-every 1
```


![loss functions](experiments/fp1_004_mdd_ec/loss_curves/all_folds_combined_loss_curves.png)

---

### MDD Fp1 EC 003 (Nov 14)

```bash
python main.py train --skip-ica \
  --channel Fp1 \
  --batch-size 32 \
  --checkpoint-dir=checkpoints_fp1_003_mdd \
  --dataset mdd \
  --condition ec
```

**Directory:** `experiments/fp1_003_mdd_mqm`

![loss functions](experiments/fp1_003_mdd_mqm/loss_curves/all_folds_combined_loss_curves.png)

**Analysis:**

**Positive observations:**
- Training loss consistently decreases across all folds → model is learning
- Folds 3, 6, 7, 8, 9 show decent convergence with stable eval loss

**Concerning patterns:**
- **Significant overfitting** in folds 4, 5, and 10 (large train/eval loss gap)
- **High fold variance**: Eval loss ranges from 0.1-0.2 (some folds) to 0.5-0.8 (others)
  - Suggests either highly variable data splits or model sensitivity to subject composition
- **Eval loss instability**: Oscillations in folds 1, 4, 5, 8, 10
  - Possible causes: batch size issues, learning rate too high, limited/noisy validation data

**Implemented improvements:**
- Added regularization (L2 weight decay)
- Added spatial dropout (nn.Dropout2d)
- Considering reducing model size

**Key insight:**
Given subject-level splits, high fold variance is expected. EEG signals vary dramatically between individuals due to:
- Electrode impedance differences
- Skull thickness variations
- Baseline neural patterns
- Individual physiological manifestations of depression

This reframes the problem: inconsistent eval performance is less about traditional overfitting and more about the challenge of **cross-subject generalization** in EEG-based classification.

### CANE Fp1 EC 003 (Nov 14)

**Notes:**
- Dropout 0.5 doesn't work well for CANE dataset
- Using checkpoint directory: `checkpoints_fp1_003_cane_dqr` (to avoid overwriting)

```bash
python main.py train --skip-ica \
  --channel Fp1 \
  --batch-size 32 \
  --checkpoint-dir=checkpoints_fp1_003_cane \
  --dataset cane \
  --condition ec
```

**Second attempt with lower dropout:**
- Using checkpoint directory: `checkpoints_fp1_003_cane_xhv`
