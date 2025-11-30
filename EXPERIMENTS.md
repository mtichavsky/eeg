## Planned experiments

### MDD+CANE EC+EO Fp1

```bash
python main.py train --skip-ica \
--channel Fp1 \
--batch-size 32 \
--checkpoint-dir=experiments/both_fp1_ec+eo_005 \
--dataset both \
--condition ec+eo \
--n-folds 6 \
--dropout 0.5 \
--weight-decay 1e-4 \
--val-every 1
```

### MDD EC+EO T3 lower batch size

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

### CANE EC+EO T7 batch size=8

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

## Experiments overview:

### Nov 28

#### CANE EC dropout and L2 / T7

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

### Nov 26

Both EC+EO
I labeled this as beta
This is the experiment in experiments that's not mentioned here
python main.py train --skip-ica   --channel Fp1   --batch-size 32   --checkpoint-dir=experiments/fp1_005_mdd_ec   --dataset mdd   --condition ec+eo   --n-folds 6   --dropout 0.5   --weight-decay 1e-4   --val-every 1

I'd do it once again with my current logic, see what it does and consult with CLAUDE

#### MDD EC dropout and L2

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


### Nov 14

#### CANE dropout=0.5 doesn't really work, need to check why.

- /home/milan/Documents/diplomka/code/venv/bin/python code/main.py train --skip-ica --channel Fp1 --batch-size 32 --checkpoint-dir=checkpoints_fp1_003_cane --dataset cane --condition ec
- Using 'checkpoints_fp1_003_cane_dqr'

#### Pure MDD attempt

- `python main.py train --skip-ica --channel Fp1 --batch-size 32 --checkpoint-dir=checkpoints_fp1_003_mdd --dataset mdd --condition ec`
- `experiments/fp1_003_mdd_mqm` directory

![loss functions](experiments/fp1_003_mdd_mqm/loss_curves/all_folds_combined_loss_curves.png)

The good news: Training loss consistently decreases across all folds, showing the model is learning. 
Several folds (3, 6, 7, 8, 9) show decent convergence with eval loss stabilizing in a reasonable range.
 
The concerning patterns:
- Significant overfitting in multiple folds — particularly folds 4, 5, and 10 where the gap between train and eval loss 
  grows substantially. 
- High variance across folds — the eval loss behavior is quite inconsistent. Some folds stabilize around 0.1-0.2,
  others hover around 0.5-0.8. This suggests either the data splits have very different characteristics, or the model 
  is sensitive to which subjects/samples end up in which fold.
- Eval loss instability — lots of oscillation in the validation curves (folds 1, 4, 5, 8, 10), which could indicate 
  batch size issues, learning rate too high, or simply that some folds have limited/noisy validation data.

**Changes implemented based on this:** Regularization added, spatial dropout added, considering making model smaller to
avoid overfitting.

Given that you're doing subject-level splits, the high fold variance makes more sense now. EEG signals vary a lot
between individuals — electrode impedance, skull thickness, baseline neural patterns, how anxiety manifests
physiologically. Some subjects are just harder to classify than others.
This reframes the problem a bit. The inconsistent eval performance across folds might be less about overfitting in the
traditional sense and more about poor cross-subject generalization — a notoriously hard problem in EEG-based
classification.

#### CANE with lower dropout
-  python main.py train --skip-ica --channel Fp1 --batch-size 32 --checkpoint-dir=checkpoints_fp1_003_cane --dataset cane --condition ec
- Using 'checkpoints_fp1_003_cane_xhv' instead to avoid overwriting.
