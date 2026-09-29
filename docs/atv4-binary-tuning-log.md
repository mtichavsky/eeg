# AllTransformerV4 binary tuning log (2026-09-10 → 2026-09-21)

Handoff file: everything needed to pick this work up again.

## Goal

Binary (`--class-mode 2`), all-channel (`--channel all`), `--condition ec+eo`, `--dataset all`
**AllTransformerV4** to **80 % chunk accuracy** (10-fold aggregate, the number in the paper),
with **balanced sensitivity/specificity** (not 95/50). Hyperparameters before architecture.
Always `--n-folds 10`; cut configs, never folds.

**Status: 80 % NOT reached. Best = 78.96 % (`all_052c`).** The 16-experiment budget that was
granted for this is spent; more runs need a fresh go-ahead.

## Best configuration (`experiments/all_052c_all_ec+eo`)

```
--model AllTransformerV4 --dataset all --class-mode 2 --channel all --condition ec+eo
--n-folds 10 --batch-size 64 --val-every 1 --focal-loss --focal-gamma 2.0
--optimizer adamw --lr 3e-4 --dropout 0.1 --weight-decay 3e-2 --epochs 200 --patience 50
```

| Metric (10-fold) | Value |
|---|---|
| Chunk acc, mean of folds (paper's headline) | **78.96 ± 5.19** |
| Chunk acc, pooled over all 8404 chunks | 78.83 |
| Subject acc | 81.79 ± 8.33 |
| Sens / spec (chunk, fold mean) | 85.18 / 68.97 |
| Confusion (TN, FP, FN, TP) | 2251, 1029, 750, 4374 |
| Per-dataset chunk acc CANE / MDD / SAD | 72.24 / 88.96 / 71.40 |
| Worst / best fold | 72.43 / 88.07 |

`all_051b` (wd 1e-2 instead of 3e-2) is equivalent: 78.75 %. Run-to-run noise is ≈1 pt
(anchor 76.46 vs replicate 75.43), so anything within ~1–2 pts of 052c is a tie.

## All configurations tried (all 10-fold, ec+eo, binary, focal γ=2 unless noted)

Chunk = mean of per-fold chunk accuracy; Pooled = all chunks summed over folds.

| Run | opt | dropout | wd | lr | epochs/pat. | other | Chunk | Sens / Spec | Pooled |
|---|---|---|---|---|---|---|---|---|---|
| aug03 anchor | adam | 0.1 | 1e-4 | 1e-4 | 200/50 | | 76.46 | 81.6 / 68.4 | 76.59 |
| 041_f70 (replicate) | adam | 0.1 | 1e-4 | 1e-4 | 200/50 | | 75.43 | 80.9 / 66.6 | 75.20 |
| **Round 1** | | | | | | | | | |
| 031a | adam | 0.3 | 1e-3 | 1e-4 | 200/50 | | 67.89 | 64.5 / 72.4 | 68.05 |
| 031b | adam | 0.5 | 3e-3 | 1e-4 | 200/50 | | 63.66 | 57.0 / 74.3 | 63.18 |
| 032a | adamw | 0.3 | 1e-2 | 1e-4 | 200/50 | | 75.35 | 76.6 / 72.9 | 74.96 |
| 032b | adamw | 0.5 | 5e-2 | 1e-4 | 200/50 | | 73.51 | 74.1 / 71.9 | 73.52 |
| **Batch 1** (60 epochs) | | | | | | | | | |
| 050a | adam | 0.1 | 1e-4 | 1e-4 | 60/60 | | 73.38 | 75.4 / 69.0 | 73.45 |
| 050b | adamw | 0.3 | 1e-2 | 1e-4 | 60/60 | | 70.50 | 71.5 / 68.0 | 70.70 |
| 050c | adamw | 0.1 | 1e-2 | 1e-4 | 60/60 | | 72.90 | 79.1 / 62.6 | 72.93 |
| 050d | adamw | 0.3 | 1e-2 | 3e-4 | 60/60 | | 74.89 | 78.7 / 69.0 | 74.74 |
| 050e | | | | | | γ=0 | **failed** (node `luna201.fzu.cz`, no numpy) — never trained | | |
| 050f | adamw | 0.3 | 1e-2 | 1e-4 | 60/60 | `--select-metric balanced` | 69.24 | 67.3 / 70.9 | 69.60 |
| **Batch 2** (200 ep) | | | | | | | | | |
| 051a | adamw | 0.3 | 1e-2 | 3e-4 | 200/50 | | 77.86 | 81.6 / 72.0 | 77.61 |
| 051b | adamw | 0.1 | 1e-2 | 3e-4 | 200/50 | | 78.75 | 84.4 / 69.4 | 78.64 |
| 051c | adamw | 0.3 | 1e-2 | 1e-3 | 200/50 | | 76.39 | 82.2 / 66.9 | 76.18 |
| 051d | adamw | 0.1 | 1e-3 | 3e-4 | 200/50 | | 77.06 | 84.6 / 64.7 | 76.87 |
| 051e | adamw | 0.3 | 1e-2 | 3e-4 | 200/50 | γ=0 | 73.83 | 80.2 / 63.6 | 73.56 |
| **Batch 3** | | | | | | | | | |
| 052a | adamw | 0.1 | 1e-2 | 3e-4 | 200/50 | γ=3 | 76.74 | 84.1 / 65.8 | 76.61 |
| 052b | adamw | 0.1 | 1e-2 | 3e-4 | 200/50 | γ=4 | 76.59 | 85.2 / 63.1 | 76.29 |
| **052c** | adamw | 0.1 | **3e-2** | 3e-4 | 200/50 | | **78.96** | 85.2 / 69.0 | **78.83** |
| 052d | adamw | 0.0 | 1e-2 | 3e-4 | 200/50 | | 76.73 | 83.0 / 66.4 | 76.62 |
| **Batch 4** | | | | | | | | | |
| 053a | adamw | 0.1 | 3e-2 | 3e-4 | 200/50 | batch 32 | 77.39 | 83.5 / 66.9 | 77.30 |
| 053b | adamw | 0.1 | 1e-1 | 3e-4 | 200/50 | | 78.29 | 85.1 / 67.2 | 77.99 |

Also on file: `all_041_all_ec+eo_f30` (freq cutoff 30 Hz, else anchor config) = 69.79 % — cropping
gamma hurts, so that route is closed.

## What the sweep established

- **lr 3e-4 is the main lever** (+2.5 pts vs 1e-4 at the same regularization: 032a → 051a);
  lr 1e-3 is worse.
- **Regularization plateau.** dropout 0.1 > 0.3 (+0.9) and > 0.0 (−2.0); wd 1e-2 … 1e-1 all
  78–79 %; wd 1e-3 is worse (77.1, train acc 92.6 %).
- **Focal loss matters.** γ=0 (weighted CE) costs 4 pts, mostly specificity (72.0 → 63.6);
  γ=3/4 cost ~2 pts vs γ=2.
- **Adam + heavy regularization collapses.** 031a/031b: 1–3 folds predict "healthy" for every
  chunk from epoch 1 and never recover. AdamW is stable.
- **Shorter schedule hurts.** Several folds peak at epochs 40–150 and get cut off. (My original
  "cosine never anneals" hypothesis was tested and refuted.)
- **`--select-metric balanced` hurts** chunk accuracy and does not fix the balance.
- **Batch size 32 is worse** (77.4).
- **Sens/spec stays ≈85/69 in every good config.** The balance requirement is unmet and nothing
  tried moved it.

## Diagnostics worth remembering

- **Dataset-prior baseline** (predict each dataset's majority class) = 62.5 % pooled. 052c gains
  +16.4 pp over it, of which **+14.1 pp is MDD** (89 % vs 53.8 % base). CANE (72.2 % vs 72.0 %
  base) is essentially at its majority rate; SAD 71.4 % vs 55.5 %, +2.1 pp overall. To reach 80 %
  pooled, CANE would have to reach ≈75 %; hyperparameters alone have not done it.
  (`helpers-print/dataset_prior_baseline.py <run dirs>`)
- **Best-epoch selection is optimistic.** The checkpoint is chosen on the fold that is reported,
  and val accuracy swings 5–10 pts epoch to epoch. Round 1, 032a: mean best-epoch 75.4 % vs
  mean of last 10 epochs ≈ 67 %. Standard for this project/literature, but an 80 % headline
  would partly reflect selection luck.
- **Tuning bias.** ~20 configs were evaluated on the same 10 folds that are reported (already
  disclosed in the paper's limitations). The other models in the paper were, as far as I know,
  not swept this way — confirm before submission.
- Runs before 2026-09-19 have swapped per-dataset sens/spec in `results.txt` (they are PPV/NPV);
  accuracies are fine. 052c and later are correct.

## Not tried yet (ideas, roughly by expected value)

1. Healthy-class re-weighting / oversampling aimed at CANE and at the sens/spec gap
   (weighted sampler is already on; focal α = inverse class frequency).
2. CANE-specific preprocessing: `--skip-artifact-removal`, or checking why healthy CANE chunks
   are called pathological (~35 % specificity on CANE).
3. Longer patience / epochs (best epochs reach 100+ in some folds), possibly with lr 3e-4 and a
   longer cosine horizon.
4. Data augmentation (currently `disabled`); label smoothing; lr warmup; gradient clipping.
5. Architecture changes (model width/depth/heads, `d_model=64`, 2 layers now) — the user wanted
   hyperparameters exhausted first, so this needs their go-ahead.
6. Replicate 052c / 051b with another training seed to quantify noise properly.

## How to resume

**Code:** branch `exp/atv4-round2` (pushed to origin, **not merged** to `main`). It adds:
`--select-metric {accuracy,balanced}` (+ `thesis/metrics.py::selection_score`, tests in
`tests/test_select_metric.py`), a `CODE_DIR` override in `metacentrum/train_job.pbs`,
`cl_luna=False` in the PBS `select` line, and the launcher `run-atv4-round2-meta.sh`
(`batch1()`…`batch4()`; add a `batch5()` and a `case` entry for new configs).

**Cluster (Metacentrum, PBS):**
- Verify access first: `ssh -o BatchMode=yes -o ConnectTimeout=15 metacentrum 'hostname; qstat -u $USER | head'`.
  At the end of the last session SSH returned `Permission denied (publickey…)`; the user must log
  in once from a separate terminal, then retry.
- Run this branch from its own worktree so the shared clone is never switched:
  `/storage/brno2/home/tichavskym/eeg-atv4-r2` (`git pull --ff-only origin exp/atv4-round2` there
  after pushing). It reuses the venv in `/storage/brno2/home/tichavskym/eeg/.venv`.
- Submit: `ssh metacentrum 'cd /storage/brno2/home/tichavskym/eeg-atv4-r2 && ./run-atv4-round2-meta.sh <N>'`.
- Each config ≈ 25–30 min; 4–6 jobs run in parallel. **Confirm with the user before submitting.**
- Excluded clusters (venv/GPU problems): `fobos`, `grogu`, `luna`.
- Quirk: the job template pre-creates `experiments/<name>/`, so `main.py` writes checkpoints and
  `results.txt` to a random-suffix sibling `<name>_<xyz>/`; `stdout.log` stays in `<name>/`.
  Fetch both:
  ```bash
  rsync -a metacentrum:/storage/brno2/home/tichavskym/experiments/binary/8channel/<name>_<xyz>/ experiments/<name>/
  rsync -a metacentrum:/storage/brno2/home/tichavskym/experiments/binary/8channel/<name>/stdout.log experiments/<name>/
  ```
- PBS job logs (`all_05*.o<id>`) are in the cluster worktree dir and were **not** downloaded.

**Local results:** `experiments/all_05{0,1,2,3}*_all_ec+eo/` (16 complete runs; git-ignored).
Round 1 (`all_031*`, `all_032*`) is in `experiments/binary/8channel/`. Old anchor:
`/home/milan/eeg/experiments/results.txt`.

## Paper state (thesis worktree, another session's scratchpad)

`/tmp/claude-1000/-home-milan-eeg-code/79a7a9fd-0329-4320-aa4d-9e62291521d2/scratchpad/thesis-wt/`
(edits are **uncommitted** there and `/tmp` may be wiped — copy them out if you want to keep them).
- `paper/paper.tex`, `paper/NEW_NUMBERS.md`, and `obrazky-figures/{confusion_matrix_binary,per_dataset_comparison}.png`
  now use `all_052c`. PDF rebuilds cleanly (`make pdf`).
- Narrative flipped: AllTransformerV4 is now the best overall (79.0 / 81.8), in-ear CNN-AttnS
  77.0 / 79.2; gap 2.0 / 2.6 pts, not significant (paired: chunk p=0.37, subject p=0.48).
- Not done: the thesis chapters (`projekt-07-results.tex`, etc.) still have old numbers; the paired
  in-ear test is not in `paper.tex` (only a red TODO); the repo's `helpers-print/plot_*.py` are
  stale. Figure scripts that reproduce the paper's PNGs exactly are saved in
  `docs/atv4-tuning-figs/` (`plot_cm_binary.py old|new <out>`, `plot_per_dataset.py old|new <out>`;
  need `seaborn`, which only system `python3` has).
- Original `paper.tex` backup: `/tmp/claude-1000/-home-milan-eeg-code/b48ca4fe-e873-486b-b12d-662e45cb136f/scratchpad/paper.tex.orig`.

## Assistant memory

`~/.claude/projects/-home-milan-eeg-code/memory/atv4-tuning-goal.md` and
`ten-fold-cv-is-mandatory.md` carry the goal, best config and the "10-fold only" rule.
