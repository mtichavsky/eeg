# Explainability: grouped SHAP for the spectrogram models

Methodology record for the SHAP section of the paper. Plan:
`docs/plans/2026-09-26-shap-explainability.md`. Code in `thesis/explain/`, driven by
`python main.py explain <run-dir> --experiment {shap-bands,shap-channels,shap-grid,shap-all}`,
tested in `tests/test_shap.py` and `tests/test_explainability.py`. `shap` is pinned to
`0.52.0` in `pyproject.toml`.

## What is computed

For each fold, the fold's best checkpoint explains that fold's validation chunks (out-of-fold,
same 10-fold subject-independent folds as training). Per chunk, grouped Shapley values of the
**logit margin** `logit_1 - logit_0` (log-odds of pathological) for one of three games:

| Game | Players | Estimator |
|---|---|---|
| bands | `sub_delta, delta, theta, alpha, beta, gamma` (all channels, all frames) | exact, 64 coalitions |
| channels | 8 electrodes (whole `(F, T)` plane) | exact, 256 coalitions |
| grid | 8 electrodes x 6 bands = 48 cells | antithetic permutations |

Players partition the input (`sub_delta` = bins 0-1, below delta), so efficiency
`sum(phi) = f(x) - E f(b)` accounts for the whole prediction. It is checked for every chunk
independently of the library; exact games raise above 1e-4.

**Value function** (interventional): absent players are filled from a background chunk,
present ones from the explained chunk, and the output is averaged over the K background
chunks. `shap` explains the on/off vector (explained point = ones, masker background = zeros).

**Background**: K = 32 training-fold chunks **of the explained chunk's own dataset**,
label-stratified, one chunk per person where possible. Validation data is never used. The
`--baseline mean` robustness variant uses the dataset's mean training spectrogram (K = 1).

## Geometry

Channel order `Fp1, Fp2, C3, Cz, C4, T7, T8, O2/Oz` (`CANONICAL_CHANNEL_ORDER`). Spectrogram
`(72, 41)` at 70 Hz; bin `i` is centred at `i x 0.9766 Hz`. Bands in bins: sub-delta 0-1, delta
2-4, theta 5-8, alpha 9-13, beta 14-30, gamma 31-71. Band widths differ (3 vs 41 bins), so
`per_bin` importance is reported alongside the totals.

For the in-ear model, the `cane` slot holds **IDUN** (real in-ear) data; results name it `idun`,
and MDD/SAD are labelled as synthetic T8-T7 derivations.

## Gate

`harness.verify_baseline` re-evaluates each rebuilt fold and compares chunk accuracy with the
metrics stored in the checkpoint. It allows a difference of up to 1.5 chunks, so one borderline
prediction flipped by CPU/GPU numerics is tolerated. A larger mismatch means the folds were
rebuilt wrongly (the fold then holds different subjects), and that fold is skipped with an
error. Fold building is shared with training (`build_cv_folds` in
`thesis/data_preparation.py`), so the rebuild uses the same code path.

## Outputs

`<run>/explain/shap_<game>[_mean][_random][_partial].json` holds (`_partial` marks runs
restricted with `--folds`; they never overwrite, or get compared with, full runs), per player, `mean_abs_phi` (importance)
and `mean_phi` (direction), each per fold plus mean and std over folds. The subject is the unit:
chunks are averaged per person, then over persons in the fold, then over folds. It also has
splits by dataset, by true class and by correct/incorrect, per-dataset paired tests (Wilcoxon and
Nadeau-Bengio corrected t, BH within the game), the max efficiency residual, and background
provenance. The grid also stores the Monte Carlo SE from repeated seeds. The matching
`_chunks.npz` holds the per-chunk rows (phi, band log-power, f(x), E f(b), labels, predictions).

Derived runs compare themselves with their reference run, when it has already been written:
`--baseline mean` gives `baseline_agreement` (Spearman against the sample baseline), and
`--random-init` gives `randomisation_check`. The randomisation check (Adebayo et al. 2018)
passes if the random model's mean |f(x) - E f(b)| is below 10% of the trained model's (rho is
reported only: it stays high because profile order follows input spectral power).

## Checkpoints explained

`experiments/all_041_inear_ec+eo_f70` (CNN-AttnS) and `experiments/all_041_all_ec+eo_f70`
(AllTransformerV4). These are the f70 reruns from the cutoff study, **not** the Table II runs,
which kept no checkpoints. Launch on MetaCentrum with `./run-shap-meta.sh`, and draw the figures
with `helpers-print/plot_shap.py`.

## Caveats for the paper

These values explain the model, not the brain. Grouped features are correlated, and composite
inputs can be off-manifold. The values depend on the background, which is why the same-dataset
background and the mean-baseline check are used. Folds vary, so std over folds is reported.
