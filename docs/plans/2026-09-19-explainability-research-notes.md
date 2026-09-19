# Explainability research notes (consolidated, 2026-09-19)

Single source of truth for the explainability work behind Section VII of
`thesis-text/paper/paper.tex`. It replaces and merges three earlier files, now deleted:

- `2026-08-16-explainability-experiments.md` (first catalogue E1–E9, never reviewed)
- `2026-09-14-explainability-design.md` (reviewed design: what was kept, dropped, and why)
- `2026-09-14-explainability-session-state.md` (status, decision log, next steps, cluster notes)

Read the TL;DR and §2–§4 first when picking the work back up. §9 keeps the still-useful technical
facts from the Aug 16 catalogue, so nothing has to be re-derived.

Companion plans that are **not** merged here: `2026-09-14-bipolar-surrogate-ablation.md` (plan A)
and `2026-09-14-frequency-cutoff-ablation.md` (plan B, now in `main` via PR #83).

**Last updated: 2026-09-19 (late evening), after experiments A and B and the six single-dataset runs finished.**

## TL;DR (2026-09-19 late evening)

1. **A (bipolar montage) and B (30 Hz cutoff) are done.** A: no montage is significantly better
   (min BH p = 0.50). B: cutting at 30 Hz costs 9.5 pp (CNN-AttnS in-ear, significant) and 5.6 pp
   (AllTransformerV4, not significant at 10 folds), so the models use information above 30 Hz.
2. **The pooled accuracy is MDD-driven, at every cutoff.** Against a dataset-prior baseline
   (always predict each dataset's majority class), the gain is +7 to +18 pp, MDD supplies
   +12 to +17 of it, and CANE sits at or below its base rate. At 30 Hz the MDD contribution
   survives (+12 to +14 pp) while CANE and SAD fall further below base rate (§4.3).
3. **Single-dataset runs (objection 1) are done and change the reading of CANE (§3.6).** MDD
   trained alone reaches 90.5 / 92.7 % balanced accuracy (EC / EO), so its result is not a
   cross-dataset shortcut. CANE alone has plain accuracy *below* its 72 % majority rate but
   balanced accuracy of 68.6 % (EC, BH p = 0.009) and 65.4 % (EO, p = 0.051), about the same as in
   the combined model. SAD alone: 69.6 / 65.3 % (p = 0.020 / 0.009), 5 to 8 pp above its combined
   value. So a weak anxiety signal exists in both setups; "at or below the majority rate" was a
   statement about a metric the model is not trained for.
4. **Still open for objection 1: the four-class decomposition only.** Approved 2026-09-19: add the
   per-dataset confusion matrix to the training code (branch `feat/per-dataset-confusion-matrix`,
   implemented by a subagent, PR pending) and rerun the 4-class models on Metacentrum after it
   merges. The dataset-identity probe and shuffled-label control are **not planned** (§7 item 2.3).
   The fold-rebuild harness was **dropped** (user decision: too much code complexity).
   **Objection 4 (biomarkers: band shuffle, asymmetry swap) is out of scope for today**; the user
   may pick it up in a later session.
5. **Paper Section VII is up to date for A, B, the decomposition and the single-dataset runs**
   (thesis-text PR #4, open). Whitham 2007 is cited. Everything still to do is in red.
6. **Known bug:** per-dataset sensitivity/specificity are mislabelled (§3.5). Fix is PR #91
   (merged, by the parallel session code-c0); accuracy numbers are unaffected.
7. **Cutoff curve (50 / 40 Hz, in-ear CNN-AttnS) is done (§4.5).** Chunk accuracy 77.1 → 74.0 →
   72.7 → 67.5 at 70 / 50 / 40 / 30 Hz. The loss is gradual and already significant at 50 Hz
   (−3.1 pp), not a single cliff, so the dependence is spread over 30–70 Hz.

---

## 1. Goal and reviewer objections

Section VII is not meant to produce heatmaps for their own sake. Its job is to **defend the
paper's existing claims against the objections a reviewer will raise.** Every experiment must
answer one of these:

| # | Reviewer objection | Claim it threatens | Covered by |
|---|---|---|---|
| 1 | "The model learned which dataset a recording came from, not pathology." | Every accuracy, 4-class especially | Prior-baseline decomposition (§3.4, §4.3) and single-dataset runs (§3.6); probe, shuffled labels and 4-class still open (§7) |
| 2 | "Why would one in-ear channel match 8 channels?" | Headline in-ear result | A (done: §3) |
| 3 | "It's reading artifacts (muscle, eye movement), not brain." | Clinical validity | B (muscle, partly: §4.4), A's EC/EO split (eye movement, not a valid test: §3.3) |
| 4 | "Does it use the known biomarkers (alpha asymmetry, beta power)?" | Scientific credibility | C (candidate), asymmetry swap (candidate); **out of scope for today** |

---

## 2. Status at a glance (2026-09-19 late evening)

| Exp. | What | Code | Runs | Result |
|---|---|---|---|---|
| **A** | Bipolar surrogate ablation (Fp2−Fp1 / C4−C3 / T8−T7 / in-ear) × EC / EO | ✅ In `main` (PR #82) | ✅ 8 runs, 2026-09-18. `experiments/all_040_*` | §3: no montage significantly better |
| **B** | Retrain at 30 Hz spectrogram cutoff vs 70 Hz | ✅ In `main` (PR #83 flag, PR #89 launcher + summary) | ✅ 4 runs, 2026-09-19, ~30 min wall. `experiments/all_041_*`. ✅ curve points 50 / 40 Hz (in-ear CNN-AttnS, `all_042_inear_ec+eo_f{50,40}`, jobs 23807556/57, PR #90; done 2026-09-19) | §4: 30 Hz costs 5.6 to 9.5 pp; §4.5: 50 Hz −3.1, 40 Hz −4.4 pp |
| **C** | Post-hoc band shuffle on saved checkpoints | ❌ | n/a (inference only) | Undecided (§6). Now unblocked: B saved 70 Hz AllTransformerV4 checkpoints |
| Obj. 1 | Dataset-identity checks | ✅ Prior-baseline script, single-dataset launcher and summary in `main` (PR #88, merged) | ✅ 6 single-dataset runs `{cane,sad,mdd}_041_t8-t7_{ec,eo}` (jobs 23807482–87, run by code-c0) done, synced to `experiments/`. Probe / shuffled labels / 4-class not built | §3.4, §3.6, §4.3 |
| Bug | `main.py:259` swapped args | ✅ Fix + `tests/test_per_dataset_eval.py` in PR #91 (merged) | n/a; the 6 single-dataset runs predate the fix (their summary uses chunk metrics, unaffected) | §3.5 |

**PRs (mtichavsky/eeg):** #87 docs (this file + CLAUDE.md Metacentrum section) open; #88 objection-1 tools, #89 cutoff launcher, #90
cutoff-curve launcher, #91 per-dataset metrics fix and #92 balanced-accuracy summary **merged**; #92 (code-c0) adds the
balanced-accuracy test of §3.6 to `single_dataset_summary.py`.
**PR (mtichavsky/thesis-text):** #4 open, holds Section VII (shortcut, montage, cutoff, planned)
plus the user's own prose edits. After #4 merges the user must
`git checkout -- paper/paper.tex` in their checkout before switching branches.

---

## 3. Results of experiment A (analysed 2026-09-19)

Setup: CNN-AttnS, Table II hyperparameters unchanged (`--batch-size 64 --epochs 100 --lr 1e-4
--dropout 0.05 --weight-decay 5e-5 --focal-loss --patience 20`), `--dataset all --class-mode 2`,
10-fold, binary. Produced with `helpers-print/bipolar_ablation_summary.py`. Fold identity across
the three headset montages was **verified** from the `Val subjects (N): [...]` log lines (all four
comparisons OK), so the paired tests below are valid.

### 3.1 Headline table (mean ± std over 10 folds, %)

| Montage | Cond | Chunk acc | Subject acc | Chunk sens | Chunk spec |
|---|---|---|---|---|---|
| Fp2−Fp1 | EC | 73.96 ± 4.80 | 73.89 ± 8.38 | 80.67 | 63.66 |
| Fp2−Fp1 | EO | 76.64 ± 5.34 | 74.86 ± 7.44 | 75.55 | 78.40 |
| C4−C3 | EC | 72.71 ± 6.24 | 70.99 ± 6.06 | 78.83 | 63.00 |
| C4−C3 | EO | 72.62 ± 9.87 | 72.00 ± 11.86 | 70.53 | 75.47 |
| T8−T7 | EC | 76.34 ± 5.82 | 78.04 ± 6.31 | 78.90 | 71.71 |
| T8−T7 | EO | 78.38 ± 5.70 | 76.56 ± 5.60 | 82.95 | 70.54 |
| in-ear (IDUN) | EC | 75.33 ± 5.96 | 73.04 ± 9.27 | 73.61 | 74.18 |
| in-ear (IDUN) | EO | 78.34 ± 7.18 | 74.41 ± 9.64 | 79.69 | 74.01 |

For reference, the Table II in-ear CNN-AttnS row (ec+eo) is 77.0 chunk / 79.2 subject.
In-ear rows use different folds (IDUN has 53 subjects vs CANE's 63; 153 vs 164 validation
subjects in total) and are descriptive only.

### 3.2 Paired statistics across headset montages (chunk acc, BH over 3 comparisons per condition)

| Cond | Comparison | Mean diff | Wilcoxon p | Nadeau–Bengio p | BH-adj. p |
|---|---|---|---|---|---|
| EC | Fp2−Fp1 vs C4−C3 | +1.25 pp | 0.770 | 0.692 | 0.692 |
| EC | Fp2−Fp1 vs T8−T7 | −2.37 pp | 0.275 | 0.439 | 0.659 |
| EC | C4−C3 vs T8−T7 | −3.62 pp | 0.160 | 0.248 | 0.659 |
| EO | Fp2−Fp1 vs C4−C3 | +4.02 pp | 0.131 | 0.347 | 0.505 |
| EO | Fp2−Fp1 vs T8−T7 | −1.74 pp | 0.492 | 0.505 | 0.505 |
| EO | C4−C3 vs T8−T7 | −5.77 pp | 0.084 | 0.168 | 0.504 |

**Nothing is significant** (smallest BH-adjusted p = 0.50). Fold SDs are 5–10 pp with about
17 validation subjects per fold, so differences under about 5 pp cannot be resolved.

### 3.3 Findings

1. **T7/T8 is not shown to be special, and neither is any other pair.** T8−T7 is nominally best
   in both conditions and C4−C3 nominally worst, but every gap is within noise. The honest
   claim for objection 2 is "a single interhemispheric bipolar channel reaches about 73–78 %
   chunk accuracy regardless of location". That supports "one channel suffices" but does **not**
   support "T7/T8 was necessary". The paper currently justifies T7/T8 only by citation
   (Tremmel 2024, Moumane 2024); this experiment neither confirms nor refutes that.
2. **Real in-ear (IDUN) does not beat the headset surrogate.** T8−T7 vs in-ear: 76.3 vs 75.3 (EC),
   78.4 vs 78.3 (EO). On MDD and SAD the two runs get identical inputs, and their per-dataset
   accuracies still differ by 1–3 pp (MDD 88.8 vs 87.7 EC, 90.3 vs 91.2 EO; SAD 62.8 vs 59.9 EC,
   62.4 vs 60.9 EO). Those gaps calibrate the **noise floor of about 1–3 pp** from different folds
   plus unseeded training.
3. **The eye-movement flag printed by the summary script is misleading.** The script says
   Fp2−Fp1's EO > EC (+2.7 pp) is "consistent with eye movement". EO ≥ EC is broadly present
   (T8−T7 +2.0, in-ear +3.0, C4−C3 −0.1), so Fp2−Fp1 is not special, and EC/EO runs do **not**
   share folds (`same_folds=False`), so no paired test is possible. Do not cite this check as
   evidence for or against eye-movement leakage. Fix or remove the message in the script.
4. **The accuracy comes almost entirely from MDD (new, and important for objection 1).** See §3.4.

### 3.4 Per-dataset breakdown and the dataset-prior baseline

Per-dataset chunk accuracy vs a **"dataset-prior" baseline**: a classifier that knows only which
dataset a chunk came from and always predicts that dataset's majority class (uses validation
class balance, so it is if anything generous to the baseline). Pooled over folds, corrected
counts (see §3.5):

| Run | MDD acc / base | CANE acc / base | SAD acc / base | Overall acc | Overall prior baseline | Gain | Gain from MDD / CANE / SAD (pp) |
|---|---|---|---|---|---|---|---|
| Fp2−Fp1 EC | 83.4 / 53.5 | 70.8 / 71.7 | 58.6 / 57.6 | 74.0 | 62.6 | +11.4 | +11.6 / −0.4 / +0.1 |
| Fp2−Fp1 EO | 91.5 / 56.9 | 69.2 / 72.3 | 59.2 / 59.6 | 76.5 | 64.3 | +12.2 | +13.6 / −1.4 / −0.0 |
| C4−C3 EC | 90.1 / 53.5 | 62.2 / 71.7 | 58.2 / 57.6 | 72.5 | 62.6 | +9.9 | +14.3 / −4.5 / +0.1 |
| C4−C3 EO | 87.5 / 56.9 | 62.6 / 72.3 | 63.9 / 59.6 | 72.6 | 64.3 | +8.3 | +12.1 / −4.4 / +0.6 |
| T8−T7 EC | 88.8 / 53.5 | 70.3 / 71.7 | 62.8 / 57.6 | 76.4 | 62.6 | +13.9 | +13.7 / −0.6 / +0.7 |
| T8−T7 EO | 90.3 / 56.9 | 72.6 / 72.3 | 62.4 / 59.6 | 78.0 | 64.3 | +13.7 | +13.2 / +0.1 / +0.4 |
| in-ear EC | 87.7 / 52.7 | 65.9 / 68.4 | 59.9 / 55.9 | 75.2 | 59.0 | +16.2 | +16.5 / −0.9 / +0.7 |
| in-ear EO | 91.2 / 56.0 | 69.4 / 68.1 | 60.9 / 57.7 | 78.3 | 60.6 | +17.7 | +16.7 / +0.5 / +0.5 |

Reading:

- **MDD is the only dataset where the model beats its prior**, by about 30–37 pp. On **CANE it is
  at or below the majority-class rate** (CANE is about 72 % pathological in the headset data, 68 %
  in IDUN) and on **SAD it is 0–5 pp above it**.
- Essentially **100 % of the gain over the dataset-prior baseline comes from MDD**.
- A chunk-level accuracy of 72–78 % therefore should not be read as "detects both depression and
  anxiety at about 75 %". It is consistent with "distinguishes MDD well, and otherwise leans on
  each dataset's base rate".
- Caveats: this is the **single-channel Exp A models only**. The Table II 8-channel
  AllTransformerV4 numbers have not been decomposed this way yet (cheap, see §7). CNN-AttnS
  chunk-level specificity on CANE is 30–58 %, i.e. many healthy CANE subjects are called
  pathological. The single-dataset runs of §3.6 now test the cross-dataset version of this:
  MDD alone reaches about 90 % balanced accuracy, so its accuracy is not a shortcut from mixing
  datasets. A within-MDD recording effect remains untested.

**Update (same day, from `helpers-print/dataset_prior_baseline.py`, PR #88):** the Table II binary
runs show the same pattern. 7 binary runs with per-dataset checkpoint data gain +7 to +18 pp over
the dataset-prior baseline, MDD contributes +12 to +17 pp, and CANE is at or below its base rate in
every one. The headline `atv4-lowlr_oex` (no checkpoints; counts reconstructed from `results.txt`)
gains +14.2 pp: MDD +13.7, CANE −1.4, SAD +1.8. **4-class runs cannot be decomposed from stored
metrics**: they store per-dataset accuracy only, not per-dataset class balance.

### 3.5 Bug found: per-dataset sensitivity/specificity are wrong (accuracy is fine)

`main.py:259` calls `compute_per_dataset_metrics(preds_arr, labels_arr, ...)`, but the function
signature is `(y_true, y_pred, subjects, subject_dataset_map)` (`thesis/metrics.py:321`). The
**arguments are swapped**. Consequences:

- Per-dataset **accuracy is unaffected** (symmetric), so all per-dataset accuracy numbers in
  `results.txt`, checkpoints and the paper's accuracy tables stay valid.
- The per-dataset **"sensitivity" is really precision (PPV), and "specificity" is really NPV**.
  Stored `tp`/`tn` are correct; stored `fp` is the true FN and stored `fn` is the true FP.
- Verified on all 80 folds of Exp A: per-dataset `(tp, tn, fp, fn)` summed over datasets equals
  `(TP, TN, FN, FP)` of the chunk confusion matrix in the same checkpoint. The symptom that
  exposed it: per-dataset "positives" summed to the number of *predicted* positives, and varied
  across montages with identical validation chunks.
- Anything that reports per-dataset sens/spec (the `Per-Dataset Chunk Metrics` table in every
  `results.txt`, `docs/per_dataset_comparison.png` if it uses those columns, any paper text)
  should be recomputed. It is recoverable from stored checkpoints by swapping `fp`↔`fn` (that is
  what §3.4 did). **Fix is PR #91** (merged, §7 item 1).

### 3.6 Single-dataset runs (objection 1; run by code-c0, analysed 2026-09-19 late evening)

CNN-AttnS, `--channel T8-T7`, Exp A hyperparameters, one dataset per run, EC and EO separately,
10-fold (`{mdd,cane,sad}_041_t8-t7_{ec,eo}`, jobs 23807482–87; CANE from the headset, as in
`all_040_t8-t7_*`). Summary script: `helpers-print/single_dataset_summary.py`: plain accuracy vs the fold's majority
rate (in `main`) and, since PR #92 (merged, by code-c0), a "Balanced accuracy vs chance (50%)" section
(pooled value, fold mean ± sd, Wilcoxon, Nadeau–Bengio, BH over these 6 runs). The balanced-accuracy
columns below come from that section: I recomputed them independently first, and ran PR #92's
script on the synced data, which reproduces them exactly (fold means and BH p-values).

| Dataset | Cond | Chunk acc (fold mean ± sd) | Majority rate (pooled) | Balanced acc, fold mean ± sd (pooled) | Folds > 50 % | BH p (NB) | Combined-run bal. acc (`all_040_t8-t7`, other folds) |
|---|---|---|---|---|---|---|---|
| MDD | EC | 90.20 ± 8.77 | 50.7 | 89.3 ± 10.2 (90.5) | 10/10 | < 0.001 | 88.6 |
| MDD | EO | 92.59 ± 8.87 | 53.0 | 92.3 ± 9.6 (92.7) | 10/10 | < 0.001 | 90.1 |
| CANE | EC | 66.78 ± 10.12 | 71.7 | 68.4 ± 10.6 (68.6) | 9/10 | 0.009 | 66.6 |
| CANE | EO | 64.49 ± 19.89 | 72.3 | 66.9 ± 16.4 (65.4) | 9/10 | 0.051 | 64.6 |
| SAD | EC | 70.49 ± 15.18 | 50.0 | 70.5 ± 15.2 (69.6) | 9/10 | 0.020 | 61.6 |
| SAD | EO | 65.69 ± 9.51 | 50.0 | 65.7 ± 9.5 (65.3) | 9/10 | 0.009 | 60.1 |

Plain accuracy vs the fold's majority rate (script output): MDD +37.1 / +38.6 pp (BH p 0.0002 /
0.0001), CANE −4.7 / −8.2 pp (p 0.45 / 0.45), SAD +20.5 / +15.7 pp (p 0.025 / 0.012).

Reading:

1. **MDD is the positive control and it works.** Without any other dataset to be confused with,
   MDD reaches about 90 %. So the high MDD accuracy in the combined runs is **not** a
   dataset-identity shortcut. A within-MDD confound (e.g. a recording difference between groups)
   cannot be ruled out by this.
2. **CANE: the "at or below the majority rate" reading needs correcting.** Plain accuracy is below
   the 72 % majority rate, but balanced accuracy is 68.4 / 66.9 % (significantly above 50 % in EC,
   borderline in EO with 9 of 10 folds above), and the combined model has **the same** CANE
   balanced accuracy (66.6 / 64.6). Training is class-balanced (WeightedRandomSampler plus class
   weights in the focal loss), so the model is optimised for balanced accuracy, and the majority
   rate is a demanding yardstick for it on a 72 % pathological dataset. Any dataset-prior rule has
   balanced accuracy 50 % by construction, which makes balanced accuracy the fair within-dataset
   test. Conclusion: a **weak anxiety/comorbid signal is present** and is not created or destroyed
   by pooling with MDD. The earlier hypothesis "signal exists and the combined model fails to use
   it" is not supported. The strength (about 65–68 %) is far below MDD's.
3. **SAD alone is 5–8 pp better than SAD inside the pooled run** (70.5 / 65.7 vs 61.6 / 60.1
   balanced). Different folds, so descriptive only. Consistent with pooled training costing SAD
   something, but not tested.
4. **Caveats.** Folds are small (SAD: 4 to 6 validation subjects), so per-fold values are
   coarse (SDs 10–20 pp). CANE EO is the noisiest (SD 16–20 pp).
   Chunk-level sensitivity/specificity here are reliable (they come from the pooled `chunk`
   metrics, not the buggy per-dataset ones). A model that reads the *recording session*
   (equipment, site) within a single dataset would also pass this test; the test removes only the
   between-dataset shortcut.

---

## 4. Experiment B: 30 Hz cutoff retrain (objection 3, muscle artifact). Done 2026-09-19

- **Question:** does the result depend on the frequency range where muscle activity (EMG)
  dominates? Anxious subjects plausibly tense their jaw and forehead more.
- **Method:** retrain with the spectrogram cropped at 30 Hz, shape `(31, 41)` instead of
  `(72, 41)`. Retraining rather than post-hoc occlusion, so the model never sees an unnatural
  input.
- **Runs (4, all `ec+eo`, 10-fold, binary, Table II hyperparameters):** launched with
  `run-freq-cutoff-meta.sh` at commit `7206375-dirty` (the cluster clone has untracked files;
  the branch was later rebased, so the label no longer resolves).
  - `all_041_inear_ec+eo_f70`, `_f30`: CNN-AttnS with the A hyperparameters, in-ear (IDUN for CANE).
  - `all_041_all_ec+eo_f70`, `_f30`: AllTransformerV4 `--epochs 200 --lr 1e-4 --dropout 0.1
    --weight-decay 1e-4 --patience 50 --batch-size 64 --focal-loss`.
- **Why rerun 70 Hz:** torch isn't seeded, and Table II was trained at commit `454c49e-dirty`.
  Rerunning gives identical folds and code, so 30-vs-70 is paired. Fold identity was **verified**
  for both models (10/10 folds identical validation subjects).
- **Code:** `--freq-cutoff` flag; the cutoff lives on the `SpectrogramDataset` instance (not a
  module global, since forkserver DataLoader workers wouldn't see a runtime override). Parameter
  counts drop at 30 Hz: CNNAttnS 79,490 → 57,986; AllTransformerV4 113,474 → 91,970. The CNN can't
  be built below ~21 Hz (22 rows builds, 21 fails), so 20 Hz is impossible.
- **Correction found during implementation:** the earlier plan and `EXPLAINABILITY.md` said
  `create_model(strict=False)` would *silently* drop a mismatched layer. It doesn't: PyTorch raises
  on shape mismatches regardless of `strict`.

### 4.1 Results (mean ± std over 10 folds, %; `helpers-print/freq_cutoff_summary.py`)

| Model | Cut | Chunk acc | Subj acc | Chunk sens | Chunk spec | Params |
|---|---|---|---|---|---|---|
| CNN-AttnS (in-ear) | 70 | 77.06 ± 4.48 | 76.37 ± 6.91 | 82.37 | 69.81 | 79,490 |
| CNN-AttnS (in-ear) | 30 | 67.54 ± 5.42 | 67.91 ± 6.84 | 69.53 | 64.73 | 57,986 |
| AllTransformerV4 | 70 | 75.43 ± 4.63 | 75.55 ± 6.00 | 80.90 | 66.63 | 113,474 |
| AllTransformerV4 | 30 | 69.79 ± 6.36 | 69.74 ± 9.01 | 70.38 | 67.71 | 91,970 |

Paired 70 minus 30 (positive = 70 Hz better), BH-adjusted over the two models within each metric:

| Model | Metric | Diff | Bootstrap 95 % CI | Wilcoxon (BH) | Nadeau–Bengio (BH) |
|---|---|---|---|---|---|
| CNN-AttnS | Chunk acc | +9.52 pp | [+6.17, +12.62] | 0.0039 (**0.0078**) | 0.0043 (**0.0087**) |
| CNN-AttnS | Subject acc | +8.46 pp | [+2.59, +13.73] | 0.031 (0.0625) | 0.085 (0.171) |
| CNN-AttnS | Chunk sens | +12.83 pp | [+7.65, +18.11] | 0.0020 (**0.0039**) | 0.012 (**0.024**) |
| CNN-AttnS | Chunk spec | +5.08 pp | [−1.07, +10.84] | 0.160 (0.320) | 0.304 (0.609) |
| AllTransformerV4 | Chunk acc | +5.63 pp | [+0.92, +10.58] | 0.106 (0.106) | 0.176 (0.176) |
| AllTransformerV4 | Subject acc | +5.81 pp | [−1.81, +13.80] | 0.275 | 0.373 |
| AllTransformerV4 | Chunk sens | +10.53 pp | [+1.59, +20.04] | 0.131 | 0.187 |
| AllTransformerV4 | Chunk spec | −1.08 pp | [−6.53, +3.80] | 0.770 | 0.795 |

**Reproducibility of the 70 Hz reruns vs Table II** (chunk / subject): CNN-AttnS 77.06 vs 77.0 /
76.37 vs 79.2; AllTransformerV4 75.43 vs 76.5 / 75.55 vs 78.6. Chunk accuracy reproduces within
about 1 pp, subject accuracy is 2.8–3.0 pp lower. Combined with A, run-to-run noise is about
1–3 pp on chunk and up to 3 pp on subject accuracy.

### 4.2 Per-dataset chunk accuracy (accuracy only; per-dataset sens/spec are mislabelled, §3.5)

| Model | Cut | MDD | CANE (IDUN in-ear) | SAD |
|---|---|---|---|---|
| CNN-AttnS | 70 / 30 | 88.6 / 81.6 (+7.0) | 67.5 / 58.4 (+9.1) | 64.0 / 46.9 (+17.0) |
| AllTransformerV4 | 70 / 30 | 88.5 / 83.2 (+5.3) | 67.0 / 61.7 (+5.3) | 64.2 / 57.4 (+6.8) |

The drop is **not confined to MDD**.

### 4.3 Decomposition against the dataset-prior baseline (pooled over folds, from `helpers-print/dataset_prior_baseline.py`, PR #88)

| Run | MDD acc / base | CANE acc / base | SAD acc / base | Overall acc | Prior | Gain | Gain from MDD / CANE / SAD (pp) |
|---|---|---|---|---|---|---|---|
| in-ear f70 | 89.0 / 53.3 | 66.8 / 68.2 | 64.3 / 54.3 | 77.1 | 58.9 | +18.2 | +17.1 / −0.5 / +1.6 |
| in-ear f30 | 81.8 / 53.3 | 57.6 / 68.2 | 47.6 / 54.3 | 67.7 | 58.9 | +8.8 | +13.7 / −3.9 / −1.0 |
| AllTransformerV4 f70 | 89.3 / 53.8 | 66.2 / 72.0 | 64.4 / 55.5 | 75.2 | 62.5 | +12.7 | +14.2 / −2.7 / +1.2 |
| AllTransformerV4 f30 | 84.1 / 53.8 | 61.7 / 72.0 | 57.5 / 55.5 | 70.1 | 62.5 | +7.7 | +12.2 / −4.8 / +0.3 |

Reading:
- Of the gain lost at 30 Hz (in-ear −9.4 pp, AllTransformerV4 −5.0 pp), the loss is spread over
  the three datasets (in-ear MDD −3.4, CANE −3.4, SAD −2.6; AllTransformerV4 −2.0, −2.1, −0.9),
  not concentrated in MDD.
- At 30 Hz the **MDD signal mostly survives** (still +28 to +30 pp above its base rate, contributing
  +12 to +14 pp overall), whereas CANE falls about 10 pp **below** its majority rate and SAD to
  +2.0 pp (AllTransformerV4) or −6.7 pp (in-ear) relative to its own. So even at 30 Hz the residual accuracy is an MDD result.
- Objection 3 is therefore only **partly** answered: the models do use frequencies above 30 Hz on
  all three datasets, which is what EMG reliance would predict, but it is also what neural
  beta/gamma reliance or group differences in muscle tone would predict. B cannot separate them.

### 4.4 What B does and doesn't show (as written in the paper)

- Shows: dependence on the 30–70 Hz range, significant for CNN-AttnS on chunk accuracy and
  sensitivity, same direction and a bootstrap CI excluding zero for AllTransformerV4 but not
  significant at 10 folds.
- Doesn't show: that the dependence is muscle. Whitham et al. (2007) put EMG contamination from
  ~20 Hz, so 30 Hz removes the EMG-*dominated* range, not all EMG. The 30 Hz cut also removes about
  21.5K parameters (confound), and it is one cutoff, not a curve.
- Paper status: added as subsection "Reliance on High Frequencies" (`sec:cutoff`, `tab:cutoff`) in
  thesis-text PR #4. Whitham 2007 is a red `[add citation]` placeholder, not in `references.bib`.

### 4.5 Cutoff curve for CNN-AttnS (in-ear), 70 / 50 / 40 / 30 Hz (done 2026-09-19)

Jobs 23807556 (50 Hz) and 23807557 (40 Hz), exit 0, `all_042_inear_ec+eo_f{50,40}`, launched by
`run-freq-cutoff-curve-meta.sh` (PR #90). Same code, hyperparameters and folds as the 70 / 30 Hz
runs (fold identity **verified** against `all_041_inear_ec+eo_f70` for all three cutoffs).
`helpers-print/freq_cutoff_curve_summary.py`:

| Cutoff | Chunk acc | Subj acc | Params | 70 Hz minus cutoff (95 % CI) | Wilcoxon p (BH over 3) | NB p |
|---|---|---|---|---|---|---|
| 70 | 77.06 ± 4.48 | 76.37 ± 6.91 | 79,490 | | | |
| 50 | 73.95 ± 4.47 | 71.85 ± 6.82 | 69,250 | +3.11 pp [1.61, 4.98] | 0.0020 (0.0059) | 0.0478 |
| 40 | 72.67 ± 3.70 | 71.39 ± 6.56 | 63,106 | +4.39 pp [1.77, 6.65] | 0.0195 (0.0195) | 0.0462 |
| 30 | 67.54 ± 5.42 | 67.91 ± 6.84 | 57,986 | +9.52 pp [6.17, 12.62] | 0.0039 (0.0059) | 0.0043 |

Per-dataset chunk accuracy (CANE column = IDUN): MDD 88.6 → 87.9 → 86.4 → 81.6; CANE 67.5 → 61.8 →
63.2 → 58.4; SAD 64.0 → 56.4 → 51.6 → 46.9.

Reading:

- **Monotone and gradual.** Every band removed costs accuracy; no single cliff at 30 Hz. Even 50 Hz
  costs 3.1 pp (Wilcoxon significant after BH; Nadeau–Bengio p = 0.048 before BH, about 0.048
  after BH over three cutoffs, i.e. borderline). 30 Hz is clearly significant on both tests
  (NB BH ≈ 0.013).
- **MDD is robust down to 40 Hz** (−2.2 pp) and only loses noticeably at 30 Hz (−7.0); CANE and SAD
  lose accuracy earlier. SAD ends near chance (46.9 %, below its 50 % base rate).
- **Same limits as §4.4:** cannot separate muscle from neural beta/gamma, the parameter count
  falls with the cutoff (79 K → 58 K), and folds are small. Per-dataset SDs are 10–20 pp.

---

## 5. Decision log (what we considered and why)

1. **Starting proposal:** Phase 1 = surrogate in-ear on Fp1−Fp2 / C3−C4 / T3−T4. Phase 2 =
   all-channel model with each region dropped (frontal / central / temporal / occipital).
2. **Phase 1 kept and refined into A.** CANE must use the headset for non-temporal pairs, so a
   headset `T8-T7` run is needed next to the IDUN `in-ear` run. EC and EO run separately.
3. **Phase 2 dropped.** Neighbouring electrodes carry overlapping signal (small non-significant
   drops expected). Regions differ in size (2 / 3 / 2 / 1 channels). The occipital channel is O2 in
   MDD/SAD but Oz in CANE. It overlaps A's spatial question. **One variant kept as optional:**
   AllTransformerV4 on T7+T8 only (directly supports the in-ear claim).
4. **Post-hoc region removal dropped:** can't apply to the single-channel in-ear model and repeats
   A's question.
5. **Band analysis, first version rejected:** filling bands with a training-set mean, plus a
   "band drop minus width-matched random-window drop" control. Judged clumsy (model never saw such
   inputs) and arbitrary.
6. **Simplified to band shuffle (C).** Then asked whether retraining without a band is simpler:
   easier to write and avoids unnatural inputs, but it answers "is the band *needed*", not "does
   *this model* use it". Neighbouring bands compensate and run-to-run noise gets added.
   **Resolution:** retrain where the question is sharp (B: 30 Hz cut for muscle artifact), keep
   post-hoc shuffle (C) as an optional supplement for the biomarker question.
7. **Literature check** showed the field is split on post-hoc permutation (see §12). An earlier
   claim that post-hoc shuffle was "the best" was too confident.
8. **Cutoff clarification:** the code cuts at **70 Hz** (`thesis/stft.py`, band-pass 1–70 Hz), not 60.
9. **Aug 16 catalogue superseded** (§9): E1–E9 as written are not the plan. What survives is the
   technical facts, the harness gate and the print-budget/LaTeX conventions.

### Dropped, and why

| Idea | Reason |
|---|---|
| Leave-one-region-out, retrained (original Phase 2) | Neighbouring electrodes overlap, so small non-significant drops likely. Overlaps with A. |
| Post-hoc region removal | Can't apply to the one-channel in-ear model; repeats A's location question. |
| Filling occluded bands with a training-set mean or zeros | Unrealistic input the model never saw (`log1p|STFT|`, zero = "silence"). |
| "Band drop minus random-window drop" width control | Arbitrary correction layered on the previous one. |
| Retrain once per removed band (5 runs × models) | Neighbouring bands compensate, results muted, run-to-run noise added. |
| 20 Hz cutoff | CNN can't be built below ~21 Hz; would also remove beta. |

---

## 6. Experiment C: post-hoc band shuffle (objection 4). Candidate, undecided

- **Method:** load each `fold_N_best.pth` and evaluate it on its own validation fold (still
  10-fold CV, no retraining). In every chunk, replace one band's spectrogram rows with the same
  rows from a **random other chunk** (permutation importance). The accuracy drop per band shows
  what the trained model relies on.
- **Bands** (bin `i` ≈ `i × 0.977 Hz`): delta rows 2–4, theta 5–8, alpha 9–13, beta 14–30, gamma
  31–71.
- **Intuition:** the model learns nothing; it is a test of a frozen model. Swap alpha for a random
  chunk's alpha: if accuracy drops the model used alpha, if not it ignored it.
- **For:** explains the exact Table II models; EEG precedent (Schirrmeister 2017).
- **Against:** bands are correlated, so shuffling produces unrealistic spectrograms (Hooker 2021).
  Without retraining, information loss can't be separated from unfamiliar input (ROAR 2019).
- **Cost:** the shuffle is a few lines. The real work is a harness that rebuilds each fold's
  validation set and verifies the unmodified evaluation reproduces the checkpoint's stored
  `val_metrics` (the "harness gate", §9.4).
- **If done:** report per band **and per dataset** (`val_metrics["per_dataset"]`; given §3.4 the
  per-dataset view is essential, since an MDD-only effect would otherwise hide inside the pooled
  number), cite Schirrmeister as precedent and Hooker 2021 as limitation, let B independently back
  "not muscle". Skip fill values and width-matched controls.

---

## 7. Open items and suggested next steps (in order, as of 2026-09-19 evening)

**Needs the user (merging):** mtichavsky/eeg #87; mtichavsky/thesis-text #4 (then
`git checkout -- paper/paper.tex` in the thesis checkout). Whitham 2007 is already in
`references.bib` and cited in `sec:cutoff` (done in PR #4).

1. **Per-dataset metrics fix: PR #91 is merged** (code-c0: `main.py:259` argument order, tests, and
   the single-fold crash in `bipolar_ablation_summary.py`). Regenerate anything using per-dataset sens/spec (`per_dataset_comparison.png` if it uses
   those columns; the paper caption says accuracy only, so probably unaffected; unverified).
   `helpers-print/dataset_prior_baseline.py` already handles both orientations.
2. **Objection 1, cheapest first (the priority).**
   1. **Single-dataset CV runs: DONE** (§3.6). Version `041` is shared with B's
      `all_041_*_f{70,30}` runs by coincidence; the families are told apart by the dataset prefix and
      the `_f70`/`_f30` suffix. Result: MDD alone ≈ 90 % (positive control), CANE alone
      68.4 / 66.9 % balanced accuracy (weak signal, same as in the combined model), SAD alone
      70.5 / 65.7 %. The balanced-accuracy test is in PR #92
      (`feat/single-dataset-balanced-accuracy`, code-c0, merged; output verified), so Table V
      reproduces from the committed script.
   2. **Four-class shortcut analysis: IN PROGRESS (approved 2026-09-19).** 4-class runs stored
      per-dataset accuracy only, not class balance. Route chosen (no harness): store the full
      per-dataset confusion matrix and class counts at training time, extend
      `dataset_prior_baseline.py` to print the 4-class decomposition, and rerun the 4-class
      AllTransformerV4 and in-ear CNN-AttnS (10-fold) on Metacentrum. A subagent is implementing it
      on branch `feat/per-dataset-confusion-matrix` (PR to follow, with a launcher
      `run-fourclass-perdataset-meta.sh`). **Jobs are submitted only after that PR is merged**
      (the cluster clone runs `main`). Optional, not started: a CANE-only anxiety vs comorbid run
      (CANE has 18 anxiety / 26 comorbid, no shortcut from other datasets).
   3. **Dataset-identity probe and shuffled-label retraining: NOT PLANNED (user, 2026-09-19).**
      Objection 1 is considered covered by the prior-baseline decomposition, the single-dataset
      runs (§3.6) and, once done, the 4-class decomposition. My reasoning: a linear probe would very
      likely find the source dataset trivially decodable from pooled features (recording equipment
      differs), which is uninformative about whether the classifier *uses* it, and it needs the
      dropped harness. A shuffled-label retraining (a CV-protocol sanity check, needs one training
      flag) remains an optional cheap extra. The paper says both were not run.
3. **Extend B: intermediate cutoffs DONE** (50 and 40 Hz, in-ear CNN-AttnS; §4.5, fetched to
   `experiments/all_042_inear_ec+eo_f{50,40}`, summary with
   `helpers-print/freq_cutoff_curve_summary.py --root experiments`). **No further cutoff experiments
   planned (user, 2026-09-19):** the curve is reported together with the muscle discussion in the
   paper (`sec:cutoff`), and a frontal-vs-temporal comparison (EMG is strongest frontally) and a
   cutoff below 30 Hz are listed there as untested. AllTransformerV4's +5.6 pp is under-powered at
   10 folds and can't be improved by adding folds (10-fold is mandatory), so the in-ear model
   carries the significant result.
4. **Fold-rebuild harness: DROPPED (user decision 2026-09-19).** A post-hoc harness that
   rebuilds folds and re-evaluates checkpoints adds code complexity. Preferred route for any new
   metric is to add it to the training/eval code and rerun on Metacentrum; that is what item 2.2
   does for the per-dataset confusion matrix. C, the asymmetry swap and the probe are dropped with
   the harness unless the user asks again (§9.4 kept for reference).
5. **OUT OF SCOPE FOR TODAY (2026-09-19; the user may pick it up in another session), objection 4:**
   C (band shuffle). If revived: report per band **and per dataset**.
6. **OUT OF SCOPE FOR TODAY (same), alpha asymmetry swap:** swap Fp1↔Fp2, C3↔C4, T7↔T8 in AllTransformerV4 inputs (channel
   indices in §9.1). Cheap once the harness exists (Aug 16 name: E2).
7. **`run-explainability-analysis.sh`:** one script for every post-training analysis, taking the
   runs directory, checking each run has 10 `fold_N_best.pth` and a `results.txt`, running A's and
   B's summary scripts (including fold-identity checks) plus the dataset-prior baseline, writing to
   `docs/explainability-results/YYYY-MM-DD/`, with placeholders for C and the swap.
8. **Optional:** AllTransformerV4 on T7+T8 only (needs multi-channel subset support in
   `--channel`, which doesn't exist).
9. **Housekeeping:** fix the summary script's eye-movement message (§3.3 item 3); clean up the
   untracked Aug 16 leftovers (`thesis/explain/`, `tests/test_explainability.py`,
   `run-explainability.sh`, `helpers-print/plot_explainability.py`, `EXPLAINABILITY.md`);
   there is no `EXPERIMENTS.md` in the repo, although `CLAUDE.md` says to log experiments there.
10. **Section VII:** already holds A, B, the decomposition and the single-dataset runs
    (thesis-text PR #4). When item 2.2 produces results, replace the matching red bullet in `sec:planned`; the
    biomarker bullet (objection 4) stays red as out of scope for now. Print budget in §9.5.

---

## 8. Shared statistics protocol

- Compare per fold against the same fold's baseline, never against pooled means.
- Wilcoxon signed-rank as primary (n = 10 folds). Also the Nadeau–Bengio corrected t-test, since
  CV folds share training data and the naive paired t-test finds too many significant differences.
- Benjamini–Hochberg correction within each family of comparisons.
- Confirm fold identity from the `Val subjects (N): [...]` log lines before any paired test.
  Note: EC and EO runs of the same montage do **not** share folds; in-ear (IDUN) runs don't share
  folds with headset runs.
- With about 17 subjects per validation fold, 1–2 point differences are unlikely to be
  significant, and run-to-run noise is about 1–3 pp (§3.3). Report nulls honestly.
- Effect sizes: mean difference with a bootstrap 95 % CI; state plainly that CV folds are not
  independent.
- **10-fold CV is mandatory:** cut configurations, never folds.

---

## 9. Technical facts retained from the Aug 16 catalogue

### 9.1 Corrected facts (the older `EXPLAINABILITY.md` is stale on these)

| Stale claim | Actual (verified) |
|---|---|
| Input `(B, 8, 129, 41)` | `(B, 8, 72, 41)`: `EXPECTED_SPECTROGRAM_SHAPE`, `thesis/stft.py`, after the 70 Hz crop |
| Channel order `Fp1, Fp2, T7, T8, C3, C4, Cz, Oz` | `Fp1, Fp2, C3, Cz, C4, T7, T8, O2/Oz`: `CANONICAL_CHANNEL_ORDER`, `thesis/dataset.py:39` (re-checked 2026-09-19) |
| `create_model(strict=False)` silently drops mismatched layers | PyTorch raises on shape mismatch regardless. But `strict=False` *does* let a stale checkpoint load partially with missing/unexpected keys, so harnesses must load with `strict=True` |

So **channel index → electrode is `0=Fp1, 1=Fp2, 2=C3, 3=Cz, 4=C4, 5=T7, 6=T8, 7=O2/Oz`**. The
temporal pair is indices 5 and 6. Mirror swap = `(0,1) (2,4) (5,6)`, `Cz` and `O2/Oz` fixed.

### 9.2 Architecture facts

- `AllTransformerV4` (`thesis/model.py`, ~744–784): per-channel shared CNN → `(B, 8, 10, 416)` →
  `proj: Linear(416→64)` → `+chan_embedding +time_embedding` → flatten to **80 tokens**, ordered
  **`token = channel*10 + time_frame`** → 2-layer `TransformerEncoder`, `nhead=4`,
  `norm_first=True` → **mean-pool over all 80 tokens**, no CLS token. 113,474 params.
- `CNNAttnS` (`model.py:312`; forward inherited from `CNNLSTM.forward`): single channel, **10 time
  tokens only**, 1 encoder layer, `d_model=64`. No channel axis: its only ablation axes are
  frequency and time.
- Frequency axis: bin `i` is centred at `i × 0.9765625 Hz`. Band → bin sets
  `[i for i in range(72) if lo <= i*0.9765625 < hi]`: delta 2–4 (3 bins), theta 5–8 (4),
  alpha 9–13 (5), beta 14–30 (17), gamma 31–71 (41). **Widths are wildly unequal**, so raw
  occlusion drops are not comparable across bands.
- Inputs are `log1p|STFT|`, so zero is not neutral ("silence", far outside the data distribution).

### 9.3 Checkpoint situation

- `code/models/*.pth` and everything under `/home/milan/eeg/experiments/` predate the Jul-30 STFT
  fix (`proj.weight` is `(64, 864)`, current code needs `(64, 416)`). **Unusable.**
- `/home/milan/eeg/experiments/{binary,4class}/` holds the valid paper runs (10/10 folds for every
  model of interest; `thesis-text/paper/experiments/` does not exist, an earlier version of these
  notes had the wrong path), **except** the headline 8-channel binary row (76.5/78.6, Table II) from
  `atv4-lowlr_oex` (`lr=1e-4, epochs=200, patience=50`): its checkpoints were **not kept**, only
  `results.txt`. The dir with checkpoints (`binary/8channel/alltransformer-binary`) is the earlier
  `lr=5e-4` run at 75.47/75.93. Hence B's 70 Hz AllTransformerV4 rerun.
- Exp A checkpoints (`experiments/all_040_*`) are valid current-code CNNAttnS checkpoints, usable
  for C on the single-channel model.
- B checkpoints (2026-09-19): `experiments/all_041_all_ec+eo_{f70,f30}/` (AllTransformerV4, 10
  folds each) and `experiments/all_041_inear_ec+eo_{f70,f30}/` (CNN-AttnS). `_f70` AllTransformerV4
  is a valid current-code replacement for the lost Table II checkpoints (75.4 vs 76.5 chunk acc).
- `RANDOM_SEED = 42` (`main.py:60`) seeds **only** fold assignment (one `np.random.RandomState`);
  there is no `torch.manual_seed`. If run-to-run noise ever matters more than the ~1–3 pp seen
  here, add torch seeding in `train()`.

### 9.4 Harness requirements for any post-hoc analysis (C, asymmetry swap, probes)

- New package `thesis/explain/` keeps `main.py` thin. A `harness.py` **reuses**
  `get_datasets_for_fold` (`thesis/data_preparation.py:589`), `create_model`
  (`thesis/model_factory.py:272`), `eval_epoch` (`main.py:186`) and `collate_spectrograms`
  (`thesis/dataset.py:1628`). Load with `strict=True` and assert empty missing/unexpected keys.
- **The harness gate:** for every fold, the *unablated* `eval_epoch` output must reproduce
  `torch.load(ckpt)["val_metrics"]` (checkpoints store the full metrics dict, `main.py:404-414`).
  If it doesn't, fold reconstruction or preprocessing is wrong and every downstream number is
  void. Cross-check the validation set against the `Val subjects (N): [...]` log lines
  (`thesis/data_preparation.py:642`).
- Band→bin sets and channel-token masks must be derived from `thesis/stft.py` constants, never
  hardcoded.
- Post-hoc channel removal for `AllTransformerV4` is cleanest as a `channel_mask` argument on
  `forward` (select kept tokens after the flatten, before the transformer; `None` must be
  bit-identical to the current forward). Mean-pool renormalises automatically, which beats zeroing
  a channel's spectrogram (that leaves `chan_embedding[c]` and a "silent channel" cue).
- Project conventions apply: `logging` not `print`, mypy annotations, `make format && make
  typecheck`.

### 9.5 Aug 16 experiment catalogue: fate of each item

| Aug 16 | What | Fate |
|---|---|---|
| E1 | Leave-one-channel-out / region ablation (post-hoc, AllTransformerV4) | Dropped (§5.3–5.4) |
| E2 | Hemispheric mirror probe | **Kept** as the asymmetry swap (§7.6) |
| E3 | Channel × band occlusion grid with mean fill and width-matched control | Dropped in that form; simplified to C |
| E4 | Integrated Gradients + attention rollout, Spearman agreement | Not scoped. If ever revived: hand-rolled IG (no `captum` dependency, baseline must be in-distribution, completeness axiom as a unit test); attention rollout needs a forward-pre-hook because `nn.TransformerEncoderLayer` uses `need_weights=False` and `norm_first=True` requires re-applying `norm1`; mean-pool readout means per-token influence is the column mean of the rollout matrix. Attention is not explanation (Jain & Wallace 2019), so ablation stays primary |
| E5 | Reduced-montage retraining (all-8, top-4, frontal, temporal, best single) | Mostly dropped; only T7+T8 kept as optional (§7.7) |
| E6 | Correlate CNN-AttnS band profile with AllTransformerV4's T7/T8 profile | Depends on C |
| E7 | XAI sanity checks (Adebayo et al. 2018) on random-init and shuffled-label models | Worth doing if C or any attribution is used |
| E8 | Site-confound audit: linear probe + within-CANE anxiety-vs-comorbid retrain | **Kept**, now objection 1 (§7.2) |
| E9 | Shuffled-label control retrain | Kept as part of objection 1 (§7.2) |

Print budget when Section VII is written: **≤ 1 page**. One composite figure or one compact
table plus dense prose with inline numbers; full results go to the thesis and JSON artifacts. Match
`paper.tex` LaTeX conventions: `[!t]` floats, caption before `\label` for tables and after the
graphic for figures, `\hline` (no booktabs), `\renewcommand{\arraystretch}{1.3}`, `\small`, group
headers as `\multicolumn{N}{@{}l}{\textit{…}}`, values as `$76.5{\pm}4$`, thin space in `113\,K`.
Plot scripts follow `helpers-print/` conventions: data hardcoded under a `# ── Data ──…` banner,
matplotlib PNG at 600 dpi with `bbox_inches="tight"` written to `docs/`, then copied by hand to
`thesis-text/obrazky-figures/`.

---

## 10. Metacentrum notes

Exp A ran on Metacentrum (PBS) because no personal GPU is available. Setup: repo cloned +
`poetry install`'d into a shared `$STORAGE_HOME/eeg/.venv` on `/storage/brno2` (persistent, not
`$SCRATCHDIR`, which is wiped per job); one `qsub` per experiment via `metacentrum/train_job.pbs`
using `run-bipolar-meta.sh` (and `run-round1-atv4-meta.sh` for AllTransformerV4); walltime 5 h
(plenty; heavy 8-channel AllTransformerV4 `ec+eo` configs finished in ~40–60 min). Launching from
`$STORAGE_HOME/eeg-code` on the frontend, e.g. `./run-bipolar-meta.sh` (required 8),
`./run-bipolar-meta.sh --with-ec+eo` (+3 optional). Results are rsynced from
`/storage/brno2/home/tichavskym/experiments/`.

**Two clusters excluded from scheduling** (`cl_fobos=False:cl_grogu=False` in the job template's
`#PBS -l select=...` line, PR #86, merged at `cd921b6`):
- `fobos` (ZCU): can't see packages installed into the shared `.venv` (numpy/torch
  `ModuleNotFoundError` though the venv is intact from the frontend); reproduced on two fobos nodes.
- `grogu` (CERIT-SC): `sm_120` (Blackwell) GPUs, no compiled kernels in `torch==2.11.0+cu126`
  (`CUDA error: no kernel image is available`). Revisit if torch is upgraded.

**B run notes (2026-09-19):** `./run-freq-cutoff-meta.sh` submitted 4 jobs (job names with `+`
are accepted by qsub). Wall times: in-ear CNN-AttnS 6–7 min, AllTransformerV4 `ec+eo` ~28 min.
Quirks worth knowing:
- The job template creates `experiments/<name>/` and tees `stdout.log` into it, so `main.py` finds
  the directory existing and writes checkpoints, `results.txt` and logs to a **random-suffix
  sibling** (`<name>_buf`, `_xwn`, `_gzn`, `_ubd`). Fetch both:
  `rsync -a metacentrum:/storage/brno2/home/tichavskym/experiments/<name><suffix>/ experiments/<name>/`
  and `rsync -a metacentrum:.../experiments/<name>/stdout.log experiments/<name>/`.
- The cluster clone `/storage/brno2/home/tichavskym/eeg` has 36 untracked files (old PBS `.o`
  files, `get-pip.py`, `diag_modules.pbs`), so `git_commit` reads `<hash>-dirty`. Harmless.
  The clone was left on `metacentrum-cluster-exclusions` (`cbdaaf3`), which is now behind `main`:
  `git fetch && git checkout` the branch to run before launching, and do not switch branches
  while jobs are queued or running (jobs read the code when they start).
- Subagents can run this end to end (submit, monitor with `qstat`, rsync), but a long monitor
  can hit the session usage limit; the jobs themselves are unaffected.

Also: the frontend's local `/tmp` has a tiny per-user quota (977 MB, separate from the large
`/storage/brno2` quota). `poetry install`'s CUDA wheel downloads will `EDQUOT` there unless
`TMPDIR` points at `/storage/brno2` first.

---

## 11. Key file index

| Path | What |
|---|---|
| `docs/plans/2026-09-14-bipolar-surrogate-ablation.md` | Plan A |
| `docs/plans/2026-09-14-frequency-cutoff-ablation.md` | Plan B (in `main`) |
| `run-bipolar.sh`, `helpers-print/bipolar_ablation_summary.py`, `tests/test_bipolar_channels.py` | A implementation |
| `run-bipolar-meta.sh`, `metacentrum/train_job.pbs` | A launch on Metacentrum (one `qsub` per config) |
| `experiments/all_040_{fp2-fp1,c4-c3,t8-t7,inear}_{ec,eo}/` | A results (local, synced 2026-09-18) |
| `tests/test_freq_cutoff.py`, `thesis/stft.py` (`spectrogram_shape`, `num_freq_bins`) | B implementation (flag) |
| `run-freq-cutoff-meta.sh`, `helpers-print/freq_cutoff_summary.py`, `tests/test_freq_cutoff_summary.py` | B launcher and summary (PR #89, merged) |
| `experiments/all_041_{inear,all}_ec+eo_{f70,f30}/` | B results (local, synced 2026-09-19) |
| `helpers-print/dataset_prior_baseline.py`, `helpers-print/single_dataset_summary.py`, `run-single-dataset-meta.sh` | Objection-1 tools (PR #88, merged; single-dataset runs submitted 2026-09-19) |
| `thesis/dataset.py` (`BIPOLAR_CHANNELS`, `bipolar_pair`, `CANONICAL_CHANNEL_ORDER`) | Channel specs |
| `main.py:186` (`eval_epoch`), `main.py:259`, `thesis/metrics.py:321` | Per-dataset metrics and the swapped-argument bug |
| `../experiments/binary/in-ear/cnnattns-binary_wve/` | Table II in-ear run, 10 checkpoints |
| `../experiments/binary/8channel/alltransformer-binary/` | AllTransformerV4 lr 5e-4 run with checkpoints (fallback for C) |
| `../experiments/results.txt` | Table II AllTransformerV4 config (`atv4-lowlr_oex`, no checkpoints) |
| `thesis/explain/`, `tests/test_explainability.py`, `run-explainability.sh`, `helpers-print/plot_explainability.py`, `EXPLAINABILITY.md` | Untracked Aug 16 leftovers (§7.8) |

Design point from A worth remembering: on MDD/SAD, `T8-T7` is bit-identical to `in-ear`, so those
two runs differ only in CANE's source. The three headset montages take CANE from the 8-channel
headset (raw voltages subtracted first, no CAR, per commit `ef3258c`); `in-ear` takes it from
IDUN. Physical caveat: electrode spacing differs (T7–T8 ≈ 80 % of the ear-to-ear arc, C3–C4 ≈ 40 %,
Fp1–Fp2 much less).

---

## 12. Resources

**In-ear surrogate (already cited in the paper):** Tremmel et al. 2024; Moumane et al. 2024. These
are what A tests.

**Post-hoc permutation importance (C):**
- [Breiman 2001, "Random Forests", *Machine Learning* 45:5–32](https://doi.org/10.1023/A:1010933404324): origin of permutation importance.
- [Fisher, Rudin & Dominici 2019, "All Models are Wrong, but Many are Useful", *JMLR* 20(177)](https://jmlr.org/papers/v20/18-760.html): "model reliance".
- [Schirrmeister et al. 2017, *Human Brain Mapping*](https://arxiv.org/abs/1703.05051): EEG precedent, frozen ConvNet with spectral amplitudes perturbed per frequency (Supplementary A.5.2).
- [Molnar, *Interpretable Machine Learning*, ch. 23](https://christophm.github.io/interpretable-ml-book/feature-importance.html): overview incl. the correlated-feature drawback.

**Critiques of post-hoc permutation (these motivate B):**
- [Hooker, Mentch & Zhou 2021, "Unrestricted permutation forces extrapolation", *Statistics and Computing* 31:82](https://arxiv.org/abs/1905.03151).
- [Hooker, Erhan, Kindermans & Kim 2019, ROAR, NeurIPS](https://arxiv.org/abs/1806.10758): remove and retrain.

**Muscle artifact (B):**
- [Whitham et al. 2007, *Clinical Neurophysiology* 118(8)](https://pubmed.ncbi.nlm.nih.gov/17574912/): EEG above 20 Hz contaminated by EMG.
- [Goncharova et al. 2003, *Clinical Neurophysiology* 114(9)](https://doi.org/10.1016/S1388-2457(03)00093-2).

**Statistics:**
- [Nadeau & Bengio 2003, *Machine Learning* 52:239–281](https://doi.org/10.1023/A:1024068626366): corrected resampled t-test for CV.

**Attribution sanity (if E4/E7 are revived):** Adebayo et al. 2018 (sanity checks for saliency
maps); Jain & Wallace 2019 (attention is not explanation); Abnar & Zuidema 2020 (attention
rollout).
