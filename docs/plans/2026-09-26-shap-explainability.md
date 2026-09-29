# Plan: grouped SHAP for the spectrogram models

**Goal:** Replace the paper's current explainability material (§VII, or at least §VII-A) with
SHAP attributions for the two headline models, AllTransformerV4 (8-ch) and CNN-AttnS (in-ear).
Attributions are grouped by **frequency band**, **electrode** and **electrode × band**, and are
computed out-of-fold under the same 10-fold subject-independent protocol as every other number
in the paper.

**Background / rationale:** `thesis-text/paper/SHAP_EXPLAINED.md`.

**Approach in one sentence:** Reuse the existing `thesis/explain/` harness for fold
reconstruction, checkpoint loading and the baseline-verification gate. Add one module with a
small **value-function wrapper** that turns a player on/off vector into masked spectrograms
(vectorised on the GPU) and returns the averaged model output. The `shap` library does all the
Shapley math: `shap.explainers.Exact` for the band and channel games, and
`shap.explainers.Permutation` for the 48-player channel × band grid. No Shapley algorithm is
hand-written.

---

## 0. State of the codebase (verified 2026-09-26, branch `exp/atv4-round2`)

What exists and gets reused:

| Piece | Where | Reuse |
|---|---|---|
| Run-config parser | `thesis/explain/harness.py::read_run_config` | as-is |
| Fold rebuild + strict checkpoint load | `harness.py::build_folds`, `iter_folds`, `load_fold_model` | as-is |
| Baseline verification gate | `harness.py::verify_baseline` | run before every SHAP fold |
| Training-fold mean spectrogram | `harness.py::reference_spectrogram` | mean-baseline variant |
| Band → bin mapping | `thesis/explain/masks.py::band_bins`, `EEG_BANDS` | defines band players |
| Channel order | `thesis/dataset.py:40` `CANONICAL_CHANNEL_ORDER` = `Fp1, Fp2, C3, Cz, C4, T7, T8, O2/Oz` | channel players |
| IG + completeness check | `thesis/explain/attribution.py` | pattern to copy, and a cross-check |
| Stats (Wilcoxon, NB-corrected t, BH, Spearman) | `thesis/explain/stats.py` | per-dataset comparisons, rank agreement |
| Plot helper | `helpers-print/plot_explainability.py` | extend or copy for SHAP figures |

Gaps found. These must be fixed before anything runs:

1. **No `explain` subcommand exists.** `thesis/cli.py` only defines `train` and `run`, yet
   `run-explainability.sh` calls `python main.py explain …`. It was never committed on any
   branch (`git log --all -S'add_parser("explain"'` is empty). It has to be added.
2. **`AllTransformerV4.forward` takes no `channel_mask`** (`thesis/model.py`), while
   `harness.AblatedModel` passes one. SHAP does not need it, because it masks by replacement,
   but the channel-ablation experiment is currently broken. Note it and leave it alone.
3. **`build_folds` / `load_fold_model` ignore `freq_cutoff`.** They always use
   `EXPECTED_SPECTROGRAM_SHAPE` (70 Hz). That is fine for the f70 runs this plan targets, but
   the 30/40/50 Hz runs cannot be explained until `spectrogram_shape(config.raw["freq_cutoff"])`
   is threaded through.
4. **Checkpoints.** The Table II headline ATV4 run (`atv4-lowlr_oex`) kept none. Checkpoints
   that fit the current code exist for:
   - `experiments/all_041_all_ec+eo_f70/` (AllTransformerV4, binary, 75.4% chunk)
   - `experiments/all_041_inear_ec+eo_f70/` (CNN-AttnS, binary, in-ear, 77.1% chunk)

   `experiments/all_043_*_4class/` has **no** `.pth` files. 4-class SHAP needs a rerun
   (Phase 6).

---

## Phase 1: Plumbing (½ day)

1. **Add the `explain` subcommand** in `thesis/cli.py` and dispatch it in `main.py`:
   ```
   python main.py explain <run-dir> --experiment {shap-bands,shap-channels,shap-grid,shap-pixel,shap-all}
          [--background-size 32] [--max-chunks-per-fold 400] [--folds 1,2,...]
          [--baseline {sample,mean}] [--grid-permutations 64] [--seed 42]
   ```
   Write outputs to `<run-dir>/explain/shap_<experiment>.json` plus
   `<run-dir>/explain/shap_<experiment>_chunks.npz`.
   Keep the existing experiment names (`channels`, `bands`, `attribution`, …) in the
   `choices` list if they are wired at the same time. That is optional and out of scope here.
2. **Thread `freq_cutoff`** from `RunConfig.raw` into `build_folds` and `load_fold_model`,
   using `spectrogram_shape()` from `thesis/stft.py`. Add a test that a f30 run config yields
   the `(31, 41)`-type shape.
3. **Add the dependency:** `shap` as an **optional Poetry group `explain`** in
   `pyproject.toml` (like the existing `api` group), and update `poetry.lock`. Do **not** add
   it to `requirements.in`: that file is the API container's runtime. Training and the API
   must not import it. It is imported lazily inside `thesis/explain/shap_values.py` (the module name must not be
   `shap.py`, which would shadow the library), in tests, and in plotting.

**Done when:** `python main.py explain experiments/all_041_inear_ec+eo_f70 --experiment
shap-bands --folds 1 --max-chunks-per-fold 8` runs end to end, and the verification gate
logs `OK` for fold 1.

---

## Phase 2: Core module `thesis/explain/shap_values.py` (½–1 day)

### 2.1 Player groups: `thesis/explain/groups.py` (or inside `masks.py`)

```python
@dataclass(frozen=True)
class PlayerSet:
    names: list[str]
    masks: torch.Tensor   # bool, (M, C, F, T): True where the player's features live
```

Builders:
- `band_players(n_channels, freq_bins, time_frames)`: players `sub_delta, delta, theta,
  alpha, beta, gamma`. **`sub_delta` = every bin not covered by `band_bins()`** (bins 0–1).
  Assert that the masks **partition** the input: every pixel belongs to exactly one player.
- `channel_players(...)`: 8 players, one full `(F, T)` plane each.
- `grid_players(...)`: channel × band, 48 players.
- Unit invariant: `masks.sum(0) == 1` everywhere. If a pixel is left out, efficiency breaks
  silently. That is why sub-δ exists.

### 2.2 Value-function wrapper (the only EEG-specific core logic)

```python
def make_value_fn(model, x, background, players, target_fn, max_batch=8192):
    # x: (C, F, T), one chunk; background: (K, C, F, T); players.masks: (M, C, F, T)
    K = background.shape[0]
    def f(switches: np.ndarray) -> np.ndarray:        # (S, M) of 0/1, supplied by shap
        on = (torch.as_tensor(switches).float() @ players.masks.flatten(1).float())
        on = on.view(-1, *x.shape).bool()                        # (S, C, F, T)
        values = []
        for batch in on.split(max(1, max_batch // K)):           # coalitions per batch
            composite = torch.where(batch[:, None], x, background[None])  # (s, K, C, F, T)
            out = target_fn(model(composite.flatten(0, 1)))
            values.append(out.view(len(batch), K).mean(1))       # average over background
        return torch.cat(values).cpu().numpy()
    return f
```
- Composite input `where(on, x, b)`, **interventional**: present players come from `x`,
  absent ones from the same background chunk `b`, then averaged over the K background chunks.
- Composites are built **inside** the batch loop, so at most `max_batch` spectrograms exist
  at once (the channel game would otherwise materialise 256 × 32 = 8,192 per call). Runs
  under `torch.no_grad()` in `eval()` mode.
- `target_fn` for binary: `logits[:,1] − logits[:,0]` (log-odds of pathological). For 4-class:
  all logits, giving `(S, n_classes)`.

### 2.3 Shapley values via the `shap` library

The trick: explain the **on/off vector**, not the spectrogram. The explained point is all ones
(the real chunk); the masker's background is all zeros (every player replaced by background).

```python
M = len(players.names)
masker = shap.maskers.Independent(np.zeros((1, M)))
f = make_value_fn(model, x, background, players, target_fn)
explainer = (shap.explainers.Exact(f, masker) if M <= 10
             else shap.explainers.Permutation(f, masker, seed=seed))
result = explainer(np.ones((1, M)), **({} if M <= 10 else {"max_evals": ...}))
phi, base = result.values[0], result.base_values[0]   # base == mean_k f(b_k)
```
- **Bands (6) and channels (8): `Exact`**, which evaluates all `2^M` coalitions. Cost per
  chunk: `2^M × K` forward passes. Bands: 64 × 32 = 2,048. Channels: 256 × 32 = 8,192.
- **Grid (48): `Permutation`**, which samples permutations and walks each one forwards and
  backwards (antithetic), so efficiency still holds exactly. Each permutation therefore costs
  about 2 × M coalitions, not M + 1. Convert `--grid-permutations` into the library's
  `max_evals` explicitly, and log the number of coalitions actually evaluated on the first
  chunk to confirm the conversion. Use `K=8` or `--baseline mean` to keep the cost down. The library returns no standard error, so estimate it by repeating the
  grid run with 3–5 seeds on a subset of chunks and store the per-cell SE. The paper should
  only interpret grid cells whose SE is well below |φ|.
- One explainer is built per chunk (the wrapper closes over `x`). The library overhead is
  negligible next to the forward passes.
- Pin the `shap` version; `Exact`/`Permutation` call signatures have changed between releases.

### 2.4 Background sampling

```python
def sample_background(context: FoldContext, k: int, seed: int,
                      dataset: str | None) -> Tensor
```
- **Headline: same-dataset background.** Each chunk is explained against K chunks from **its
  own dataset**, so φ means "different from a typical chunk of the same recording setup".
  Absent players are filled from recordings with the same hardware (or, for in-ear, the same
  real/synthetic origin), which keeps dataset identity out of φ. The paper already treats
  dataset identity as a possible shortcut (§VII, dataset-prior decomposition).
- Per fold, draw one background per dataset (3 per fold, K = 32 each), and pick the one that
  matches the chunk's dataset via `subject_dataset_map`. Same compute as a single pooled
  background.
- From `context.train_loader` **only** (never validation). Within the dataset, **stratify by
  label** (healthy / pathological), using the subject IDs from the batch tuple
  `(inputs, labels, subject_ids)`. Draw from as many different subjects as possible.
- `dataset=None` gives the pooled background (stratified by dataset × label), used only by
  the optional Phase 8 comparison.
- The `--baseline mean` variant uses a **per-dataset** mean training spectrogram as a single
  `K=1` background. `reference_spectrogram(context)` averages over all datasets, so add a
  `dataset` filter to it.

### 2.5 Efficiency check (non-negotiable, like IG's completeness check)

```python
residual = φ.sum(-1) − (f(x) − mean_k f(b_k))
```
- Compute it ourselves from `f(np.ones)` and `f(np.zeros)`, independent of the library's
  `base_values`. This also guards against a masker misconfiguration.
- Exact: `max |residual|` must be < 1e-4. Otherwise raise.
- Permutation: exact per permutation too. Log the max.
- Store `max_efficiency_residual` in the JSON so the paper can quote it.

### 2.6 Pixel-level Expected Gradients (optional, for the attribution-vs-frequency figure)

- `shap.GradientExplainer(model, background)` (Expected Gradients). The library is a
  dependency anyway now, so no hand-rolled version.
- Reduce `|φ|` over channels and time to a 72-long curve per fold. Keep the signed version
  too.

---

## Phase 3: Experiment runner `thesis/explain/experiments.py::run_shap(...)` (1 day)

For each fold from `iter_folds(config, device, folds)`:

1. `verify_baseline(context, config, device)`. **Skip the fold and log an error on a
   mismatch.** Without this, the fold's subjects may be the wrong ones.
2. Sample one background per dataset (2.4).
3. Iterate `context.loader` up to `--max-chunks-per-fold` and collect `x, y, subject_ids`.
   If truncated, subsample **evenly across subjects** (cap chunks per subject, e.g.
   `ceil(max_chunks / n_subjects)`), then check the dataset × label mix is preserved.
4. Compute φ (exact or permutation), `f(x)`, `E f(b)` and the predictions.
5. Append per-chunk rows: `fold, subject, dataset, label, pred, fx, fbase, φ[M]` → npz.

Aggregations written to JSON, all as **per-fold values plus mean ± std over folds**. The unit
is the **subject**: average φ over a subject's chunks first, then over subjects within the
fold, then over folds. Otherwise subjects with many chunks dominate.

- `mean_abs_phi[player]`: global importance
- `mean_phi[player]`: net direction toward pathological
- the same two **split by dataset** (MDD / CANE / SAD), **by true class**, and **by
  correct/incorrect**. For the **in-ear** model the "cane" label is really **IDUN** (real
  in-ear recordings), while MDD and SAD are **synthetic in-ear derivations**. Name them that
  way in the JSON and the plots; a different band profile for real vs synthetic in-ear is a
  result worth reporting.
- `mean_abs_phi_per_bin[player]` = `mean_abs_phi / n_bins` (bands only, for the width caveat)
- `beeswarm` inputs: per-chunk band log-power (mean of `x` over the player's mask) next to φ
- `max_efficiency_residual`, `background_size`, `baseline`, `n_chunks_per_fold`, `seed`,
  git SHA

Statistics (reuse `stats.py`):
- Per-dataset differences in band importance: Wilcoxon over folds, plus the Nadeau–Bengio
  corrected t, with BH correction within the band family.
- Rank agreement between the SHAP channel/band ranking and the existing IG / occlusion
  rankings: `spearman`. The existing `EXPLAINABILITY.md` already argues that agreement across
  methods is the strong result.
- Sample-vs-mean baseline agreement: Spearman on `mean_abs_phi`. This is the robustness line
  for the paper.

Sanity control (reuse the idea of `run_sanity_checks`): run `shap-bands` once on a
**randomly re-initialised** model (`_reinitialise`). Its band profile must differ clearly from
the trained one (Adebayo et al. 2018 model-randomisation test). **Pass criterion, fixed in
advance:** Spearman ρ between the trained and random `mean_abs_phi` band profiles is below
0.5 on the fold average, and the trained model's efficiency gap `f(x) − E f(b)` is larger.
Report ρ in one sentence.

---

## Phase 4: Tests `tests/test_shap.py` (½ day; write these first, TDD)

All tests run on CPU with tiny synthetic tensors, with no datasets and no checkpoints. The
Shapley math belongs to the library, so the tests target **our** code: the wrapper, the
masks, the background sampler, and whether the library is wired up correctly.

1. **Wrapper endpoints:** `f(ones) == target(model(x))` and `f(zeros) == mean_k
   target(model(b_k))`. This is the most likely place for a bug.
2. **Linear model, analytic answer:** `f(x) = Σ w·x`. With a mean baseline, φ_group =
   Σ_{pixels in group} w·(x − μ). The full pipeline must match to 1e-6. This catches a wrong
   mask/player mapping that the efficiency check alone would miss.
3. **Efficiency:** random small CNN, 6 band players → `|Σφ − (f(x) − E f(b))| < 1e-5`.
4. **Dummy:** a model that zeroes channel 3 internally → φ_channel3 == 0.
5. **Partition invariant:** every builder's masks sum to 1 over players at every pixel,
   including sub-δ for the real 72-bin geometry.
6. **Permutation vs Exact:** on 6 players, `Permutation` with a large `max_evals` is close to
   `Exact`. Confirms the grid configuration (masker, `max_evals`) before trusting 48 players.
7. **Background never touches validation, and matches the dataset:** the sampler, given a
   fold, returns only training subject IDs, all from the requested dataset, with both labels
   represented. Use a small fake `FoldContext` (tiny in-memory loaders with made-up
   subject IDs); unit tests must never load real data.
8. **CLI smoke test:** argparse accepts `explain <dir> --experiment shap-bands`.

Then run `make format` and `make typecheck`, as `CLAUDE.md` requires.

---

## Phase 5: Runs (compute)

Rough costs, to be confirmed on the first fold. Assume about 20k spectrogram forwards/s for
ATV4 on one GPU, and more for CNN-AttnS. Use `--max-chunks-per-fold 400` (stratified), which
gives about 4k chunks total.

| Run | Model | Players | Evals/chunk | Est. time |
|---|---|---|---|---|
| shap-bands | CNN-AttnS in-ear | 6, exact, K=32 | 2,048 | ~5 min |
| shap-bands | ATV4 8-ch | 6, exact, K=32 | 2,048 | ~10 min |
| shap-channels | ATV4 8-ch | 8, exact, K=32 | 8,192 | ~30 min |
| shap-grid | ATV4 8-ch | 48, 64 perm (antithetic), K=8 | ~50k | ~3 h |
| shap-pixel | both | Expected Gradients, 64 samples | 64 fwd+bwd | ~10 min |
| mean-baseline robustness | both, bands + channels | K=1 | ÷32 | minutes |
| random-init control | both, bands | exact | 2,048 | ~10 min |

Add a `run-shap-meta.sh` in the style of `run-freq-cutoff-meta.sh` for MetaCentrum. The
existing `run-explainability.sh` `posthoc` stage can call the new experiments too.

---

## Phase 6: Optional 4-class (1 training run + ~1 h SHAP)

- Rerun `all_043_all_ec+eo_4class` and `all_043_inear_ec+eo_4class` with the same configs.
  Confirm first that `--save-every` / best-checkpoint saving is enabled so that `.pth` files
  are kept.
- Run `shap-bands` with a per-class target (`n_classes` outputs). The interesting comparison
  is φ toward *anxiety* vs φ toward *comorbid* **within CANE only**. If the two profiles are
  indistinguishable, that explains the anxiety/comorbid confusion in the paper.

---

## Phase 7: Figures and paper text

New `helpers-print/plot_shap.py`, which reads the JSON/npz and writes PNGs to
`thesis-text/obrazky-figures/`:

1. `shap_bands.png`: grouped bars of mean |φ| per band, two panels (in-ear | 8-ch), hue =
   dataset, error bars = std over folds. Add a per-bin variant as an inset or a second row.
2. `shap_channel_band.png`: 8 × 6 heatmap of signed mean φ (diverging colormap centred at 0),
   with cells hatched where SE ≥ |φ|/2.
3. `shap_beeswarm_inear.png`: band log-power (colour) against φ (x) for CNN-AttnS.
4. (optional) `shap_frequency_curve.png`: pixel EG |φ| against frequency for both models,
   with a vertical line at 30 Hz.

Paper edits in `paper.tex`:
- Remove §VII-A (or all of §VII) as decided. Fix the dangling `\ref{sec:shortcut}` in §VI-C
  (line ~707) and the roadmap sentence in the Introduction (line ~149).
- New subsection **"SHAP Attribution"**: method paragraph (players, exact enumeration,
  same-dataset training-fold background, logit margin, out-of-fold, efficiency residual, which checkpoints
  are explained, namely the `all_041_*_f70` reruns and not the Table II runs), Fig. bands,
  Fig. heatmap, and interpretation tied to §VII-B (T7/T8 share) and §VII-C (γ).
- Caveats paragraph: explains the model not the brain, correlated features, baseline
  dependence (including which datasets the background is drawn from), fold variability.
- Add bib entries: `shapley1953`, `lundberg2017shap`, `sundararajan2017ig`,
  `erion2021expected`, `adebayo2018sanity`.

---

## Phase 8: Optional robustness add-ons (only after Phase 7 is done)

None of these block the main results. They answer likely reviewer questions and are mostly
library calls or reuse existing code. Pick them up once the figures and text are finished.

1. **Kernel SHAP equivalence (test).** On the 6-band game, `shap.KernelExplainer` with
   `nsamples` ≥ 2⁶ matches `Exact` to 1e-5. Shows that our values are exactly what Kernel SHAP
   converges to. Closest prior work (STF-KernelSHAP, *Computers* 2026) uses Kernel SHAP only
   because its channel × time × frequency cells are too many to enumerate.
2. **Kernel SHAP vs Permutation on the grid.** On a subset of chunks (e.g. 50 per fold, 2
   folds), run `KernelExplainer` next to `Permutation` for the 48 players and report Spearman
   over the cells. Paper sentence: *"Kernel SHAP estimates the same interventional Shapley
   values; with ≤ 8 players we enumerate all coalitions, and for the 48-player grid Kernel SHAP
   and permutation estimates agree (ρ = …)."*
3. **Faithfulness: deletion curve.** Remove players in SHAP order vs random order and plot
   the drop in the logit margin. The value function is already there, so this is a few extra
   calls to `f`. Follows EEG-Xplain (Ma & Wang, arXiv 2026). (Rank agreement with occlusion
   and IG is already in Phase 3.)
4. **Background-size stability.** On one fold, K = 16 / 32 / 64 and three reseeded K = 32
   backgrounds; report rank agreement of `mean_abs_phi`. Answers Yuan et al. (arXiv 2022):
   SHAP rankings shift with the background, most for mid-importance features.
5. **Promote pixel-level Expected Gradients (2.6) from optional to reported**, as a second,
   independent method. Methods disagree strongly across architectures (EEG-Xplain), so
   cross-method agreement is the strongest defence.
6. **One sentence in the paper against attention maps as explanations** (Grad-CAM /
   attention did not hold up on transformers in EEG-Xplain).
7. **Mixed-dataset background comparison.** Rerun `shap-bands` and `shap-channels` with the
   pooled background (`dataset=None`, stratified by dataset × label). Report per band/channel
   the difference **pooled − same-dataset** in `mean_abs_phi`, which estimates how much of
   each player's credit comes from dataset identity rather than from the healthy vs
   pathological difference. Agreement is a robustness sentence ("band findings are not a
   dataset shortcut"); disagreement goes into the shortcut discussion. Expect the largest
   gap for the in-ear model (real IDUN vs synthetic MDD/SAD). Same compute as the headline
   runs.
8. **Bib entries** for whatever of the above gets cited: STF-KernelSHAP (*Computers* 2026,
   doi:10.3390/computers15070428), Contreras et al. 2024 (spectral-zones SHAP), EEG-Xplain
   (arXiv 2609.15687), Ravindran & Contreras-Vidal 2023 (*Sci. Rep.*), Yuan et al. 2022
   (arXiv 2204.11351).

---

## Order of work and checkpoints

0. **Install check:** `pip install shap && python -c "import shap"` in the project env
   (numpy 2.4.2, Python 3.12; `shap` depends on numba, which lags new numpy releases). If it
   fails, stop and pick a compatible `shap`/numpy pin before writing any code.
1. Phase 4 tests 1–6 (red) → Phase 2 core (green).
2. Phase 1 plumbing → smoke test on fold 1 with 8 chunks. **Check that the gate reads OK.**
3. Phase 3 runner → full `shap-bands` on in-ear, then read the numbers before running
   anything larger. If γ does not show up at all despite the cutoff result, stop and debug
   the masking first. **If γ dominates**, first check whether preprocessing notches 50 Hz. If
   it does not, rerun the band game with γ split into 30–45 / 45–70 Hz (7 players, exact,
   cheap) to rule out mains noise before interpreting γ.
4. Phase 5 remaining runs → Phase 7 figures and text.
5. Phase 6 only if time allows and the supervisor wants 4-class.
6. Phase 8 add-ons last, in the listed order, as time allows.

**Estimated effort:** about 2.5–3.5 working days of implementation and writing, plus about 4–5 h of
GPU time, not counting the optional 4-class retraining.

## Risks

| Risk | Mitigation |
|---|---|
| Fold reconstruction drifts, so SHAP explains the wrong subjects | `verify_baseline` gate per fold, refuse on mismatch |
| Off-manifold composites (γ from one chunk, α from another) | same-dataset, label-stratified background, mean-baseline robustness check, state it as a caveat |
| Dataset identity leaks into φ (hardware, real vs synthetic in-ear) | same-dataset background for all headline numbers; pooled-background comparison in Phase 8 |
| γ dominance just reflects band width | per-bin normalisation plus the width-matched argument from the occlusion code |
| Grid estimates too noisy | report SE, hatch uncertain cells, raise the permutation count only for the grid |
| Reviewers ask "why not DeepSHAP / KernelSHAP?" | DeepExplainer does not reliably support attention/LayerNorm. KernelSHAP approximates the same values we compute exactly. Say so in one sentence |
