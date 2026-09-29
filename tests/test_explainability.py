"""
Tests for the explainability package.

Everything here runs on synthetic tensors, so the suite is usable on a machine with neither
the EEG datasets nor a GPU. The end-to-end test substitutes fake folds and models for the real
ones, exercising the full ablation pipeline down to the statistics.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from thesis.dataset import (
    CANONICAL_CHANNEL_ORDER,
    num_model_channels,
    parse_channel_subset,
    resolve_channel_names,
)
from thesis.explain import attribution as attr
from thesis.explain import experiments as explain_experiments
from thesis.explain import harness
from thesis.explain.masks import (
    CHANNEL_REGIONS,
    EEG_BANDS,
    band_bins,
    channel_mask,
    matched_control_windows,
    mirror_channels,
    occlude,
    spectrogram_reference,
)
from thesis.explain.stats import (
    benjamini_hochberg,
    bootstrap_ci,
    nadeau_bengio_corrected_t,
    paired_wilcoxon,
    spearman,
)
from thesis.model import AllTransformerV4
from thesis.stft import EXPECTED_SPECTROGRAM_SHAPE, FREQ_BIN_WIDTH_HZ, NUM_FREQ_BINS

SPEC_SHAPE = EXPECTED_SPECTROGRAM_SHAPE
N_CHANNELS = len(CANONICAL_CHANNEL_ORDER)


@pytest.fixture(scope="module")
def model() -> AllTransformerV4:
    """A deterministic eight-channel model in eval mode."""
    torch.manual_seed(0)
    net = AllTransformerV4(
        input_shape=SPEC_SHAPE, in_channels=N_CHANNELS, num_classes=2, dropout=0.0, rnn_hidden=64
    )
    net.eval()
    return net


class TestChannelSpec:
    """Parsing of the ``--channel`` montage spec."""

    def test_all_expands_to_every_channel(self) -> None:
        assert parse_channel_subset("all") == list(range(N_CHANNELS))
        assert num_model_channels("all") == N_CHANNELS

    def test_single_and_inear_are_not_montages(self) -> None:
        for spec in ("in-ear", "Fp1", None):
            assert parse_channel_subset(spec) is None
            assert num_model_channels(spec) == 1

    def test_montage_is_sorted_into_canonical_order(self) -> None:
        # A model channel index must mean the same electrode however the user typed it.
        assert parse_channel_subset("T8,T7") == parse_channel_subset("T7,T8") == [5, 6]

    def test_occipital_aliases_resolve_per_dataset(self) -> None:
        from thesis.dataset import CANE_CHANNEL_ORDER, MDD_CHANNEL_ORDER

        for alias in ("O2", "Oz", "O2/Oz"):
            assert resolve_channel_names(f"T7,{alias}", MDD_CHANNEL_ORDER) == ["T7", "O2"]
            assert resolve_channel_names(f"T7,{alias}", CANE_CHANNEL_ORDER) == ["T7", "Oz"]

    @pytest.mark.parametrize("spec", ["T9,T7", "T7,T7", "Fp1,", "T7,"])
    def test_invalid_specs_rejected(self, spec: str) -> None:
        with pytest.raises(ValueError):
            parse_channel_subset(spec)


class TestBands:
    """Frequency-band to bin-index mapping."""

    def test_bands_are_disjoint_and_ordered(self) -> None:
        seen: set[int] = set()
        previous_max = -1
        for band in EEG_BANDS:
            bins = band_bins(band)
            assert bins, f"{band} is empty"
            assert not seen & set(bins), f"{band} overlaps a lower band"
            assert min(bins) > previous_max
            seen |= set(bins)
            previous_max = max(bins)

    def test_bins_lie_inside_their_band(self) -> None:
        for band, (low, high) in EEG_BANDS.items():
            for index in band_bins(band):
                assert low <= index * FREQ_BIN_WIDTH_HZ < high

    def test_alpha_covers_the_expected_frequencies(self) -> None:
        # Alpha is the band the depression literature cares about; pin it down explicitly.
        assert band_bins("alpha") == [9, 10, 11, 12, 13]

    def test_no_bin_exceeds_the_spectrogram(self) -> None:
        for band in EEG_BANDS:
            assert max(band_bins(band)) < NUM_FREQ_BINS


class TestChannelMask:
    """Boolean channel masks."""

    def test_drop_and_keep_are_complementary(self) -> None:
        assert torch.equal(channel_mask(drop=[5]), channel_mask(keep=[0, 1, 2, 3, 4, 6, 7]))

    def test_regions_cover_every_channel_exactly_once(self) -> None:
        covered = [index for indices in CHANNEL_REGIONS.values() for index in indices]
        assert sorted(covered) == list(range(N_CHANNELS))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"keep": [1], "drop": [2]},  # both
            {},  # neither
            {"drop": [N_CHANNELS]},  # out of range
            {"drop": list(range(N_CHANNELS))},  # would empty the sequence
        ],
    )
    def test_invalid_masks_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            channel_mask(**kwargs)


class TestOcclusion:
    """Input occlusion and the mirror transform."""

    def test_reference_is_the_mean_over_batches(self) -> None:
        x = torch.randn(6, N_CHANNELS, *SPEC_SHAPE)
        assert torch.allclose(spectrogram_reference([x[:4], x[4:]]), x.mean(0), atol=1e-5)

    def test_reference_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError):
            spectrogram_reference([])

    def test_occlusion_writes_a_rectangle_not_a_diagonal(self) -> None:
        x = torch.randn(4, N_CHANNELS, *SPEC_SHAPE)
        reference = torch.zeros(N_CHANNELS, *SPEC_SHAPE)
        channels, bins = [1, 5], band_bins("alpha")

        out = occlude(x, reference, channels=channels, bins=bins)

        for channel in channels:
            assert torch.equal(out[:, channel][:, bins], torch.zeros(4, len(bins), SPEC_SHAPE[1]))
        untouched = [c for c in range(N_CHANNELS) if c not in channels]
        assert torch.equal(out[:, untouched], x[:, untouched])
        other_bins = [b for b in range(SPEC_SHAPE[0]) if b not in bins]
        assert torch.equal(out[:, channels][:, :, other_bins], x[:, channels][:, :, other_bins])

    def test_occlusion_does_not_mutate_its_input(self) -> None:
        x = torch.randn(2, N_CHANNELS, *SPEC_SHAPE)
        original = x.clone()
        occlude(x, torch.zeros(N_CHANNELS, *SPEC_SHAPE), channels=[0], bins=[1, 2])
        assert torch.equal(x, original)

    def test_occlusion_rejects_mismatched_reference(self) -> None:
        with pytest.raises(ValueError):
            occlude(torch.randn(2, N_CHANNELS, *SPEC_SHAPE), torch.zeros(3, 4, 5))

    def test_mirror_swaps_lateral_pairs_and_is_an_involution(self) -> None:
        x = torch.randn(3, N_CHANNELS, *SPEC_SHAPE)
        mirrored = mirror_channels(x)

        for left, right in ((0, 1), (2, 4), (5, 6)):  # Fp1/Fp2, C3/C4, T7/T8
            assert torch.equal(mirrored[:, left], x[:, right])
        for midline in (3, 7):  # Cz, O2/Oz
            assert torch.equal(mirrored[:, midline], x[:, midline])
        assert torch.equal(mirror_channels(mirrored), x)

    def test_mirror_preserves_the_pooled_spectrum(self) -> None:
        # The probe is only interpretable if it changes lateralisation and nothing else.
        x = torch.randn(3, N_CHANNELS, *SPEC_SHAPE)
        assert torch.allclose(mirror_channels(x).sum(dim=1), x.sum(dim=1), atol=1e-5)

    def test_mirror_rejects_wrong_channel_count(self) -> None:
        with pytest.raises(ValueError):
            mirror_channels(torch.randn(2, 3, *SPEC_SHAPE))


class TestMatchedControls:
    """Width-matched control windows for band occlusion."""

    def test_controls_are_contiguous_and_avoid_the_band(self) -> None:
        rng = np.random.default_rng(0)
        for band in EEG_BANDS:
            bins = band_bins(band)
            for window in matched_control_windows(len(bins), 20, rng, exclude=bins):
                assert len(window) == len(bins)
                assert window == list(range(window[0], window[0] + len(window)))
                assert not set(window) & set(bins)

    def test_gamma_admits_no_control(self) -> None:
        # Gamma spans 41 of 72 bins, so no disjoint window of equal width exists. The caller
        # must fall back to a per-bin comparison rather than silently getting a bad control.
        bins = band_bins("gamma")
        assert matched_control_windows(len(bins), 20, np.random.default_rng(0), bins) == []

    def test_rejects_nonpositive_width(self) -> None:
        with pytest.raises(ValueError):
            matched_control_windows(0, 5, np.random.default_rng(0))


class TestModelMasking:
    """The ``channel_mask`` argument added to AllTransformerV4."""

    def test_none_mask_is_bit_identical_to_the_unmasked_forward(
        self, model: AllTransformerV4
    ) -> None:
        x = torch.randn(3, N_CHANNELS, *SPEC_SHAPE)
        with torch.no_grad():
            assert torch.equal(model(x), model(x, channel_mask=None))
            assert torch.equal(model(x), model(x, channel_mask=torch.ones(8, dtype=torch.bool)))

    def test_dropping_a_channel_changes_the_output(self, model: AllTransformerV4) -> None:
        x = torch.randn(3, N_CHANNELS, *SPEC_SHAPE)
        with torch.no_grad():
            assert not torch.equal(model(x), model(x, channel_mask=channel_mask(drop=[5])))

    def test_dropped_channel_cannot_influence_the_output(self, model: AllTransformerV4) -> None:
        # The point of token dropping over zeroing: the channel's content is truly irrelevant.
        x = torch.randn(2, N_CHANNELS, *SPEC_SHAPE)
        other = x.clone()
        other[:, 5] = torch.randn(2, *SPEC_SHAPE)
        mask = channel_mask(drop=[5])
        with torch.no_grad():
            assert torch.allclose(model(x, channel_mask=mask), model(other, channel_mask=mask))

    def test_encode_returns_the_pooled_embedding(self, model: AllTransformerV4) -> None:
        with torch.no_grad():
            pooled = model.encode(torch.randn(3, N_CHANNELS, *SPEC_SHAPE))
        assert pooled.shape == (3, 64)

    def test_rejects_malformed_masks(self, model: AllTransformerV4) -> None:
        x = torch.randn(2, N_CHANNELS, *SPEC_SHAPE)
        with pytest.raises(ValueError):
            model(x, channel_mask=torch.ones(3, dtype=torch.bool))
        with pytest.raises(ValueError):
            model(x, channel_mask=torch.zeros(N_CHANNELS, dtype=torch.bool))


class TestAttribution:
    """Integrated Gradients and attention rollout."""

    def test_integrated_gradients_satisfies_completeness(self, model: AllTransformerV4) -> None:
        x = torch.randn(2, N_CHANNELS, *SPEC_SHAPE)
        baseline = torch.zeros(N_CHANNELS, *SPEC_SHAPE)
        attributions = attr.integrated_gradients(model, x, baseline, target=1, n_steps=64)

        assert attributions.shape == x.shape
        error = attr.completeness_error(model, x, baseline, attributions, target=1)
        assert float(error.max()) < 1e-3

    def test_more_steps_reduce_completeness_error(self, model: AllTransformerV4) -> None:
        x = torch.randn(2, N_CHANNELS, *SPEC_SHAPE)
        baseline = torch.zeros(N_CHANNELS, *SPEC_SHAPE)
        errors = [
            float(
                attr.completeness_error(
                    model,
                    x,
                    baseline,
                    attr.integrated_gradients(model, x, baseline, 1, n_steps=steps),
                    1,
                ).max()
            )
            for steps in (4, 64)
        ]
        assert errors[1] < errors[0]

    def test_per_sample_targets_are_supported(self, model: AllTransformerV4) -> None:
        x = torch.randn(3, N_CHANNELS, *SPEC_SHAPE)
        baseline = torch.zeros(N_CHANNELS, *SPEC_SHAPE)
        targets = torch.tensor([0, 1, 0])
        attributions = attr.integrated_gradients(model, x, baseline, targets, n_steps=32)
        assert (
            float(attr.completeness_error(model, x, baseline, attributions, targets).max()) < 1e-3
        )

    def test_rejects_bad_arguments(self, model: AllTransformerV4) -> None:
        x = torch.randn(2, N_CHANNELS, *SPEC_SHAPE)
        with pytest.raises(ValueError):
            attr.integrated_gradients(model, x, torch.zeros(N_CHANNELS, *SPEC_SHAPE), 1, 0)
        with pytest.raises(ValueError):
            attr.integrated_gradients(model, x, torch.zeros(2, 3), 1)

    def test_attention_capture_yields_one_matrix_per_layer(self, model: AllTransformerV4) -> None:
        x = torch.randn(2, N_CHANNELS, *SPEC_SHAPE)
        with attr.capture_attention(model) as captured:
            with torch.no_grad():
                model(x)

        n_tokens = N_CHANNELS * model._W_prime
        assert len(captured) == len(model.transformer.layers)
        for weights in captured:
            assert weights.shape == (
                2,
                model.transformer.layers[0].self_attn.num_heads,
                n_tokens,
                n_tokens,
            )
            assert torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-5)

    def test_hooks_are_removed_on_exit(self, model: AllTransformerV4) -> None:
        x = torch.randn(2, N_CHANNELS, *SPEC_SHAPE)
        with attr.capture_attention(model) as captured:
            with torch.no_grad():
                model(x)
        captured_count = len(captured)

        with torch.no_grad():
            model(x)
        assert len(captured) == captured_count

    def test_rollout_is_row_stochastic(self, model: AllTransformerV4) -> None:
        x = torch.randn(2, N_CHANNELS, *SPEC_SHAPE)
        with attr.capture_attention(model) as captured:
            with torch.no_grad():
                model(x)

        rollout = attr.attention_rollout(captured)
        assert rollout.shape == (2, 80, 80)
        assert bool((rollout >= 0).all())
        assert torch.allclose(rollout.sum(-1), torch.ones(2, 80), atol=1e-5)

    def test_rollout_influence_is_a_distribution(self, model: AllTransformerV4) -> None:
        x = torch.randn(2, N_CHANNELS, *SPEC_SHAPE)
        with attr.capture_attention(model) as captured:
            with torch.no_grad():
                model(x)

        influence = attr.rollout_token_influence(attr.attention_rollout(captured))
        per_channel = attr.tokens_to_channels(influence, N_CHANNELS)
        assert per_channel.shape == (2, N_CHANNELS)
        assert torch.allclose(per_channel.sum(-1), torch.ones(2), atol=1e-5)

    def test_rollout_rejects_bad_arguments(self) -> None:
        with pytest.raises(ValueError):
            attr.attention_rollout([])
        with pytest.raises(ValueError):
            attr.attention_rollout([torch.rand(1, 1, 4, 4)], residual_weight=1.5)

    def test_token_to_channel_mapping_follows_the_layout(self) -> None:
        # Token i belongs to channel i // W'; channel 5 owns tokens 50..59.
        scores = torch.zeros(1, 80)
        scores[0, 50:60] = 1.0
        assert attr.tokens_to_channels(scores, N_CHANNELS)[0].tolist() == [0, 0, 0, 0, 0, 10, 0, 0]

    def test_token_to_channel_rejects_indivisible_counts(self) -> None:
        with pytest.raises(ValueError):
            attr.tokens_to_channels(torch.zeros(1, 81), N_CHANNELS)


class TestStats:
    """Paired statistics."""

    BASELINE = [0.80, 0.78, 0.82, 0.79, 0.81, 0.77, 0.83, 0.80, 0.79, 0.81]
    ABLATED = [0.72, 0.70, 0.75, 0.71, 0.73, 0.69, 0.76, 0.72, 0.71, 0.74]

    def test_wilcoxon_detects_a_consistent_drop(self) -> None:
        result = paired_wilcoxon(self.ABLATED, self.BASELINE)
        assert result["mean_drop"] > 0
        assert result["p_value"] < 0.05
        assert result["n"] == 10

    def test_identical_sequences_give_p_one(self) -> None:
        result = paired_wilcoxon(self.BASELINE, self.BASELINE)
        assert result["p_value"] == 1.0
        assert result["mean_drop"] == 0.0

    def test_drop_sign_convention(self) -> None:
        # Positive mean_drop must mean "the ablation hurt".
        assert paired_wilcoxon(self.BASELINE, self.ABLATED)["mean_drop"] < 0

    def test_nadeau_bengio_is_more_conservative_than_naive(self) -> None:
        from scipy import stats as scipy_stats

        corrected = nadeau_bengio_corrected_t(self.ABLATED, self.BASELINE, 0.1)
        naive = scipy_stats.ttest_rel(self.BASELINE, self.ABLATED)
        assert corrected["p_value"] > float(naive.pvalue)
        assert abs(corrected["t_statistic"]) < abs(float(naive.statistic))

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1])
    def test_nadeau_bengio_rejects_bad_fraction(self, fraction: float) -> None:
        with pytest.raises(ValueError):
            nadeau_bengio_corrected_t(self.ABLATED, self.BASELINE, fraction)

    def test_benjamini_hochberg_is_monotone_and_conservative(self) -> None:
        raw = [0.001, 0.01, 0.04, 0.2, 0.6]
        result = benjamini_hochberg(raw)

        assert all(q >= p for q, p in zip(result.q_values, raw))
        ordered = [q for _, q in sorted(zip(raw, result.q_values))]
        assert ordered == sorted(ordered)
        assert result.n_rejected == sum(result.rejected)

    def test_benjamini_hochberg_preserves_input_order(self) -> None:
        result = benjamini_hochberg([0.6, 0.001, 0.04])
        assert result.q_values[1] < result.q_values[0]

    def test_bootstrap_ci_brackets_the_mean(self) -> None:
        drops = [b - a for a, b in zip(self.ABLATED, self.BASELINE)]
        result = bootstrap_ci(drops)
        assert result["lower"] <= result["mean"] <= result["upper"]

    def test_bootstrap_is_reproducible(self) -> None:
        assert bootstrap_ci([1.0, 2.0, 3.0]) == bootstrap_ci([1.0, 2.0, 3.0])

    def test_spearman_ranks(self) -> None:
        assert spearman([1, 2, 3, 4], [1, 2, 3, 4])["rho"] == pytest.approx(1.0)
        assert spearman([1, 2, 3, 4], [4, 3, 2, 1])["rho"] == pytest.approx(-1.0)

    @pytest.mark.parametrize("func", [paired_wilcoxon, lambda a, b: spearman(a, b)])
    def test_length_mismatch_rejected(self, func) -> None:
        with pytest.raises(ValueError):
            func([1.0, 2.0, 3.0], [1.0, 2.0])


class TestRunConfig:
    """Recovering a run's configuration from results.txt."""

    CONFIG = """10-Fold Cross-Validation Results
========================================

Configuration:
  command: train
  condition: ec+eo
  channel: all
  class_mode: 2
  dataset: all
  model: AllTransformerV4
  n_folds: 10
  batch_size: 64
  dropout: 0.1
  skip_artifact_removal: False
  chunk_duration: 10.0

Fold Results:
  fold 1: 0.77
"""

    def test_parses_the_configuration_block(self, tmp_path: Path) -> None:
        (tmp_path / "results.txt").write_text(self.CONFIG)
        config = harness.read_run_config(tmp_path)

        assert config.model == "AllTransformerV4"
        assert config.channel == "all"
        assert config.num_classes == 2
        assert config.conditions == ["EC", "EO"]
        assert config.n_folds == 10
        assert config.batch_size == 64
        assert config.dropout == pytest.approx(0.1)
        assert config.skip_artifact_removal is False

    def test_stops_at_the_end_of_the_block(self, tmp_path: Path) -> None:
        (tmp_path / "results.txt").write_text(self.CONFIG)
        assert "fold" not in harness.read_run_config(tmp_path).raw

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            harness.read_run_config(tmp_path)

    def test_incomplete_block_raises(self, tmp_path: Path) -> None:
        (tmp_path / "results.txt").write_text("Configuration:\n  model: X\n")
        with pytest.raises(ValueError, match="missing configuration keys"):
            harness.read_run_config(tmp_path)


class TestAblatedModel:
    """The wrapper that applies an ablation inside the forward pass."""

    def test_applies_the_input_transform(self, model: AllTransformerV4) -> None:
        x = torch.randn(2, N_CHANNELS, *SPEC_SHAPE)
        wrapped = harness.AblatedModel(model, input_transform=mirror_channels)
        wrapped.eval()
        with torch.no_grad():
            assert torch.equal(wrapped(x), model(mirror_channels(x)))

    def test_forwards_the_channel_mask(self, model: AllTransformerV4) -> None:
        x = torch.randn(2, N_CHANNELS, *SPEC_SHAPE)
        mask = channel_mask(drop=[5])
        wrapped = harness.AblatedModel(model, channel_mask=mask)
        wrapped.eval()
        with torch.no_grad():
            assert torch.equal(wrapped(x), model(x, channel_mask=mask))

    def test_is_a_passthrough_without_an_ablation(self, model: AllTransformerV4) -> None:
        x = torch.randn(2, N_CHANNELS, *SPEC_SHAPE)
        wrapped = harness.AblatedModel(model)
        wrapped.eval()
        with torch.no_grad():
            assert torch.equal(wrapped(x), model(x))


class _FakeSpectrogramDataset(Dataset):
    """Synthetic (spectrogram, label, subject) triples with a learnable channel signal."""

    def __init__(self, n_subjects: int = 8, chunks: int = 4, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.items: list[tuple[torch.Tensor, int, str]] = []
        for subject in range(n_subjects):
            label = subject % 2
            for _ in range(chunks):
                spec = torch.randn(N_CHANNELS, *SPEC_SHAPE, generator=generator)
                # Put the class signal in T7 only, so ablating it must hurt.
                spec[5] += label * 3.0
                self.items.append((spec, label, f"S{subject}"))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        return self.items[index]


class _FakeFolds:
    """Stands in for CVFolds, returning the same synthetic data for every fold."""

    use_raw_eeg = False

    def datasets_for_fold(
        self, fold: int, dataset_type: str
    ) -> tuple[Dataset, Dataset, dict[str, str]]:
        train = _FakeSpectrogramDataset(seed=100 + fold)
        val = _FakeSpectrogramDataset(seed=fold)
        return train, val, {f"S{i}": "mdd" for i in range(8)}


class _SignalModel(nn.Module):
    """Reads only channel 5, so channel ablation has a known ground truth."""

    def forward(self, x: torch.Tensor, channel_mask: torch.Tensor | None = None) -> torch.Tensor:
        if channel_mask is not None:
            x = x[:, channel_mask.to(x.device)]
            if not bool(channel_mask[5]):
                # Channel 5 gone: nothing to go on, so predict class 0 always.
                return torch.stack(
                    [
                        torch.ones(x.shape[0], device=x.device),
                        torch.zeros(x.shape[0], device=x.device),
                    ],
                    dim=1,
                )
            index = int(channel_mask[:5].sum())
        else:
            index = 5
        # Centre on 1.5, midway between the label-0 and label-1 means, so the
        # decision is driven by the injected signal rather than by noise.
        evidence = x[:, index].mean(dim=(1, 2)) - 1.5
        return torch.stack([-evidence, evidence], dim=1)


class TestEndToEnd:
    """The full ablation pipeline, with folds and checkpoints faked out."""

    @pytest.fixture
    def patched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(explain_experiments, "build_folds", lambda config: _FakeFolds())
        monkeypatch.setattr(
            harness,
            "load_fold_model",
            lambda config, fold, device, strict=True: (
                _SignalModel(),
                {"val_metrics": None},
            ),
        )

    @pytest.fixture
    def config(self, tmp_path: Path) -> harness.RunConfig:
        return harness.RunConfig(
            model="AllTransformerV4",
            dataset="all",
            condition="ec+eo",
            channel="all",
            class_mode="2",
            n_folds=3,
            batch_size=8,
            dropout=0.0,
            chunk_duration=10.0,
            skip_artifact_removal=False,
            checkpoint_dir=tmp_path,
        )

    def test_channel_ablation_finds_the_informative_channel(
        self, patched: None, config: harness.RunConfig
    ) -> None:
        result = explain_experiments.run_channel_ablation(config, torch.device("cpu"))
        ablations = result["ablations"]

        assert result["baseline_mean"] == pytest.approx(1.0)
        assert len(result["baseline_per_fold"]) == 3

        # Dropping T7 destroys accuracy; dropping anything else is harmless.
        assert ablations["drop_T7"]["mean"] < 0.6
        for name in CANONICAL_CHANNEL_ORDER:
            if name != "T7":
                assert ablations[f"drop_{name}"]["mean"] == pytest.approx(1.0)

        # Keeping only T7 is as good as the full montage; keeping only Fp1 is not.
        assert ablations["only_T7"]["mean"] == pytest.approx(1.0)
        assert ablations["only_Fp1"]["mean"] < 0.6

        # The temporal region carries the signal, matching the single-channel result.
        assert ablations["drop_region_temporal"]["mean"] < 0.6
        assert ablations["drop_region_frontal"]["mean"] == pytest.approx(1.0)

    def test_statistics_are_attached_to_every_ablation(
        self, patched: None, config: harness.RunConfig
    ) -> None:
        result = explain_experiments.run_channel_ablation(config, torch.device("cpu"))
        entry = result["ablations"]["drop_T7"]

        assert len(entry["per_fold"]) == 3
        assert {"wilcoxon", "corrected_t", "drop_ci", "q_value", "significant"} <= entry.keys()
        assert entry["drop_ci"]["lower"] <= entry["drop_ci"]["mean"] <= entry["drop_ci"]["upper"]
        assert 0.0 <= entry["q_value"] <= 1.0

    def test_result_is_json_serialisable(
        self, patched: None, config: harness.RunConfig, tmp_path: Path
    ) -> None:
        import json

        result = explain_experiments.run_channel_ablation(config, torch.device("cpu"))
        destination = harness.write_results(result, tmp_path / "explain" / "channels.json")
        assert json.loads(destination.read_text())["channel_order"] == CANONICAL_CHANNEL_ORDER

    def test_band_ablation_runs_for_both_montages(
        self, patched: None, config: harness.RunConfig
    ) -> None:
        result = explain_experiments.run_band_ablation(
            config, torch.device("cpu"), n_control_draws=2
        )

        assert result["n_channels"] == N_CHANNELS
        for band in EEG_BANDS:
            assert f"band_{band}" in result["ablations"]
            assert f"band_{band}_T7" in result["ablations"]
        assert result["control_windows"]["gamma"] == []

    def test_channel_ablation_rejects_single_channel_runs(self, config: harness.RunConfig) -> None:
        config.channel = "in-ear"
        with pytest.raises(ValueError, match="eight-channel"):
            explain_experiments.run_channel_ablation(config, torch.device("cpu"))
        with pytest.raises(ValueError, match="eight-channel"):
            explain_experiments.run_hemisphere_mirror(config, torch.device("cpu"))

    def test_sanity_check_runs_on_random_weights(
        self, patched: None, config: harness.RunConfig
    ) -> None:
        result = explain_experiments.run_sanity_checks(config, torch.device("cpu"))

        assert "note" in result
        assert result["channel_order"] == CANONICAL_CHANNEL_ORDER
        for name in CANONICAL_CHANNEL_ORDER:
            assert f"drop_{name}" in result["ablations"]

    def test_site_probe_reports_all_three_baselines(
        self, patched: None, config: harness.RunConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Two cohorts, split by subject, so the probe has something to separate.
        monkeypatch.setattr(
            _FakeFolds,
            "datasets_for_fold",
            lambda self, fold, dataset_type: (
                _FakeSpectrogramDataset(seed=100 + fold),
                _FakeSpectrogramDataset(seed=fold),
                {f"S{i}": ("mdd" if i < 4 else "cane") for i in range(8)},
            ),
        )
        monkeypatch.setattr(
            explain_experiments,
            "_collect_embeddings",
            lambda context, control, device: (
                np.random.default_rng(0).normal(size=(32, 8)),
                np.random.default_rng(1).normal(size=(32, 8)),
                # Ordered by cohort, exactly as the unshuffled validation loader emits them.
                ["mdd"] * 16 + ["cane"] * 16,
                [f"S{i // 4}" for i in range(32)],
            ),
        )
        monkeypatch.setattr(harness, "load_fold_model", lambda *a, **k: (_SignalModel(), {}))

        result = explain_experiments.run_site_probe(config, torch.device("cpu"))
        assert {"trained_backbone", "random_backbone", "majority_baseline"} == result.keys()
        assert len(result["trained_backbone"]["per_fold"]) == 3
