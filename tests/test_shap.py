"""
Tests for grouped SHAP (``thesis.explain.groups``, ``shap_values`` and ``experiments``).

All tests run on CPU with tiny synthetic tensors: no datasets, no checkpoints. The Shapley
arithmetic belongs to the ``shap`` library, so the tests target our code: the player masks, the
value-function wrapper, the background sampler, and whether the library is wired up correctly.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

pytest.importorskip("shap")

from thesis.cli import get_arg_parser  # noqa: E402
from thesis.dataset import collate_spectrograms  # noqa: E402
from thesis.explain import experiments as shap_experiments  # noqa: E402
from thesis.explain import harness  # noqa: E402
from thesis.explain.groups import (  # noqa: E402
    SUB_DELTA,
    PlayerSet,
    band_partition,
    band_players,
    build_players,
    channel_players,
    grid_players,
)
from thesis.explain.shap_values import (  # noqa: E402
    Background,
    _recording_key,
    binary_margin,
    explain_chunk,
    make_value_fn,
    mean_background,
    person_of,
    reinitialise,
    sample_background,
)

F, T = 72, 41


class _SmallCNN(nn.Module):
    """A tiny non-linear spectrogram classifier."""

    def __init__(self, in_channels: int, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=5, stride=3)
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.conv(x)).mean(dim=(2, 3))
        return self.head(h)


class _LinearModel(nn.Module):
    """``margin = sum(w * x)`` in float64, so Shapley values have a closed form."""

    def __init__(self, shape: tuple[int, ...], seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.w = nn.Parameter(torch.randn(*shape, generator=generator, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        margin = (x.double() * self.w).flatten(1).sum(dim=1)
        return torch.stack([torch.zeros_like(margin), margin], dim=1)


class _IgnoresChannel(_SmallCNN):
    """Zeroes one channel before the CNN, so that channel is a dummy player."""

    def __init__(self, dropped: int) -> None:
        super().__init__(in_channels=8)
        self.dropped = dropped

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x[:, self.dropped] = 0.0
        return super().forward(x)


def _chunks(n: int, channels: int, seed: int) -> torch.Tensor:
    return torch.randn(n, channels, F, T, generator=torch.Generator().manual_seed(seed))


# ── 5. Player partition ───────────────────────────────────────────────────────


class TestPlayers:
    @pytest.mark.parametrize("channels", [1, 8])
    @pytest.mark.parametrize("freq_bins", [72, 31])
    def test_band_players_partition_the_input(self, channels: int, freq_bins: int) -> None:
        players = band_players(channels, freq_bins, T)
        assert bool((players.masks.sum(dim=0) == 1).all())

    @pytest.mark.parametrize("builder", [channel_players, grid_players])
    def test_multichannel_players_partition_the_input(self, builder) -> None:
        players = builder(8, F, T)
        assert bool((players.masks.sum(dim=0) == 1).all())

    def test_sub_delta_holds_the_bins_below_delta(self) -> None:
        partition = band_partition(F)
        assert list(partition) == [SUB_DELTA, "delta", "theta", "alpha", "beta", "gamma"]
        assert partition[SUB_DELTA] == [0, 1]
        assert sum(len(b) for b in partition.values()) == F

    def test_empty_bands_are_dropped_below_the_gamma_edge(self) -> None:
        assert "gamma" not in band_partition(31)

    def test_gamma_game_splits_gamma_at_45_hz(self) -> None:
        partition = band_partition(F, split_gamma=True)
        assert list(partition)[-2:] == ["gamma_low", "gamma_high"]
        assert "gamma" not in partition
        assert sum(len(b) for b in partition.values()) == F
        assert max(partition["gamma_low"]) * 0.9766 < 45.0 <= min(partition["gamma_high"]) * 0.9766
        players = build_players("gamma", 8, F, T)
        assert bool((players.masks.sum(dim=0) == 1).all())

    def test_grid_has_48_players_at_70_hz(self) -> None:
        players = grid_players(8, F, T)
        assert len(players) == 48
        assert players.names[0] == "Fp1:sub_delta"
        assert players.names[-1] == "O2/Oz:gamma"

    def test_single_channel_rejects_channel_games(self) -> None:
        with pytest.raises(ValueError):
            build_players("channels", 1, F, T)
        with pytest.raises(ValueError):
            build_players("grid", 1, F, T)

    def test_overlapping_masks_are_rejected(self) -> None:
        masks = torch.ones(2, 1, 2, 2, dtype=torch.bool)
        with pytest.raises(ValueError, match="partition"):
            PlayerSet(["a", "b"], masks, [2, 2])


# ── 1. Wrapper endpoints ──────────────────────────────────────────────────────


class TestValueFunction:
    def test_all_on_is_the_real_chunk_and_all_off_the_background_mean(self) -> None:
        model = _SmallCNN(8).eval()
        x = _chunks(1, 8, seed=1)[0]
        background = _chunks(4, 8, seed=2)
        players = band_players(8, F, T)
        f = make_value_fn(model, x, background, players, max_batch=3)

        with torch.no_grad():
            expected_on = binary_margin(model(x[None])).item()
            expected_off = binary_margin(model(background)).mean().item()
        values = f(np.stack([np.ones(len(players)), np.zeros(len(players))]))
        assert values[0] == pytest.approx(expected_on, abs=1e-6)
        assert values[1] == pytest.approx(expected_off, abs=1e-6)

    def test_one_player_on_swaps_in_exactly_its_pixels(self) -> None:
        model = _LinearModel((8, F, T)).eval()
        x = _chunks(1, 8, seed=1)[0]
        background = _chunks(1, 8, seed=2)
        players = band_players(8, F, T)
        f = make_value_fn(model, x, background, players)

        switches = np.zeros((1, len(players)))
        switches[0, 3] = 1  # alpha
        composite = torch.where(players.masks[3], x, background[0])
        expected = binary_margin(model(composite[None])).item()
        assert f(switches)[0] == pytest.approx(expected, abs=1e-9)


# ── 2. Linear model, analytic answer ──────────────────────────────────────────


class TestLinearModel:
    @pytest.mark.parametrize("game", ["bands", "channels"])
    @pytest.mark.parametrize("k", [1, 3])
    def test_shapley_values_match_the_closed_form(self, game: str, k: int) -> None:
        # For f(x) = sum(w * x) with an interventional background, phi_group is
        # sum over the group's pixels of w * (x - mean_k b_k).
        model = _LinearModel((8, F, T)).eval()
        x = _chunks(1, 8, seed=3)[0]
        background = _chunks(k, 8, seed=4)
        players = build_players(game, 8, F, T)  # type: ignore[arg-type]

        result = explain_chunk(model, x, background, players)

        contribution = model.w.detach() * (x.double() - background.double().mean(dim=0))
        expected = [float(contribution[m].sum()) for m in players.masks]
        np.testing.assert_allclose(result.phi, expected, atol=1e-6)


# ── 3. Efficiency, 4. dummy, 6. permutation versus exact ──────────────────────


class TestShapleyProperties:
    def test_efficiency_holds_for_a_nonlinear_model(self) -> None:
        model = _SmallCNN(8).eval()
        result = explain_chunk(model, _chunks(1, 8, 5)[0], _chunks(4, 8, 6), band_players(8, F, T))
        assert abs(result.residual) < 1e-5
        assert result.phi.sum() == pytest.approx(result.fx - result.fbase, abs=1e-5)
        assert result.n_evals >= 2**6

    def test_ignored_channel_gets_zero(self) -> None:
        model = _IgnoresChannel(dropped=3).eval()
        result = explain_chunk(
            model, _chunks(1, 8, 7)[0], _chunks(2, 8, 8), channel_players(8, F, T)
        )
        assert result.phi[3] == pytest.approx(0.0, abs=1e-6)
        assert np.abs(result.phi).max() > 1e-4

    def test_permutation_estimate_approaches_exact(self) -> None:
        model = _SmallCNN(8).eval()
        x, background = _chunks(1, 8, 9)[0], _chunks(3, 8, 10)
        players = band_players(8, F, T)

        exact = explain_chunk(model, x, background, players, estimator="exact")
        sampled = explain_chunk(
            model, x, background, players, estimator="permutation", n_permutations=300, seed=1
        )
        scale = np.abs(exact.phi).max()
        np.testing.assert_allclose(sampled.phi, exact.phi, atol=0.05 * scale)
        # Antithetic sampling keeps efficiency exact per permutation.
        assert abs(sampled.residual) < 1e-5
        # max_evals = n_permutations * (2M + 1) coalitions.
        assert sampled.n_evals >= 300 * (2 * len(players))

    def test_permutation_requires_a_permutation_count(self) -> None:
        with pytest.raises(ValueError, match="n_permutations"):
            explain_chunk(
                _SmallCNN(8).eval(),
                _chunks(1, 8, 1)[0],
                _chunks(1, 8, 2),
                grid_players(8, F, T),
            )


# ── 7. Background sampling ────────────────────────────────────────────────────


class _FakeDataset(Dataset):
    """In-memory ``(spectrogram, label, subject)`` triples."""

    def __init__(self, items: list[tuple[torch.Tensor, int, str]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        return self.items[index]


def _fake_split(
    prefix: str, n_people: int, chunks: int, channels: int, seed: int
) -> tuple[_FakeDataset, list[str]]:
    """People alternate healthy/pathological; the first half is MDD, the rest CANE."""
    generator = torch.Generator().manual_seed(seed)
    items, datasets = [], []
    for person in range(n_people):
        dataset = "mdd" if person < n_people // 2 else "cane"
        for condition in ("EC", "EO"):
            for _ in range(chunks):
                x = torch.randn(channels, F, T, generator=generator)
                items.append((x, person % 2, f"{prefix}{person} {condition}"))
                datasets.append(dataset)
    return _FakeDataset(items), datasets


class TestBackground:
    def test_draws_only_the_requested_dataset_with_both_labels(self) -> None:
        train, datasets = _fake_split("T", n_people=12, chunks=3, channels=1, seed=0)
        background = sample_background(train, datasets, k=4, seed=1, dataset_key="cane")

        assert background.chunks.shape == (4, 1, F, T)
        assert set(background.datasets) == {"cane"}
        assert set(background.labels) == {0, 1}
        assert background.labels.count(0) == background.labels.count(1)
        # One chunk per person while people remain.
        assert len({person_of(s) for s in background.subjects}) == 4
        for index, subject in zip(background.indices, background.subjects):
            assert datasets[index] == "cane"
            assert train[index][2] == subject

    def test_tops_up_when_there_are_fewer_people_than_the_quota(self) -> None:
        train, datasets = _fake_split("T", n_people=4, chunks=3, channels=1, seed=0)
        background = sample_background(train, datasets, k=8, seed=1, dataset_key="mdd")
        assert len(background.subjects) == 8
        assert set(background.labels) == {0, 1}

    def test_pooled_background_is_stratified_by_dataset_and_label(self) -> None:
        train, datasets = _fake_split("T", n_people=12, chunks=2, channels=1, seed=0)
        background = sample_background(train, datasets, k=8, seed=1, dataset_key=None)
        strata = list(zip(background.datasets, background.labels))
        assert {s: strata.count(s) for s in set(strata)} == {
            ("cane", 0): 2,
            ("cane", 1): 2,
            ("mdd", 0): 2,
            ("mdd", 1): 2,
        }

    def test_draw_is_reproducible(self) -> None:
        train, datasets = _fake_split("T", n_people=12, chunks=3, channels=1, seed=0)
        a = sample_background(train, datasets, k=6, seed=3, dataset_key="mdd")
        b = sample_background(train, datasets, k=6, seed=3, dataset_key="mdd")
        assert a.indices == b.indices

    def test_unknown_dataset_raises(self) -> None:
        train, datasets = _fake_split("T", n_people=4, chunks=1, channels=1, seed=0)
        with pytest.raises(ValueError, match="sad"):
            sample_background(train, datasets, k=2, seed=0, dataset_key="sad")

    def test_mean_background_is_the_dataset_mean(self) -> None:
        train, datasets = _fake_split("T", n_people=6, chunks=2, channels=1, seed=0)
        background = mean_background(train, datasets, "mdd", seed=0)
        expected = torch.stack([x for (x, _, _), d in zip(train.items, datasets) if d == "mdd"])
        torch.testing.assert_close(background.chunks[0], expected.mean(dim=0))

    def test_recording_key_walks_subsets_and_concats(self) -> None:
        class _Flat(_FakeDataset):
            def __init__(self, items: list[tuple[torch.Tensor, int, str]]) -> None:
                super().__init__(items)
                self.index = [(i // 2, i % 2) for i in range(len(items))]

        flat_a, _ = _fake_split("A", 2, 1, 1, 0)
        flat_b, _ = _fake_split("B", 2, 1, 1, 1)
        a, b = _Flat(flat_a.items), _Flat(flat_b.items)
        combined = ConcatDataset([Subset(a, [0, 1, 2]), Subset(b, [3])])
        keys = [_recording_key(combined, i) for i in range(4)]
        assert keys[0] == keys[1] == (id(a), 0)
        assert keys[2] == (id(a), 1)
        assert keys[3] == (id(b), 1)

    def test_person_strips_the_condition(self) -> None:
        assert person_of("H S12 EC") == "H S12"
        assert person_of("MDD S3 EO") == "MDD S3"
        assert person_of("1003") == "1003"


def test_reinitialise_changes_weights_reproducibly() -> None:
    model_a, model_b = _SmallCNN(8), _SmallCNN(8)
    before = [p.clone() for p in model_a.parameters()]
    reinitialise(model_a, seed=1)
    reinitialise(model_b, seed=1)
    for old, new, twin in zip(before, model_a.parameters(), model_b.parameters()):
        assert not torch.equal(old, new)
        assert torch.equal(new, twin)


# ── Runner, end to end with faked folds ───────────────────────────────────────


def test_choose_chunks_spreads_across_people_and_keeps_the_mix() -> None:
    persons = [f"p{i // 10}" for i in range(60)]  # six people, ten chunks each
    strata = [f"mdd/{(i // 10) % 2}" for i in range(60)]
    chosen, mix = shap_experiments.choose_chunks(persons, strata, max_chunks=18)
    counts = {p: sum(persons[i] == p for i in chosen) for p in set(persons)}
    assert set(counts.values()) == {3}
    assert mix["before"] == mix["after"]

    chosen, _ = shap_experiments.choose_chunks(persons, strata, max_chunks=3)
    assert len(chosen) == 3
    assert len({persons[i] for i in chosen}) == 3


class TestRunShap:
    @pytest.fixture
    def config(self, tmp_path: Path) -> harness.RunConfig:
        return harness.RunConfig(
            model="AllTransformerV4",
            dataset="all",
            condition="ec+eo",
            channel="all",
            class_mode="2",
            n_folds=2,
            batch_size=8,
            dropout=0.0,
            chunk_duration=10.0,
            skip_artifact_removal=False,
            checkpoint_dir=tmp_path,
        )

    @pytest.fixture
    def drawn(self, monkeypatch: pytest.MonkeyPatch) -> list[Background]:
        """Fake two folds; record every background the runner draws."""

        def contexts(config, device, folds, fold_numbers):
            for fold in fold_numbers if fold_numbers is not None else range(2):
                train, train_datasets = _fake_split("T", 8, 2, 8, seed=10 + fold)
                val, val_datasets = _fake_split("V", 4, 2, 8, seed=fold)
                yield harness.FoldContext(
                    fold=fold,
                    model=_SmallCNN(8, seed=fold).eval(),
                    loader=DataLoader(val, batch_size=4, collate_fn=collate_spectrograms),
                    subject_dataset_map={},
                    checkpoint={},
                    train_dataset=train,
                    val_datasets=val_datasets,
                    train_datasets=train_datasets,
                )

        draws: list[Background] = []

        def recording_sampler(*args, **kwargs) -> Background:
            background = sample_background(*args, **kwargs)
            draws.append(background)
            return background

        monkeypatch.setattr(shap_experiments, "build_folds", lambda config: None)
        monkeypatch.setattr(shap_experiments, "iter_folds", contexts)
        monkeypatch.setattr(
            shap_experiments, "verify_baseline", lambda context, config, device: (True, 0.0)
        )
        monkeypatch.setattr(shap_experiments, "sample_background", recording_sampler)
        return draws

    def test_bands_end_to_end(
        self, config: harness.RunConfig, drawn: list[Background], tmp_path: Path
    ) -> None:
        options = shap_experiments.ShapOptions(
            game="bands", background_size=4, max_chunks_per_fold=6
        )
        payload, arrays = shap_experiments.run_shap(config, torch.device("cpu"), options)

        # Backgrounds come from training people only, one draw per dataset per fold.
        assert len(drawn) == 4
        assert all(s.startswith("T") for b in drawn for s in b.subjects)
        assert {b.datasets[0] for b in drawn} == {"mdd", "cane"}

        # Four validation people per fold, capped at 6 // 4 = 1 chunk each.
        assert arrays["phi"].shape == (8, 6)
        assert payload["max_efficiency_residual"] < 1e-5
        assert payload["estimator"] == "exact"
        importance = payload["importance"]
        assert set(importance["by_dataset"]) == {"mdd", "cane"}
        assert importance["all"]["mean_abs_phi"]["n_folds"] == 2
        assert set(importance["all"]["mean_abs_phi"]["mean"]) == set(payload["players"])
        assert "per_bin" in importance
        assert "mdd_vs_cane" in payload["dataset_differences"]["tests"] or (
            "cane_vs_mdd" in payload["dataset_differences"]["tests"]
        )
        json.dumps(payload)  # serialisable

    def test_grid_reports_a_monte_carlo_error(
        self, config: harness.RunConfig, drawn: list[Background]
    ) -> None:
        options = shap_experiments.ShapOptions(
            game="grid",
            background_size=2,
            max_chunks_per_fold=2,
            grid_permutations=2,
            grid_se_seeds=2,
            grid_se_chunks=1,
            folds=[0],
        )
        payload, arrays = shap_experiments.run_shap(config, torch.device("cpu"), options)
        assert arrays["phi"].shape == (2, 48)
        assert payload["estimator"] == "permutation"
        assert payload["coalitions_first_chunk"] >= 2 * 2 * 48
        se = payload["monte_carlo_se"]["mean_phi_se"]
        assert len(se) == 48 and all(v >= 0 for v in se.values())

    def test_in_ear_runs_name_idun(self, config: harness.RunConfig) -> None:
        config.channel = "in-ear"
        names = shap_experiments.dataset_names(config)
        assert "idun" in names and "cane" not in names
        assert shap_experiments._public_key(config, "cane") == "idun"

    def test_four_class_is_not_supported(self, config: harness.RunConfig) -> None:
        config.class_mode = "4"
        with pytest.raises(NotImplementedError):
            shap_experiments.run_shap(
                config, torch.device("cpu"), shap_experiments.ShapOptions(game="bands")
            )

    def test_comparisons_attach_to_derived_runs(
        self, config: harness.RunConfig, drawn: list[Background], tmp_path: Path
    ) -> None:
        options = shap_experiments.ShapOptions(
            game="bands", background_size=2, max_chunks_per_fold=4
        )
        trained, _ = shap_experiments.run_shap(config, torch.device("cpu"), options)
        harness.write_explain_results(trained, tmp_path / "explain" / "shap_bands.json")

        random_options = shap_experiments.ShapOptions(
            game="bands", background_size=2, max_chunks_per_fold=4, random_init=True
        )
        random, _ = shap_experiments.run_shap(config, torch.device("cpu"), random_options)
        shap_experiments.attach_comparisons(random, tmp_path / "explain", random_options)
        check = random["randomisation_check"]
        assert set(check["per_fold"]) == {"1", "2"}
        assert isinstance(check["passed"], bool)
        assert check["passed"] == (
            check["random_mean_abs_gap"]
            < check["gap_ratio_threshold"] * check["trained_mean_abs_gap"]
        )


# ── 8. CLI ────────────────────────────────────────────────────────────────────


class TestCli:
    def test_explain_parses(self) -> None:
        args = get_arg_parser().parse_args(
            ["explain", "some/run", "--experiment", "shap-bands", "--folds", "1,3"]
        )
        assert args.command == "explain"
        assert args.run_dir == Path("some/run")
        assert args.folds == [0, 2]
        assert args.baseline == "sample"
        assert args.background_size == 32

    def test_choices_match_the_runner(self) -> None:
        parser = get_arg_parser()
        explain = parser._subparsers._group_actions[0].choices["explain"]  # type: ignore[union-attr]
        experiment = next(a for a in explain._actions if a.dest == "experiment")
        assert set(experiment.choices) == set(shap_experiments.EXPERIMENT_GAMES)

    def test_rejects_zero_fold(self) -> None:
        with pytest.raises(SystemExit):
            get_arg_parser().parse_args(
                ["explain", "run", "--experiment", "shap-bands", "--folds", "0"]
            )


# ── Review follow-ups ─────────────────────────────────────────────────────────


def test_partial_runs_get_their_own_file_and_skip_comparisons(tmp_path: Path) -> None:
    options = shap_experiments.ShapOptions(game="bands", baseline="mean", folds=[0])
    assert options.stem == "shap_bands_mean_partial"
    assert shap_experiments.ShapOptions(game="bands").stem == "shap_bands"

    (tmp_path / "shap_bands.json").write_text("{}")  # would crash a comparison if read
    payload: dict = {}
    shap_experiments.attach_comparisons(payload, tmp_path, options)
    assert payload == {}


@pytest.mark.parametrize("value", ["0", "-3"])
def test_cli_rejects_non_positive_counts(value: str) -> None:
    with pytest.raises(SystemExit):
        get_arg_parser().parse_args(
            ["explain", "run", "--experiment", "shap-grid", "--grid-se-chunks", value]
        )


class TestVerifyBaseline:
    """The fold gate tolerates one flipped chunk, not a different fold."""

    @staticmethod
    def _context(n_chunks: int, stored_accuracy: float) -> harness.FoldContext:
        val, datasets = _fake_split("V", 1, n_chunks // 2, 1, 0)
        return harness.FoldContext(
            fold=0,
            model=_SmallCNN(1),
            loader=DataLoader(val, batch_size=4, collate_fn=collate_spectrograms),
            subject_dataset_map={},
            checkpoint={"val_metrics": {"chunk": {"accuracy": stored_accuracy}}},
            train_dataset=val,
            val_datasets=datasets,
            train_datasets=datasets,
        )

    @pytest.mark.parametrize(("offset", "matched"), [(0, True), (1, True), (3, False)])
    def test_tolerance_is_about_one_chunk(
        self, monkeypatch: pytest.MonkeyPatch, offset: int, matched: bool
    ) -> None:
        context = self._context(100, stored_accuracy=0.5 + offset / 100)
        monkeypatch.setattr(
            harness, "evaluate", lambda context, config, device: {"chunk": {"accuracy": 0.5}}
        )
        ok, _ = harness.verify_baseline(context, None, torch.device("cpu"))  # type: ignore[arg-type]
        assert ok is matched


def test_index_datasets_labels_each_part_of_a_combined_fold() -> None:
    mdd, _ = _fake_split("M", 2, 1, 1, 0)
    sad, _ = _fake_split("S", 2, 1, 1, 1)

    class _Folds:
        mdd_flat_dataset, cane_flat_dataset, sad_flat_dataset = mdd, None, sad
        dataset_type = "all"

    combined = ConcatDataset([Subset(mdd, [0, 1]), Subset(sad, [0, 1, 2])])
    keys = harness.index_datasets(combined, _Folds())  # type: ignore[arg-type]
    assert keys == ["mdd", "mdd", "sad", "sad", "sad"]
