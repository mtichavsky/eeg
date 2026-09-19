"""Tests for helpers-print/freq_cutoff_summary.py using synthetic checkpoints and logs."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from thesis.model_factory import create_model

HELPERS_DIR = Path(__file__).resolve().parents[1] / "helpers-print"


def _load_module() -> ModuleType:
    """Import helpers-print/freq_cutoff_summary.py (the directory name is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "freq_cutoff_summary", HELPERS_DIR / "freq_cutoff_summary.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["freq_cutoff_summary"] = module
    spec.loader.exec_module(module)
    return module


fcs = _load_module()

SUBJECTS = [[f"mdd:H S{fold}{i} EC" for i in range(3)] for fold in range(1, 11)]


def _write_run(
    run_dir: Path,
    chunk_accs: list[float],
    *,
    cutoff: float | None = 70.0,
    subjects: list[list[str]] | None = None,
    n_state: int = 12,
    git_commit: str = "abc1234",
) -> None:
    """Write a synthetic run directory: fold checkpoints, results.txt and a training log."""
    run_dir.mkdir(parents=True)
    for i, acc in enumerate(chunk_accs, start=1):
        payload: dict[str, object] = {
            "model_state_dict": {"w": torch.zeros(n_state), "b": torch.zeros(3, 2)},
            "val_metrics": {
                "chunk": {"accuracy": acc, "recall": acc + 0.05, "specificity": acc - 0.05},
                "subject": {"accuracy": acc + 0.02, "recall": 0.7, "specificity": 0.6},
                "per_dataset": {
                    "mdd": {"accuracy": acc},
                    "cane": {"accuracy": acc - 0.1},
                    "sad": {"accuracy": acc + 0.1},
                },
            },
        }
        if cutoff is not None:
            payload["freq_cutoff_hz"] = cutoff
        torch.save(payload, run_dir / f"fold_{i}_best.pth")

    freq = "n/a" if cutoff is None else cutoff
    (run_dir / "results.txt").write_text(
        "10-Fold Cross-Validation Results\n"
        "================================================================================\n\n"
        "Configuration:\n"
        "  command: train\n"
        f"  freq_cutoff: {freq}\n"
        f"  git_commit: {git_commit}\n\n"
        "Per-fold best validation accuracy:\n"
    )
    folds = SUBJECTS if subjects is None else subjects
    lines = [
        f"[2026-09-19 10:00:00,000 INFO data_preparation.get_datasets_for_fold] "
        f"Val subjects ({len(s)}): {s!r}"
        for s in folds
    ]
    (run_dir / "training_ec+eo_all_20260919_100000.log").write_text("\n".join(lines) + "\n")


ACCS_70 = [0.80, 0.78, 0.82, 0.75, 0.77, 0.79, 0.81, 0.76, 0.74, 0.80]


class TestLoadRun:
    def test_metrics_provenance_and_param_count(self, tmp_path: Path) -> None:
        _write_run(tmp_path / "r", ACCS_70)
        run = fcs.load_run(tmp_path / "r")
        assert run.n_folds == 10
        assert run.chunk_acc == pytest.approx(ACCS_70)
        assert run.chunk_sens == pytest.approx([a + 0.05 for a in ACCS_70])
        assert run.subject_acc == pytest.approx([a + 0.02 for a in ACCS_70])
        assert run.per_dataset["cane"] == pytest.approx([a - 0.1 for a in ACCS_70])
        assert run.param_count == 12 + 6
        assert run.ckpt_cutoffs == [70.0] * 10
        assert run.config["freq_cutoff"] == "70.0"
        assert run.config["git_commit"] == "abc1234"

    def test_folds_ordered_numerically(self, tmp_path: Path) -> None:
        """fold_10 must come after fold_9, not after fold_1."""
        _write_run(tmp_path / "r", [float(i) / 100 for i in range(1, 11)])
        run = fcs.load_run(tmp_path / "r")
        assert run.chunk_acc == pytest.approx([i / 100 for i in range(1, 11)])

    def test_checkpoint_without_cutoff_key(self, tmp_path: Path) -> None:
        _write_run(tmp_path / "r", ACCS_70, cutoff=None)
        run = fcs.load_run(tmp_path / "r")
        assert run.ckpt_cutoffs == [None] * 10

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        (tmp_path / "r").mkdir()
        with pytest.raises(FileNotFoundError):
            fcs.load_run(tmp_path / "r")


class TestCountParameters:
    @pytest.mark.parametrize(
        ("model", "channels", "shape", "expected"),
        [
            ("CNNAttnS", 1, (72, 41), 79_490),
            ("CNNAttnS", 1, (31, 41), 57_986),
            ("AllTransformerV4", 8, (31, 41), 91_970),
        ],
    )
    def test_matches_documented_counts(
        self, model: str, channels: int, shape: tuple[int, int], expected: int
    ) -> None:
        net = create_model(
            model,
            spec_shape=shape,
            dropout=0.1,
            num_classes=2,
            device=torch.device("cpu"),
            in_channels=channels,
        )
        assert fcs.count_parameters(net.state_dict()) == expected
        assert sum(p.numel() for p in net.parameters()) == expected


class TestParseResultsConfig:
    def test_missing_file(self, tmp_path: Path) -> None:
        assert fcs.parse_results_config(tmp_path / "nope.txt") == {}

    def test_stops_at_end_of_config_block(self, tmp_path: Path) -> None:
        _write_run(tmp_path / "r", ACCS_70)
        config = fcs.parse_results_config(tmp_path / "r" / "results.txt")
        assert set(config) == {"command", "freq_cutoff", "git_commit"}


class TestPairedComparison:
    def test_sign_convention_is_f70_minus_f30(self) -> None:
        f70 = [0.80] * 10
        f30 = [0.78, 0.77, 0.79, 0.76, 0.78, 0.77, 0.79, 0.76, 0.78, 0.77]
        res = fcs.paired_comparison(f70, f30)
        assert res.mean_diff == pytest.approx(np.mean(np.asarray(f70) - np.asarray(f30)))
        assert res.mean_diff > 0
        assert res.ci_low <= res.mean_diff <= res.ci_high
        assert res.wilcoxon_p < 0.05
        assert res.t_stat > 0

    def test_identical_runs_give_no_evidence_of_difference(self) -> None:
        res = fcs.paired_comparison(ACCS_70, ACCS_70)
        assert res.mean_diff == 0.0
        assert res.wilcoxon_p == 1.0
        assert res.nb_p == 1.0

    def test_fold_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            fcs.paired_comparison([0.8, 0.7], [0.8])

    def test_bootstrap_ci_is_deterministic(self) -> None:
        diffs = np.asarray([0.01, -0.02, 0.03, 0.0, 0.02, 0.01, -0.01, 0.02, 0.0, 0.01])
        assert fcs.bootstrap_ci(diffs) == fcs.bootstrap_ci(diffs)


class TestFoldIdentity:
    def test_identical_folds_pass(self, tmp_path: Path) -> None:
        _write_run(tmp_path / "f70", ACCS_70)
        _write_run(tmp_path / "f30", ACCS_70, cutoff=30.0)
        assert fcs.check_fold_identity(
            fcs.load_run(tmp_path / "f70"), fcs.load_run(tmp_path / "f30")
        )

    def test_different_folds_fail(self, tmp_path: Path) -> None:
        other = [list(s) for s in SUBJECTS]
        other[4] = other[4][::-1] + ["mdd:H S99 EC"]
        _write_run(tmp_path / "f70", ACCS_70)
        _write_run(tmp_path / "f30", ACCS_70, cutoff=30.0, subjects=other)
        assert not fcs.check_fold_identity(
            fcs.load_run(tmp_path / "f70"), fcs.load_run(tmp_path / "f30")
        )

    def test_missing_log_fails(self, tmp_path: Path) -> None:
        _write_run(tmp_path / "f70", ACCS_70)
        _write_run(tmp_path / "f30", ACCS_70, cutoff=30.0)
        next((tmp_path / "f30").glob("training_*.log")).unlink()
        assert not fcs.check_fold_identity(
            fcs.load_run(tmp_path / "f70"), fcs.load_run(tmp_path / "f30")
        )


def _write_all_runs(root: Path, *, break_folds_for: str | None = None) -> None:
    """Write all four runs with a small f70 > f30 gap; optionally break one model's folds."""
    for model, by_cutoff in fcs.RUNS.items():
        for cutoff, dir_name in by_cutoff.items():
            accs = (
                ACCS_70
                if cutoff == 70
                else [a - 0.02 - 0.005 * (i % 3) for i, a in enumerate(ACCS_70)]
            )
            subjects = None
            if model == break_folds_for and cutoff == 30:
                subjects = [list(s) for s in SUBJECTS]
                subjects[0] = ["mdd:H S1 EC"]
            _write_run(root / dir_name, accs, cutoff=float(cutoff), subjects=subjects)


class TestMain:
    def test_full_report(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _write_all_runs(tmp_path)
        sys.argv = ["freq_cutoff_summary.py", "--root", str(tmp_path)]
        fcs.main()
        out = capsys.readouterr().out
        assert "Run provenance" in out
        assert "Main table" in out
        assert "Paired per-fold differences" in out
        assert "Per-dataset chunk accuracy" in out
        assert "Table II" in out
        assert "SKIPPED" not in out
        # 70 Hz rerun mean chunk acc is 78.20; Table II CNNAttnS is 77.0 -> +1.20.
        assert "(+1.20)" in out
        # The mislabelled per-dataset sens/spec must not be reported.
        assert "Per-dataset sensitivity/specificity are intentionally not reported" in out

    def test_broken_folds_skip_paired_stats_for_that_model_only(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write_all_runs(tmp_path, break_folds_for="CNNAttnS (in-ear)")
        sys.argv = ["freq_cutoff_summary.py", "--root", str(tmp_path)]
        fcs.main()
        out = capsys.readouterr().out
        assert "MISMATCH" in out
        assert "CNNAttnS (in-ear): SKIPPED (fold-identity check failed)" in out
        assert any(
            line.lstrip().startswith("AllTransformerV4") and "diff = " in line
            for line in out.splitlines()
        )
        assert not any(
            line.lstrip().startswith("CNNAttnS") and "diff = " in line for line in out.splitlines()
        )

    def test_missing_runs_are_skipped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        sys.argv = ["freq_cutoff_summary.py", "--root", str(tmp_path)]
        fcs.main()
        assert "[skip]" in capsys.readouterr().out

    def test_check_folds_only(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _write_all_runs(tmp_path)
        sys.argv = ["freq_cutoff_summary.py", "--root", str(tmp_path), "--check-folds"]
        fcs.main()
        out = capsys.readouterr().out
        assert "OK:" in out
        assert "Main table" not in out
