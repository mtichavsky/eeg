"""Tests for the ``--select-metric`` checkpoint-selection criterion."""

import pytest

from thesis.cli import get_arg_parser
from thesis.metrics import selection_score


def _metrics(accuracy: float, recall: float, specificity: float) -> dict[str, float]:
    return {"accuracy": accuracy, "recall": recall, "specificity": specificity}


def test_accuracy_is_default_and_ignores_balance() -> None:
    m = _metrics(0.9, 1.0, 0.0)
    assert selection_score(m) == 0.9
    assert selection_score(m, "accuracy") == 0.9


def test_balanced_is_mean_of_sensitivity_and_specificity() -> None:
    assert selection_score(_metrics(0.9, 1.0, 0.0), "balanced") == 0.5
    assert selection_score(_metrics(0.7, 0.8, 0.6), "balanced") == pytest.approx(0.7)


def test_balanced_prefers_balanced_epoch_over_degenerate_one() -> None:
    """A collapsed always-positive epoch must not outrank a balanced one on the balanced metric."""
    collapsed = _metrics(0.61, 1.0, 0.0)
    balanced = _metrics(0.58, 0.6, 0.55)
    assert selection_score(collapsed) > selection_score(balanced)
    assert selection_score(collapsed, "balanced") < selection_score(balanced, "balanced")


def test_unknown_metric_raises() -> None:
    with pytest.raises(ValueError):
        selection_score(_metrics(0.5, 0.5, 0.5), "f1")


def test_cli_flag_default_and_choices() -> None:
    parser = get_arg_parser()
    base = ["train", "--model", "AllTransformerV4"]
    assert parser.parse_args(base).select_metric == "accuracy"
    assert parser.parse_args([*base, "--select-metric", "balanced"]).select_metric == "balanced"
    with pytest.raises(SystemExit):
        parser.parse_args([*base, "--select-metric", "f1"])
