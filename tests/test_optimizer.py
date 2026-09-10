"""Tests for the selectable training optimizer (--optimizer CLI flag).

These tests verify:
- The CLI defaults to "adam" (existing behaviour must reproduce bit-for-bit) and accepts
  "adamw" as the decoupled-weight-decay alternative for the regularization sweep.
- ``create_optimizer()`` constructs the expected ``torch.optim`` class for each choice.
"""

import pytest
import torch
import torch.nn as nn

from thesis.cli import get_arg_parser
from thesis.model_factory import create_optimizer


class TestOptimizerCLIFlag:
    """--optimizer argument parsing."""

    def test_default_is_adam(self):
        """Omitting --optimizer must default to "adam" so existing configs are unaffected."""
        parser = get_arg_parser()
        args = parser.parse_args(["train", "--model", "CNNLSTM"])
        assert args.optimizer == "adam"

    def test_accepts_adamw(self):
        parser = get_arg_parser()
        args = parser.parse_args(["train", "--model", "CNNLSTM", "--optimizer", "adamw"])
        assert args.optimizer == "adamw"

    def test_rejects_invalid_choice(self):
        parser = get_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["train", "--model", "CNNLSTM", "--optimizer", "sgd"])


class TestCreateOptimizer:
    """create_optimizer() builds the expected torch.optim class."""

    def _params(self):
        model = nn.Linear(4, 2)
        return model.parameters()

    def test_adam_is_default_and_constructs_adam(self):
        optimizer = create_optimizer("adam", self._params(), lr=1e-4, weight_decay=1e-4)
        assert isinstance(optimizer, torch.optim.Adam)
        assert not isinstance(optimizer, torch.optim.AdamW)

    def test_adamw_constructs_adamw(self):
        optimizer = create_optimizer("adamw", self._params(), lr=1e-4, weight_decay=1e-4)
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_unknown_optimizer_raises(self):
        with pytest.raises(ValueError):
            create_optimizer("sgd", self._params(), lr=1e-4, weight_decay=1e-4)

    def test_lr_and_weight_decay_are_applied(self):
        optimizer = create_optimizer("adamw", self._params(), lr=5e-3, weight_decay=2e-2)
        param_group = optimizer.param_groups[0]
        assert param_group["lr"] == pytest.approx(5e-3)
        assert param_group["weight_decay"] == pytest.approx(2e-2)
