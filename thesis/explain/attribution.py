"""
Gradient- and attention-based attribution for the spectrogram models.

Two mechanism-level methods that corroborate the ablation results:

* **Integrated Gradients** (Sundararajan et al., 2017) attributes the predicted logit back to
  input pixels along a straight path from a baseline. Hand-rolled rather than pulled from
  ``captum``: the model has a single tensor input, so the whole method is a few lines, and it
  keeps the dependency list unchanged.
* **Attention rollout** (Abnar & Zuidema, 2020) composes attention across layers while
  accounting for residual connections.

Ablation stays the primary evidence. Attention weights are not an explanation on their own
(Jain & Wallace, 2019), and gradient attribution answers a different question than "what
happens to accuracy if this channel is gone". Agreement between the three is the result worth
reporting.
"""

import logging
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import cast

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def integrated_gradients(
    model: nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    target: torch.Tensor | int,
    n_steps: int = 32,
) -> torch.Tensor:
    """
    Integrated Gradients attribution of a target logit to the input spectrogram.

    Approximates the path integral with the Riemann midpoint rule, which converges faster than
    the left-endpoint rule for the same step count.

    :param torch.nn.Module model: Model to attribute; put in ``eval()`` mode by the caller.
    :param torch.Tensor x: Input batch of shape ``(N, C, F, T)``.
    :param torch.Tensor baseline: Reference input, either ``(C, F, T)`` (broadcast over the
        batch) or the same shape as ``x``. Use the training-set mean spectrogram, matching the
        occlusion reference, so that ablation and attribution share a notion of "absent".
    :param target: Class index per sample, or a single index applied to the whole batch.
    :param int n_steps: Number of interpolation steps.
    :return: Attributions with the same shape as ``x``. Summing them gives the model's logit
        change from baseline to input (the completeness axiom), up to discretisation error.
    :rtype: torch.Tensor
    :raises ValueError: If ``n_steps`` is not positive or the baseline shape is incompatible.
    """
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")

    if baseline.dim() == x.dim() - 1:
        baseline = baseline.unsqueeze(0).expand_as(x)
    if baseline.shape != x.shape:
        raise ValueError(
            f"Baseline shape {tuple(baseline.shape)} incompatible with input {tuple(x.shape)}"
        )

    baseline = baseline.to(device=x.device, dtype=x.dtype)
    if isinstance(target, int):
        target = torch.full((x.shape[0],), target, dtype=torch.long, device=x.device)
    target = target.to(x.device)

    delta = x - baseline
    total = torch.zeros_like(x)

    for step in range(n_steps):
        alpha = (step + 0.5) / n_steps  # midpoint rule
        point = (baseline + alpha * delta).detach().requires_grad_(True)
        logits = model(point)
        selected = logits.gather(1, target.view(-1, 1)).sum()
        (grad,) = torch.autograd.grad(selected, point)
        total += grad.detach()

    return delta.detach() * total / n_steps


def completeness_error(
    model: nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    attributions: torch.Tensor,
    target: torch.Tensor | int,
) -> torch.Tensor:
    """
    Per-sample violation of the Integrated Gradients completeness axiom.

    Attributions must sum to ``f(x) - f(baseline)`` for the target logit. A large residual
    means too few integration steps, and is the sanity check to run before trusting any
    attribution map.

    :param torch.nn.Module model: The attributed model.
    :param torch.Tensor x: Input batch.
    :param torch.Tensor baseline: Reference input, as passed to :func:`integrated_gradients`.
    :param torch.Tensor attributions: Output of :func:`integrated_gradients`.
    :param target: Class index per sample, or one index for the batch.
    :return: Absolute error per sample, shape ``(N,)``.
    :rtype: torch.Tensor
    """
    if baseline.dim() == x.dim() - 1:
        baseline = baseline.unsqueeze(0).expand_as(x)
    baseline = baseline.to(device=x.device, dtype=x.dtype)
    if isinstance(target, int):
        target = torch.full((x.shape[0],), target, dtype=torch.long, device=x.device)
    target = target.to(x.device)

    with torch.no_grad():
        gap = (model(x) - model(baseline)).gather(1, target.view(-1, 1)).squeeze(1)
    attributed = attributions.flatten(start_dim=1).sum(dim=1)
    return cast(torch.Tensor, (attributed - gap).abs())


@contextmanager
def capture_attention(model: nn.Module) -> Iterator[list[torch.Tensor]]:
    """
    Capture per-layer self-attention weights from a ``nn.TransformerEncoder``.

    ``nn.TransformerEncoderLayer.forward`` calls its ``self_attn`` with ``need_weights=False``
    hardcoded, so the weights are never computed on the normal path and a plain forward hook
    captures nothing. This re-runs attention from a forward-pre-hook on each layer.

    The models use ``norm_first=True``, so attention operates on ``norm1(x)``, not on the raw
    layer input; the hook reproduces that. Weights are therefore over LayerNorm'd tokens.

    :param torch.nn.Module model: A model exposing ``model.transformer.layers``.
    :yield: List that fills with one ``(B, nhead, S, S)`` tensor per layer, in layer order,
        each time the model is called. Clear it between batches.
    :raises AttributeError: If the model has no ``transformer.layers``.
    """
    layers = cast(Iterable[nn.TransformerEncoderLayer], model.transformer.layers)  # type: ignore[union-attr]
    captured: list[torch.Tensor] = []
    handles = []

    def make_hook(
        layer: nn.TransformerEncoderLayer,
    ) -> Callable[[nn.Module, tuple[torch.Tensor, ...]], None]:
        def pre_hook(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            normed = layer.norm1(args[0])  # norm_first=True
            _, weights = layer.self_attn(
                normed,
                normed,
                normed,
                need_weights=True,
                average_attn_weights=False,  # keep heads separate: (B, nhead, S, S)
            )
            captured.append(weights.detach())

        return pre_hook

    try:
        for layer in layers:
            handles.append(layer.register_forward_pre_hook(make_hook(layer)))
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def attention_rollout(
    layer_weights: Sequence[torch.Tensor], residual_weight: float = 0.5
) -> torch.Tensor:
    """
    Compose per-layer attention into end-to-end token influence.

    Each layer's attention is averaged over heads and mixed with the identity to account for
    the residual connection, then the layers are multiplied together:
    ``R = (w*A_L + (1-w)*I) @ ... @ (w*A_1 + (1-w)*I)``.

    :param layer_weights: One ``(B, nhead, S, S)`` tensor per layer, in layer order, as
        produced by :func:`capture_attention`.
    :param float residual_weight: Weight on attention versus the residual path. ``0.5`` is the
        original paper's equal split.
    :return: Rollout matrix ``(B, S, S)``; ``R[b, i, j]`` is the influence of token ``j`` on
        token ``i``. Rows sum to 1.
    :rtype: torch.Tensor
    :raises ValueError: If no layers are given or ``residual_weight`` is outside [0, 1].
    """
    if not layer_weights:
        raise ValueError("Need at least one layer of attention weights")
    if not 0.0 <= residual_weight <= 1.0:
        raise ValueError(f"residual_weight must be in [0, 1], got {residual_weight}")

    rollout: torch.Tensor | None = None
    for weights in layer_weights:
        averaged = weights.mean(dim=1).to(torch.float64)  # (B, S, S), heads averaged
        eye = torch.eye(averaged.shape[-1], device=averaged.device, dtype=averaged.dtype)
        mixed = residual_weight * averaged + (1.0 - residual_weight) * eye
        mixed = mixed / mixed.sum(dim=-1, keepdim=True)
        rollout = mixed if rollout is None else torch.bmm(mixed, rollout)

    assert rollout is not None
    return rollout.to(torch.float32)


def rollout_token_influence(rollout: torch.Tensor) -> torch.Tensor:
    """
    Reduce a rollout matrix to a per-token influence score on the model's readout.

    ``AllTransformerV4`` reads out by mean-pooling every token — there is no CLS row to index.
    Influence on the pooled vector is therefore the column mean of the rollout matrix, since
    each output token enters the pool with equal weight.

    :param torch.Tensor rollout: Rollout matrix ``(B, S, S)``.
    :return: Per-token influence ``(B, S)``, summing to 1 per sample.
    :rtype: torch.Tensor
    """
    return rollout.mean(dim=1)


def tokens_to_channels(token_scores: torch.Tensor, n_channels: int) -> torch.Tensor:
    """
    Sum per-token scores into per-channel scores.

    Relies on ``AllTransformerV4``'s token layout, where token ``i`` is
    ``(channel, time) = (i // W', i % W')``.

    :param torch.Tensor token_scores: Scores of shape ``(B, n_channels * W')``.
    :param int n_channels: Number of EEG channels.
    :return: Per-channel scores of shape ``(B, n_channels)``.
    :rtype: torch.Tensor
    :raises ValueError: If the token count is not divisible by the channel count.
    """
    n_tokens = token_scores.shape[-1]
    if n_tokens % n_channels != 0:
        raise ValueError(f"{n_tokens} tokens is not divisible by {n_channels} channels")
    return token_scores.reshape(*token_scores.shape[:-1], n_channels, -1).sum(dim=-1)
