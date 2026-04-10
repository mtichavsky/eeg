# Plan: ChannelFormer — 8-channel EEG Transformer

**Save to**: `docs/plans/2026-04-10-channel-former.md`

## Context

`SmallerAllV3` (~1M params) is dominated by a single bottleneck: `Linear(6912→128)` which projects concatenated per-channel CNN features into a manageable vector before the LSTM. This 885K-param layer is added precisely because concatenating 8×16×54=6912 features is expensive. The result is a model that likely overfits before it learns generalizable cross-channel patterns.

The fix: instead of flattening all channels at each time step into one huge vector, treat every **(channel, timestep)** pair as an independent token of dim 864 (=16×54). A shared `Linear(864→64)` projects all 80 tokens to `d_model`, separate spatial and temporal embeddings encode position, and a 2-layer `TransformerEncoder` lets inter-channel attention emerge naturally — matching the approach already proven to work in `SmallerAttn` for single channels.

**Estimated params**: ~142K (7× reduction from SmallerAllV3 ~1M).

## Architecture: `ChannelFormer`

```
Input: (B, 8, 129, 41)

1. Per-channel CNN  [shared weights, 16K params]
   reshape → (B*8, 1, 129, 41)
   Conv2d(1→32, 10×10, stride=2) + ReLU + MaxPool(2×2,s=1) + Dropout2d
   Conv2d(32→16, 5×5) + ReLU + MaxPool(2×2,s=1) + Dropout2d
   → (B*8, 16, 54, 10)   # C_feat=16, H'=54, W'=10

2. Per-token projection  [55K params]
   reshape to (B, 8, 10, 16*54=864)
   Linear(864 → d_model=64)  ← shared across all (chan, time) positions
   → (B, 8, 10, 64)

3. Positional embeddings
   chan_embedding: Embedding(8, 64)   [512 params]
   time_embedding: Embedding(10, 64)  [640 params]
   tokens += chan_embedding[c][None,:,None,:]  # (1,8,1,64)
   tokens += time_embedding[t][None,None,:,:]  # (1,1,10,64)

4. Flatten + Dropout
   → (B, 80, 64)   # 80 tokens = 8 chan × 10 time steps
   Dropout(0.3)

5. TransformerEncoder  [67K params]
   2 × TransformerEncoderLayer(d_model=64, nhead=4,
                                dim_feedforward=128, norm_first=True)
   → (B, 80, 64)

6. Mean pool over tokens → (B, 64)

7. Classifier  [2.6K params]
   Linear(64→32) + ReLU → Linear(32→16) + ReLU → Linear(16→num_classes)
```

## Inheritance strategy

Inherit from `Smaller` using `rnn_type="ATTENTION"` and `in_channels=1`. This gives us for free:
- CNN layers: `conv1, conv2, pool1, pool2, dropout1, dropout2d_2`
- `self.proj = Linear(864, d_model)` — exactly our token projection (correct dims!)
- `self.fc1/fc2/out` — correct classifier sizes
- `_forward_conv_layers()` method

We then:
1. `del self.pos_embed` → replace with `chan_embedding` + `time_embedding`
2. Override `self.transformer` (1 layer → `num_layers=2`)

## Files to modify

| File | Change |
|------|--------|
| `thesis/model.py` | Add `ChannelFormer` class after `SmallerAllV3` (~90 lines) |
| `thesis/model.py` | Add `"ChannelFormer": (ChannelFormer, 64)` to `MODEL_REGISTRY` |

No changes needed to `model_factory.py`, `main.py`, or data pipeline.

## Implementation steps

### 1. Add `ChannelFormer` class (thesis/model.py, after line 742)

```python
class ChannelFormer(Smaller):
    """
    8-channel EEG classifier: shared per-channel CNN → (channel×time) token Transformer.

    Each (channel, timestep) pair becomes one token after the CNN backbone.
    Separate spatial (channel) and temporal (timestep) embeddings are added,
    then 2 TransformerEncoder layers learn inter-channel and inter-frame relationships.
    Mean pooling over all 80 tokens feeds the classifier.

    Compared to SmallerAllV3: eliminates the 885K-param bottleneck projection.
    Parameter count: ~142K (d_model=64) vs ~1M for SmallerAllV3.

    Architecture::

        (B, n_ch, H, W)
        → Per-channel CNN (shared): (B*n_ch, 1, H, W) → (B*n_ch, 16, H', W')
        → Reshape: (B, n_ch, W', C_feat*H'=864)
        → token_proj: (B, n_ch, W', d_model)
        → + chan_embedding[c] + time_embedding[t]
        → flatten: (B, n_ch*W', d_model)
        → TransformerEncoder(2 layers)
        → mean pool: (B, d_model)
        → classifier (→32→16→num_classes)

    :param tuple input_shape: Spectrogram shape (H, W), typically (129, 41).
    :param int in_channels: Number of EEG channels (default 8).
    :param int num_classes: Output classes (default 2).
    :param float dropout: Dropout probability (default 0.3).
    :param int rnn_hidden: d_model for the Transformer (default 64).
    :param int num_layers: Number of TransformerEncoder layers (default 2).
    """

    def __init__(
        self,
        input_shape: tuple[int, int],
        in_channels: int = 8,
        num_classes: int = 2,
        dropout: float = 0.3,
        rnn_hidden: int = 64,
        num_layers: int = 2,
        **kwargs: object,
    ) -> None:
        # Inherit CNN backbone + proj + classifier from Smaller's ATTENTION branch.
        # in_channels=1: CNN processes one EEG channel at a time (shared weights).
        # self.proj = Linear(C_feat*H', rnn_hidden) matches our token projection dims.
        super().__init__(
            input_shape,
            in_channels=1,
            rnn_type="ATTENTION",
            rnn_hidden=rnn_hidden,
            dropout=dropout,
            num_classes=num_classes,
        )

        self._n_eeg_channels = in_channels
        d_model = rnn_hidden

        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            cnn_out = self._forward_conv_layers(dummy)
            _, C_feat, H_prime, W_prime = cnn_out.shape

        self._C_feat = C_feat
        self._H_prime = H_prime
        self._W_prime = W_prime
        self._d_model = d_model

        # Replace 1D temporal pos_embed with 2D spatial+temporal embeddings.
        del self.pos_embed
        self.chan_embedding = nn.Embedding(in_channels, d_model)
        self.time_embedding = nn.Embedding(W_prime, d_model)

        # Replace 1-layer transformer with num_layers version.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=max(1, d_model // 16),
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        logger.info(
            f"ChannelFormer: n_eeg_channels={in_channels}, C_feat={C_feat}, "
            f"H_prime={H_prime}, W_prime={W_prime}, "
            f"tokens={in_channels * W_prime}, d_model={d_model}, num_layers={num_layers}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: per-channel CNN → (chan×time) tokens → Transformer → classify.

        :param torch.Tensor x: Input of shape (B, n_ch, H, W).
        :return: Class logits of shape (B, num_classes).
        :rtype: torch.Tensor
        """
        B, n_ch, H, W = x.shape

        # 1. Per-channel CNN (weight-shared across channels).
        x_flat = x.reshape(B * n_ch, 1, H, W)
        cnn_out = self._forward_conv_layers(x_flat)          # (B*n_ch, C_feat, H', W')
        _, C_feat, H_prime, W_prime = cnn_out.shape

        # 2. Reshape to per-(channel, time) tokens.
        tokens = cnn_out.reshape(B, n_ch, C_feat, H_prime, W_prime)
        tokens = tokens.permute(0, 1, 4, 2, 3).contiguous()  # (B, n_ch, W', C_feat, H')
        tokens = tokens.reshape(B, n_ch, W_prime, C_feat * H_prime)  # (B, n_ch, W', 864)

        # 3. Project to d_model (reuses self.proj from Smaller's ATTENTION branch).
        tokens = self.proj(tokens)                            # (B, n_ch, W', d_model)

        # 4. Add 2D positional embeddings.
        chan_idx = torch.arange(n_ch, device=x.device)
        time_idx = torch.arange(W_prime, device=x.device)
        tokens = tokens + self.chan_embedding(chan_idx)[None, :, None, :]   # (1,n_ch,1,d)
        tokens = tokens + self.time_embedding(time_idx)[None, None, :, :]  # (1,1,W',d)

        # 5. Flatten to token sequence + dropout.
        tokens = tokens.reshape(B, n_ch * W_prime, self._d_model)  # (B, 80, d_model)
        tokens = self.dropout(tokens)

        # 6. Transformer encoder.
        tokens = self.transformer(tokens)                     # (B, 80, d_model)

        # 7. Mean pool + classify.
        pooled = tokens.mean(dim=1)                           # (B, d_model)
        out = F.relu(self.fc1(pooled))
        out = F.relu(self.fc2(out))
        return self.out(out)
```

### 2. Register in MODEL_REGISTRY (thesis/model.py, line 755)

```python
    "SmallerAllV3": (SmallerAllV3, 100),
    "ChannelFormer": (ChannelFormer, 64),   # ← add this line
```

## Verification

```bash
# Quick sanity check — test mode, single fold
poetry run python main.py train \
  --model ChannelFormer \
  --channel all \
  --dataset all \
  --condition ec \
  --class-mode 2 \
  --test-mode \
  --n-folds 2 \
  --checkpoint-dir experiments/all3-channelformer-smoke

# Full binary run (compare against SmallerAllV3 all3-022 baseline: 78.56% chunk)
poetry run python main.py train \
  --model ChannelFormer \
  --channel all \
  --dataset all \
  --condition ec \
  --class-mode 2 \
  --n-folds 6 \
  --checkpoint-dir experiments/all3-023-all-binary-channelformer

# 4-class run
poetry run python main.py train \
  --model ChannelFormer \
  --channel all \
  --dataset all \
  --condition ec \
  --class-mode 4 \
  --n-folds 6 \
  --checkpoint-dir experiments/all3-023-all-4class-channelformer
```

Check `cv_results.txt` for chunk accuracy, subject accuracy, sensitivity, specificity. Target: ≥78.5% chunk accuracy with significantly fewer params (~142K vs ~1M).
