# SmallerAllV3: Time-Aware Cross-Channel Attention (~0.89M params)

## Context

Three architectural weaknesses in the existing multichannel models prompted this design:

1. **`SmallerAll` / `CNN_LSTM_DepCapAll` (Conv3d)**: a single 3D kernel collapses all 8 channels in one shot — rigid, no ongoing cross-channel reasoning after the first layer.
2. **`SmallerAllV2` cross-channel attention**: `tokens = x_ch.mean(dim=4)` pools away the time axis *before* computing channel importance, so the model produces one global set of channel weights for the whole clip — a channel informative only in certain time windows gets the same weight as one informative everywhere.
3. **No EEGNet-style inductive bias**: EEGNet shows the right prior for multi-channel EEG is spectral extraction *per channel* followed by a *per-filter* depthwise spatial convolution across channels.

The fix: **preserve the time axis through the cross-channel attention step** so channel importance is computed per time frame, not globally.

---

## Architecture: `SmallerAllV3`

```
Input: (B, 8, 129, 41)

Step 1 — Per-channel spectral CNN  (Smaller backbone, weight-shared)
  reshape: (B*8, 1, 129, 41)
  Conv2d(1→32, 10×10, stride=2) → MaxPool → Dropout2d
  Conv2d(32→16,  5×5,  stride=1) → MaxPool → Dropout2d
  reshape back: (B, 8, 16, 54, 10)     [C_feat=16, H'=54, W'=10]

Step 2 — Time-aware cross-channel self-attention
  permute + reshape: (B, 8, 16, 54, 10)
                   → (B, 10, 8, 16, 54)        keep W'=10 as second dim
                   → (B*10, 8, 864)             864 = 16*54; fold B and W' together

  chan_proj  Linear(864 → 128):   (B*10, 8, 128)
  TransformerEncoder (1 layer, d_model=128, nhead=8, ff=256, norm_first):
                                  (B*10, 8, 128)   ← self-attention across 8 channel tokens
  chan_gate  Linear(128 → 1):     (B*10, 8, 1)
  softmax over dim=1 (channels):  (B*10, 8, 1)
  weighted sum over channels:     (B*10, 128)
  reshape:                         (B, 10, 128)    ← temporal sequence, already channel-merged

Step 3 — Temporal aggregation
  LSTM(input=128, hidden=330):   last hidden → (B, 330)

Step 4 — Classifier  (CNN_LSTM_DepCap-style head)
  Dropout → Linear(330→64) → ReLU → Linear(64→32) → ReLU → Linear(32→num_classes)
```

---

## How the Attention Works in Detail

After the per-channel CNN each EEG clip is represented as a 5D tensor `(B, 8, 16, 54, 10)`:
- axis 1 = 8 EEG channels
- axes 2–3 = 16 feature maps × 54 frequency bins (spatial/spectral features per channel)
- axis 4 = 10 time frames (W', the temporal axis)

**The reshape trick** folds batch and time together:

```
(B, 8, 16, 54, 10)  permute→  (B, 10, 8, 16, 54)  reshape→  (B*10, 8, 864)
```

We now have `B × 10` independent "mini-batches", each containing exactly **8 tokens** — one per EEG channel — describing that channel's spectral content *at one specific time frame*.

**Within the TransformerEncoder**, each of the 8 channel tokens can attend to all other 7:

```
Q_i = W_Q · token_i      (what channel i is looking for)
K_j = W_K · token_j      (what channel j offers)
score_{ij} = Q_i · K_j / √d   (how relevant is channel j to channel i)
attention weights = softmax(scores) over j
new_token_i = Σ_j attention_weight_{ij} · V_j
```

The model can learn things like:
- "when the frontal channels (Fp1, Fp2) show a particular pattern at time t, amplify the temporal channels (T7, T8) at the same time t"
- "at time steps where all channels are noisy together, down-weight all of them"

**The gating layer** then produces a single scalar per channel per time step:

```
gate_i = sigmoid-free_softmax( Linear(128→1)(token_i) )   over i=1..8
merged_t = Σ_i  gate_i  ·  token_i              ∈ R^128
```

This gives `(B*10, 128)` — one 128-dim vector for each of the 10 time steps — which is reshaped to `(B, 10, 128)`: the temporal sequence fed to the LSTM.

**Key difference from SmallerAllV2:**

| | SmallerAllV2 | SmallerAllV3 |
|---|---|---|
| Shape before channel attention | `(B, 8, 16, 54)` — time pooled away | `(B*10, 8, 864)` — time kept |
| Channel weights | One set per clip | One set **per time frame** |
| Can learn "channel X matters only at time t"? | ✗ | ✓ |
| Output of channel step | `(B, 16, H', W')` — needs extra reshape for LSTM | `(B, 10, 128)` — already a temporal sequence |

---

## Parameter Budget

With `input_shape=(129,41)`, `in_channels=8`, `chan_d_model=128`, `rnn_hidden=330`:

| Component | Params |
|---|---|
| Per-channel CNN (32→16 filters, weight-shared) | ~16,048 |
| `chan_proj` Linear(864, 128) | ~110,848 |
| TransformerEncoder (d_model=128, nhead=8, ff=256, 1 layer) | ~132,480 |
| `chan_gate` Linear(128, 1) | 129 |
| LSTM(128 → 330) | ~605,880 |
| Classifier (330→64→32→num_classes) | ~23,330 |
| **Total** | **~888,715 ≈ 889K** |

---

## Implementation

### `thesis/model.py` — only file to modify

**1. Add `SmallerAllV3` class** after `SmallerAllV2Attn` (before `MODEL_REGISTRY`):

```python
class SmallerAllV3(Smaller):
    """
    Multi-channel EEG model: per-channel weight-shared CNN + time-aware
    cross-channel self-attention + temporal LSTM.

    Compared to SmallerAllV2, channel importance is computed *per time frame*
    (W' axis preserved during attention) so the model can learn that a channel
    is informative only in certain time windows.

    :param tuple input_shape: Spectrogram shape (H, W).
    :param int in_channels: EEG channels (default 8).
    :param str rnn_type: "LSTM", "GRU", or "ATTENTION".
    :param int rnn_hidden: Hidden size for LSTM/GRU or d_model for attention (default 330).
    :param float dropout: Dropout probability.
    :param int num_classes: Output classes.
    :param int chan_d_model: Token dimension for cross-channel attention (default 128).
    """

    def __init__(
        self,
        input_shape: tuple[int, int],
        in_channels: int = 8,
        rnn_type: str = "LSTM",
        rnn_hidden: int = 330,
        dropout: float = 0.3,
        num_classes: int = 2,
        chan_d_model: int = 128,
    ) -> None:
        super().__init__(
            input_shape,
            in_channels=1,       # CNN sees one channel at a time (shared weights)
            rnn_type=rnn_type,
            rnn_hidden=rnn_hidden,
            dropout=dropout,
            num_classes=num_classes,
        )
        # Override classifier to use DepCap-style head (→64→32→num_classes)
        self.fc1 = nn.Linear(rnn_hidden, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, num_classes)

        self._n_eeg_channels = in_channels

        # Dummy pass to get CNN output dims
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            cnn_out = self._forward_conv_layers(dummy)
            _, C_feat, H_prime, W_prime = cnn_out.shape

        self.chan_proj = nn.Linear(C_feat * H_prime, chan_d_model)
        nhead = max(1, chan_d_model // 16)
        chan_layer = nn.TransformerEncoderLayer(
            d_model=chan_d_model,
            nhead=nhead,
            dim_feedforward=chan_d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.chan_transformer = nn.TransformerEncoder(chan_layer, num_layers=1)
        self.chan_gate = nn.Linear(chan_d_model, 1)

        # Override RNN: input size is now chan_d_model, not rnn_input_size
        if rnn_type.upper() == "LSTM":
            self.rnn = nn.LSTM(
                input_size=chan_d_model, hidden_size=rnn_hidden, batch_first=True
            )
        elif rnn_type.upper() == "GRU":
            self.rnn = nn.GRU(
                input_size=chan_d_model, hidden_size=rnn_hidden, batch_first=True
            )
        elif rnn_type.upper() == "ATTENTION":
            self.proj = nn.Linear(chan_d_model, rnn_hidden)
            self.pos_embed = nn.Parameter(torch.zeros(1, W_prime, rnn_hidden))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=rnn_hidden,
                nhead=max(1, rnn_hidden // 16),
                dim_feedforward=rnn_hidden * 2,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self._W_prime = W_prime
        self._C_feat = C_feat
        self._H_prime = H_prime

        logger.info(
            f"SmallerAllV3: n_eeg_channels={in_channels}, C_feat={C_feat}, "
            f"H_prime={H_prime}, W_prime={W_prime}, chan_d_model={chan_d_model}, "
            f"nhead={nhead}, rnn_hidden={rnn_hidden}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: per-channel CNN → time-aware cross-channel attention → LSTM.

        :param torch.Tensor x: Input (B, n_ch, H, W).
        :return: Logits (B, num_classes).
        :rtype: torch.Tensor
        """
        B, n_ch, H, W = x.shape

        # 1. Per-channel CNN (shared weights)
        x_flat = x.reshape(B * n_ch, 1, H, W)
        cnn_out = self._forward_conv_layers(x_flat)          # (B*8, 16, H', W')
        _, C_feat, H_prime, W_prime = cnn_out.shape
        x_ch = cnn_out.reshape(B, n_ch, C_feat, H_prime, W_prime)

        # 2. Time-aware cross-channel attention
        # Keep the W' time axis: fold B and W' together so each time step
        # gets its own independent attention computation across 8 channels
        x_t = x_ch.permute(0, 4, 1, 2, 3).contiguous()     # (B, W', n_ch, C_feat, H')
        tokens = x_t.reshape(B * W_prime, n_ch, C_feat * H_prime)
        tokens = self.chan_proj(tokens)                       # (B*W', n_ch, chan_d_model)
        tokens = self.chan_transformer(tokens)                # (B*W', n_ch, chan_d_model)

        weights = torch.softmax(self.chan_gate(tokens), dim=1)   # (B*W', n_ch, 1)
        merged = (tokens * weights).sum(dim=1)                   # (B*W', chan_d_model)
        seq = merged.reshape(B, W_prime, -1)                     # (B, W', chan_d_model)

        # 3. Temporal aggregation
        if self.rnn_type == "ATTENTION":
            seq = self.proj(seq) + self.pos_embed
            attn_out = self.transformer(seq)
            last_hidden = attn_out.mean(dim=1)
        else:
            rnn_out, _ = self.rnn(seq)
            last_hidden = rnn_out[:, -1, :]

        # 4. Classifier
        x = self.dropout(last_hidden)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
```

**2. Add to `MODEL_REGISTRY`:**

```python
"SmallerAllV3": (SmallerAllV3, 330),
```

### No other files need changes

`main.py` already routes `in_channels=8` when `--channel all`. `model_factory.py` generic instantiation works as-is.

---

## Verification

```bash
# Parameter count
poetry run python -c "
from thesis.model import SmallerAllV3
m = SmallerAllV3((129, 41), in_channels=8)
all_p, train_p = m.count_parameters()
print(f'Total: {all_p:,}  Trainable: {train_p:,}')
"

# Forward pass shape
poetry run python -c "
import torch; from thesis.model import SmallerAllV3
m = SmallerAllV3((129, 41), in_channels=8)
print(m(torch.randn(4, 8, 129, 41)).shape)  # expect (4, 2)
"

# 4-class and attention variants
poetry run python -c "
import torch; from thesis.model import SmallerAllV3
print(SmallerAllV3((129,41), num_classes=4)(torch.randn(4,8,129,41)).shape)
print(SmallerAllV3((129,41), rnn_type='ATTENTION', rnn_hidden=128)(torch.randn(4,8,129,41)).shape)
"
```
