import logging
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from thesis.deformer import Deformer

logger = logging.getLogger(__name__)


class CNN_LSTM_DepCap(nn.Module):
    def __init__(
        self,
        input_shape,
        in_channels=1,
        rnn_type="LSTM",
        rnn_hidden=100,
        dropout=0.3,
        num_classes=2,
    ):
        """
        Initialize the CNN-LSTM model for depression detection from spectrograms.

        Implements the architecture from the paper:
         - Conv2D(64, 10x10, stride=2), ReLU -> MaxPool(2x2, stride=1)
         - Conv2D(32, 5x5, stride=1), ReLU -> MaxPool(2x2, stride=1) - paper uses 15x15 kernel
         - Reshape to sequence (time=W) and pass through LSTM/GRU (hidden=100)
         - Dense 64 -> Dense 32 -> Output 2 classes


        :param tuple input_shape: Tuple of (height, width) for the input spectrogram.
        :param int in_channels: Number of input channels (default=1 for one channel EEG).
        :param str rnn_type: Type of RNN to use - "LSTM" or "GRU".
        :param int rnn_hidden: Hidden size for RNN layer.
        :param float dropout: Dropout probability for regularization.
        :param int num_classes: Number of output classes of the classifier.
        """
        super().__init__()

        # Conv layers
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(10, 10), stride=2, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.dropout2d_1 = nn.Dropout2d(dropout)
        # The paper works with kernel size 15x15, my spectrogram is too small for this
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(5, 5), stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=1)
        self.dropout2d_2 = nn.Dropout2d(dropout)
        self.relu = nn.ReLU(inplace=True)

        # Calculate RNN input size by doing a dummy forward pass through conv layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, *input_shape)
            conv_output = self._forward_conv_layers(dummy_input)
            _, C, H, W = conv_output.shape
            rnn_input_size = H * C  # Each time step has H*C features

        # Initialize RNN with correct input size
        self.rnn_type = rnn_type.upper()
        self.rnn: Union[nn.LSTM, nn.GRU]
        if self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size=rnn_input_size, hidden_size=rnn_hidden, batch_first=True)
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(input_size=rnn_input_size, hidden_size=rnn_hidden, batch_first=True)
        else:
            raise ValueError("rnn_type must be 'LSTM' or 'GRU'")

        logger.info(
            f"Initialized {self.rnn_type} with input_size={rnn_input_size}, "
            f"hidden_size={rnn_hidden}, batch_first=True"
        )

        # Classifier head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(rnn_hidden, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, num_classes)

    def _forward_conv_layers(self, x):
        """
        Pass input through convolutional layers only.

        :param torch.Tensor x: Input tensor of shape (B, C, H, W).
        :return: Output tensor after convolution and pooling layers.
        :rtype: torch.Tensor
        """
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout2d_1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2d_2(x)
        return x

    def count_parameters(self) -> tuple[int, int]:
        """
        Count the number of parameters in the model.

        :return: Tuple of (all_params, trainable_params).
        :rtype: tuple[int, int]
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        all = sum(p.numel() for p in self.parameters())
        return all, trainable

    def forward(self, x):
        """
        Forward pass through the network.

        :param torch.Tensor x: Input tensor of shape (B, C, H, W) where B=batch size,
                               C=channels (typically 1 for mono spectrograms),
                               H=height (frequency bins), W=width (time frames).
        :return: Class probabilities of shape (B, num_classes) after softmax.
        :rtype: torch.Tensor
        """
        # Pass through convolutional layers
        x = self._forward_conv_layers(x)  # -> (B, 32, H', W')

        # TODO check the reshape
        # Reshape into sequence: treat width (time) as sequence dimension
        # Each time step contains all features from height and channels
        B, C, H, W = x.shape
        seq = x.permute(0, 3, 2, 1).contiguous()  # (B, W, H, C)
        seq = seq.view(B, W, H * C)  # (B, seq_len=W, features=H*C)

        # RNN outputs hidden states for all time steps
        rnn_out, _ = self.rnn(seq)  # rnn_out: (B, seq_len, hidden_size)

        # Take the last time step's hidden state as the sequence representation
        last_hidden = rnn_out[:, -1, :]  # (B, hidden_size)

        # Pass through classifier layers
        x = self.dropout(last_hidden)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return logits


class Smaller(nn.Module):
    Pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
    Dropout1 = nn.Dropout2d

    @classmethod
    def Conv1(cls, in_channels: int) -> nn.Module:
        """
        Create the first convolutional layer.

        :param int in_channels: Number of input channels.
        :return: Configured Conv2d layer.
        :rtype: nn.Module
        """
        return nn.Conv2d(in_channels, 32, kernel_size=(10, 10), stride=2, padding=0)

    def __init__(
        self,
        input_shape,
        in_channels=1,
        rnn_type="LSTM",
        rnn_hidden=64,
        dropout=0.3,
        num_classes=2,
    ):
        """
        Initialize the reduced model for depression detection from spectrograms.

        Implements a smaller version of the CNN-LSTM architecture (~50% fewer parameters):
         - Conv2D(32, 10x10, stride=2), ReLU -> MaxPool(2x2, stride=1)
         - Conv2D(16, 5x5, stride=1), ReLU -> MaxPool(2x2, stride=1)
         - Reshape to sequence (time=W) and pass through LSTM/GRU (hidden=64)
         - Dense 32 -> Dense 16 -> Output 2 classes


        :param tuple input_shape: Tuple of (height, width) for the input spectrogram.
        :param int in_channels: Number of input channels (default=1 for one channel EEG).
        :param str rnn_type: Type of RNN to use - "LSTM" or "GRU".
        :param int rnn_hidden: Hidden size for RNN layer (default=64).
        :param float dropout: Dropout probability for regularization.
        :param int num_classes: Number of output classes of the classifier.
        """
        super().__init__()

        self.conv1 = self.Conv1(in_channels)  # Create layer with proper in_channels
        self.pool1 = self.Pool1
        self.dropout1 = self.Dropout1(dropout)

        self.conv2 = nn.Conv2d(32, 16, kernel_size=(5, 5), stride=1, padding=0)  # 32->16
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=1)
        self.dropout2d_2 = nn.Dropout2d(dropout)
        self.relu = nn.ReLU(inplace=True)

        # Calculate RNN input size by doing a dummy forward pass through conv layers
        with torch.no_grad():
            # For 3D CNN (SmallerAll): input is (B, C=1, D=in_channels, H, W)
            # For 2D CNN (Smaller): input is (B, C=in_channels, H, W)
            if isinstance(self.conv1, nn.Conv3d):
                # 3D case: channels become depth dimension
                dummy_input = torch.zeros(1, 1, in_channels, *input_shape)
            else:
                # 2D case: standard input
                dummy_input = torch.zeros(1, in_channels, *input_shape)
            conv_output = self._forward_conv_layers(dummy_input)
            _, C, H, W = conv_output.shape
            rnn_input_size = H * C  # Each time step has H*C features

        # Initialize temporal aggregator with correct input size
        self.rnn_type = rnn_type.upper()
        if self.rnn_type == "LSTM":
            self.rnn: nn.LSTM | nn.GRU = nn.LSTM(
                input_size=rnn_input_size, hidden_size=rnn_hidden, batch_first=True
            )
            logger.info(
                f"Initialized LSTM with input_size={rnn_input_size}, "
                f"hidden_size={rnn_hidden}, batch_first=True"
            )
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(input_size=rnn_input_size, hidden_size=rnn_hidden, batch_first=True)
            logger.info(
                f"Initialized GRU with input_size={rnn_input_size}, "
                f"hidden_size={rnn_hidden}, batch_first=True"
            )
        elif self.rnn_type == "ATTENTION":
            d_model = rnn_hidden  # reuse rnn_hidden as d_model
            nhead = max(1, d_model // 16)
            self.proj = nn.Linear(rnn_input_size, d_model)
            self.pos_embed = nn.Parameter(torch.zeros(1, W, d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                batch_first=True,
                norm_first=True,  # Pre-norm for more stable training
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
            logger.info(
                f"Initialized TransformerEncoder with d_model={d_model}, nhead={nhead}, "
                f"seq_len={W}, input_size={rnn_input_size}"
            )
        else:
            raise ValueError("rnn_type must be 'LSTM', 'GRU', or 'ATTENTION'")

        # Classifier head - reduced
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(rnn_hidden, 32)
        self.fc2 = nn.Linear(32, 16)
        self.out = nn.Linear(16, num_classes)

    def _forward_conv_layers(self, x):
        """
        Pass input through convolutional layers only.

        :param torch.Tensor x: Input tensor of shape (B, C, H, W).
        :return: Output tensor after convolution and pooling layers.
        :rtype: torch.Tensor
        """
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2d_2(x)
        return x

    def count_parameters(self) -> tuple[int, int]:
        """
        Count the number of parameters in the model.

        :return: Tuple of (all_params, trainable_params).
        :rtype: tuple[int, int]
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        all = sum(p.numel() for p in self.parameters())
        return all, trainable

    def forward(self, x):
        """
        Forward pass through the network.

        :param torch.Tensor x: Input tensor of shape (B, C, H, W) where B=batch size,
                               C=channels (typically 1 for mono spectrograms),
                               H=height (frequency bins), W=width (time frames).
        :return: Class probabilities of shape (B, num_classes) after softmax.
        :rtype: torch.Tensor
        """
        # Pass through convolutional layers
        x = self._forward_conv_layers(x)  # -> (B, 32, H', W')

        # TODO check the reshape
        # Reshape into sequence: treat width (time) as sequence dimension
        # Each time step contains all features from height and channels
        B, C, H, W = x.shape
        seq = x.permute(0, 3, 2, 1).contiguous()  # (B, W, H, C)
        seq = seq.view(B, W, H * C)  # (B, seq_len=W, features=H*C)

        # Temporal aggregation: LSTM/GRU takes last hidden, attention uses mean pooling
        if self.rnn_type == "ATTENTION":
            seq = self.proj(seq) + self.pos_embed
            attn_out = self.transformer(seq)  # (B, seq_len, d_model)
            last_hidden = attn_out.mean(dim=1)  # (B, d_model)
        else:
            rnn_out, _ = self.rnn(seq)  # (B, seq_len, hidden_size)
            last_hidden = rnn_out[:, -1, :]  # (B, hidden_size)

        # Pass through classifier layers
        x = self.dropout(last_hidden)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return logits


class SmallerAll(Smaller):
    Pool1 = nn.AdaptiveMaxPool3d((1, None, None))
    Dropout1 = nn.Dropout3d

    @classmethod
    def Conv1(cls, in_channels: int) -> nn.Module:
        """
        Create the first convolutional layer (3D version).

        Takes 1 input channel with depth=in_channels (8 EEG channels).
        Kernel spans all channels in depth so it processes all EEG channels
        simultaneously in a single convolution operation.

        :param int in_channels: Number of EEG channels (becomes depth dimension).
        :return: Configured Conv3d layer for 3D spectrograms.
        :rtype: nn.Module
        """
        return nn.Conv3d(
            1,  # Input channels = 1 (depth dimension holds the 8 EEG channels)
            32,  # Output channels
            kernel_size=(in_channels, 10, 10),  # Depth=in_channels to span all EEG channels
            stride=(1, 2, 2),  # No stride in depth, stride 2 in H and W
            padding=0,
        )

    def forward(self, x):
        """
        Forward pass through the network with 3D CNN.

        :param torch.Tensor x: Input tensor of shape (B, C, H, W) where B=batch size,
                               C=channels (8 for multi-channel EEG),
                               H=height (frequency bins), W=width (time frames).
        :return: Class probabilities of shape (B, num_classes) after softmax.
        :rtype: torch.Tensor
        """
        # Reshape for 3D conv: (B, 8, H, W) → (B, 1, 8, H, W)
        # This makes the 8 channels become the depth dimension
        x = x.unsqueeze(1)  # (B, 1, 8, H, W)

        # Call parent's forward method which expects 4D output from conv layers
        return super().forward(x)

    def _forward_conv_layers(self, x):
        """
        Pass input through convolutional layers (3D→2D transition).

        :param torch.Tensor x: Input tensor of shape (B, C, D, H, W) for 3D input.
        :return: Output tensor after convolution and pooling (2D).
        :rtype: torch.Tensor
        """
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        x = x.squeeze(2)  # Remove depth dimension: (B, C, 1, H, W) → (B, C, H, W)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2d_2(x)
        return x


class SmallerAttn(Smaller):
    """Smaller model with self-attention replacing LSTM.

    Uses a TransformerEncoder with learned positional embeddings over the
    CNN-produced time frames. d_model is controlled via rnn_hidden (default 128).
    """

    def __init__(
        self,
        input_shape: tuple[int, int],
        in_channels: int = 1,
        rnn_type: str = "attention",
        rnn_hidden: int = 128,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        """
        Initialize Smaller with self-attention temporal aggregation.

        :param tuple input_shape: Tuple of (height, width) for the input spectrogram.
        :param int in_channels: Number of input channels (default=1).
        :param str rnn_type: Ignored; always forces "attention".
        :param int rnn_hidden: d_model for the attention layer (default=128).
        :param float dropout: Dropout probability.
        :param int num_classes: Number of output classes.
        """
        super().__init__(
            input_shape,
            in_channels,
            rnn_type="attention",
            rnn_hidden=rnn_hidden,
            dropout=dropout,
            num_classes=num_classes,
        )


class SmallerAllAttn(SmallerAll):
    """SmallerAll model (8-channel Conv3d backbone) with self-attention replacing LSTM.

    Identical to SmallerAttn but uses the 3D CNN backbone from SmallerAll
    for multi-channel EEG input.
    """

    def __init__(
        self,
        input_shape: tuple[int, int],
        in_channels: int = 8,
        rnn_type: str = "attention",
        rnn_hidden: int = 128,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        """
        Initialize SmallerAll with self-attention temporal aggregation.

        :param tuple input_shape: Tuple of (height, width) for the input spectrogram.
        :param int in_channels: Number of EEG channels (default=8).
        :param str rnn_type: Ignored; always forces "attention".
        :param int rnn_hidden: d_model for the attention layer (default=128).
        :param float dropout: Dropout probability.
        :param int num_classes: Number of output classes.
        """
        super().__init__(
            input_shape,
            in_channels,
            rnn_type="attention",
            rnn_hidden=rnn_hidden,
            dropout=dropout,
            num_classes=num_classes,
        )


class SmallerAllV2(Smaller):
    """
    Multi-channel EEG model: per-channel weight-shared CNN + cross-channel
    self-attention + temporal LSTM.

    Processes each EEG channel independently through the Smaller CNN (shared
    weights), then uses self-attention across the 8 channel tokens to produce
    soft channel-importance weights before merging into the LSTM path.

    :param tuple input_shape: Spectrogram shape (H, W), e.g. (129, 41).
    :param int in_channels: Number of EEG channels (default 8).
    :param str rnn_type: "LSTM", "GRU", or "ATTENTION".
    :param int rnn_hidden: Hidden size for LSTM/GRU, or d_model for attention.
    :param float dropout: Dropout probability.
    :param int num_classes: Number of output classes.
    :param int chan_d_model: Token dimension for cross-channel attention (default 64).
    """

    def __init__(
        self,
        input_shape: tuple[int, int],
        in_channels: int = 8,
        rnn_type: str = "LSTM",
        rnn_hidden: int = 64,
        dropout: float = 0.3,
        num_classes: int = 2,
        chan_d_model: int = 64,
    ) -> None:
        # Build parent with in_channels=1: conv1 is Conv2d(1→32), correct for
        # per-channel processing. Parent's dummy pass computes rnn_input_size from
        # a single-channel input — matches our channel-merged output (B, 16, H', W').
        super().__init__(
            input_shape,
            in_channels=1,
            rnn_type=rnn_type,
            rnn_hidden=rnn_hidden,
            dropout=dropout,
            num_classes=num_classes,
        )
        self._n_eeg_channels = in_channels

        # Second dummy pass to get CNN output dims for channel attention modules
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            cnn_out = self._forward_conv_layers(dummy)
            _, C_feat, H_prime, _ = cnn_out.shape

        # Cross-channel attention
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

        logger.info(
            f"SmallerAllV2: n_eeg_channels={in_channels}, C_feat={C_feat}, "
            f"H_prime={H_prime}, chan_d_model={chan_d_model}, nhead={nhead}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: per-channel CNN → cross-channel attention → LSTM → classifier.

        :param torch.Tensor x: Input of shape (B, C_eeg, H, W).
        :return: Logits of shape (B, num_classes).
        :rtype: torch.Tensor
        """
        B, n_ch, H, W = x.shape

        # 1. Per-channel CNN (shared weights)
        x_flat = x.reshape(B * n_ch, 1, H, W)
        cnn_out = self._forward_conv_layers(x_flat)          # (B*8, 16, H', W')
        _, C_feat, H_prime, W_prime = cnn_out.shape
        x_ch = cnn_out.reshape(B, n_ch, C_feat, H_prime, W_prime)

        # 2. Cross-channel self-attention
        tokens = x_ch.mean(dim=4)                            # pool W': (B, 8, 16, H')
        tokens = tokens.reshape(B, n_ch, C_feat * H_prime)   # (B, 8, 864)
        tokens = self.chan_proj(tokens)                       # (B, 8, chan_d_model)
        tokens = self.chan_transformer(tokens)                # (B, 8, chan_d_model)

        # Soft channel weights, sum-to-1 over the 8 channels
        weights = torch.softmax(self.chan_gate(tokens), dim=1)          # (B, 8, 1)
        weights = weights.unsqueeze(-1).unsqueeze(-1)                   # (B, 8, 1, 1, 1)
        merged = (x_ch * weights).sum(dim=1)                            # (B, 16, H', W')

        # 3. LSTM / Attention + Classifier (same as Smaller.forward() tail)
        seq = merged.permute(0, 3, 2, 1).contiguous()       # (B, W', H', C_feat)
        seq = seq.view(B, W_prime, H_prime * C_feat)         # (B, W', rnn_input_size)

        if self.rnn_type == "ATTENTION":
            seq = self.proj(seq) + self.pos_embed
            attn_out = self.transformer(seq)
            last_hidden = attn_out.mean(dim=1)
        else:
            rnn_out, _ = self.rnn(seq)
            last_hidden = rnn_out[:, -1, :]

        x = self.dropout(last_hidden)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class SmallerAllV2Attn(SmallerAllV2):
    """
    SmallerAllV2 with temporal self-attention replacing LSTM.

    Cross-channel attention is unchanged. After channel merging, temporal
    aggregation uses a TransformerEncoder instead of LSTM/GRU.
    d_model is controlled via rnn_hidden (default 128).
    """

    def __init__(
        self,
        input_shape: tuple[int, int],
        in_channels: int = 8,
        rnn_type: str = "attention",
        rnn_hidden: int = 128,
        dropout: float = 0.3,
        num_classes: int = 2,
        chan_d_model: int = 64,
    ) -> None:
        """
        Initialize SmallerAllV2 with temporal self-attention.

        :param tuple input_shape: Spectrogram shape (H, W).
        :param int in_channels: Number of EEG channels (default 8).
        :param str rnn_type: Ignored; always forces "attention".
        :param int rnn_hidden: d_model for temporal attention (default 128).
        :param float dropout: Dropout probability.
        :param int num_classes: Number of output classes.
        :param int chan_d_model: Token dimension for cross-channel attention (default 64).
        """
        super().__init__(
            input_shape,
            in_channels=in_channels,
            rnn_type="attention",
            rnn_hidden=rnn_hidden,
            dropout=dropout,
            num_classes=num_classes,
            chan_d_model=chan_d_model,
        )


class SmallerAllV3(Smaller):
    """
    Multi-channel EEG model: per-channel weight-shared CNN + full-channel concat
    projection + temporal LSTM.

    Previous V3 used a soft-gate weighted sum to merge channels, which collapsed
    8 channels into 1 and effectively reduced to single-channel performance.  This
    version concatenates all channel feature vectors at each time step so the LSTM
    always sees all 8 channels simultaneously.

    Architecture::

        (B, n_ch, H, W)
        → Per-channel CNN (shared weights): (B*n_ch, 1, H, W) → (B*n_ch, 16, H', W')
        → Reshape + permute: (B, W', n_ch, 16, H')
        → Concat channels: (B, W', n_ch * 16 * H')
        → Linear projection: (B, W', chan_d_model)   ← all channels visible to LSTM
        → LSTM: last hidden (B, rnn_hidden)
        → Classifier (→64→32→num_classes)

    :param tuple input_shape: Spectrogram shape (H, W).
    :param int in_channels: EEG channels (default 8).
    :param str rnn_type: "LSTM", "GRU", or "ATTENTION".
    :param int rnn_hidden: Hidden size for LSTM/GRU or d_model for attention (default 100).
    :param float dropout: Dropout probability.
    :param int num_classes: Output classes.
    :param int chan_d_model: Projection dimension after channel concat (default 128).
    """

    def __init__(
        self,
        input_shape: tuple[int, int],
        in_channels: int = 8,
        rnn_type: str = "LSTM",
        rnn_hidden: int = 100,
        dropout: float = 0.3,
        num_classes: int = 2,
        chan_d_model: int = 128,
    ) -> None:
        """
        Initialize SmallerAllV3.

        :param tuple input_shape: Spectrogram shape (H, W).
        :param int in_channels: Number of EEG channels (default 8).
        :param str rnn_type: "LSTM", "GRU", or "ATTENTION".
        :param int rnn_hidden: Hidden size for LSTM/GRU, or d_model for attention (default 100).
        :param float dropout: Dropout probability.
        :param int num_classes: Number of output classes.
        :param int chan_d_model: Projection dimension after channel concat (default 128).
        """
        super().__init__(
            input_shape,
            in_channels=1,  # CNN sees 1 channel at a time (shared weights)
            rnn_type=rnn_type,
            rnn_hidden=rnn_hidden,
            dropout=dropout,
            num_classes=num_classes,
        )
        # Override Smaller's classifier head
        self.fc1 = nn.Linear(rnn_hidden, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, num_classes)

        self._n_eeg_channels = in_channels

        # Compute CNN output dims from a dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            cnn_out = self._forward_conv_layers(dummy)
            _, C_feat, H_prime, W_prime = cnn_out.shape

        self._C_feat = C_feat
        self._H_prime = H_prime
        self._W_prime = W_prime

        # Project concatenated channel features down to a manageable size.
        # Input: n_ch * C_feat * H_prime (e.g. 8*16*54 = 6912), output: chan_d_model.
        concat_dim = in_channels * C_feat * H_prime
        self.chan_proj = nn.Linear(concat_dim, chan_d_model)

        # Override parent's RNN to accept chan_d_model as input size
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

        logger.info(
            f"SmallerAllV3: n_eeg_channels={in_channels}, C_feat={C_feat}, "
            f"H_prime={H_prime}, W_prime={W_prime}, concat_dim={concat_dim}, "
            f"chan_d_model={chan_d_model}, rnn_hidden={rnn_hidden}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: per-channel CNN → concat all channels → project → LSTM.

        :param torch.Tensor x: Input tensor of shape (B, n_ch, H, W).
        :return: Class logits of shape (B, num_classes).
        :rtype: torch.Tensor
        """
        B, n_ch, H, W = x.shape

        # 1. Per-channel CNN (weight-shared across channels)
        x_flat = x.reshape(B * n_ch, 1, H, W)
        cnn_out = self._forward_conv_layers(x_flat)  # (B*n_ch, C_feat, H', W')
        _, C_feat, H_prime, W_prime = cnn_out.shape

        # 2. Concat all channels at each time step
        # (B*n_ch, C_feat, H', W') → (B, n_ch, C_feat, H', W')
        # → permute to (B, W', n_ch, C_feat, H') → (B, W', n_ch * C_feat * H')
        x_ch = cnn_out.reshape(B, n_ch, C_feat, H_prime, W_prime)
        x_t = x_ch.permute(0, 4, 1, 2, 3).contiguous()  # (B, W', n_ch, C_feat, H')
        seq = x_t.reshape(B, W_prime, n_ch * C_feat * H_prime)

        # Project to chan_d_model so all 8 channels feed into LSTM together
        seq = F.relu(self.chan_proj(seq))  # (B, W', chan_d_model)

        # 3. Temporal aggregation
        if self.rnn_type == "ATTENTION":
            seq = self.proj(seq) + self.pos_embed
            attn_out = self.transformer(seq)
            last_hidden = attn_out.mean(dim=1)
        else:
            rnn_out, _ = self.rnn(seq)
            last_hidden = rnn_out[:, -1, :]  # (B, rnn_hidden)

        # 4. Classifier
        x = self.dropout(last_hidden)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class AllTransformerV4(Smaller):
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
        → proj (shared Linear): (B, n_ch, W', d_model)
        → + chan_embedding[c] + time_embedding[t]
        → flatten: (B, n_ch*W', d_model)
        → TransformerEncoder(num_layers)
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
        """
        Initialize AllTransformerV4.

        :param tuple input_shape: Spectrogram shape (H, W).
        :param int in_channels: Number of EEG channels (default 8).
        :param int num_classes: Number of output classes (default 2).
        :param float dropout: Dropout probability (default 0.3).
        :param int rnn_hidden: Transformer d_model (default 64).
        :param int num_layers: Number of TransformerEncoder layers (default 2).
        """
        # Inherit CNN backbone + proj + classifier from Smaller's ATTENTION branch.
        # in_channels=1: CNN processes one EEG channel at a time (shared weights).
        # self.proj = Linear(C_feat*H', rnn_hidden) is reused as the token projection.
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

        # Replace 1D temporal pos_embed with 2D spatial + temporal embeddings.
        del self.pos_embed
        self.chan_embedding = nn.Embedding(in_channels, d_model)
        self.time_embedding = nn.Embedding(W_prime, d_model)

        # Replace 1-layer transformer (from Smaller) with num_layers version.
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
            f"AllTransformerV4: n_eeg_channels={in_channels}, C_feat={C_feat}, "
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
        cnn_out = self._forward_conv_layers(x_flat)  # (B*n_ch, C_feat, H', W')
        _, C_feat, H_prime, W_prime = cnn_out.shape

        # 2. Reshape to per-(channel, time) tokens.
        tokens = cnn_out.reshape(B, n_ch, C_feat, H_prime, W_prime)
        tokens = tokens.permute(0, 1, 4, 2, 3).contiguous()  # (B, n_ch, W', C_feat, H')
        tokens = tokens.reshape(B, n_ch, W_prime, C_feat * H_prime)  # (B, n_ch, W', 864)

        # 3. Project to d_model (self.proj from Smaller's ATTENTION branch).
        tokens = self.proj(tokens)  # (B, n_ch, W', d_model)

        # 4. Add 2D positional embeddings.
        chan_idx = torch.arange(n_ch, device=x.device)
        time_idx = torch.arange(W_prime, device=x.device)
        tokens = tokens + self.chan_embedding(chan_idx)[None, :, None, :]  # (1,n_ch,1,d)
        tokens = tokens + self.time_embedding(time_idx)[None, None, :, :]  # (1,1,W',d)

        # 5. Flatten to token sequence + dropout.
        tokens = tokens.reshape(B, n_ch * W_prime, self._d_model)  # (B, 80, d_model)
        tokens = self.dropout(tokens)

        # 6. Transformer encoder.
        tokens = self.transformer(tokens)  # (B, 80, d_model)

        # 7. Mean pool + classify.
        pooled = tokens.mean(dim=1)  # (B, d_model)
        out = F.relu(self.fc1(pooled))
        out = F.relu(self.fc2(out))
        return self.out(out)



# Models that consume raw EEG time-series (batch, channels, time) instead of spectrograms.
# These bypass the STFT pipeline and use FlattenedRawEEGDataset.
RAW_EEG_MODELS: frozenset[str] = frozenset({"Deformer", "DeformerS"})

# Model registry: maps model names to (model_class, default_rnn_hidden).
# rnn_hidden is None for raw-EEG models that don't use an RNN.
MODEL_REGISTRY: dict[str, tuple[type[nn.Module], int | None]] = {
    "CNN_LSTM_DepCap": (CNN_LSTM_DepCap, 100),
    "Smaller": (Smaller, 64),
    "SmallerAll": (SmallerAll, 64),
    "SmallerAttn": (SmallerAttn, 128),
    "SmallerAllAttn": (SmallerAllAttn, 128),
    "SmallerAllV2": (SmallerAllV2, 64),
    "SmallerAllV2Attn": (SmallerAllV2Attn, 128),
    "SmallerAllV3": (SmallerAllV3, 100),
    "AllTransformerV4": (AllTransformerV4, 64),
    "Deformer": (Deformer, None),
    "DeformerS": (Deformer, None),
}
