import logging
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

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


# Model registry: maps model names to (model_class, default_rnn_hidden)
MODEL_REGISTRY: dict[str, tuple[type[nn.Module], int]] = {
    "CNN_LSTM_DepCap": (CNN_LSTM_DepCap, 100),
    "Smaller": (Smaller, 64),
    "SmallerAll": (SmallerAll, 64),
}
