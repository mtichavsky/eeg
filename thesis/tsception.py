# =============================================================================
# TSception: Capturing Temporal Dynamics and Spatial Asymmetry from EEG
# =============================================================================
# Source:  https://github.com/yi-ding-cs/TSception
# Paper:   Yi Ding, Neethu Robinson, Qiuhao Zeng, Dou Chen, Aung Aung Phyo Wai,
#          Tih-Shih Lee, Cuntai Guan.
#          "TSception: Capturing Temporal Dynamics and Spatial Asymmetry from EEG
#          for Emotion Recognition." IEEE Transactions on Affective Computing, 2022.
#          DOI: 10.1109/TAFFC.2022.3169001
#
# The ``TSception`` class in this file is derived from Models.py in the repository
# above (adapted: mypy annotations, formatting). It is redistributed under the
# CBCR License 1.0 (reproduced below), which governs the ``TSception`` class
# independently of the MIT License that covers the rest of this project.
#
# ``TSceptionWrapper`` is original project code under the project's MIT License.
#
# ----------------------------- CBCR License 1.0 ------------------------------
# Copyright 2022 Centre for Brain Computing Research (CBCR)
#
# Redistribution and use for non-commercial purpose in source and binary
# forms, with or without modification, are permitted provided that the
# following conditions are met:
#   1. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#   2. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   3. Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# =============================================================================
"""
TSception: Multi-Scale Temporal + Asymmetric Spatial CNN for EEG emotion recognition.

Vendored from https://github.com/yi-ding-cs/TSception (CBCR License 1.0).

Adaptation for this project:
- ``TSceptionWrapper`` accepts ``(batch, channels, time)`` matching the project's raw-EEG
  convention, reshapes to ``(batch, 1, channels, time)`` before forwarding to ``TSception``.
- Constructor signature matches the ``num_chan / num_classes / dropout / **config`` pattern
  expected by ``create_model()`` for raw-EEG models.
"""

import torch
import torch.nn as nn


class TSception(nn.Module):
    """
    Multi-scale temporal inception + asymmetric spatial CNN.

    Input shape: ``(batch, 1, num_channels, num_time)``.
    Three temporal branches with kernels at 0.5 / 0.25 / 0.125 × sampling_rate are
    concatenated along the time axis; two spatial branches (global and hemisphere)
    are concatenated along the channel axis; a flatten + FC head produces logits.

    :param int num_classes: Number of output classes.
    :param tuple[int, int] input_size: ``(num_channels, num_time)``.
    :param int sampling_rate: EEG sampling rate in Hz; determines temporal kernel sizes.
    :param int num_T: Number of temporal inception filters per branch.
    :param int num_S: Number of spatial filters per branch.
    :param int hidden: Hidden units in the first FC layer.
    :param float dropout_rate: Dropout applied between FC layers.
    """

    def conv_block(
        self,
        in_chan: int,
        out_chan: int,
        kernel: tuple[int, int],
        step: int | tuple[int, int],
        pool: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_chan,
                out_channels=out_chan,
                kernel_size=kernel,
                stride=step,
                padding=0,
            ),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)),
        )

    def __init__(
        self,
        num_classes: int,
        input_size: tuple[int, int],
        sampling_rate: int,
        num_T: int,
        num_S: int,
        hidden: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 8

        self.Tception1 = self.conv_block(
            1, num_T, (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool
        )
        self.Tception2 = self.conv_block(
            1, num_T, (1, int(self.inception_window[1] * sampling_rate)), 1, self.pool
        )
        self.Tception3 = self.conv_block(
            1, num_T, (1, int(self.inception_window[2] * sampling_rate)), 1, self.pool
        )

        self.Sception1 = self.conv_block(
            num_T, num_S, (int(input_size[-2]), 1), 1, int(self.pool * 0.25)
        )
        self.Sception2 = self.conv_block(
            num_T,
            num_S,
            (int(input_size[-2] * 0.5), 1),
            (int(input_size[-2] * 0.5), 1),
            int(self.pool * 0.25),
        )
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)

        size = self.get_size(input_size)
        self.fc = nn.Sequential(
            nn.Linear(size[1], hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param torch.Tensor x: Input of shape ``(batch, 1, channels, time)``.
        :return: Logits of shape ``(batch, num_classes)``.
        :rtype: torch.Tensor
        """
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_ = z
        z = self.Sception2(out)
        out_ = torch.cat((out_, z), dim=2)
        out = self.BN_s(out_)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out

    def get_size(self, input_size: tuple[int, int]) -> torch.Size:
        """Run a dummy forward pass to determine the flattened FC input size."""
        data = torch.ones((1, 1, input_size[-2], int(input_size[-1])))
        y = self.Tception1(data)
        out = y
        y = self.Tception2(data)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(data)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_final = z
        z = self.Sception2(out)
        out_final = torch.cat((out_final, z), dim=2)
        out = self.BN_s(out_final)
        out = out.view(out.size()[0], -1)
        return out.size()


# ---------------------------------------------------------------------------
# Project-compatible wrapper (MIT License — original code)
# ---------------------------------------------------------------------------


class TSceptionWrapper(nn.Module):
    """
    Adapter that makes ``TSception`` compatible with the project's raw-EEG model interface.

    The project passes ``(batch, channels, time)`` tensors to raw-EEG models.
    This wrapper unsqueezes dim 1 to produce ``(batch, 1, channels, time)`` before
    forwarding to the inner ``TSception`` module.

    The constructor signature matches the ``num_chan / num_classes / dropout / **config``
    pattern used by ``create_model()`` for all raw-EEG models.

    :param int num_chan: Number of EEG channels (e.g. 8 for all-channel mode).
    :param int num_classes: Number of output classes (2 or 4).
    :param float dropout: Dropout rate applied between the two FC layers.
    :param int num_T: Temporal inception filter count (default: 9, paper value).
    :param int num_S: Spatial filter count per branch (default: 6, paper value).
    :param int hidden: Hidden units in the first FC layer.
        ``TSception`` uses 128 (paper default);
        ``TSceptionS`` uses 32 (paper's cross-dataset recommendation).
    :param int sampling_rate: EEG sampling rate in Hz, used to compute temporal kernel
        sizes as ``int(fraction × sampling_rate)``. Defaults to 250 Hz (MDD / IDUN).
    :param int num_time: Number of time samples per chunk. Used to compute FC input size
        via a dummy forward pass. Default: 2500 (10 s × 250 Hz).
    """

    def __init__(
        self,
        num_chan: int,
        num_classes: int,
        dropout: float,
        num_T: int = 9,
        num_S: int = 6,
        hidden: int = 128,
        sampling_rate: int = 250,
        num_time: int = 2500,
    ) -> None:
        super().__init__()
        self.inner = TSception(
            num_classes=num_classes,
            input_size=(num_chan, num_time),
            sampling_rate=sampling_rate,
            num_T=num_T,
            num_S=num_S,
            hidden=hidden,
            dropout_rate=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param torch.Tensor x: Raw EEG of shape ``(batch, channels, time)``.
        :return: Logits of shape ``(batch, num_classes)``.
        :rtype: torch.Tensor
        """
        # (batch, channels, time) → (batch, 1, channels, time)
        return self.inner(x.unsqueeze(1))

    def count_parameters(self) -> tuple[int, int]:
        """
        :return: ``(total_params, trainable_params)`` tuple.
        :rtype: tuple[int, int]
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
