# =============================================================================
# LGGNet: A Neurologically Inspired Graph Neural Network for EEG
# =============================================================================
# Source:  https://github.com/yi-ding-cs/LGG
# Paper:   Yi Ding, Neethu Robinson, Chengxuan Tong, Qiuhao Zeng, Cuntai Guan.
#          "LGGNet: Learning From Local-Global-Graph Representations for Brain-
#          Computer Interface." IEEE Transactions on Neural Networks and
#          Learning Systems, 2023.
#          DOI: 10.1109/TNNLS.2023.3236635
#
# This file is a modified port of networks.py and layers.py from the repository
# above. It is redistributed under the CBCR License 1.0 (reproduced below),
# which governs this file independently of the MIT License that covers the
# rest of this project.
#
# Adaptations for this project:
#   - Aggregator promoted from plain class to nn.Module
#   - Constructor refactored: input_size tuple → separate num_chan / num_time args
#   - dropout_rate parameter renamed to dropout for uniform create_model() dispatch
#   - get_adj fixed: torch.eye created on x.device (removes global DEVICE dependency)
#   - channel_order applied in forward() to reorder channels per graph topology
#   - Full mypy-compatible type annotations added
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# Graph topologies for the 8-channel canonical ordering used in this project:
# Fp1(0), Fp2(1), T7(2), T8(3), C3(4), C4(5), Cz(6), Oz(7)
#
# "frontal"   — anatomical grouping: frontal / temporal / central / occipital
# "hemisphere"— symmetric grouping: left-hemisphere / right-hemisphere / midline
GRAPH_TOPOLOGIES: dict[str, dict[str, list[int]]] = {
    "frontal": {
        # Groups: {Fp1,Fp2} / {T7,T8} / {C3,C4,Cz} / {Oz}
        "channel_order": [0, 1, 2, 3, 4, 5, 6, 7],
        "idx_graph": [2, 2, 3, 1],
    },
    "hemisphere": {
        # Groups: {Fp1,T7,C3} / {Fp2,T8,C4} / {Cz,Oz}
        "channel_order": [0, 2, 4, 1, 3, 5, 6, 7],
        "idx_graph": [3, 3, 2],
    },
}


class PowerLayer(nn.Module):
    """Log-transformed power via average pooling: log(avg_pool(x²))."""

    def __init__(self, dim: int, length: int, step: int) -> None:
        super().__init__()
        self.dim = dim
        self.pooling = nn.AvgPool2d(kernel_size=(1, length), stride=(1, step))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(self.pooling(x.pow(2)).clamp(min=1e-6))


class GraphConvolution(nn.Module):
    """Single spectral GCN layer: H' = ReLU(A · (X·W − b))."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight, gain=1.414)
        if bias:
            self.bias: Parameter | None = Parameter(
                torch.zeros((1, 1, out_features), dtype=torch.float32)
            )
        else:
            self.bias = None

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        output = torch.matmul(x, self.weight) - self.bias
        return F.relu(torch.matmul(adj, output))


class Aggregator(nn.Module):
    """Mean-pools EEG channels into brain-region node embeddings."""

    def __init__(self, idx_area: list[int]) -> None:
        super().__init__()
        self.chan_in_area = idx_area
        self.area = len(idx_area)
        # Precompute cumulative start indices for slicing
        self.idx = self._get_idx(idx_area)

    def _get_idx(self, chan_in_area: list[int]) -> list[int]:
        # Start index of each region: [0, 3, 6] for [3, 3, 2]
        idx_: list[int] = [0]
        for n in chan_in_area[:-1]:
            idx_.append(idx_[-1] + n)
        return idx_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, features)
        data: list[torch.Tensor] = []
        for i in range(self.area):
            region = x.narrow(1, self.idx[i], self.chan_in_area[i])
            data.append(region.mean(dim=1))
        return torch.stack(data, dim=1)  # (batch, num_regions, features)


class LGGNet(nn.Module):
    """
    Local-Global Graph Network for raw EEG classification (TNNLS 2023).

    Accepts raw EEG (batch, num_chan, num_time) and classifies via a two-block
    pipeline: (1) multi-scale temporal CNN with log-power, (2) local-global
    graph convolution over anatomically defined brain regions.
    """

    def __init__(
        self,
        num_classes: int,
        num_chan: int,
        num_time: int,
        sampling_rate: int,
        num_T: int,
        out_graph: int,
        dropout: float,
        pool: int,
        pool_step_rate: float,
        graph_type: str,
    ) -> None:
        super().__init__()

        if graph_type not in GRAPH_TOPOLOGIES:
            raise ValueError(
                f"Unknown graph_type '{graph_type}'. Choose from: {list(GRAPH_TOPOLOGIES.keys())}"
            )

        topo = GRAPH_TOPOLOGIES[graph_type]
        self.channel_order: list[int] = topo["channel_order"]
        self.idx_graph: list[int] = topo["idx_graph"]
        self.brain_area: int = len(self.idx_graph)
        self.pool = pool

        window_sizes = [0.5, 0.25, 0.125]
        self.Tception1 = self._temporal_learner(
            1, num_T, (1, int(window_sizes[0] * sampling_rate)), pool, pool_step_rate
        )
        self.Tception2 = self._temporal_learner(
            1, num_T, (1, int(window_sizes[1] * sampling_rate)), pool, pool_step_rate
        )
        self.Tception3 = self._temporal_learner(
            1, num_T, (1, int(window_sizes[2] * sampling_rate)), pool, pool_step_rate
        )

        self.BN_t = nn.BatchNorm2d(num_T)
        self.OneXOneConv = nn.Sequential(
            nn.Conv2d(num_T, num_T, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.AvgPool2d((1, 2)),
        )
        self.BN_t_ = nn.BatchNorm2d(num_T)

        features = self._get_size_temporal(num_chan, num_time)

        self.local_filter_weight = Parameter(
            torch.FloatTensor(num_chan, features), requires_grad=True
        )
        nn.init.xavier_uniform_(self.local_filter_weight)
        self.local_filter_bias = Parameter(
            torch.zeros((1, num_chan, 1), dtype=torch.float32), requires_grad=True
        )

        self.aggregate = Aggregator(self.idx_graph)

        self.global_adj = Parameter(
            torch.FloatTensor(self.brain_area, self.brain_area), requires_grad=True
        )
        nn.init.xavier_uniform_(self.global_adj)

        self.bn = nn.BatchNorm1d(self.brain_area)
        self.bn_ = nn.BatchNorm1d(self.brain_area)
        self.GCN = GraphConvolution(features, out_graph)

        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.brain_area * out_graph, num_classes),
        )

    def _temporal_learner(
        self,
        in_chan: int,
        out_chan: int,
        kernel: tuple[int, int],
        pool: int,
        pool_step_rate: float,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=(1, 1)),
            PowerLayer(dim=-1, length=pool, step=int(pool_step_rate * pool)),
        )

    def _get_size_temporal(self, num_chan: int, num_time: int) -> int:
        """Infer feature dimension per channel via a CPU dummy forward."""
        dummy = torch.ones((1, 1, num_chan, num_time))
        out = self.Tception1(dummy)
        out = torch.cat((out, self.Tception2(dummy)), dim=-1)
        out = torch.cat((out, self.Tception3(dummy)), dim=-1)
        out = self.BN_t(out)
        out = self.OneXOneConv(out)
        out = self.BN_t_(out)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(out.size(0), out.size(1), -1)
        return int(out.size(-1))

    def _local_filter_fun(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        w = w.unsqueeze(0).repeat(x.size(0), 1, 1)
        return F.relu(torch.mul(x, w) - self.local_filter_bias)

    def _self_similarity(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, nodes, features) → (batch, nodes, nodes)
        return torch.bmm(x, x.permute(0, 2, 1))

    def _get_adj(self, x: torch.Tensor) -> torch.Tensor:
        adj = self._self_similarity(x)
        num_nodes = adj.size(-1)
        adj = F.relu(adj * (self.global_adj + self.global_adj.t()))
        adj = adj + torch.eye(num_nodes, device=x.device)
        rowsum = adj.sum(dim=-1)
        # Avoid division by zero for isolated nodes
        rowsum = rowsum.masked_fill(rowsum == 0, 1.0)
        d_inv_sqrt = rowsum.pow(-0.5)
        d_mat = torch.diag_embed(d_inv_sqrt)
        return torch.bmm(torch.bmm(d_mat, adj), d_mat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_chan, time) — reorder channels per graph topology
        x = x[:, self.channel_order, :].unsqueeze(1)  # (B, 1, num_chan, time)

        out = self.Tception1(x)
        out = torch.cat((out, self.Tception2(x)), dim=-1)
        out = torch.cat((out, self.Tception3(x)), dim=-1)
        out = self.BN_t(out)
        out = self.OneXOneConv(out)
        out = self.BN_t_(out)

        out = out.permute(0, 2, 1, 3)
        out = out.reshape(out.size(0), out.size(1), -1)  # (B, num_chan, features)

        out = self._local_filter_fun(out, self.local_filter_weight)
        out = self.aggregate(out)  # (B, brain_area, features)

        adj = self._get_adj(out)
        out = self.bn(out)
        out = self.GCN(out, adj)
        out = self.bn_(out)

        out = out.view(out.size(0), -1)
        return self.fc(out)

    def count_parameters(self) -> tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
