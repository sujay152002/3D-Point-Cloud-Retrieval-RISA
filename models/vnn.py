"""Vector Neuron Network (VNN) encoder for SO(3)-invariant point cloud encoding.

Implements the VN-DGCNN variant from:
    Deng et al., "Vector Neurons: A General Framework for SO(3)-Equivariant Networks", ICCV 2021.

Each feature is a 3D vector; equivariance is maintained by replacing scalar
nonlinearities with VN-LeakyReLU (projects onto a learned direction).
Global invariance is obtained via VN-StdFeature (Gram-matrix pooling).

Output: (B, 256) SO(3)-invariant global descriptor.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── VN primitives ─────────────────────────────────────────────────────────────

class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.map = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        # x: (B, C_in, 3, N)  →  (B, C_out, 3, N)
        return self.map(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, negative_slope=0.2):
        super().__init__()
        self.negative_slope = negative_slope
        self.bias = nn.Parameter(torch.empty(1, in_channels, 3, 1))
        nn.init.uniform_(self.bias, -1 / in_channels ** 0.5, 1 / in_channels ** 0.5)

    def forward(self, x):
        # x: (B, C, 3, N)
        d = x + self.bias
        norm = d.norm(dim=2, keepdim=True).clamp(min=1e-8)
        d_hat = d / norm
        dot = (x * d_hat).sum(dim=2, keepdim=True)
        pos = F.relu(dot) * d_hat
        neg = self.negative_slope * (x - dot * d_hat)
        return pos + neg


class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, negative_slope=0.2):
        super().__init__()
        self.linear = VNLinear(in_channels, out_channels)
        self.bn     = nn.BatchNorm1d(out_channels * 3)
        self.act    = VNLeakyReLU(out_channels, negative_slope)

    def forward(self, x):
        # x: (B, C_in, 3, N)
        x = self.linear(x)
        B, C, _, N = x.shape
        x = self.bn(x.reshape(B, C * 3, N)).reshape(B, C, 3, N)
        return self.act(x)


class VNStdFeature(nn.Module):
    """Invariant pooling via Gram-matrix diagonalisation."""
    def __init__(self, in_channels):
        super().__init__()
        self.vn1 = VNLinearLeakyReLU(in_channels, in_channels // 2)
        self.vn2 = VNLinearLeakyReLU(in_channels // 2, 3)

    def forward(self, x):
        # x: (B, C, 3, N)
        z = self.vn1(x)
        z = self.vn2(z)                          # (B, 3, 3, N)
        z = z.mean(dim=-1)                       # (B, 3, 3)
        # Gram-Schmidt to get an orthonormal frame
        u0 = F.normalize(z[:, 0], dim=-1)
        u1 = z[:, 1] - (z[:, 1] * u0).sum(-1, keepdim=True) * u0
        u1 = F.normalize(u1, dim=-1)
        u2 = torch.cross(u0, u1, dim=-1)
        frame = torch.stack([u0, u1, u2], dim=1)  # (B, 3, 3)
        # Project x into the canonical frame → invariant
        # x: (B, C, 3, N)  frame: (B, 3, 3)
        x_inv = torch.einsum("bcvn,bwv->bcwn", x, frame)
        return x_inv


# ── Graph helpers ─────────────────────────────────────────────────────────────

def _knn(x, k):
    # x: (B, C, 3, N) — use first channel for distance
    pts = x[:, 0]                                # (B, 3, N)
    inner = -2 * torch.bmm(pts.transpose(1, 2), pts)
    sq    = (pts ** 2).sum(1, keepdim=True)
    dist  = sq + inner + sq.transpose(1, 2)
    return dist.topk(k, dim=-1, largest=False).indices  # (B, N, k)


def _get_graph_feature_vn(x, k):
    # x: (B, C, 3, N)
    B, C, _, N = x.shape
    idx = _knn(x, k)                             # (B, N, k)
    base = torch.arange(B, device=x.device).view(B, 1, 1) * N
    idx_flat = (idx + base).view(-1)
    x_t = x.permute(0, 3, 1, 2).reshape(B * N, C, 3)  # (B*N, C, 3)
    nbr = x_t[idx_flat].view(B, N, k, C, 3).permute(0, 3, 4, 1, 2)  # (B,C,3,N,k)
    ctr = x.unsqueeze(-1).expand_as(nbr)
    edge = torch.cat([nbr - ctr, ctr], dim=1)   # (B, 2C, 3, N, k)
    return edge.max(dim=-1).values               # (B, 2C, 3, N)


# ── Encoder ───────────────────────────────────────────────────────────────────

class VNNEncoder(nn.Module):
    """VN-DGCNN encoder. Returns a (B, 256) SO(3)-invariant embedding."""

    name = "VNN"

    def __init__(self, k=20, out_dim=512):
        super().__init__()
        self.k = k

        self.conv1 = VNLinearLeakyReLU(2,   64)
        self.conv2 = VNLinearLeakyReLU(128, 64)
        self.conv3 = VNLinearLeakyReLU(128, 128)
        self.conv4 = VNLinearLeakyReLU(256, 256)

        self.std = VNStdFeature(512)

        # After std, features are (B, 512, 3, N) → flatten last two dims
        self.fc = nn.Sequential(
            nn.Linear(512 * 3, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        # x: (B, 3, N)
        B, _, N = x.shape
        # Lift to VN format: (B, 1, 3, N)
        x = x.unsqueeze(1)

        x1 = self.conv1(_get_graph_feature_vn(x,  self.k))   # (B, 64, 3, N)
        x2 = self.conv2(_get_graph_feature_vn(x1, self.k))   # (B, 64, 3, N)
        x3 = self.conv3(_get_graph_feature_vn(x2, self.k))   # (B, 128, 3, N)
        x4 = self.conv4(_get_graph_feature_vn(x3, self.k))   # (B, 256, 3, N)

        xc = torch.cat([x1, x2, x3, x4], dim=1)              # (B, 512, 3, N)
        xi = self.std(xc)                                     # (B, 512, 3, N)

        g  = xi.amax(dim=-1)                                  # (B, 512, 3)
        g  = g.reshape(B, -1)                                 # (B, 512*3)
        return self.fc(g)                                     # (B, out_dim)
