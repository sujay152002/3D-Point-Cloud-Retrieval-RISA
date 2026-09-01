import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import networkx as nx
from einops import rearrange, repeat
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import vstack, coo_matrix
import torch.fft

EPS = 1e-6


class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x_out = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        return x_out


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super(VNLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1 - self.negative_slope) * (
                    mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d))
        return x_out


class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, negative_slope=0.2):
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.negative_slope = negative_slope

        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.batchnorm = VNBatchNorm(out_channels, dim=dim)

        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        p = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)

        p = self.batchnorm(p)

        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)

        dotprod = (p * d).sum(dim=2, keepdims=True)

        mask = (dotprod >= 0).float()

        d_norm_sq = (d * d).sum(dim=2, keepdims=True)

        normalized_dotprod = dotprod / (d_norm_sq + EPS)

        result = mask * p + (1 - mask) * (p - normalized_dotprod * d)

        x_out = self.negative_slope * p + (1 - self.negative_slope) * result
        return x_out


class VNLinearAndLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2):
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        self.linear = VNLinear(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)

        # BatchNorm
        self.use_batchnorm = use_batchnorm
        if use_batchnorm != 'none':
            self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Conv
        x = self.linear(x)
        # InstanceNorm
        if self.use_batchnorm != 'none':
            x = self.batchnorm(x)
        # LeakyReLU
        x_out = self.leaky_relu(x)
        return x_out


class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super(VNBatchNorm, self).__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # norm = torch.sqrt((x*x).sum(2))
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn

        return x

class VNMaxPool(nn.Module):
    def __init__(self, in_channels):
        super(VNMaxPool, self).__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(dim=2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        output = x.gather(dim=4, index=idx.unsqueeze(-1).expand(-1, -1, 3, -1, 1))
        output = output.squeeze(-1)
        return output

class VNStdFeature(nn.Module):
    def __init__(self, in_channels, dim=4, normalize_frame=False, share_nonlinearity=False, negative_slope=0.2):
        super(VNStdFeature, self).__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame

        self.vn1 = VNLinearLeakyReLU(in_channels, in_channels // 2, dim=dim, share_nonlinearity=share_nonlinearity,
                                     negative_slope=negative_slope)
        self.vn2 = VNLinearLeakyReLU(in_channels // 2, in_channels // 4, dim=dim, share_nonlinearity=share_nonlinearity,
                                     negative_slope=negative_slope)
        if normalize_frame:
            self.vn_lin = nn.Linear(in_channels // 4, 2, bias=False)
        else:
            self.vn_lin = nn.Linear(in_channels // 4, 3, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        z0 = x
        z0 = self.vn1(z0)
        z0 = self.vn2(z0)
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)

        if self.normalize_frame:
            # make z0 orthogonal. u2 = v2 - proj_u1(v2)
            v1 = z0[:, 0, :]
            # u1 = F.normalize(v1, dim=1)
            v1_norm = torch.sqrt((v1 * v1).sum(1, keepdims=True))
            u1 = v1 / (v1_norm + EPS)
            v2 = z0[:, 1, :]
            v2 = v2 - (v2 * u1).sum(1, keepdims=True) * u1
            # u2 = F.normalize(u2, dim=1)
            v2_norm = torch.sqrt((v2 * v2).sum(1, keepdims=True))
            u2 = v2 / (v2_norm + EPS)

            # compute the cross product of the two output vectors
            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)

        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)

        return x_std, z0

def knn(x, k):
    # 计算点之间的距离, 形状为 [b, n, n]
    dist = torch.cdist(x.transpose(1, 2), x.transpose(1, 2))  # 计算点对点距离 [b, n, n]

    # 使用 torch.topk 获取每个点的最近邻索引
    _, idx = torch.topk(dist, k=k, largest=False)  # [b, n, k]

    return idx  # 返回每个点的 k 个最近邻索引

def get_graph_feature(x, k=20, idx=None, x_coord=None,return_cloud=False):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = rearrange(x,'b c v n -> b (c v) n')
    if idx is None:
        if x_coord is None:
            idx = knn(x, k=k)
        else:
            idx = knn(x_coord, k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)

    feature = feature.permute(0, 3, 4, 1, 2).contiguous()
    x = x.permute(0, 3, 4, 1, 2).contiguous()

    x = torch.cat((feature - x, x),dim=1)

    return x

def get_graph_feature_cross(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    cross = torch.cross(feature, x, dim=-1)

    feature = torch.cat((feature - x, x, cross), dim=3).permute(0, 3, 4, 1, 2).contiguous()

    return feature

def nd_get_graph_feature(x, k=20, idx=None, x_coord=None):
    batch_size = x.size(0)
    D = x.size(2)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is None:  # dynamic knn graph
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            x_coord = x_coord.view(batch_size, -1, num_points)
            idx = knn(x_coord, k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // D

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, D)
    x = x.view(batch_size, num_points, 1, num_dims, D).repeat(1, 1, k, 1, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()

    return feature


class DASFTModule(nn.Module):
    """
    Direction-Aware Spherical Fourier Transform (DASFT) module.
    Corresponds to Section 3.4 and the global branch in Figure 2.
    Computes the global directional response spectrum $f_{DASFT}(\mathcal{P})$.

    Args:
        in_channels (int): Input feature channels C.
        spec_dim (int): Spectrum dimension M.
        f_min (float): Minimum frequency.
        f_max (float): Maximum frequency.
        N_dir (int): Number of spherical sampling directions $N_{dir}$.
        mode (str): Frequency sampling mode ('linear' or 'log').
        chunk_size (int): Chunk size for memory-efficient computation.
    """
    def __init__(self, in_channels, spec_dim, f_min=0.0, f_max=12.0, N_dir=36, mode='linear', chunk_size=6):
        super(DASFTModule, self).__init__()
        self.spec_dim = spec_dim
        self.N_dir = N_dir
        self.chunk_size = chunk_size

        self.norm = nn.LayerNorm(in_channels)

        # 1. Init frequency bins (r in Eq. 19)
        if mode == 'linear':
            freq_bins = torch.linspace(f_min, f_max, spec_dim)
        elif mode == 'log':
            freq_bins = torch.exp(torch.linspace(math.log(f_min + 1e-6), math.log(f_max + 1e-6), spec_dim))
        else:
            raise ValueError("DASFT mode must be 'linear' or 'log'")
        self.register_buffer('freq_bins', freq_bins)

        # 2. Init spherical directions (omega in Eq. 17) using Fibonacci sampling
        directions = self.sample_directions(N_dir)
        self.register_buffer('directions', directions)

    @staticmethod
    def sample_directions(N_dir, dtype=torch.float32):
        """Sample N_dir directions uniformly on a sphere using Fibonacci spiral."""
        indices = torch.arange(0, N_dir, dtype=dtype) + 0.5
        phi = torch.acos(1 - 2 * indices / N_dir)
        theta = math.pi * (1 + 5 ** 0.5) * indices

        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)
        return torch.stack([x, y, z], dim=1) # [N_dir, 3]

    def forward(self, x):
        """
        Args:
            x: (B, C, 3, N) Input point cloud features (P in Figure 2).
        Returns:
            G: (B, M, C) Global directional energy spectrum.
        """
        b, c, _, n = x.shape
        m = self.spec_dim
        N_dir = self.N_dir

        # Init output spectrum G: (B, C, M)
        G = torch.zeros((b, c, m), device=x.device, dtype=x.dtype)

        num_chunks = (N_dir + self.chunk_size - 1) // self.chunk_size

        # Reshape for batch matmul: (B, C, 3, N) -> (B*C, N, 3)
        x_reshaped = x.reshape(b * c, 3, n).permute(0, 2, 1).contiguous()

        for i in range(num_chunks):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, N_dir)
            # Chunk of directions: (chunk, 3) -> (3, chunk)
            dirs_chunk = self.directions[start_idx:end_idx].transpose(0, 1)
            
            # Dot product <P, Omega>: (B*C, N, 3) @ (3, chunk) -> (B*C, N, chunk)
            dots = torch.matmul(x_reshaped, dirs_chunk)
            dots = dots.view(b, c, n, -1) # (B, C, N, chunk)

            # Phase term: exp(-i * r * <P, Omega>) (Eq. 18)
            # freq_bins: (M) -> (1, 1, M, 1, 1)
            # dots: (B, C, N, chunk) -> (B, C, 1, N, chunk)
            # phase: (B, C, M, N, chunk)
            phase = self.freq_bins.view(1, 1, m, 1, 1) * dots.unsqueeze(2)
            
            # Energy spectrum E = |sum(exp(ix))|^2 = (sum_cos)^2 + (sum_sin)^2 (Eq. 19)
            # Sum over N points (dim=3)
            E = torch.cos(phase).sum(dim=3).pow(2) + torch.sin(phase).sum(dim=3).pow(2)

            # Accumulate energy over direction chunks
            G += E.sum(dim=-1) 

        # Spherical average (Eq. 21)
        G = G / N_dir

        # (B, C, M) -> (B, M, C) and normalize
        G = G.transpose(1, 2)
        G = self.norm(G)
        
        return G


class L2DPOperator(nn.Module):
    """
    Learnable Local Dot Product (L2DP) operator.
    Corresponds to Section 3.3 and Figure 3.
    Captures local rotation-invariant features via relative position dot products.
    
    Args:
        in_channels (int): Input channels (half of VNN features).
        out_channels (int): Output channels.
        k (int): Number of nearest neighbors K.
        aggregation_mode (str): 'dlp' or 'sap'.
    """
    def __init__(self, in_channels, out_channels, k, aggregation_mode='dlp', dropout=0.1):
        super(L2DPOperator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.aggregation_mode = aggregation_mode

        # FFN_L: Processes concatenated invariants (Eq. 12)
        # Input dim is 2 * in_channels due to concatenation of relative and absolute terms.
        self.ffn_l = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels * 2),
            nn.GELU()
        )

        # Aggregation mapping phi (Eq. 13 & 14)
        if aggregation_mode == 'dlp':
            # Direct Linear Projection (Eq. 13)
            self.agg_linear = nn.Linear(in_channels * 2 * k, out_channels, bias=False)
            self.agg_norm = nn.LayerNorm(out_channels)
        elif aggregation_mode == 'sap':
            # Statistic-Aware Projection (Eq. 14)
            # Concatenates max, avg, var statistics.
            self.agg_linear = nn.Linear(in_channels * 2 * 3, out_channels, bias=False)
            self.agg_dropout = nn.Dropout(dropout)
        else:
            raise ValueError(f"Unknown aggregation mode: {aggregation_mode}")

    def forward(self, x_graph):
        """
        Args:
            x_graph: (B, 2*C_in, 3, N, K) VNN graph features.
                     First half is (x_k - x_j), second half is x_j.
        """
        batch_size, _, _, num_points, _ = x_graph.shape

        # 1. Decompose VNN graph features
        # feature_diff: v_k - v_j (g_jk - v_j in paper)
        # feature_center: v_j (v_j in paper)
        feature_diff, feature_center = torch.chunk(x_graph, 2, dim=1)

        # 2. Compute atomic dot-product invariants (Figure 3)
        # Relative geometric feature (Eq. 11): <v_j, g_jk - v_j>
        dot_rel = (feature_center * feature_diff).sum(dim=2) # (B, C_in, N, K)
        # Positional encoding: <v_j, v_j>
        dot_pos = (feature_center * feature_center).sum(dim=2) # (B, C_in, N, K)

        # 3. Concatenate and pass through FFN_L (Eq. 12)
        invariants = torch.cat([dot_rel, dot_pos], dim=1) # (B, 2*C_in, N, K)
        x = self.ffn_l(invariants) # (B, 2*C_in, N, K)

        # 4. Feature aggregation phi (Eq. 15)
        if self.aggregation_mode == 'dlp':
            # DLP: Flatten K dimension and linearly project
            # (B, 2*C_in, N, K) -> (B, N, 2*C_in*K)
            x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, num_points, -1)
            x = self.agg_linear(x)
            x = self.agg_norm(x) # (B, N, C_out)
        else:
            # SAP: Compute statistics (max, avg, var)
            # All have shape (B, 2*C_in, N)
            x_max = x.max(dim=-1)[0]
            x_avg = x.mean(dim=-1)
            x_var = x.var(dim=-1, unbiased=False)
            
            # Concatenate statistics: (B, 2*C_in*3, N)
            x_sap = torch.cat([x_max, x_avg, x_var], dim=1)
            x = x_sap.transpose(1, 2) # (B, N, 2*C_in*3)
            x = self.agg_linear(x)
            x = self.agg_dropout(x) # (B, N, C_out)

        return x


class CrossAttention(nn.Module):
    """
    Standard Cross-Attention module for fusing local (Q) and global (K, V) features.
    """
    def __init__(self, q_dim, kv_dim, inner_dim, out_dim, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = (inner_dim // num_heads) ** -0.5

        self.to_q = nn.Linear(q_dim, inner_dim, bias=False)
        # K, V are projected from the same global features
        self.to_kv = nn.Linear(kv_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x_q, x_kv):
        """
        Args:
            x_q: (B, N, q_dim) Local features as Query.
            x_kv: (B, M, kv_dim) Global features as Key/Value.
        """
        H = self.num_heads

        # 1. Project and split heads
        q = self.to_q(x_q) # (B, N, inner_dim)
        k, v = self.to_kv(x_kv).chunk(2, dim=-1) # (B, M, inner_dim)

        # (B, N/M, inner_dim) -> (B, H, N/M, head_dim)
        q = rearrange(q, 'b n (h d) -> b h n d', h=H)
        k = rearrange(k, 'b m (h d) -> b h m d', h=H)
        v = rearrange(v, 'b m (h d) -> b h m d', h=H)

        # 2. Compute attention scores
        # (B, H, N, head_dim) @ (B, H, head_dim, M) -> (B, H, N, M)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)

        # 3. Aggregate values
        # (B, H, N, M) @ (B, H, M, head_dim) -> (B, H, N, head_dim)
        out = torch.matmul(attn, v)
        
        # 4. Merge heads and project output
        # (B, H, N, head_dim) -> (B, N, inner_dim)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class DiPVNet(nn.Module):
    """
    DiPVNet Layer (corresponds to Figure 2).
    Integrates L2DP operator and DASFT module via cross-attention.
    """
    def __init__(self, in_channels, out_channels, k, spec_dim=32, N_dir=36, aggregation_mode='dlp', dropout=0.1):
        super(DiPVNet, self).__init__()
        self.in_channels = in_channels
        
        # 1. L2DP operator (Local branch)
        self.l2dp = L2DPOperator(in_channels, out_channels, k, aggregation_mode, dropout)
        
        # 2. DASFT module (Global branch)
        self.dasft = DASFTModule(in_channels, spec_dim, N_dir=N_dir, mode='linear')

        # 3. Cross Attention (Fusion)
        inner_dim = out_channels * 2
        self.cross_attn = CrossAttention(
            q_dim=out_channels,    # Query from L2DP
            kv_dim=in_channels,    # Key/Value from DASFT
            inner_dim=inner_dim,
            out_dim=out_channels,
            dropout=dropout
        )

        # FFN for fused features
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels * 4, out_channels),
            nn.Dropout(dropout)
        )

        self.norm_local = nn.LayerNorm(out_channels)
        self.norm_fused = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x_graph):
        """
        Args:
            x_graph: (B, 2*C_in, 3, N, K) VNN graph features.
        """
        # --- 1. L2DP branch ---
        local_feat = self.l2dp(x_graph) # (B, N, C_out)
        q_feat = self.norm_local(self.relu(local_feat))

        # --- 2. DASFT branch ---
        # Extract center point features: (B, 2*C_in, 3, N, K) -> (B, C_in, 3, N)
        center_point_feats = x_graph[:, self.in_channels:, :, :, 0]
        
        # Compute global spectrum: (B, C_in, 3, N) -> (B, M, C_in)
        # Note: DASFT output is (B, M, C), used as KV here.
        global_spec = self.dasft(center_point_feats)
        kv_feat = global_spec

        # --- 3. Fusion via Cross-Attention ---
        # Q: (B, N, C_out), KV: (B, M, C_in) -> (B, N, C_out)
        fused_feat = self.cross_attn(q_feat, kv_feat)

        # Residual connection + FFN
        fused_feat = fused_feat + local_feat
        fused_feat = fused_feat + self.ffn(self.norm_fused(fused_feat))
        
        # Transpose for downstream tasks: (B, N, C_out) -> (B, C_out, N)
        return fused_feat.transpose(1, 2), local_feat.transpose(1, 2)
