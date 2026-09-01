"""DGCNN (Dynamic Graph CNN) encoder for point cloud encoding.

Builds a dynamic k-NN graph at each layer and applies edge convolutions.
Outputs a 512-dim global feature via max+avg pooling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    # x input: (Batch, Channels, Num_points)
    inner = x.unsqueeze(-1)
    outer = x.unsqueeze(2)
    diff = inner - outer
    dist = (diff ** 2).sum(dim=1)
    idx = dist.topk(k=k, dim=-1, largest=False)[1]
    return idx

def get_graph_feature(x, k=20, idx=None):
    """
    Construct edge features for graph convolution.
    
    Args:
        x: (B, C, N) feature tensor
        k: number of neighbors
        idx: (B, N, k) precomputed neighbor indices (optional)
    
    Returns:
        (B, 2C, N, k) edge features [f_i - f_j, f_i]
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    device = x.device
    
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        idx = knn(x, k=k)  # (B, N, k)
    
    # Create batch indices for gathering
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    
    _, num_dims, _ = x.size()
    
    # Transpose for indexing: (B, C, N) -> (B, N, C)
    x = x.transpose(2, 1).contiguous()
    
    # Gather neighbor features: (B*N*k, C)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    
    # Repeat center features: (B, N, 1, C) -> (B, N, k, C)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    # Edge features: [neighbor - center, center]
    # (B, N, k, C*2) -> (B, C*2, N, k)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    
    return feature

class DGCNNEncoder(nn.Module):
    def __init__(self, k=20, emb_dims=1024):
        super(DGCNNEncoder, self).__init__()
        self.k = k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1, nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2, nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3, nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4, nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                   self.bn5, nn.LeakyReLU(0.2))
        
        # Project 1024D (max+avg) down to 512D to match PointNet/Transformer suite
        self.linear_out = nn.Linear(emb_dims*2, 512)

    def forward(self, x):
        batch_size = x.size(0)
        
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        
        # Global Aggregation
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        
        return self.linear_out(x)