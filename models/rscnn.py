"""RS-CNN (Relation-Shape CNN) encoder for point cloud encoding.

Multi-scale message passing with relation-shape convolutions.
Outputs a 512-dim global feature.
"""
"""RS-CNN (Relation-Shape CNN) encoder for point cloud encoding.

Multi-scale message passing with relation-shape convolutions.
Outputs a 512-dim global feature.
"""
import torch
import torch.nn as nn
from .rscnn_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction

class RSCNNEncoder(nn.Module):
    def __init__(self, input_channels=0):
        super().__init__()
        
        self.SA_modules = nn.ModuleList()
        # Level 1: SA MSG (output 512 channels so Level 2 conv gets 512+3=515)
        self.SA_modules.append(
            PointNetSetAbstractionMsg(npoint=512, radius_list=[0.23], nsample_list=[48],
                                      in_channel=input_channels, mlp_list=[[512]])
        )
        # Level 2: SA MSG
        self.SA_modules.append(
            PointNetSetAbstractionMsg(npoint=128, radius_list=[0.32], nsample_list=[64],
                                      in_channel=512, mlp_list=[[512]])
        )
        # Level 3: Global Pooling
        self.SA_modules.append(
            PointNetSetAbstraction(npoint=None, radius=None, nsample=None, 
                                   in_channel=512, mlp=[1024], group_all=True)
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512)  # Add BN after final projection for stability
        )

    def forward(self, x):
        # Input x: (B, 3, N)
        xyz = x.permute(0, 2, 1)  # (B, N, 3)
        features = None
        
        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        
        # After group_all: features is (B, 1024, 1)
        return self.fc_layer(features.squeeze(-1))