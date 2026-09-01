import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

    
from .DiPVNet_layer import *

## Rotation utilities for input augmentation and testing (Original code was using torch3d)
class Rotate:
        def __init__(self, R):
            self.R = R

        def to(self, device):
            self.R = self.R.to(device)
            return self

        def transform_points(self, points):
            return torch.bmm(points, self.R.transpose(1, 2))

def random_rotations(batch_size, device=None):
    q = torch.randn(batch_size, 4, device=device)
    q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(dim=1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
        2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y),
    ], dim=1).view(batch_size, 3, 3)

class RotateAxisAngle(Rotate):
    def __init__(self, angle, axis="Z", degrees=True):
        if degrees:
            angle = angle * torch.pi / 180.0
        c = torch.cos(angle)
        s = torch.sin(angle)
        z = torch.zeros_like(angle)
        o = torch.ones_like(angle)

        axis = axis.upper()
        if axis == "Z":
            R = torch.stack([
                c, -s, z,
                s,  c, z,
                z,  z, o,
            ], dim=1).view(-1, 3, 3)
        elif axis == "Y":
            R = torch.stack([
                c,  z, s,
                z,  o, z,
                -s,  z, c,
            ], dim=1).view(-1, 3, 3)
        elif axis == "X":
            R = torch.stack([
                o,  z,  z,
                z,  c, -s,
                z,  s,  c,
            ], dim=1).view(-1, 3, 3)
        else:
            raise ValueError(f"Unsupported rotation axis: {axis}")

        super().__init__(R)


class DiPVNet_cls(nn.Module):
    """
    DiPVNet Classification Network for ModelNet40.
    
    This architecture leverages both rotation-equivariant Vector Neurons (VNN) 
    and our proposed rotation-invariant DiPVNet layers (L2DP + DASFT).
    
    Args:
        num_class (int): Number of classification categories (default: 40 for ModelNet40).
        knn (int): Number of nearest neighbors for local graph construction.
        N_dir (int): Number of spherical directions for DASFT global perception.
        aggr (str): Aggregation mode for L2DP operator ('dlp' or 'sap').
        use_local (bool): Whether to use local features in the final fusion (usually True).
    """
    def __init__(self, num_class=40, knn=12, N_dir=36, aggr='dlp', use_local=True):
        super(DiPVNet_cls, self).__init__()

        self.knn = knn
        self.use_local = use_local

        # --- 1. VNN Backbone Layers (Rotation-Equivariant) ---
        self.conv1 = VNLinearLeakyReLU(4, 32) 
        self.conv2 = VNLinearLeakyReLU(64, 32)
        self.conv3 = VNLinearLeakyReLU(64, 64)
        self.conv4 = VNLinearLeakyReLU(128, 128)
        self.conv5 = VNLinearLeakyReLU(256, 256, dim=4, share_nonlinearity=True)

        # --- 2. Global Invariant Feature Extractor ---
        self.std_feature = VNStdFeature(512, dim=4, normalize_frame=False)

        # --- 3. DiPVNet Invariant Layers (L2DP + DASFT) ---
        self.IV_Flow1 = DiPVNet(16, 32, knn, spec_dim=64, N_dir=N_dir, aggregation_mode=aggr)
        self.IV_Flow2 = DiPVNet(16, 32, knn, spec_dim=64, N_dir=N_dir, aggregation_mode=aggr)
        self.IV_Flow3 = DiPVNet(32, 64, knn, spec_dim=32, N_dir=N_dir, aggregation_mode=aggr)
        self.IV_Flow4 = DiPVNet(64, 128, knn, spec_dim=32, N_dir=N_dir, aggregation_mode=aggr)

        # --- 4. Feature Fusion & Classification Head ---
        if self.use_local:
            self.FFN_fused = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.LayerNorm(256)
            )
            self.FFN_local = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.LayerNorm(256)
            )
        else:
            self.FFN_fused = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.LayerNorm(512)
            )
            self.FFN_local = None

        self.pool1 = VNMaxPool(32)
        self.pool2 = VNMaxPool(32)
        self.pool3 = VNMaxPool(64)
        self.pool4 = VNMaxPool(128)

        self.linear1 = nn.Linear(256 * 12 + 1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, num_class)

    def forward(self, x, rot_mode='so3'):
        batch_size = x.size(0)
        num_points = x.size(1)

        # --- 1. Input Preparation & Augmentation ---
        trot = self.get_trot(batch_size, x.device, rot_mode)
        x_rot = rearrange(x, 'b n (c v) -> b (c n) v', v=3)
        x_rot = trot.transform_points(x_rot)
        x_rot = rearrange(x_rot, 'b (c n) v -> b c v n', n=num_points)

        # --- 2. Hierarchical Feature Extraction ---
        x = get_graph_feature(x_rot, k=self.knn) 
        x = self.conv1(x)                        
        feat1, local_feat1 = self.IV_Flow1(x)    
        x1 = self.pool1(x)                       

        x = get_graph_feature(x1, k=self.knn)
        x = self.conv2(x)
        feat2, local_feat2 = self.IV_Flow2(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.knn)
        x = self.conv3(x)
        feat3, local_feat3 = self.IV_Flow3(x)
        x3 = self.pool3(x)

        x = get_graph_feature(x3, k=self.knn)
        x = self.conv4(x)
        feat4, local_feat4 = self.IV_Flow4(x)
        x4 = self.pool4(x)

        # --- 3. Feature Aggregation & Global Invariant Extraction ---
        final_x = torch.cat((x1, x2, x3, x4), dim=1)
        final_x = self.conv5(final_x)

        x_mean = final_x.mean(dim=-1, keepdim=True).expand(final_x.size())
        eq_xfeat = torch.cat((final_x, x_mean), 1)
        eq_xfeat, trans = self.std_feature(eq_xfeat) 
        eq_xfeat = eq_xfeat.view(batch_size, -1, num_points)

        x1_std = F.adaptive_max_pool1d(eq_xfeat, 1).view(batch_size, -1)
        x2_std = F.adaptive_avg_pool1d(eq_xfeat, 1).view(batch_size, -1)
        eq_xfeat_pooled = torch.cat((x1_std, x2_std), 1)

        # --- 4. DiPVNet Invariant Feature Fusion ---
        final_feat = torch.cat((feat1, feat2, feat3, feat4), dim=1)
        final_feat = self.FFN_fused(final_feat.transpose(-1, -2)).transpose(-1, -2)
        x1_fused = F.adaptive_max_pool1d(final_feat, 1).view(batch_size, -1)
        x2_fused = F.adaptive_avg_pool1d(final_feat, 1).view(batch_size, -1)
        feat_fused_pooled = torch.cat((x1_fused, x2_fused), 1) 

        if self.use_local:
            final_local_feat = torch.cat((local_feat1, local_feat2, local_feat3, local_feat4), dim=1)
            proc_local = self.FFN_local(final_local_feat.transpose(-1, -2)).transpose(-1, -2)
            x1_local = F.adaptive_max_pool1d(proc_local, 1).view(batch_size, -1)
            x2_local = F.adaptive_avg_pool1d(proc_local, 1).view(batch_size, -1)
            feat_local_pooled = torch.cat((x1_local, x2_local), 1) 
            iv_xfeat = torch.cat((feat_fused_pooled, feat_local_pooled), 1) 
        else:
            iv_xfeat = feat_fused_pooled 

        feat = torch.cat((eq_xfeat_pooled, iv_xfeat), 1)

        # --- 5. Classification ---
        x = F.leaky_relu(self.bn1(self.linear1(feat)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x, None

    def get_trot(self, batch_size, device, mode):
        rot = random_rotations(batch_size)
        if mode == 'z':
            trot = RotateAxisAngle(angle=torch.rand(batch_size) * 360, axis="Z", degrees=True).to(device)
        elif mode == 'so3':
            trot = Rotate(R=rot).to(device)
        return trot


class DiPVNet_ScanObjectNN_cls(nn.Module):
    """
    DiPVNet Classification Network for ScanObjectNN.
    Tailored for real-world noisy point clouds.
    """
    def __init__(self, num_class=15, knn=20, N_dir=36, aggr='sap', use_local=True):
        super(DiPVNet_ScanObjectNN_cls, self).__init__()

        self.knn = knn
        self.use_local = use_local

        # --- 1. VNN Backbone Layers ---
        self.conv1 = VNLinearLeakyReLU(2, 32)
        self.conv2 = VNLinearLeakyReLU(64, 32)
        self.conv3 = VNLinearLeakyReLU(64, 64)
        self.conv4 = VNLinearLeakyReLU(128, 128)
        self.conv5 = VNLinearLeakyReLU(256, 256, dim=4, share_nonlinearity=True)

        # --- 2. Global Invariant Feature Extractor ---
        self.std_feature = VNStdFeature(512, dim=4, normalize_frame=False)

        # --- 3. DiPVNet Invariant Layers (L2DP + DASFT) ---
        self.IV_Flow1 = DiPVNet(16, 32, knn, spec_dim=64, N_dir=N_dir, aggregation_mode=aggr)
        self.IV_Flow2 = DiPVNet(16, 32, knn, spec_dim=64, N_dir=N_dir, aggregation_mode=aggr)
        self.IV_Flow3 = DiPVNet(32, 64, knn, spec_dim=32, N_dir=N_dir, aggregation_mode=aggr)
        self.IV_Flow4 = DiPVNet(64, 128, knn, spec_dim=32, N_dir=N_dir, aggregation_mode=aggr)

        # --- 4. Feature Fusion & Classification Head ---
        if self.use_local:
            self.FFN_fused = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.LayerNorm(256)
            )
            self.FFN_local = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.LayerNorm(256)
            )
        else:
            self.FFN_fused = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.LayerNorm(512)
            )
            self.FFN_local = None

        self.pool1 = VNMaxPool(32)
        self.pool2 = VNMaxPool(32)
        self.pool3 = VNMaxPool(64)
        self.pool4 = VNMaxPool(128)

        fusion_dim = 1024 if use_local else 512
        self.linear1 = nn.Linear(256 * 12 + fusion_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, num_class)

    def forward(self, x, rot_mode='so3'):
        batch_size = x.size(0)
        num_points = x.size(1)

        # --- 1. Input Augmentation & Reshape ---
        trot = self.get_trot(batch_size, x.device, rot_mode)
        
        if x.shape[-1] == 3:
             x_rot = trot.transform_points(x) 
             x_rot = rearrange(x_rot, 'b n v -> b 1 v n') 
        else:
             x_rot = rearrange(x, 'b n (c v) -> b (c n) v', v=3)
             x_rot = trot.transform_points(x_rot)
             x_rot = rearrange(x_rot, 'b (c n) v -> b c v n', n=num_points)

        # --- 2. Hierarchical Flow ---
        x = get_graph_feature(x_rot, k=self.knn)
        x = self.conv1(x)
        feat1, local_feat1 = self.IV_Flow1(x)
        x1 = self.pool1(x)

        x = get_graph_feature(x1, k=self.knn)
        x = self.conv2(x)
        feat2, local_feat2 = self.IV_Flow2(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.knn)
        x = self.conv3(x)
        feat3, local_feat3 = self.IV_Flow3(x)
        x3 = self.pool3(x)

        x = get_graph_feature(x3, k=self.knn)
        x = self.conv4(x)
        feat4, local_feat4 = self.IV_Flow4(x)
        x4 = self.pool4(x)

        # --- 3. Global Aggregation ---
        final_x = torch.cat((x1, x2, x3, x4), dim=1)
        final_x = self.conv5(final_x)

        x_mean = final_x.mean(dim=-1, keepdim=True).expand(final_x.size())
        eq_xfeat = torch.cat((final_x, x_mean), 1)
        eq_xfeat, trans = self.std_feature(eq_xfeat)
        eq_xfeat = eq_xfeat.view(batch_size, -1, num_points)

        x1_std = F.adaptive_max_pool1d(eq_xfeat, 1).view(batch_size, -1)
        x2_std = F.adaptive_avg_pool1d(eq_xfeat, 1).view(batch_size, -1)
        eq_xfeat_pooled = torch.cat((x1_std, x2_std), 1)

        # --- 4. DiPVNet Feature Fusion ---
        final_feat = torch.cat((feat1, feat2, feat3, feat4), dim=1)
        final_feat = self.FFN_fused(final_feat.transpose(-1, -2)).transpose(-1, -2)
        x1_fused = F.adaptive_max_pool1d(final_feat, 1).view(batch_size, -1)
        x2_fused = F.adaptive_avg_pool1d(final_feat, 1).view(batch_size, -1)
        feat_fused_pooled = torch.cat((x1_fused, x2_fused), 1)

        if self.use_local:
            final_local_feat = torch.cat((local_feat1, local_feat2, local_feat3, local_feat4), dim=1)
            proc_local = self.FFN_local(final_local_feat.transpose(-1, -2)).transpose(-1, -2)
            x1_local = F.adaptive_max_pool1d(proc_local, 1).view(batch_size, -1)
            x2_local = F.adaptive_avg_pool1d(proc_local, 1).view(batch_size, -1)
            feat_local_pooled = torch.cat((x1_local, x2_local), 1)
            iv_xfeat = torch.cat((feat_fused_pooled, feat_local_pooled), 1)
        else:
            iv_xfeat = feat_fused_pooled

        feat = torch.cat((eq_xfeat_pooled, iv_xfeat), 1)

        # --- 5. Classification ---
        x = F.leaky_relu(self.bn1(self.linear1(feat)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x, None

    def get_trot(self, batch_size, device, mode):
        rot = random_rotations(batch_size)
        if mode == 'z':
            trot = RotateAxisAngle(angle=torch.rand(batch_size) * 360, axis="Z", degrees=True).to(device)
        elif mode == 'so3':
            trot = Rotate(R=rot).to(device)
        return trot


class DiPVNetEncoder(DiPVNet_ScanObjectNN_cls):
    """Repo-compatible DiPVNet encoder wrapper.

    The copied DiPVNet classification model expects xyz as (B, N, 3) and
    returns (logits, None).  The project trainer expects encoders to receive
    (B, 3, N) and return a single global encoding tensor.
    """
    def __init__(self, output_dim=512, knn=20, N_dir=36, aggr='sap', use_local=True):
        super().__init__(
            num_class=output_dim,
            knn=knn,
            N_dir=N_dir,
            aggr=aggr,
            use_local=use_local,
        )
        self.name = "dipvnet"

    def forward(self, x, rot_mode='so3'):
        if x.dim() != 3 or x.size(1) != 3:
            raise ValueError(f"DiPVNetEncoder expects input of shape (B, 3, N), got {tuple(x.shape)}")

        x = x.permute(0, 2, 1).contiguous()
        encoding, _ = super().forward(x, rot_mode=rot_mode)
        return encoding


class DiPVNet_partseg(nn.Module):
    """
    DiPVNet Part Segmentation Network for ShapeNet.
    Encoder-Decoder architecture.
    """
    def __init__(self, num_part=50, knn=40, N_dir=36, aggr='dlp'):
        super(DiPVNet_partseg, self).__init__()

        self.n_knn = knn

        # --- Encoder: VNN Backbone ---
        self.conv1 = VNLinearLeakyReLU(4, 16)
        self.conv2 = VNLinearLeakyReLU(16, 16)
        self.conv3 = VNLinearLeakyReLU(32, 32)
        self.conv4 = VNLinearLeakyReLU(32, 32)
        self.conv5 = VNLinearLeakyReLU(64, 64)

        # --- Encoder: DiPVNet Invariant Flows ---
        self.IV_Flow1 = DiPVNet(8, 16, knn, spec_dim=64, N_dir=N_dir, aggregation_mode=aggr)
        self.IV_Flow2 = DiPVNet(8, 16, knn, spec_dim=64, N_dir=N_dir, aggregation_mode=aggr)
        self.IV_Flow3 = DiPVNet(16, 32, knn, spec_dim=32, N_dir=N_dir, aggregation_mode=aggr)
        self.IV_Flow4 = DiPVNet(16, 32, knn, spec_dim=32, N_dir=N_dir, aggregation_mode=aggr)
        self.IV_Flow5 = DiPVNet(32, 64, knn, spec_dim=32, N_dir=N_dir, aggregation_mode=aggr)

        self.IV_Feat_FFN = nn.Linear(320, 320)

        self.pool1 = VNMaxPool(16)
        self.pool2 = VNMaxPool(32)
        self.pool3 = VNMaxPool(64)

        self.conv6 = VNLinearLeakyReLU(112, 256, dim=4, share_nonlinearity=True)
        self.std_feature = VNStdFeature(512, dim=4, normalize_frame=False)
        
        # --- Decoder & Segmentation Head ---
        self.conv7 = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv8 = nn.Sequential(
            nn.Conv1d(3696, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.dp1 = nn.Dropout(p=0.4)
        self.conv9 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.dp2 = nn.Dropout(p=0.4)
        self.conv10 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv11 = nn.Conv1d(128, num_part, kernel_size=1, bias=False)

    def forward(self, x, l, rot_mode='so3'):
        batch_size = x.size(0)
        num_points = x.size(2)

        # --- 1. Input Preparation & Augmentation ---
        trot = self.get_trot(batch_size, x.device, rot_mode)
        x_rot = rearrange(x, 'b (c v) n -> b (c n) v', v=3) 
        x_rot = trot.transform_points(x_rot)                
        x_rot = rearrange(x_rot, 'b (c n) v -> b c v n', n=num_points) 

        # --- 2. Encoder (Hierarchical Feature Extraction) ---
        x = get_graph_feature(x_rot, k=self.n_knn)
        x = self.conv1(x)
        feat1, local_feat1 = self.IV_Flow1(x)
        lc_seg_feat1 = self.get_seg_feat(local_feat1, batch_size)
        seg_feat1 = self.get_seg_feat(feat1, batch_size)
        
        x = self.conv2(x)
        feat2, local_feat2 = self.IV_Flow2(x)
        lc_seg_feat2 = self.get_seg_feat(local_feat2, batch_size)
        seg_feat2 = self.get_seg_feat(feat2, batch_size)
        x1 = self.pool1(x)

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        feat3, local_feat3 = self.IV_Flow3(x)
        lc_seg_feat3 = self.get_seg_feat(local_feat3, batch_size)
        seg_feat3 = self.get_seg_feat(feat3, batch_size)
        
        x = self.conv4(x)
        feat4, local_feat4 = self.IV_Flow4(x)
        lc_seg_feat4 = self.get_seg_feat(local_feat4, batch_size)
        seg_feat4 = self.get_seg_feat(feat4, batch_size)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        feat5, local_feat5 = self.IV_Flow5(x)
        lc_seg_feat5 = self.get_seg_feat(local_feat5, batch_size)
        seg_feat5 = self.get_seg_feat(feat5, batch_size)
        x3 = self.pool3(x)

        # --- 3. Global Equivariant Feature Aggregation ---
        x123 = torch.cat((x1, x2, x3), dim=1) 
        final_x = self.conv6(x123)
        
        x_mean = final_x.mean(dim=-1, keepdim=True).expand(final_x.size())
        eq_xfeat = torch.cat((final_x, x_mean), 1)
        eq_xfeat, z0 = self.std_feature(eq_xfeat) 
        
        x123 = torch.einsum('bijm,bjkm->bikm', x123, z0).view(batch_size, -1, num_points)

        eq_xfeat = eq_xfeat.view(batch_size, -1, num_points)
        x_global = eq_xfeat.max(dim=-1, keepdim=True)[0] 

        # --- 4. Decoder Feature Assembly ---
        l = l.view(batch_size, -1, 1)
        l = self.conv7(l) 

        seg_feat = torch.cat((seg_feat1, seg_feat2, seg_feat3, seg_feat4, seg_feat5), dim=1)
        lc_seg_feat = torch.cat((lc_seg_feat1, lc_seg_feat2, lc_seg_feat3, lc_seg_feat4, lc_seg_feat5), dim=1)
        
        global_context = torch.cat((x_global, seg_feat, lc_seg_feat, l), dim=1) 
        global_context = global_context.repeat(1, 1, num_points) 

        x_combined = torch.cat((global_context, x123), dim=1)

        final_feat = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        final_local_feat = torch.cat((local_feat1, local_feat2, local_feat3, local_feat4, local_feat5), dim=1)
        
        iv_xfeat = torch.cat((final_feat, final_local_feat), 1)
        iv_xfeat = self.IV_Feat_FFN(iv_xfeat.transpose(-1, -2)).transpose(-1, -2)
        
        iv_xfeat_max = F.adaptive_max_pool1d(iv_xfeat, 1).view(batch_size, -1, 1).expand(-1, -1, num_points)
        iv_xfeat_avg = F.adaptive_avg_pool1d(iv_xfeat, 1).view(batch_size, -1, 1).expand(-1, -1, num_points)

        feat = torch.cat((x_combined, final_local_feat, iv_xfeat, iv_xfeat_max, iv_xfeat_avg), 1)

        # --- 5. Segmentation Head ---
        x = self.conv8(feat)
        x = self.dp1(x)
        x = self.conv9(x)
        x = self.dp2(x)
        x = self.conv10(x)
        x = self.conv11(x)

        return x.transpose(-1, -2) 

    def get_seg_feat(self, x, batch_size):
        x1 = F.adaptive_max_pool1d(x[:batch_size, :, :], 1).view(batch_size, -1).unsqueeze(-1)
        x2 = F.adaptive_avg_pool1d(x[:batch_size, :, :], 1).view(batch_size, -1).unsqueeze(-1)
        return torch.cat((x1, x2), 1)

    def get_trot(self, batch_size, device, mode):
        if mode == 'z':
            trot = RotateAxisAngle(angle=torch.rand(batch_size) * 360, axis="Z", degrees=True).to(device)
        elif mode == 'so3':
            rot = random_rotations(batch_size)
            trot = Rotate(R=rot).to(device)
        else:
            trot = RotateAxisAngle(angle=torch.zeros(batch_size), axis="Z", degrees=True).to(device)
        return trot
