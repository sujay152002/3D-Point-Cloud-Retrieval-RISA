# Extracted and modified from https://github.com/jiaxunguo/EnRI-GAM
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from types import SimpleNamespace

try:
    import torch_bingham
except ImportError:
    torch_bingham = None


class BinghamSampler(nn.Module):

    def __init__(self, K=1, n_s=10):  # κ1≤κ2≤κ3=0
        super().__init__()
        self.K = K        
        self.n_s = n_s
        self.init_bingham()
    
    def init_bingham(self):
        self.z_z = nn.Parameter(torch.randn(self.K, 3).float(), requires_grad=True)
        
        self.z_q = nn.Parameter(torch.randn(self.K, 4), requires_grad=True)
        
        self.z_fq = nn.Parameter(torch.randn(self.K, 4), requires_grad=True)
        
    
    @staticmethod
    def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
        """Convert a batch of unit quaternions *(w,x,y,z)* into 3×3 matrices."""
        w, x, y, z = q.unbind(-1)
        wx, wy, wz = w * x, w * y, w * z
        xx, xy, xz = x * x, x * y, x * z
        yy, yz = y * y, y * z
        zz = z * z

        R = torch.stack([
            1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy),
            2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)
        ], dim=-1).reshape(q.shape[0], 3, 3)
        return R

    # ---------------------------------------------------------------------
    def _build_Zdiag(self) -> torch.Tensor:
        """Return (K,4) tensor with ordered negative κ‑values for Bingham."""
        dZ = F.softplus(self._z_z)                       # positive      (K,3)
        Z0 = dZ[:, 0:1]
        Z1 = Z0 + dZ[:, 1:2]
        Z2 = Z1 + dZ[:, 2:3]
        Zdiag = torch.cat([Z0, Z1, Z2], dim=-1)   # κ₃, κ₂, κ₁, κ₀
        Zdiag = -1.0 * Zdiag.clamp(1e-12, 500.0)        # all ≤ 0
        return Zdiag                                    # (K,4)

    def _normalize_quaternion(self) -> torch.Tensor:
        """Map z_q → unit quaternion with non‑negative scalar part (w ≥ 0)."""
        q = self._z_q.reshape(-1, 4)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-12)
        q = ((q[:, 0:1] > 0).float() - 0.5) * 2 * q
        return q
    
    def _normalize_free_quaternion(self) -> torch.Tensor:
        """Map z_q → unit quaternion with non‑negative scalar part (w ≥ 0)."""
        q = self._z_fq.reshape(-1, 4)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-12)
        q = ((q[:, 0:1] > 0).float() - 0.5) * 2 * q
        return q
    
    
    @classmethod
    def bingham_rj(cls, E, K):
        lam = -K

        qa = lam.shape[1]
        mu = torch.zeros(qa, device=E.device)
        sigacginv = 1 + 2 * lam
        SigACG = torch.sqrt(1 / (1+2*lam))

        X = torch.zeros_like(K, device=E.device)
        rj = torch.zeros(lam.shape[0], dtype=torch.bool, device=E.device)

        while not rj.all():
            indx = torch.where(rj==0)
            yp = torch.normal(mu, SigACG[indx])
            y = yp / torch.sqrt(torch.sum(yp**2, 1, keepdim=True))
            X[indx] = y

            lratio = -torch.sum(y**2 * lam[indx], 1) - qa/2 * torch.log(torch.tensor([qa], device=E.device)) + 0.5*(qa-1) + qa/2 * torch.log(torch.sum(y**2 * sigacginv[indx], 1))
            rj[indx] = torch.log(torch.rand(len(lratio), device=E.device)) < lratio

        return torch.bmm(E.transpose(1, 2), X.unsqueeze(2)).squeeze()
    
    @classmethod
    def bingham_quaternion_sample(cls, normalized_q_output, Zbatch):

        q0 = normalized_q_output[:, 0].unsqueeze(1)  # [800, 1]
        q1 = normalized_q_output[:, 1].unsqueeze(1)  # [800, 1]
        q2 = normalized_q_output[:, 2].unsqueeze(1)  # [800, 1]
        q3 = normalized_q_output[:, 3].unsqueeze(1)  # [800, 1]
            
        row1 = torch.cat([q0, -q1, -q2, q3], dim=1)
        row2 = torch.cat([q1, q0, q3, q2], dim=1)
        row3 = torch.cat([q2, -q3, q0, -q1], dim=1)
        row4 = torch.cat([q3, q2, -q1, -q0], dim=1)
            
        z_Q = torch.stack([row1, row2, row3, row4], dim=1)
        zero_tensor = torch.ones(Zbatch.size(0), 1, dtype=Zbatch.dtype, device=Zbatch.device)*(-0.000001)
        Z_expanded = torch.cat((zero_tensor, Zbatch), dim=1)
            
        assert torch.all(Z_expanded < 0)
        z = cls.bingham_rj(z_Q, Z_expanded)
            
        return z
    
    def _sample_bingham(self, Z, q):

        Z = Z.repeat(self.n_s, 1)
        q = q.repeat(self.n_s, 1)

        quats = self.bingham_quaternion_sample(q, Z).view(-1, 4)
        
        q = quats.reshape(-1, 4)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-12)
        q = ((q[:, 0:1] > 0).float() - 0.5) * 2 * q
        
        M = q.T @ q
        
        eigval, eigvec = torch.linalg.eigh(M)
        avg_quat = eigvec[:, -1]
        avg_quat = avg_quat / avg_quat.norm()
        return avg_quat.unsqueeze(0)
    
    def update_mode(self, q):
        
        self.z_q.data.copy_(nn.Parameter(q, requires_grad=True))
        
    def forward(self):
        
        self._z_z = self.z_z
        self._z_q = self.z_q.to(self._z_z.device)

        Z = self._build_Zdiag()
        q = self._normalize_quaternion()

        quats = self._sample_bingham(Z, q)  # (K,4)
        B_paras = torch.cat([q, Z, quats], dim=-1)   
        rotated_spheres = self._quat_to_rotmat(quats)  # (K,3,3)
        rotated_spheres_mode = self._quat_to_rotmat(q)
        return rotated_spheres.squeeze(), rotated_spheres_mode.squeeze(), B_paras
    

def feat_select(feat, ind):
    assert feat.dim()==3 and ind.dim()==1
    B, C, N = feat.size()
    BNK = ind.size(0)
    K = int(BNK/(B*N))
    base = torch.arange(B, device=feat.device).view(B, 1, 1).repeat(1, 1, N*K) *N

    return torch.gather(feat, 2, (ind.view(B, 1, N*K) - base).repeat(1, C, 1)).view(B, C, N, K)

def knn(x, k, remove_self_loop=True):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    if remove_self_loop:
        idx = pairwise_distance.topk(k=k + 1, dim=-1)[1]  # (batch_size, num_points, k)
        return idx[:, :, 1:]
    else:
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
        return idx


def get_angle(v1, v2, axis=None):
    if axis is None:
        return torch.atan2(
            torch.cross(v1, v2, dim=1).norm(p=2, dim=1), (v1 * v2).sum(dim=1))
    else:
        cosine = (v1 * v2).sum(dim=1)
        cross_axis = torch.cross(v1, v2, dim=1)
        sign = torch.ones_like(cosine)
        sign[(cross_axis * axis).sum(dim=1) < 0.] = -1.
        return torch.atan2(
            cross_axis.norm(p=2, dim=1) * sign, cosine)


def point_pair_features(pos_i, pos_j, norm_i, norm_j):
    pseudo = pos_j - pos_i
    return torch.stack([
        pseudo.norm(p=2, dim=1),
        torch.cos(get_angle(norm_i, pseudo)),
        torch.cos(get_angle(norm_j, pseudo)),
        torch.cos(get_angle(norm_i, norm_j, axis=pseudo))
    ], dim=1)


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

class RIAttnConv(nn.Module):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        C_sipf: int,
        heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert C_in % heads == 0, "C_in must be divisible by heads"
        
        self.C_in = C_in
        self.C_out = C_out
        self.C_sipf = C_sipf
        self.heads = heads
        self.d_h = C_in // heads

        self.w_kv = nn.Conv2d(C_in, 2 * C_in, 1, bias=False)      # (B,2*C_in,N,K)
        
        self.bias_net_k = nn.Sequential(
            nn.Conv2d(C_sipf, C_in, 1, bias=False),
            nn.BatchNorm2d(C_in),
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, 1, bias=False),                 # (B,heads,N,K)
        )
        
        self.bias_net_v = nn.Sequential(
            nn.Conv2d(C_sipf, C_in, 1, bias=True),
            nn.BatchNorm2d(C_in),
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, 1, bias=True),                 # (B,heads,N,K)
        )
        
        self.edge_conv = nn.Sequential(nn.Conv1d(C_in*2, C_out, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(C_out),
                                       nn.LeakyReLU(negative_slope=0.2))
        
        self.dropout = dropout

    def forward(
        self, x: torch.Tensor, SiPF: torch.Tensor, edge_index: list, bs: int
    ) -> torch.Tensor:
        
        row, col = edge_index
        fs = feat_select(x, col) # 32, 64, 1024, 20
        B, _, N, K = fs.shape  # C_in == self.C_in
        sipf = SiPF.view(B, -1, K, self.C_sipf).permute(0, 3, 1, 2).contiguous() # 32, 8, 1024, 20
    

        qv = self.w_kv(fs)  # (B, 2*C_in, N, K)
        q_fs, v_fs = torch.split(qv, self.C_in, dim=1)  # 各自 (B,C_in,N,K)
        

        q_fs = q_fs.view(B, self.heads, self.d_h, N, K).permute(0, 1, 3, 4, 2)  # (B,H,N,K,d_h)
        v_fs = v_fs.view(B, self.heads, self.d_h, N, K).permute(0, 1, 3, 4, 2)  # (B,H,N,K,d_h)

        k_sipf = self.bias_net_k(sipf)
        v_sipf = self.bias_net_v(sipf)    
        k_sipf = k_sipf.view(B, self.heads, self.d_h, N, K).permute(0, 1, 3, 4, 2)  # (B,H,N,K,d_h)
        v_sipf = v_sipf.view(B, self.heads, self.d_h, N, K).permute(0, 1, 3, 4, 2)  # (B,H,N,K,d_h)

        # reshape
        q_fs_flat = q_fs.reshape(B * self.heads, N, K, self.d_h)        # (B*H,N,K,d_h)
        v_fs_flat = v_fs.reshape(B * self.heads, N, K, self.d_h)
        
        k_sipf_flat = k_sipf.reshape(B * self.heads, N, K, self.d_h)        # (B*H,N,K,d_h)
        v_sipf_flat = v_sipf.reshape(B * self.heads, N, K, self.d_h)

        attn_out = scaled_dot_product_attention(
            q_fs_flat, k_sipf_flat, v_fs_flat*v_sipf_flat,
            attn_mask=None,
            dropout_p=self.dropout,
            is_causal=False,
        )                               # (B*H,N,K,d_h)
        
        y = attn_out.view(B, self.heads, N, K, self.d_h).permute(0, 1, 4, 2, 3).reshape(B, self.C_in, N, K)  # (B,C_in,N, K)
        
        y = y.max(dim=-1, keepdim=False)[0]
        z = torch.cat([y-x, x], dim=1)  # (B,2*C_in,N)
        
        z = self.edge_conv(z)         # (B,C_out,N)
        
        return z


class RINet(nn.Module):
    def __init__(self, opt=None):
        super(RINet, self).__init__()
        if opt is None:
            opt = SimpleNamespace(k=20, dp_rate=0.5, output_channels=512)

        self.k = opt.k
        emb_dims = 1024
        self.opt = opt
        self.name = "rinet"
        
        self.n_ri_feature = 8
        
        self.bingham_sampler = BinghamSampler()
        self.init_binghamer()
    
        self.additional_channel = 0
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(emb_dims),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv1 = nn.Sequential(nn.Conv2d(self.n_ri_feature + 3 + 3, 64, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.pari_1 = RIAttnConv(64, 64, self.n_ri_feature)
        self.pari_2 = RIAttnConv(64, 128, self.n_ri_feature)
        self.pari_3 = RIAttnConv(128, 256, self.n_ri_feature)
        

        self.linear1 = nn.Linear(emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=opt.dp_rate)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=opt.dp_rate)
        self.linear3 = nn.Linear(256, opt.output_channels)
        
    def init_binghamer(self):

        rotated_spheres, rotated_spheres_mode, B_paras = self.bingham_sampler()
        
        self._rotated_spheres = rotated_spheres.detach()
        self._rotated_spheres_mode = rotated_spheres_mode.detach()
        self._B_paras = B_paras.detach()
    
    def update_binghamer(self):

        rotated_spheres, rotated_spheres_mode, B_paras = self.bingham_sampler()
        
        self._rotated_spheres_mode = rotated_spheres.detach()
        self._B_paras = B_paras.detach()
        
        self.bingham_sampler.update_mode(B_paras[:,7:])

    def _tensor_to_data(self, x):
        """Adapt repo tensor input (B, 3, N) to the original RI-Net data fields."""
        if x.dim() != 3 or x.size(1) != 3:
            raise ValueError(f"RINet expects input of shape (B, 3, N), got {tuple(x.shape)}")

        batch_size, _, num_points = x.size()
        pos = x.permute(0, 2, 1).contiguous()
        pos_flat = pos.view(-1, 3)
        normals = F.normalize(pos_flat, dim=1, eps=1e-8)

        batch = torch.arange(batch_size, device=x.device).view(-1, 1)
        batch = batch.repeat(1, num_points).view(-1)

        return SimpleNamespace(
            x=pos_flat,
            pos=pos_flat,
            pos_org=pos_flat,
            l0=normals,
            norm_org=normals,
            rotmat=torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(batch_size, 1, 1),
            batch=batch,
        )

    def forward(self, data):
        repo_tensor_input = isinstance(data, torch.Tensor)
        if repo_tensor_input:
            data = self._tensor_to_data(data)

        batch_size = data.batch.max() + 1
        BN, feat_dim = data.x.size()
        N = int(BN/batch_size)
        data.x = data.x.view(batch_size, -1, feat_dim).permute(0, 2, 1)
        
        rotmat = data.rotmat.view(batch_size,-1,3)
        pos_org = data.pos_org.view(batch_size, N, 3)
        norm_org = data.norm_org.view(batch_size, N, 3)
        rotmat_T = rotmat.transpose(1, 2)
        
        _, _, B_paras = self.bingham_sampler()  
        rotated_spheres = self._rotated_spheres_mode.to(data.x.device)
        
        pos_r = pos_org @ rotated_spheres.T
        norm_r = norm_org @ rotated_spheres.T
        
        data.pos_r = torch.bmm(pos_r, rotmat_T).view(-1, 3)
        data.l0_r  = torch.bmm(norm_r, rotmat_T).view(-1, 3)
        
        
        euc_knn_idx = knn(data.pos.view(batch_size, -1, 3).permute(0, 2, 1), k=self.k).to(data.pos.device)
        x = data.x
        SiPF, edge_index = self.get_graph_feature(data.pos, data, idx=euc_knn_idx) # (32, 8, 1024, k) [(32, 1024, k), (32, 1024, k) ]   
        
        row, col = edge_index
        SiPF = SiPF.view(batch_size, N, self.k, -1).permute(0, 3, 1, 2).contiguous() # (32, 8, 1024, k)        
        pad_x = x.unsqueeze(-1).repeat(1, 1, 1, self.k) # (32, 3, 1024, k)
        x = self.conv1(torch.cat([SiPF, feat_select(x, col) - pad_x , pad_x], dim=1))  # EdgeConv # (32, 64, 1024, k)
        x1 = x.max(dim=-1, keepdim=False)[0] # (32, 64, 1024)
       
        SiPF, edge_index = self.get_graph_feature(x1, data) # (32, 8, 1024, k) [(32, 1024, k), (32, 1024, k) ]
        x2 = self.pari_1(x1, SiPF, edge_index, bs=batch_size) # (32, 64, 1024)

        SiPF, edge_index = self.get_graph_feature(x2, data) # (32, 8, 1024, k) [(32, 1024, k), (32, 1024, k) ]
        x3 = self.pari_2(x2, SiPF, edge_index, bs=batch_size) # (32, 128, 1024)

        SiPF, edge_index = self.get_graph_feature(x3, data) # (32, 8, 1024, k) [(32, 1024, k), (32, 1024, k) ]
        x4 = self.pari_3(x3, SiPF, edge_index, bs=batch_size) # (32, 256, 1024)

        x = torch.cat((x1, x2, x3, x4), dim=1) # (32, 512, 1024)
        x = self.conv5(x) # (32, 1024, 1024)
        
        x1 = F.adaptive_max_pool1d(x, 1).squeeze() # (32, 1024)
        x2 = F.adaptive_avg_pool1d(x, 1).squeeze() # (32, 1024)

        x = torch.cat((x1, x2), 1) # (32, 2048)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # (32, 512)
        x = self.dp1(x)  
        feat = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (32, 256)
        x = self.dp2(feat)  
        x = self.linear3(x) # (32, 40)

        if repo_tensor_input:
            return x
        
        return {
            "x": x,              # (B, L, N, 3)
            "B_paras": B_paras,
            "rotated_spheres": rotated_spheres
        }
    
    def get_loss(self, parinet_out, target, writer=None, it=None, epoch=None, label_smoothing=0.0):
        
        x = parinet_out["x"]
        B_paras = parinet_out["B_paras"]
        
        V_q = B_paras[:,7:]
        V_z = B_paras[:,4:7]
        V_q_gt = self._B_paras[:,:4].to(V_q.device)
        
        ce_kwargs = {"label_smoothing": label_smoothing} if label_smoothing > 0 else {}
        loss_cls = F.cross_entropy(x, target, **ce_kwargs)
        
        pred_q_0 = V_q
        gt_q_0 = V_q_gt
        pred_z_0 = V_z
        
        entropy_bingham0 = torch_bingham.bingham_entropy(pred_z_0).reshape(-1,1)
        loss_bingham0 = - (torch_bingham.bingham_prob(pred_q_0, pred_z_0, gt_q_0) + entropy_bingham0) + 1.5
        loss_bingham0 = torch.clamp(loss_bingham0, min=0.0, max=3.0)
        
        loss_bingham0 = loss_bingham0.mean()
        
        total_loss = loss_cls + 0.8*torch.sqrt((loss_bingham0 - (loss_cls / 10))**2)
        
        if writer is not None:
            writer.add_scalar('train/loss_cls', loss_cls, it)
            writer.add_scalar('train/loss_b', loss_bingham0, it)
            
            writer.add_scalar('train/total_loss', total_loss, it)
      
        return total_loss
        

    def get_graph_feature(self, x, data, idx=None):

        if idx is None:
            idx = knn(x, k=self.k).to(x.device)  # (batch_size, num_points, k)

        batch_size = idx.size(0)
        num_points = idx.size(1)

        device = x.device
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

        idx_mat = idx + idx_base
        col = idx_mat.view(-1)
        row = (torch.arange(num_points, device=device).view(1, -1, 1).repeat(batch_size, 1, self.k) + idx_base).view(-1)
        
        
        pos_i = data.pos[row] # (32*1024*k, 3)
        pos_j = data.pos[col] # (32*1024*k, 3)
        
        norm_i = data.l0[row] # (32*1024*k, 3)
        norm_j = data.l0[col] # (32*1024*k, 3)  
  
        pos_i_r = data.pos_r[row] # (32*1024*k, 3)
        norm_i_r = data.l0_r[row] # (32*1024*k, 3)
  
        ppf = point_pair_features(pos_i=pos_i, pos_j=pos_j,
                                    norm_i=norm_i, norm_j=norm_j) # (32*1024*k, 4)
        
   
        ppf_ir = point_pair_features(pos_i=pos_i_r, pos_j=pos_i,
                                    norm_i=norm_i_r, norm_j=norm_i) # (32*1024*k, 4)
        
        ppf_jr = point_pair_features(pos_i=pos_i_r, pos_j=pos_j,
                                    norm_i=norm_i_r, norm_j=norm_j) # (32*1024*k, 4)
        
        ppf_var = ppf_jr -ppf_ir
        ppf_var = ppf_var.view(-1, self.k, 4)
         
        ppf_var_norm = ppf_var.norm(p=2, dim=1, keepdim=True)
        SiPPF = ppf_var / (ppf_var_norm + 1e-10)
        
        SiPPF = SiPPF.view(-1, 4)

        SiPF = torch.cat((ppf, SiPPF), dim=-1)
        return SiPF, [row, col]
