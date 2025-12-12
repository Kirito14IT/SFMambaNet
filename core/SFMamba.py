import torch
import torch.nn as nn
from loss import batch_episym

from mamba_ssm.modules.mamba_simple import Mamba
#********    新加的    *******
import torch.nn.functional as F
import math
#********    新加的    *******

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True) #xx[32,1,2000]
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k) [32,2000,9] [32,1000,6]

    return idx[:, :, :]

def get_graph_feature(x, k=20, idx=None):
    #x[32,128,2000,1],k=9
    # x[32,128,1000,1],k=6
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points) #x[32,128,2000]
    if idx is None:
        idx_out = knn(x, k=k) #idx_out[32,2000,9]
    else:
        idx_out = idx

    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx_out + idx_base #idx[32,2000,9] 把32个批次的标号连续了

    idx = idx.view(-1) #idx[32*2000*9] 把32个批次连在一起了 [32*1000*6]

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous() #x[32,2000,128]
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) #feature[32,2000,9,128]
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1) #x[32,2000,9,128]
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous() #feature[32,256,2000,9] 图特征
    return feature

class ResNet_Block(nn.Module):
    def __init__(self, inchannel, outchannel, pre=False):
        super(ResNet_Block, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
        )
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )

    def forward(self, x):
        x1 = self.right(x) if self.pre is True else x
        out = self.left(x)
        out = out + x1
        return torch.relu(out)

def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b,d,d)
    for batch_idx in range(X.shape[0]):
        e, v = torch.linalg.eigh(X[batch_idx, :, :].squeeze(), UPLO='U')
        bv[batch_idx,:,:] = v
    bv = bv.cuda()
    return bv

def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4 [32,1,500,4] logits[32,2,500,1]
    mask = logits[:, 0, :, 0] #[32,500] logits的第一层
    weights = logits[:, 1, :, 0] #[32,500] logits的第二层

    mask = torch.sigmoid(mask)
    weights = torch.exp(weights) * mask
    weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)

    x_shp = x_in.shape
    x_in = x_in.squeeze(1)

    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1).contiguous()

    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1).contiguous()
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1).contiguous(), wX)

    # Recover essential matrix from self-adjoing eigen

    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat

# KNN_MAX_Block
class DGCNN_MAX_Block(nn.Module):
    def __init__(self, knn_num=9, in_channel=128):
        super(DGCNN_MAX_Block, self).__init__()
        self.knn_num = knn_num
        self.in_channel = in_channel

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel*2, self.in_channel, (1, 1)),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channel, self.in_channel, (1, 1)), 
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
            )

    def forward(self, features):
        #feature[32,128,2000,1]
        B, _, N, _ = features.shape
        out = get_graph_feature(features, k=self.knn_num)
        out = self.conv(out) 
        out = out.max(dim=-1, keepdim=False)[0] #out[32,128,2000,1]
        out = out.unsqueeze(3)
        return out

#############  Mamba_Block  #############

class Mamba_Block(nn.Module):
    def __init__(self,channels,d_state):
        super(Mamba_Block,self).__init__()
        self.norm = nn.LayerNorm(channels)
        self.mamba = Mamba(d_model=channels,d_state=d_state)

    def forward(self,x):
        # x:bcn1
        data = x
        x = self.norm(x.transpose(1,2).squeeze(-1))#bnc
        out = self.mamba(x) #bnc
        out = self.norm(out + data.transpose(1,2).squeeze(-1))
        return out.transpose(1,2).unsqueeze(-1) #bcn1

#############  Mamba_Block  #############


#############  CSSM   #############

# 转置
class trans(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class CSM(nn.Module):
    def __init__(self, channels, points, out_channels = None, d_state=8):
        super(CSM, self).__init__()
        if not out_channels:
            out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size = 1)

        self.conv1 = nn.Sequential(
            nn.InstanceNorm2d(channels, eps = 1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size = 1),  # b*c*k*1
            )
        # Spatial Aggregation Mamba
        self.norm = nn.LayerNorm(points)
        self.SAM = Mamba(points,d_state) #bck
        self.conv3 = nn.Sequential(
            nn.InstanceNorm2d(out_channels, eps = 1e-3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 1)
        )

    def forward(self, x):
        # print(x.size(0))
        # x:bck1
        out = self.conv1(x).squeeze(-1) #out:bck
        out_n = self.norm(out)
        out = out + self.SAM(out_n)  # out:bck
        out = self.conv3(out.unsqueeze(-1)) #out:bck1
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out #out:bck1


class diff_pool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps = 1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size = 1))

    def forward(self, x):
        #x[32,128,2000,1]
        embed = self.conv(x)  # b*k*n*1,embed:[b,256,2000,1]
        S = torch.softmax(embed, dim = 2).squeeze(3) # b*k*n，计算点属于哪个类别的权重
        out = torch.matmul(x.squeeze(3), S.transpose(1, 2)).unsqueeze(3) #out[b,c,k]，将类别的权重乘以特征图，得到新的特征图
        return out #b*c*k*1


class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps = 1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size = 1))

    def forward(self, x_up, x_down):
        # x_up: b*c*n*1
        # x_down: b*c*k*1
        embed = self.conv(x_up)  # b*k*n*1
        S = torch.softmax(embed, dim = 1).squeeze(3)  # b*k*n，计算每个特征对每个点的贡献程度
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3) #bcn1，原来的特征乘以特征得分得到新的含有特征得分的特征图
        return out


class CSSM(nn.Module):
    def __init__(self, net_channels, depth = 6, clusters = 64):
        nn.Module.__init__(self)
        channels = net_channels
        self.layer_num = depth
        l2_nums = clusters
        self.down1 = diff_pool(channels, l2_nums)
        self.l2 = CSM(channels,l2_nums)
        self.up1 = diff_unpool(channels, l2_nums)
        self.output = nn.Conv2d(channels, 1, kernel_size = 1)
        self.shot_cut = nn.Conv2d(channels * 2, channels, kernel_size = 1)

        self.mamba_block = Mamba_Block(channels,8)

    def forward(self, data):
        # data: b*c*n*1
        x1_1 = data
        x1_2 = self.mamba_block(x1_1) #bcn1
        x_down = self.down1(x1_1)
        x2 = self.l2(x_down)
        x_up = self.up1(x1_1, x2)
        out = torch.cat([x1_1, x_up], dim = 1)
        return self.shot_cut(out) + x1_2

#############  CSSM   #############


#############  CFBM  #############

class CFBM(nn.Module):
    def __init__(self, channels,d_state = 16):
        super(CFBM, self).__init__()
        self.channels = channels
        self.cfbm = Mamba_Block(channels,d_state)

    def forward(self, x):
        #x:bcn1
        x_r = x
        x1 = self.cfbm(x)  # bcn1
        x1_f = x1.flip(dims=[2])
        x2 = self.cfbm(x1_f)
        out = x2.flip(dims=[2]) + x_r
        return out  # bcn1

#############  CFBM  #############




# --- 辅助类: 正弦位置编码 (Positional Encoding) ---
class PosEnc(nn.Module):
    """
    正弦位置编码，用于在CFBM前注入绝对坐标信息。
    遵循 model_v7.md 中的 PosEnc 说明。
    """
    def __init__(self, in_dim=4, d_model=128):
        super(PosEnc, self).__init__()
        self.d_model = d_model
        self.in_dim = in_dim
        # 确保可以直接整除
        assert d_model % in_dim == 0
        self.freq_dim = d_model // in_dim // 2 
        
        # 预计算频率因子
        div_term = torch.exp(torch.arange(0, self.freq_dim * 2, 2).float() * (-math.log(10000.0) / (self.freq_dim * 2)))
        self.register_buffer('div_term', div_term)

    def forward(self, coords):
        # coords: [B, N, 4] 或 [B, 4, N]
        if coords.shape[1] == 4 and coords.shape[2] != 4: 
             coords = coords.transpose(1, 2) #[B, N, 4]
             
        B, N, D = coords.shape
        # 初始化PE，注意这里需要permute回 [B, C, N] 以匹配特征图格式
        pe = torch.zeros(B, N, self.d_model, device=coords.device)
        
        for i in range(D):
            pos = coords[:, :, i].unsqueeze(-1)
            start_idx = i * (self.freq_dim * 2)
            pe[:, :, start_idx:start_idx + self.freq_dim * 2:2] = torch.sin(pos * self.div_term)
            pe[:, :, start_idx + 1:start_idx + self.freq_dim * 2:2] = torch.cos(pos * self.div_term)
            
        return pe.transpose(1, 2).unsqueeze(-1) # [B, C, N, 1] 匹配特征维度


#=============================================================================
# 创新模块实现 (Based on model_v7.md)
# =============================================================================

# ************ LSGA  **************
class LSGA(nn.Module):
    """
    创新点1：局部谱-几何注意力模块 (Local Spectral-Geometric Attention)
    """
    def __init__(self, in_channel=128, dim_L=128):
        super(LSGA, self).__init__()
        self.in_channel = in_channel
        self.dim_L = dim_L
        
        # 随机高斯矩阵 B, 形状 [4, 64]
        self.register_buffer('B_gauss', torch.randn(4, dim_L // 2))

        # MLP_light: Bottleneck结构 (128 -> 32 -> 128)
        r = 4
        self.mlp_light = nn.Sequential(
            nn.Linear(dim_L, dim_L // r),
            nn.ReLU(),
            nn.Linear(dim_L // r, in_channel)
        )

        # Attention Projections (使用Conv1d实现点对点的线性变换)
        self.W_Q = nn.Conv2d(in_channel, in_channel, 1)
        self.W_K = nn.Conv2d(in_channel, in_channel, 1)
        self.W_V = nn.Conv2d(in_channel, in_channel, 1)
        
        self.out_proj = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, x, coords, idx):
        """
        x: [B, C, N, 1] 特征
        coords: [B, N, 4] 原始坐标
        idx: [B, N, k] KNN索引 (注意：get_graph_feature里使用的是展平的索引，这里需要原始的k近邻索引)
        为了兼容MatchMamba的knn函数返回 [B, N, k]
        """
        B, C, N, _ = x.shape
        x = x.squeeze(-1) # [B, C, N]
        k = idx.shape[2]
        
        # 1. Gather neighbor coordinates
        device = coords.device
        
        # 构造batch索引
        batch_ids = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, k)
        # 展平索引以便gather
        idx_flat = idx # [B, N, k]
        
        # Gather coords: [B, N, k, 4]
        # coords [B, N, 4] -> neighbor_coords
        # 这种gather方式比view展平更直观
        neighbor_coords = torch.stack([coords[b, idx[b], :] for b in range(B)], dim=0)
        
        # Center coordinates: [B, N, 1, 4]
        center_coords = coords.unsqueeze(2)
        
        # Relative coordinates: delta_p [B, N, k, 4]
        delta_p = neighbor_coords - center_coords
        
        # 2. Spectral-Geometric Encoding
        # delta_p [B, N, k, 4] @ B_gauss [4, 64] -> [B, N, k, 64]
        projected = torch.matmul(delta_p, self.B_gauss)
        E_freq = torch.cat([torch.sin(2 * math.pi * projected), 
                            torch.cos(2 * math.pi * projected)], dim=-1) # [B, N, k, 128]
        
        # MLP_light processing
        E_geo = self.mlp_light(E_freq) # [B, N, k, 128]
        E_geo = E_geo.permute(0, 3, 1, 2) # [B, C, N, k]
        
        # 3. Geometry-Infused Attention
        # Gather neighbor features for K and V
        # x: [B, C, N] -> neighbor_feats [B, C, N, k]
        neighbor_feats = torch.stack([x[b, :, idx[b]] for b in range(B)], dim=0)
        
        Q = self.W_Q(x.unsqueeze(-1)) # [B, C, N, 1]
        
        # Inject geometry into K and V
        K_input = neighbor_feats + E_geo
        V_input = neighbor_feats + E_geo
        
        K = self.W_K(K_input) # [B, C, N, k]
        V = self.W_V(V_input) # [B, C, N, k]
        
        # Attention Score: Q * K^T
        # [B, C, N, 1] * [B, C, N, k] -> sum over C -> [B, 1, N, k]
        attn = torch.sum(Q * K, dim=1, keepdim=True) / math.sqrt(C)
        attn = F.softmax(attn, dim=-1) # [B, 1, N, k]
        
        # Weighted Sum: attn * V
        # [B, 1, N, k] * [B, C, N, k] -> sum over k -> [B, C, N, 1]
        out = torch.sum(attn * V, dim=-1, keepdim=True)
        
        out = self.out_proj(out)
        
        return out # [B, C, N, 1]
# ************ LSGA  **************


# ************ LSGA_SkipConnectionBlock  **************
class LSGA_SkipConnectionBlock(nn.Module):
    def __init__(self, in_channel=128):
        super(LSGA_SkipConnectionBlock, self).__init__()
        self.lsga = LSGA(in_channel=in_channel, dim_L=in_channel)
        # 初始化 Scale 为 0
        self.scale = nn.Parameter(torch.zeros(1))
    
    def forward(self, x, coords, idx):
        # x: [B, C, N, 1]
        B_out = self.lsga(x, coords, idx)
        C = x + self.scale * B_out
        return C
# ************ LSGA_SkipConnectionBlock  **************


# ************ CDCP  **************
class CDCP(nn.Module):
    """
    创新点2：跨域一致性投影模块 (Cross-Domain Consistency Projector)
    """
    def __init__(self, in_channel=128):
        super(CDCP, self).__init__()
        self.in_channel = in_channel
        
        # 预设最大点数 N的一半用于频域切片，假设最大N=2000
        self.max_freq_len = 2000 // 2 + 1
        # 可学习复数权重 [C, max_freq_len]
        self.W_gate = nn.Parameter(torch.view_as_complex(torch.randn(in_channel, self.max_freq_len, 2) * 0.02))

        self.mlp_att = nn.Sequential(
            nn.Conv1d(in_channel, in_channel // 4, 1),
            nn.ReLU(),
            nn.Conv1d(in_channel // 4, in_channel, 1),
            nn.Sigmoid()
        )
        
        self.mlp_proj = nn.Conv1d(in_channel, in_channel, 1)
        self.ln = nn.InstanceNorm1d(in_channel, affine=True)

    def forward(self, x):
        # x: [B, C, N, 1] -> [B, C, N]
        x_in = x.squeeze(-1)
        B, C, N = x_in.shape
        
        # 1. FFT
        x_fft = torch.fft.rfft(x_in, dim=-1) # [B, C, N//2 + 1]
        freq_len = x_fft.shape[-1]
        
        # 2. Spectral Gating (动态插值以适应不同的N)
        # [C, max_freq] -> [1, C, current_freq]
        # Treat complex weights as real with last dim 2 for interpolation
        W_gate_real = torch.view_as_real(self.W_gate).permute(2, 0, 1).unsqueeze(0) # [1, 2, C, max_len]
        
        # Interpolate to current frequency length
        W_gate_interp = F.interpolate(W_gate_real, size=(C, freq_len), mode='bilinear', align_corners=False)
        W_gate_interp = W_gate_interp.squeeze(0).permute(1, 2, 0).contiguous() # [C, freq_len, 2]
        
        gate_weight = torch.view_as_complex(W_gate_interp).to(x.device) # [C, freq_len]
            
        x_fft_gated = x_fft * gate_weight.unsqueeze(0)
        
        # 3. IFFT & Attention
        x_restored = torch.fft.irfft(x_fft_gated, n=N, dim=-1)
        h_att = self.mlp_att(x_restored)
        
        # 4. Calibration & Projection
        x_calibrated = x_in * h_att
        out = self.ln(x_calibrated + self.mlp_proj(x_in))
        
        return out.unsqueeze(-1) # [B, C, N, 1]
# ************ CDCP  **************

# ************ CDCP_SkipConnectionBlock  **************
class CDCP_SkipConnectionBlock(nn.Module):
    def __init__(self, in_channel=128):
        super(CDCP_SkipConnectionBlock, self).__init__()
        self.cdcp = CDCP(in_channel=in_channel)
        self.scale = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B_out = self.cdcp(x)
        C = x + self.scale * B_out
        return C
# ************ CDCP_SkipConnectionBlock  **************


# ************ GSFM  **************
class GSFM(nn.Module):
    """
    创新点3：全局谱-频率混合器 (Global Spectral-Frequency Mixer)
    """
    def __init__(self, in_channel=128):
        super(GSFM, self).__init__()
        self.max_freq_len = 2000 // 2 + 1
        self.W_spec = nn.Parameter(torch.view_as_complex(torch.randn(in_channel, self.max_freq_len, 2) * 0.02))
        
        self.mlp_channel = nn.Sequential(
            nn.Conv1d(in_channel, in_channel, 1),
            nn.ReLU(),
            nn.Conv1d(in_channel, in_channel, 1)
        )

    def forward(self, x):
        # x: [B, C, N, 1]
        x_in = x.squeeze(-1)
        B, C, N = x_in.shape
        
        # 1. Spectral Mixing
        x_fft = torch.fft.rfft(x_in, dim=-1)
        freq_len = x_fft.shape[-1]
        
        # Interpolate weights
        W_spec_real = torch.view_as_real(self.W_spec).permute(2, 0, 1).unsqueeze(0)
        W_spec_interp = F.interpolate(W_spec_real, size=(C, freq_len), mode='bilinear', align_corners=False)
        W_spec_interp = W_spec_interp.squeeze(0).permute(1, 2, 0).contiguous()
        spec_weight = torch.view_as_complex(W_spec_interp).to(x.device)

        x_fft_mixed = x_fft * spec_weight.unsqueeze(0)
        x_spec = torch.fft.irfft(x_fft_mixed, n=N, dim=-1)
        
        # 2. Channel Mixer
        out = x_spec + self.mlp_channel(x_spec)
        
        return out.unsqueeze(-1)
# ************ GSFM  **************


# ************ GSFM_SkipConnectionBlock  **************
class GSFM_SkipConnectionBlock(nn.Module):
    def __init__(self, in_channel=128):
        super(GSFM_SkipConnectionBlock, self).__init__()
        self.gsfm = GSFM(in_channel=in_channel)
        self.scale = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B_out = self.gsfm(x)
        C = x + self.scale * B_out
        return C
# ************ GSFM_SkipConnectionBlock  **************



class DS_Block(nn.Module):
    def __init__(self, initial=False, predict=False, out_channel=128, k_num=8, sampling_rate=0.5):
        super(DS_Block, self).__init__()
        self.initial = initial
        self.in_channel = 4 if self.initial is True else 6
        self.out_channel = out_channel
        self.k_num = k_num
        self.predict = predict
        self.sr = sampling_rate

        # 1. 输入嵌入层
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, (1, 1)), 
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )

        # 2. 局部阶段 (LGSE) + 创新点1 (LSGA)
        # 包含了 MatchMamba 的 ResNet 和 DGCNN
        self.LGSE_layers = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            DGCNN_MAX_Block(self.k_num * 2, self.out_channel),
        )
        # 创新点1：LSGA (全阶段启用)
        self.lsga_skip = LSGA_SkipConnectionBlock(in_channel=self.out_channel)

        # 3. 精炼阶段 (CSR)
        # 包含了 MatchMamba 的 CSSM
        self.CSR = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            CSSM(self.out_channel, clusters=256),
            CSSM(self.out_channel, clusters=256),
            CSSM(self.out_channel, clusters=256),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )

        # 4. 转换清洗阶段 (SRA) + 创新点2 (CDCP)
        # 全阶段启用
        self.SRA = CDCP_SkipConnectionBlock(in_channel=self.out_channel)

        # 5. 全局阶段 (GSSA) + 创新点3 (GSFM)
        # 包含了 MatchMamba 的 CFBM
        # 为了增强 CFBM，我们需要 PosEnc
        self.pos_enc = PosEnc(in_dim=4, d_model=self.out_channel)
        self.cfbm = CFBM(self.out_channel)
        
        # 创新点3：GSFM (全阶段启用)
        self.gsfm_skip = GSFM_SkipConnectionBlock(in_channel=self.out_channel)

        self.GSSA_post = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )

        self.dropout = nn.Dropout(p=0.3)

        # 预测头
        self.embed_1 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )
        self.linear_0 = nn.Conv2d(self.out_channel, 1, (1, 1))
        self.linear_1 = nn.Conv2d(self.out_channel, 1, (1, 1))

        if self.predict == True:
            self.embed_2 = ResNet_Block(self.out_channel, self.out_channel, pre=False)
            self.linear_2 = nn.Conv2d(self.out_channel, 2, (1, 1))


    def down_sampling(self, x, y, weights, indices, features=None, predict=False):
        B, _, N , _ = x.size()
        indices = indices[:, :int(N*self.sr)] 
        with torch.no_grad():
            y_out = torch.gather(y, dim=-1, index=indices) 
            w_out = torch.gather(weights, dim=-1, index=indices) 
        indices = indices.view(B, 1, -1, 1) 

        if predict == False:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4)) 
            return x_out, y_out, w_out
        else:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4)) 
            feature_out = torch.gather(features, dim=2, index=indices.repeat(1, 128, 1, 1)) 
            return x_out, y_out, w_out, feature_out

    def forward(self, x, y):
        # x: [B, 1, N, 4] -> 4D input
        # y: [B, N, 4] coords
        B, _, N , _ = x.size()
        out = x.transpose(1, 3).contiguous() 
        out = self.conv(out) 

        # 使用输入中的几何坐标，避免把标签 y 当作坐标导致维度错误
        coords = x[:, 0, :, :4]  # [B, N, 4]

        # ==========================================
        # 1. Local Stage (LGSE + LSGA)
        # ==========================================
        out = self.LGSE_layers(out) # 跑完 ResNet 和 DGCNN
        
        # 创新点1：LSGA (现在每次都跑)
        # 需要重新计算一次 KNN idx 传给 LSGA (因为 DGCNN 内部的 idx 拿不到)
        # DGCNN_MAX_Block 内部用的是 k_num*2, 我们这里保持一致或者用基础 k_num 都可以
        # 为了效果最好，我们用 k_num*2
        idx = knn(out.squeeze(-1), k=self.k_num * 2) 
        out = self.lsga_skip(out, coords, idx)
        
        out = self.dropout(out)
        
        # ==========================================
        # 2. Refinement Stage (CSR)
        # ==========================================
        out = self.CSR(out)
        
        out = self.dropout(out)
        # 预测权重 w0 (用于下采样)
        w0 = self.linear_0(out).view(B, -1) 

        # ==========================================
        # 3. Inter-Block Cleaning (SRA / CDCP)
        # ==========================================
        # 创新点2：CDCP (现在每次都跑)
        out_g = self.SRA(out)
        
        # ==========================================
        # 4. Global Stage (GSSA / CFBM + GSFM)
        # ==========================================
        # 创新点3：全套全局处理 (现在每次都跑)
        
        # 4.1 PosEnc + CFBM
        out_g = out_g + self.pos_enc(coords) # 注入位置编码
        out_g = self.cfbm(out_g)        # 跑 MatchMamba 的 CFBM
        
        # 4.2 GSFM
        out_g = self.gsfm_skip(out_g)   # 跑我们的 GSFM
        
        # 4.3 后处理
        out_g = self.GSSA_post(out_g)
        
        # 残差连接 (Global + Local)
        out_g = self.dropout(out_g)
        out = out_g + out

        # ==========================================
        # 5. Prediction & Downsampling
        # ==========================================
        out = self.embed_1(out)
        w1 = self.linear_1(out).view(B, -1) 

        if self.predict == False: 
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True) 
            w1_ds = w1_ds[:, :int(N*self.sr)] 
            x_ds, y_ds, w0_ds = self.down_sampling(x, y, w0, indices, None, self.predict)
            return x_ds, y_ds, [w0, w1], [w0_ds, w1_ds]
        else: 
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True) 
            w1_ds = w1_ds[:, :int(N*self.sr)] 
            x_ds, y_ds, w0_ds, out = self.down_sampling(x, y, w0, indices, out, self.predict)
            out = self.embed_2(out)
            w2 = self.linear_2(out) 
            e_hat = weighted_8points(x_ds, w2)

            return x_ds, y_ds, [w0, w1, w2[:, 0, :, 0]], [w0_ds, w1_ds], e_hat
class SFMamba(nn.Module):
    def __init__(self, config):
        super(SFMamba, self).__init__()

        self.ds_0 = DS_Block(initial=True, predict=False, out_channel=128, k_num=9, sampling_rate=config.sr)#sampling_rate=0.5
        self.ds_1 = DS_Block(initial=False, predict=True, out_channel=128, k_num=6, sampling_rate=config.sr)

    def forward(self, x, y):
        #x[32,1,2000,4],y[32,2000]
        B, _, N, _ = x.shape

        x1, y1, ws0, w_ds0 = self.ds_0(x, y) # 返回的是x_ds, y_ds, [w0, w1], [w0_ds, w1_ds]

        w_ds0[0] = torch.relu(torch.tanh(w_ds0[0])).reshape(B, 1, -1, 1) #变成0到1的权重[32,1,1000,1]
        w_ds0[1] = torch.relu(torch.tanh(w_ds0[1])).reshape(B, 1, -1, 1) #变成0到1的权重[32,1,1000,1]
        x_ = torch.cat([x1, w_ds0[0].detach(), w_ds0[1].detach()], dim=-1) #x_[32,1,1000,6] 剪枝后的特征并带上了权重信息

        x2, y2, ws1, w_ds1, e_hat = self.ds_1(x_, y1) #x_[32,1,1000,6],y1[32,1000]

        with torch.no_grad():
            y_hat = batch_episym(x[:, 0, :, :2], x[:, 0, :, 2:], e_hat) #y_hat对称极线距离
        #print(y_hat)
        return ws0 + ws1, [y, y, y1, y1, y2], [e_hat], y_hat

