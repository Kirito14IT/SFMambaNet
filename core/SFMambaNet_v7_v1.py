import torch
import torch.nn as nn
from loss import batch_episym
import torch.nn.functional as F

from model.knn import knn
from model.get_graph_feature import get_graph_feature
from model.batch_symeig import batch_symeig
from model.weighted_8points import weighted_8points
from model.mamba_block import Mamba_Block
from model.diff_pool import diff_pool
from model.diff_unpool import diff_unpool
from model.bicsm import BiCSM


class DGCNN_MAX_Block(nn.Module):
    def __init__(self, knn_num=9, in_channel=128):
        super(DGCNN_MAX_Block, self).__init__()
        self.knn_num = knn_num
        self.in_channel = in_channel

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel*2, self.in_channel, (1, 1)), #[32,128,2000,9]→[32,128,2000,3]
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channel, self.in_channel, (1, 1)), #[32,128,2000,3]→[32,128,2000,1]
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
            )

    def forward(self, features):
        #feature[32,128,2000,1]
        B, _, N, _ = features.shape
        out = get_graph_feature(features, k=self.knn_num)
        out = self.conv(out) #out[32,128,2000,1]
        out = out.max(dim=-1, keepdim=False)[0]
        out = out.unsqueeze(3)
        return out


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


class SGEA(nn.Module):
    """
    创新点：局部谱-几何注意力模块
    在 DGCNN 之后注入频域几何信息，弥补 EdgeConv 仅关注语义特征的缺陷。
    通过傅里叶特征编码相对坐标，并注入到注意力机制中。

    改进：
    1. 输入归一化稳定训练起点
    2. 频域几何编码归一化
    3. Q/K 归一化稳定注意力计算
    4. 输出 ResNet_Block 保证最低性能
    """
    def __init__(self, in_channel=128, dim_L=64):
        super(SGEA, self).__init__()
        self.in_channel = in_channel
        self.dim_L = dim_L

        # 随机高斯矩阵 B, 形状 [4, 32]
        self.register_buffer('B_gauss', torch.randn(4, dim_L // 2))

        # 🎯 新增：输入归一化（稳定训练起点）
        self.ln_in = nn.InstanceNorm2d(in_channel, eps=1e-3)

        # 🎯 新增：频域几何编码归一化（稳定 MLP 处理）
        self.ln_geo = nn.InstanceNorm2d(in_channel, eps=1e-3)

        # MLP_light: Bottleneck结构 (128 -> 16 -> 128)，r=8
        r = 8
        self.mlp_light = nn.Sequential(
            nn.Linear(dim_L, dim_L // r),
            nn.ReLU(),
            nn.Linear(dim_L // r, in_channel)
        )

        # 🎯 新增：Q/K 归一化（稳定注意力分数）
        self.ln_q = nn.InstanceNorm2d(in_channel, eps=1e-3)
        self.ln_k = nn.InstanceNorm2d(in_channel, eps=1e-3)

        # Attention Projections
        self.W_Q = nn.Conv2d(in_channel, in_channel, 1)
        self.W_K = nn.Conv2d(in_channel, in_channel, 1)
        self.W_V = nn.Conv2d(in_channel, in_channel, 1)

        self.out_proj = nn.Conv2d(in_channel, in_channel, 1)

        self.ln_out = nn.LayerNorm(in_channel)

    def forward(self, x, coords, idx):
        """
        x: [B, C, N, 1] 特征
        coords: [B, N, 4] 原始坐标
        idx: [B, N, k] KNN索引
        """
        init_x = x  # 保存原始输入
        B, C, N, _ = x.shape
        
        # 🎯 新增：输入归一化（稳定训练起点）
        x_norm = self.ln_in(x)
        x_flat = x_norm.squeeze(-1).permute(0, 2, 1).contiguous().view(B * N, C)  # [B*N, C]

        k = idx.shape[2]

        # 1. 收集邻居坐标 (向量化实现)
        device = coords.device
        coords_flat = coords.view(B * N, 4)  # [B*N, 4]
        idx_base = torch.arange(0, B, device=device).view(B, 1, 1) * N  # [B, 1, 1]
        idx_flat = (idx + idx_base).view(-1)  # [B*N*k]
        neighbor_coords_flat = coords_flat[idx_flat, :]  # [B*N*k, 4]
        neighbor_coords = neighbor_coords_flat.view(B, N, k, 4)

        # 相对坐标: delta_p [B, N, k, 4]
        center_coords = coords.unsqueeze(2)
        delta_p = neighbor_coords - center_coords

        # 2. 频域几何编码 (Spectral-Geometric Encoding)
        # delta_p [B, N, k, 4] @ B_gauss [4, 64] -> [B, N, k, 64]
        projected = torch.matmul(delta_p, self.B_gauss)
        E_freq = torch.cat([torch.sin(2 * 3.14159 * projected),
                            torch.cos(2 * 3.14159 * projected)], dim=-1)  # [B, N, k, 128]

        # MLP 处理
        E_geo = self.mlp_light(E_freq)  # [B, N, k, 128]
        E_geo = E_geo.permute(0, 3, 1, 2)  # [B, C, N, k]

        # 🎯 新增：频域几何编码归一化（稳定特征尺度）
        E_geo_norm = self.ln_geo(E_geo)

        # 3. 收集邻居特征
        neighbor_feats_flat = x_flat[idx_flat, :]  # [B*N*k, C]
        neighbor_feats = neighbor_feats_flat.view(B, N, k, C).permute(0, 3, 1, 2)  # [B, C, N, k]

        # 4. 几何注入注意力 (Geometry-Infused Attention)
        Q = self.W_Q(x_norm)  # [B, C, N, 1]

        # 🎯 新增：K/V 归一化（稳定注意力计算）
        K_input = neighbor_feats + E_geo_norm
        V_input = neighbor_feats + E_geo_norm

        K = self.W_K(K_input)  # [B, C, N, k]
        V = self.W_V(V_input)  # [B, C, N, k]

        # 🎯 Q/K 归一化后计算注意力分数
        Q_norm = self.ln_q(Q)
        K_norm = self.ln_k(K)

        # 注意力分数: Q * K -> sum over C -> [B, 1, N, k]
        attn = torch.sum(Q_norm * K_norm, dim=1, keepdim=True) / (C ** 0.5)
        attn = F.softmax(attn, dim=-1)  # [B, 1, N, k]

        # 加权求和: attn * V -> sum over k -> [B, C, N, 1]
        out = torch.sum(attn * V, dim=-1, keepdim=True)

        out = self.out_proj(out)

        # 轻量级残差 + LayerNorm
        out = out.squeeze(-1).permute(0, 2, 1)  # [B, C, N, 1] -> [B, N, C]
        out = self.ln_out(out).permute(0, 2, 1).unsqueeze(-1)  # [B, N, C] -> [B, C, N, 1]
        out = out + init_x
        
        return torch.relu(out)

class SFMSI(nn.Module):
    """
    创新点：谱-频多尺度交互层 (Spectral-Frequency Multi-Scale Interaction)

    论文公式 (14):
    F^I = Softmax(F^G (F^L)^T / sqrt(D)) F^L

    输入：
        L: 局部/聚类特征 [B, C, M, 1] - 来自 F^L 
        G: 全局特征 [B, C, N, 1] - 来自 F^G
    输出：
        交互特征 [B, C, N, 1]

    框架：
        F^G = Query, F^L = Keys & Values
    """
    def __init__(self, channels=128):
        super(SFMSI, self).__init__()
        self.channels = channels

        # 🎯 输入归一化（稳定训练起点）
        self.ln_L = nn.InstanceNorm2d(channels, eps=1e-3)
        self.ln_G = nn.InstanceNorm2d(channels, eps=1e-3)

        # 投影层：F^G -> Q, F^L -> K, V
        self.W_q = nn.Conv2d(channels, channels, 1)  # G -> Q [B, C, N, 1] -> [B, C, N]
        self.W_k = nn.Conv2d(channels, channels, 1)  # L -> K [B, C, M, 1] -> [B, C, M]
        self.W_v = nn.Conv2d(channels, channels, 1)  # L -> V [B, C, M, 1] -> [B, C, M]

        # 🎯 输出 ResNet_Block（保证最低性能）
        self.ResNet = ResNet_Block(channels, channels, pre=False)

    def forward(self, L, G):
        """
        L: 全局特征 [B, C, N, 1] - F^G 
        G: 局部/聚类特征 [B, C, M, 1] - F^L 

        论文公式: F^I = Softmax(F^G (F^L)^T / sqrt(D)) F^L
        其中 F^G (Query) 是 N 点，F^L (Keys/Values) 是 M 点
        """
        B, C, N, _ = L.shape
        _, _, M, _ = G.shape

        # SSCM 调用: sfmsi(G, x2) 其中 G=2000点, x2=256点
        # SFMSI 接收: L=G (2000点), G=x2 (256点)
        # 正确顺序: Query(L)=2000点, Keys/Values(G)=256点

        # 1. Query: F^G -> Q [B, C, N=2000]
        Q = self.W_q(L).squeeze(-1)

        # 2. Keys: F^L -> K [B, C, M=256]
        K = self.W_k(G).squeeze(-1)

        # 3. Values: F^L -> V [B, C, M=256] (需要额外投影)
        V = self.W_v(G).squeeze(-1)

        # 4. 注意力分数: A = Q @ K^T -> [B, N, M]
        # 论文公式: F^G (F^L)^T / sqrt(D)
        # Q = F^G [B, C, N], K = F^L [B, C, M]
        # 正确计算: Q.view(B, N, C) @ K.view(B, M, C).transpose(1, 2)
        Q_for_attn = Q.view(B, N, C)  # [B, N, C]
        K_for_attn = K.view(B, M, C)  # [B, M, C]
        A = torch.bmm(Q_for_attn, K_for_attn.transpose(1, 2)) / (C ** 0.5)

        # 5. Softmax 归一化
        A_softmax = F.softmax(A, dim=-1)  # [B, N, M]

        # 6. 加权求和: F^I = A_softmax @ V -> [B, N, C]
        # bmm 要求: (B, N, M) @ (B, M, C) -> (B, N, C)
        F_I = torch.bmm(A_softmax, V.transpose(1, 2))

        # 7. 变换回 [B, C, N, 1] 格式
        F_I = F_I.transpose(1, 2).contiguous()  # [B, C, N]
        F_I = F_I.view(B, C, N, 1)  # [B, C, N, 1]

        # 8. 残差连接: F^I + Q (注意 Q 已经是 3D，需要扩展)
        residual = Q.view(B, C, N, 1)
        out = self.ResNet(F_I + residual)

        return out

class SSCM(nn.Module):
    def __init__(self, net_channels, depth = 6, clusters = 64):
        nn.Module.__init__(self)
        channels = net_channels
        self.layer_num = depth
        l2_nums = clusters
        self.down1 = diff_pool(channels, l2_nums)
        self.l2 = BiCSM(channels,l2_nums)
        self.up1 = diff_unpool(channels, l2_nums)
        self.output = nn.Conv2d(channels, 1, kernel_size = 1)
        self.shot_cut = nn.Conv2d(channels * 2, channels, kernel_size = 1)

        self.BiMamba_block = Mamba_Block(channels,8)
        # self.sfmsi = SFMSI(channels)
    def forward(self, data):
        # data: b*c*n*1
        x1_1 = data
        x1_2 = self.BiMamba_block(x1_1) #bcn1
        x_down = self.down1(x1_1)
        x2 = self.l2(x_down)
        x_up = self.up1(x1_1, x2)
        # x_up = self.sfmsi(x_up, x1_2)
        out = torch.cat([x1_1, x_up], dim = 1)
        return self.shot_cut(out) + x1_2



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
        # 使用频域线性插值，避免破坏复数权重的频域特性
        # W_gate: [C, max_freq_len] (complex)
        if freq_len <= self.max_freq_len:
            # 如果当前频长小于等于最大频长，直接切片
            gate_weight = self.W_gate[:, :freq_len]
        else:
            # 如果当前频长大于最大频长，使用线性插值（在频域维度上）
            # 将复数权重转换为实数进行1D插值
            W_gate_real = torch.view_as_real(self.W_gate)  # [C, max_freq_len, 2]
            # 在频域维度上进行1D线性插值
            # Reshape to [2*C, max_freq_len] for 1D interpolation
            C = W_gate_real.shape[0]
            W_gate_reshaped = W_gate_real.permute(2, 0, 1).contiguous().view(2 * C, -1).unsqueeze(0)  # [1, 2*C, max_freq_len]
            W_gate_interp = F.interpolate(W_gate_reshaped, size=freq_len, mode='linear', align_corners=False)  # [1, 2*C, freq_len]
            W_gate_interp = W_gate_interp.squeeze(0).view(2, C, freq_len).permute(1, 2, 0).contiguous()  # [C, freq_len, 2]
            gate_weight = torch.view_as_complex(W_gate_interp)
        
        gate_weight = gate_weight.to(x.device)  # [C, freq_len]
            
        x_fft_gated = x_fft * gate_weight.unsqueeze(0)
        
        # 3. IFFT & Attention
        x_restored = torch.fft.irfft(x_fft_gated, n=N, dim=-1)
        h_att = self.mlp_att(x_restored)
        
        # 4. Calibration & Projection
        x_calibrated = x_in * h_att
        out = self.ln(x_calibrated + self.mlp_proj(x_in))
        
        return out.unsqueeze(-1) # [B, C, N, 1]

class SIGM(nn.Module):
    def __init__(self, channels,d_state = 16):
        super(SIGM, self).__init__()
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



#DGCNN_MAX_Block、ResNet_Block、SGEA、SFMSI、SSCM、CDCP、SIGM、DS_Block、SFMambaNet_v7
class DS_Block(nn.Module):
    def __init__(self, initial=False, predict=False, out_channel=128, k_num=8, sampling_rate=0.5):
        super(DS_Block, self).__init__()
        self.initial = initial
        self.in_channel = 4 if self.initial is True else 6
        self.out_channel = out_channel
        self.k_num = k_num
        self.predict = predict
        self.sr = sampling_rate

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, (1, 1)), #4或6 → 128
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )

        self.LSGFE = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            DGCNN_MAX_Block(self.k_num * 2, self.out_channel),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )

        #################################################
        self.LSGFE_LSGA_SGEA = SGEA(self.out_channel, 64)  # 第二个参数dim_L=64，节省显存

        self.LSGFE_LSGA_SSCM = nn.Sequential(
            SSCM(self.out_channel, clusters=256),
            SSCM(self.out_channel, clusters=256),
            SSCM(self.out_channel, clusters=256),
        )
        #################################################

        #################################################
        self.SGCA_SIGM = nn.Sequential(CDCP(in_channel=self.out_channel), 
                                    CDCP(in_channel=self.out_channel), 
                                    CDCP(in_channel=self.out_channel), 
                                    CDCP(in_channel=self.out_channel), 
                                    CDCP(in_channel=self.out_channel),
                                    CDCP(in_channel=self.out_channel))
        #################################################
        self.SGCA = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            SIGM(self.out_channel),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )

        self.dropout = nn.Dropout(p=0.3)

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
        indices = indices[:, :int(N*self.sr)] #indices[32,1000]剪枝剪掉一半
        with torch.no_grad():
            y_out = torch.gather(y, dim=-1, index=indices) #y_out 剪枝后保留的标签[32,1000]
            w_out = torch.gather(weights, dim=-1, index=indices) #w_out 剪枝后保留的w0[32,1000]
        indices = indices.view(B, 1, -1, 1) #indices[32,1,1000,1]

        if predict == False:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4)) #x_out 剪枝后保留的x[32,1,1000,4]
            return x_out, y_out, w_out
        else:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4)) #x_out 剪枝后保留的x[32,1,500,4]
            feature_out = torch.gather(features, dim=2, index=indices.repeat(1, 128, 1, 1)) #feature_out 剪枝后保留的features[32,128,500,1]
            return x_out, y_out, w_out, feature_out

    def forward(self, x, y):
        # x[32,1,2000,4],y[32,2000]
        # x_[32,1,1000,6],y1[32,1000]
        B, _, N , _ = x.size()
        out = x.transpose(1, 3).contiguous() #contiguous断开out与x的依赖关系。out[32,4或6,2000,1]
        out = self.conv(out) #out[32,128,2000,1]

        out = self.LSGFE(out) #out[32,128,2000,1] [32,128,1000,1]
        
        ######################################
        coords = x[:, 0, :, :4]  # [B, N, 4] 原始坐标
        idx = knn(out.squeeze(-1), k=self.k_num)  # [B, N, k] KNN索引
        out = self.LSGFE_LSGA_SGEA(out, coords, idx)
        ######################################

        out = self.LSGFE_LSGA_SSCM(out)
        out = self.dropout(out)
        w0 = self.linear_0(out).view(B, -1) #w0[32,2000]

        #############################
        out = out + self.SGCA_SIGM(out)
        #############################
        out_g = self.SGCA(out) #out_g[32,128,2000,1]
        out_g = self.dropout(out_g)
        out = out_g + out

        out = self.embed_1(out)
        w1 = self.linear_1(out).view(B, -1) #w1[32,2000]

        if self.predict == False: #剪枝，不预测
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True) #w1排序,w1_ds[32,2000],indices[32,2000]是索引
            w1_ds = w1_ds[:, :int(N*self.sr)] #w1_ds[32,1000]剪枝？剪掉一半 self.sr=0.5
            x_ds, y_ds, w0_ds = self.down_sampling(x, y, w0, indices, None, self.predict)
            #x_ds[32,1,1000,4],y_ds[32,1000],w0_ds[32,1000],ds：剪枝后？
            return x_ds, y_ds, [w0, w1], [w0_ds, w1_ds]
        else: #剪枝，出预测结果
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True) #w1排序,w1_ds[32,1000],indices[32,1000]是索引
            w1_ds = w1_ds[:, :int(N*self.sr)] #w1_ds[32,500]剪枝？剪掉一半 self.sr=0.5
            x_ds, y_ds, w0_ds, out = self.down_sampling(x, y, w0, indices, out, self.predict)
            # x_ds[32,1,500,4],y_ds[32,500],w0_ds[32,500],out[32,128,500,1]也是剪枝后,ds：剪枝后？
            out = self.embed_2(out)
            w2 = self.linear_2(out) #[32,2,500,1]
            e_hat = weighted_8points(x_ds, w2)

            return x_ds, y_ds, [w0, w1, w2[:, 0, :, 0]], [w0_ds, w1_ds], e_hat

class SFMambaNet_v7_v1(nn.Module):
    def __init__(self, config):
        super(SFMambaNet_v7_v1, self).__init__()

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

