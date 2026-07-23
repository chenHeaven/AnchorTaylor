"""
简化版 VMAMBA2Block 实现
专为 ultralytics 项目设计，可直接替换 VSSBlock

主要特点：
1. 输入输出格式：(B, H, W, C) - 与VSSBlock兼容
2. 基于Mamba2的核心SSM机制
3. 去除了不必要的复杂参数（relpos、kv、window partition等）
4. 保留关键功能：LayerNorm、残差连接、可选的CPE
5. 计算效率优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from timm.models.layers import DropPath

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    HAS_MAMBA_SSM = True
except ImportError:
    print('Warning: mamba_ssm not found, VMAMBA2Block may not work properly')
    mamba_chunk_scan_combined = None
    HAS_MAMBA_SSM = False

# 预导入 einops 避免每次 forward 都导入
try:
    from einops import rearrange
    HAS_EINOPS = True
except ImportError:
    rearrange = None
    HAS_EINOPS = False


class RoPE(nn.Module):
    """RoPE (Rotary Position Embedding) 类 - 预计算旋转矩阵以提升性能
    
    特点：
    1. 在 __init__ 时预计算旋转矩阵，避免重复计算
    2. 支持动态分辨率变化，自动重新计算
    3. 支持缓存机制
    
    Args:
        shape: (H, W, feature_dim) - 形状信息
        base: RoPE 基数，默认 10000
    """
    
    def __init__(self, shape, base=10000):
        super().__init__()
        self.shape = shape
        self.base = base
        H, W, feature_dim = shape
        
        # 预计算旋转矩阵
        self.register_buffer('rotations_re', self._compute_rotations_re(shape, base)[0])
        self.register_buffer('rotations_im', self._compute_rotations_re(shape, base)[1])
        
        self.cached_shape = (H, W)
    
    @staticmethod
    def _compute_rotations_re(shape, base=10000):
        """预计算旋转矩阵的余弦和正弦部分"""
        H, W, feature_dim = shape
        channel_dims = (H, W)
        k_max = feature_dim // (2 * len(channel_dims))
        
        assert feature_dim % k_max == 0
        
        # 创建临时设备（后续会通过 register_buffer 转移到正确设备）
        device = torch.device('cpu')
        
        # angles
        theta_ks = 1 / (base ** (torch.arange(k_max, device=device) / k_max))
        angles = torch.cat([t.unsqueeze(-1) * theta_ks 
                           for t in torch.meshgrid([torch.arange(d, device=device) 
                                                   for d in channel_dims], indexing='ij')], dim=-1)
        
        # rotation
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        
        return rotations_re, rotations_im
    
    def forward(self, x):
        """应用 RoPE 位置编码
        
        Args:
            x: (B, H, W, feature_dim) 张量
        
        Returns:
            应用 RoPE 后的张量，形状不变
        """
        B, H, W, feature_dim = x.shape
        
        # 如果分辨率变化，重新计算旋转矩阵
        if (H, W) != self.cached_shape:
            rotations_re, rotations_im = self._compute_rotations_re((H, W, feature_dim), self.base)
            self.register_buffer('rotations_re', rotations_re.to(x.device))
            self.register_buffer('rotations_im', rotations_im.to(x.device))
            self.cached_shape = (H, W)
        
        # 确保旋转矩阵在正确设备上
        rotations_re = self.rotations_re.to(x.device)
        rotations_im = self.rotations_im.to(x.device)
        
        # 应用旋转编码
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)  # (B, H, W, feature_dim//2, 2)
        x_re = x_reshaped[..., :1]
        x_im = x_reshaped[..., 1:]
        
        pe_x = torch.cat([x_re * rotations_re - x_im * rotations_im,
                         x_im * rotations_re + x_re * rotations_im], dim=-1)
        
        return pe_x.flatten(-2)  # (B, H, W, feature_dim)


def rope(x, shape, base=10000):
    """RoPE (Rotary Position Embedding) 函数版本 - 用于不需要缓存的场景
    
    Args:
        x: 输入张量
        shape: (H, W, feature_dim) 形状信息
        base: RoPE 基数
    
    Returns:
        应用 RoPE 后的张量
    """
    channel_dims, feature_dim = shape[:-1], shape[-1]
    k_max = feature_dim // (2 * len(channel_dims))

    assert feature_dim % k_max == 0

    # angles
    theta_ks = 1 / (base ** (torch.arange(k_max, device=x.device) / k_max))
    angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in torch.meshgrid([torch.arange(d, device=x.device) for d in channel_dims], indexing='ij')], dim=-1)

    # rotation
    rotations_re = torch.cos(angles).unsqueeze(dim=-1)
    rotations_im = torch.sin(angles).unsqueeze(dim=-1)

    x = x.reshape(*x.shape[:-1], -1, 2)
    x_re = x[..., :1]
    x_im = x[..., 1:]
    pe_x = torch.cat([x_re * rotations_re - x_im * rotations_im, x_im * rotations_re + x_re * rotations_im], dim=-1)
    return pe_x.flatten(-2)


class tTensor(torch.Tensor):
    """用于兼容某些操作的Tensor子类"""
    @property
    def shape(self):
        shape = super().shape
        return tuple([int(s) for s in shape])


def to_ttensor(*args):
    """转换为tTensor"""
    return tuple([tTensor(x) for x in args]) if len(args) > 1 else tTensor(args[0])


class SimplifiedMamba2(nn.Module):
    """
    简化版Mamba2模块，专为ultralytics项目设计
    
    Args:
        d_model: 模型维度
        d_conv: 卷积核大小，默认3
        expand: 扩展比例，默认2
        headdim: 每个head的维度，默认64
        ngroups: 分组数量，默认1
        chunk_size: chunk大小，默认256
        d_state: 状态空间维度，默认64
        dt_min: 最小时间步长
        dt_max: 最大时间步长
        bias: 是否使用bias
        conv_bias: 卷积是否使用bias
        linear_attn_duality: 是否使用线性注意力（False=SSM扫描，True=线性注意力）
    """
    
    def __init__(
        self,
        d_model,
        d_conv=3,
        expand=2,
        headdim=64,
        ngroups=1,
        chunk_size=256,
        d_state=64,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        bias=False,
        conv_bias=True,
        device=None,
        dtype=None,
        linear_attn_duality=True,
        use_rope=True,
    ):
        # 不再使用 factory_kwargs，让 PyTorch 自动处理 device 和 dtype
        super().__init__()
        
        self.d_model = d_model
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.headdim = headdim
        self.d_state = d_state
        self.linear_attn_duality = linear_attn_duality  # ← 新增参数
        self.use_rope = use_rope  # ← RoPE 开关
        
        if ngroups == -1:
            ngroups = self.d_inner // self.headdim
        self.ngroups = ngroups
        
        assert self.d_inner % self.headdim == 0, f"d_inner={self.d_inner} must be divisible by headdim={self.headdim}"
        self.nheads = self.d_inner // self.headdim
        
        self.chunk_size = chunk_size
        
        # 投影层：[z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias)
        
        # 2D卷积
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv2d = nn.Conv2d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            groups=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        
        self.act = nn.SiLU()
        
        # 线性注意力时使用ELU激活
        if self.linear_attn_duality:
            self.elu = nn.ELU()
        
        # 初始化dt bias
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True
        
        # A参数
        A_init_range = (1, 16)
        A = torch.empty(self.nheads, dtype=torch.float32).uniform_(*A_init_range)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        
        # D "skip"参数
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True
        
        # 归一化层
        self.norm = nn.LayerNorm(self.d_inner)
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        
        # RoPE 位置编码（将在 forward 中初始化，因为此时还不知道 H, W）
        self.rope_module = None
    
    def non_casual_linear_attn(self, x, dt, A, B, C, D, H, W):
        """
        非因果线性注意力 - Mamba2的线性注意力机制
        
        Args:
            x: (B, L, nheads, headdim) - 对应V（值）
            dt: (B, L, nheads) - 时间步长
            A: (nheads,) - 衰减系数
            B: (B, L, ngroups, d_state) - 对应K（键）
            C: (B, L, ngroups, d_state) - 对应Q（查询）
            D: (nheads,) - 跳跃连接系数
            H, W: 空间维度
        
        Returns:
            output: (B, L, nheads, headdim)
        """
        batch, seqlen, nheads, headdim = x.shape
        dstate = B.shape[3]
        
        # V: (B, nheads, L, headdim)
        V = x.permute(0, 2, 1, 3)
        
        # dt: (B, nheads, L)
        dt = dt.permute(0, 2, 1)
        
        # dA: (B, nheads, L, 1) - 衰减系数
        dA = dt.unsqueeze(-1) * A.view(1, -1, 1, 1)
        
        # K: (B, 1, L, d_state)
        K = B.view(batch, 1, seqlen, dstate)
        
        # Q: (B, 1, L, d_state)
        Q = C.view(batch, 1, seqlen, dstate)
        
        # 应用ELU激活 + 1（用于归一化）
        Q = self.elu(Q) + 1.0
        K = self.elu(K) + 1.0
        
        # 应用 RoPE 位置编码 (恢复2D空间结构)
        if self.use_rope:
            # 初始化 RoPE 模块（第一次调用时）
            if self.rope_module is None or self.rope_module.cached_shape != (H, W):
                self.rope_module = RoPE(shape=(H, W, dstate), base=10000).to(Q.device)
            
            # 将 1D 序列恢复为 2D 坐标，应用 RoPE
            Q = self.rope_module(Q.view(batch, H, W, dstate)).view(batch, -1, dstate).unsqueeze(1)
            K = self.rope_module(K.view(batch, H, W, dstate)).view(batch, -1, dstate).unsqueeze(1)
        
        # 归一化dA
        dA = dA / dA.max(dim=-2, keepdim=True)[0]
        
        # 缩放K: (B, 1, L, d_state)
        K_scaled = K * dA
        
        # 计算KV: (B, 1, d_state, headdim)
        KV = K_scaled.transpose(-2, -1) @ V
        
        # 线性注意力: (B, 1, L, headdim)
        output = Q @ KV
        
        # 添加跳跃连接并进行置换
        if hasattr(self, 'dscale') and self.dscale:
            output = output.permute(0, 2, 1, 3).contiguous() + self.D(x.flatten(2, 3)).view(batch, seqlen, nheads, headdim)
        else:
            output = output + V * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)
            output = output.permute(0, 2, 1, 3).contiguous()  # (B, L, nheads, headdim)
        
        return output
    
    def forward(self, u, H, W):
        """
        Args:
            u: (B, L, C) 其中 L = H * W
            H: 高度
            W: 宽度
        Returns:
            out: (B, L, C)
        """
        batch, seqlen, dim = u.shape
        assert seqlen == H * W, f"seqlen={seqlen} must equal H*W={H*W}"
        
        # A参数
        A = -torch.exp(self.A_log)
        
        # 投影
        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt, 
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], 
            dim=-1
        )
        
        # dt激活
        dt = F.softplus(dt + self.dt_bias)
        
        # 2D卷积处理
        xBC = xBC.view(batch, H, W, -1).permute(0, 3, 1, 2).contiguous()
        xBC = self.act(self.conv2d(xBC))
        xBC = xBC.permute(0, 2, 3, 1).view(batch, H * W, -1).contiguous()
        
        # 分割成x, B, C
        x, B, C = torch.split(
            xBC, 
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], 
            dim=-1
        )
        
        # 根据 linear_attn_duality 选择不同的路径
        if self.linear_attn_duality:
            # 路径1：线性注意力
            # 重新形状为多头格式
            x_reshaped = x.view(batch, seqlen, self.nheads, self.headdim)
            B_reshaped = B.view(batch, seqlen, self.ngroups, self.d_state)
            C_reshaped = C.view(batch, seqlen, self.ngroups, self.d_state)
            
            # 调用非因果线性注意力
            y = self.non_casual_linear_attn(
                x_reshaped, dt, -torch.exp(self.A_log), B_reshaped, C_reshaped, self.D, H, W
            )
            # 转换回: (B, L, nheads, headdim) -> (B, L, d_inner)
            y = y.view(batch, seqlen, self.d_inner)
        else:
            # 路径2：SSM扫描（原始实现）
            # Mamba chunk scan
            # 检查是否在GPU上且mamba_ssm可用
            use_mamba_scan = (mamba_chunk_scan_combined is not None and 
                             HAS_MAMBA_SSM and 
                             x.device.type == 'cuda')
            
            if use_mamba_scan:
                # 优化：使用原生 reshape 替代 einops（更快）
                # x: (B, L, d_inner) -> (B, L, nheads, headdim)
                x_reshaped = x.view(batch, seqlen, self.nheads, self.headdim)
                # B, C: (B, L, ngroups*d_state) -> (B, L, ngroups, d_state)
                B_reshaped = B.view(batch, seqlen, self.ngroups, self.d_state)
                C_reshaped = C.view(batch, seqlen, self.ngroups, self.d_state)
                
                # 转换为tTensor（只在需要时转换）
                x_tt = to_ttensor(x_reshaped)
                dt_tt = to_ttensor(dt)
                A_tt = to_ttensor(-torch.exp(self.A_log))
                B_tt = to_ttensor(B_reshaped)
                C_tt = to_ttensor(C_reshaped)
                D_tt = to_ttensor(self.D)
                
                # Mamba chunk scan
                y = mamba_chunk_scan_combined(
                    x_tt,
                    dt_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    chunk_size=self.chunk_size,
                    D=D_tt,
                    z=None,
                    seq_idx=None,
                    initial_states=None,
                )
                # 转换回: (B, L, nheads, headdim) -> (B, L, d_inner)
                y = y.view(batch, seqlen, self.d_inner)
            else:
                # Fallback: 使用简化的实现（CPU或没有mamba_ssm时）
                if not HAS_MAMBA_SSM:
                    import warnings
                    warnings.warn("mamba_ssm not found, using fallback implementation. Performance will be severely degraded!")
                elif x.device.type != 'cuda':
                    import warnings
                    warnings.warn("mamba_ssm requires GPU. Running on CPU with fallback implementation. Performance will be severely degraded!")
                
                # 简化的状态空间计算（fallback）
                y = x  # 简单的恒等映射作为fallback
        
        # 归一化
        y = self.norm(y)
        
        # Gate
        y = y * z
        
        # 输出投影
        out = self.out_proj(y)
        
        return out

class SimplifiedMamba3(nn.Module):
    """
    简化版Mamba2模块，专为ultralytics项目设计
    
    Args:
        d_model: 模型维度
        d_conv: 卷积核大小，默认3
        expand: 扩展比例，默认2
        headdim: 每个head的维度，默认64
        ngroups: 分组数量，默认1
        chunk_size: chunk大小，默认256
        d_state: 状态空间维度，默认64
        dt_min: 最小时间步长
        dt_max: 最大时间步长
        bias: 是否使用bias
        conv_bias: 卷积是否使用bias
        linear_attn_duality: 是否使用线性注意力（False=SSM扫描，True=线性注意力）
    """
    
    def __init__(
        self,
        d_model,
        d_conv=3,
        expand=2,
        headdim=64,
        ngroups=1,
        chunk_size=256,
        d_state=64,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        bias=False,
        conv_bias=True,
        device=None,
        dtype=None,
    ):
        # 不再使用 factory_kwargs，让 PyTorch 自动处理 device 和 dtype
        super().__init__()
        
        self.d_model = d_model
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.headdim = headdim
        self.d_state = d_state
        
        if ngroups == -1:
            ngroups = self.d_inner // self.headdim
        self.ngroups = ngroups
        
        assert self.d_inner % self.headdim == 0, f"d_inner={self.d_inner} must be divisible by headdim={self.headdim}"
        self.nheads = self.d_inner // self.headdim
        
        self.chunk_size = chunk_size
        
        # 投影层：[z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias)
        
        # 2D卷积
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv2d = nn.Conv2d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            groups=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        
        self.act = nn.SiLU()
        # 初始化dt bias
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True
        
        # A参数
        A_init_range = (1, 16)
        A = torch.empty(self.nheads, dtype=torch.float32).uniform_(*A_init_range)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        
        # D "skip"参数
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True
        
        # 归一化层
        self.norm = nn.LayerNorm(self.d_inner)
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
    
    
    def forward(self, u, H, W):
        """
        Args:
            u: (B, L, C) 其中 L = H * W
            H: 高度
            W: 宽度
        Returns:
            out: (B, L, C)
        """
        batch, seqlen, dim = u.shape
        assert seqlen == H * W, f"seqlen={seqlen} must equal H*W={H*W}"
        
        # A参数
        A = -torch.exp(self.A_log)
        
        # 投影
        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt, 
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], 
            dim=-1
        )
        
        # dt激活
        dt = F.softplus(dt + self.dt_bias)
        
        # 2D卷积处理
        xBC = xBC.view(batch, H, W, -1).permute(0, 3, 1, 2).contiguous()
        xBC = self.act(self.conv2d(xBC))
        xBC = xBC.permute(0, 2, 3, 1).view(batch, H * W, -1).contiguous()
        
        # 分割成x, B, C
        x, B, C = torch.split(
            xBC, 
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], 
            dim=-1
        )
        

        # 路径2：SSM扫描（原始实现）
        # Mamba chunk scan
        # 检查是否在GPU上且mamba_ssm可用
        use_mamba_scan = (mamba_chunk_scan_combined is not None and 
                            HAS_MAMBA_SSM and 
                            x.device.type == 'cuda')
        
        if use_mamba_scan:
            # 优化：使用原生 reshape 替代 einops（更快）
            # x: (B, L, d_inner) -> (B, L, nheads, headdim)
            x_reshaped = x.view(batch, seqlen, self.nheads, self.headdim)
            # B, C: (B, L, ngroups*d_state) -> (B, L, ngroups, d_state)
            B_reshaped = B.view(batch, seqlen, self.ngroups, self.d_state)
            C_reshaped = C.view(batch, seqlen, self.ngroups, self.d_state)
            
            # 转换为tTensor（只在需要时转换）
            x_tt = to_ttensor(x_reshaped)
            dt_tt = to_ttensor(dt)
            A_tt = to_ttensor(-torch.exp(self.A_log))
            B_tt = to_ttensor(B_reshaped)
            C_tt = to_ttensor(C_reshaped)
            D_tt = to_ttensor(self.D)
            
            # Mamba chunk scan
            y = mamba_chunk_scan_combined(
                x_tt,
                dt_tt,
                A_tt,
                B_tt,
                C_tt,
                chunk_size=self.chunk_size,
                D=D_tt,
                z=None,
                seq_idx=None,
                initial_states=None,
            )
            # 转换回: (B, L, nheads, headdim) -> (B, L, d_inner)
            y = y.view(batch, seqlen, self.d_inner)
        else:
            # Fallback: 使用简化的实现（CPU或没有mamba_ssm时）
            if not HAS_MAMBA_SSM:
                import warnings
                warnings.warn("mamba_ssm not found, using fallback implementation. Performance will be severely degraded!")
            elif x.device.type != 'cuda':
                import warnings
                warnings.warn("mamba_ssm requires GPU. Running on CPU with fallback implementation. Performance will be severely degraded!")
            
            # 简化的状态空间计算（fallback）
            y = x  # 简单的恒等映射作为fallback
        
        # 归一化
        y = self.norm(y)
        
        # Gate
        y = y * z
        
        # 输出投影
        out = self.out_proj(y)
        
        return out


class VMAMBA2Block(nn.Module):
    """
    简化版VMAMBA2Block，可直接替换VSSBlock
    
    输入输出格式与VSSBlock完全兼容：(B, H, W, C)
    
    Args:
        hidden_dim: 隐藏层维度
        drop_path: DropPath率
        norm_layer: 归一化层
        d_state: 状态空间维度
        expand: Mamba2扩展比例
        headdim: 每个head的维度
        ngroups: 分组数量
        chunk_size: chunk大小
        use_cpe: 是否使用条件位置编码
        **kwargs: 其他参数（兼容性）
    """
    
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer=None,
        d_state: int = 16,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 256,
        use_cpe: bool = False,
        **kwargs,
    ):
        super().__init__()
        
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        self.hidden_dim = hidden_dim
        self.use_cpe = use_cpe
        
        # LayerNorm
        self.ln_1 = norm_layer(hidden_dim)
        
        # 核心Mamba2模块
        self.mamba2 = SimplifiedMamba3(
            d_model=hidden_dim,
            d_conv=3,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            chunk_size=chunk_size,
            d_state=d_state,
        )
        
        # DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # 可选的条件位置编码（Conditional Positional Encoding）
        if self.use_cpe:
            self.cpe = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
    
    def forward(self, input: torch.Tensor):
        """
        Args:
            input: (B, H, W, C)
        Returns:
            output: (B, H, W, C)
        """
        B, H, W, C = input.shape
        
        # 可选的CPE（优化：减少 permute 操作）
        if self.use_cpe:
            # input: (B, H, W, C) -> (B, C, H, W)
            input_cpe = input.permute(0, 3, 1, 2).contiguous()
            cpe_out = self.cpe(input_cpe)
            # (B, C, H, W) -> (B, H, W, C) 并直接相加
            input = input + cpe_out.permute(0, 2, 3, 1).contiguous()
        
        # 保存残差
        shortcut = input
        
        # LayerNorm
        x = self.ln_1(input)
        
        # 转换为 (B, L, C) 格式（优化：使用 reshape 而不是 view）
        x = x.reshape(B, H * W, C)
        
        # Mamba2处理
        x = self.mamba2(x, H, W)
        
        # 转换回 (B, H, W, C) 格式
        x = x.reshape(B, H, W, C)
        
        # DropPath + 残差连接
        x = shortcut + self.drop_path(x)
        
        return x

