# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math

import cv2
import numpy as np
import torch
import torch.nn as nn
from copy import copy
from pathlib import Path
try:
    # `clip` is only required by the "world"/text-conditioning branches.
    # Keep it optional so standard detection training can run without it.
    import clip  # type: ignore
except ImportError:  # pragma: no cover
    clip = None
import numpy as np
import pandas as pd
import requests
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
from typing import Tuple
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
try:
    import selective_scan_cuda_oflex
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_oflex.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda_core
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_core.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda
except Exception as e:
    ...
# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

import matplotlib.pyplot as plt
#from scipy.misc import imread, imsave
from PIL import Image
from torch.nn import init, Sequential
from torchvision import models
import torch.nn.functional as F
import torchvision.transforms as transforms
#from .block import C3,C2f
from .wavelet import create_wavelet_filter, wavelet_transform, inverse_wavelet_transform
# from .CoE import *
from ultralytics.nn.modules.vmamba2_simplified import VMAMBA2Block

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "frequencyfuse",
    "fusehigh",
    "fuselow",
    "fusemiddle",
    "Cross_modal_Vmaba_module2",
    "ChannelAttention2",
    "SpatialAttention2",
    "CBAM3",
    "Add2",
    "Add",
    "myTransformerBlock",
    "TalorNet",
    "getreturntalorfuse",
    "getreturntalorir",
    "fusehd_fourth",
    "VMAMBA2BasicLayer",
    "getreturnir",
    "getreturnrgb",
    "getreturnfeature",
    "getreturnfeature2",
    "getreturnfuse",
    "getreturnfhigh",
    "fusehdxiaorong",
    "fusehdxiaorong2",
    "fusehdxiaorong3",
    "fusehdxiaorong4",
    "fusehdquanzhong",
    "fusehd23",
    "fusehd33",
    "fusehdwithoutlow",
    "fusehdwithoutlh",
    "fusehdwithouthl",
    "fusehdwithouthh",
)


def l1_norm(feature_map):
    """Deprecated: unused helper function"""
    pass

def softmax_fusion(feature_map_a, feature_map_b):
    """Deprecated: unused helper function"""  
    pass

class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None, 
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            #torch.arange(1, d_state + 1, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    
    #得到可以依据输入变化的参数A、B、C、D并且计算最终的输出
    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y
    


    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
        
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        #assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = y.to(x.dtype)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        d=self.ln_1(input)
        #d=d.to(torch.float32)
        c=self.self_attention(d)
        #c=c.to(torch.float32)
        y=self.drop_path(c)
        x = input+y
        return x


class VMAMBA2BasicLayer(nn.Module):
    """Placeholder for missing VMAMBA2BasicLayer implementation.

    `ultralytics/nn/modules/__init__.py` and `ultralytics/nn/tasks.py` expect
    this symbol to exist, but this repository snapshot does not define it.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "`VMAMBA2BasicLayer` is missing in this repo snapshot. "
            "If your YAML uses it, you need to implement/restore the layer."
        )

class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x):
        #print(x.shape)
        x=x.permute(0, 2, 3, 1)
        #chnnels = x.shape[-1]

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x=x.cuda()
                x=x
                x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)
        x=x.permute(0, 3, 1, 2)
        return x

def _make_vmamba_config(channels, scale):
    """Build the scale/channel-aware VMAMBA2 configuration used by fusion modules."""
    expand = 2
    d_inner = expand * channels
    preferred_headdim = min(64, max(16, channels // 2))

    if preferred_headdim >= d_inner:
        headdim = d_inner
    else:
        factors = [i for i in range(1, min(preferred_headdim, d_inner) + 1) if d_inner % i == 0]
        headdim = max(factors) if factors else 1

    if channels <= 32:
        d_state, drop_path = 16, 0.00
    elif channels <= 64:
        d_state, drop_path = 24, 0.05
    elif channels <= 128:
        d_state, drop_path = 32, 0.10
    else:
        d_state, drop_path = 48, 0.15

    chunk_size = 128 if str(scale).lower() == "p2" else 256
    return {
        "drop_path": float(drop_path),
        "d_state": int(d_state),
        "expand": expand,
        "headdim": int(headdim),
        "ngroups": 1,
        "chunk_size": int(chunk_size),
        "use_cpe": True,
    }


class _TaylorDerivativeBlock(nn.Module):
    """Approximate one Taylor derivative term without channel grouping."""

    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )

    def forward(self, term, ir_condition, rgb_condition):
        return self.net(torch.cat((term, ir_condition, rgb_condition), dim=1))


class TalorNet(nn.Module):
    """IR-primary RGB-corrected Taylor fusion in the spatial feature domain.

    IR builds the zero-order term ``g_0``. RGB is used as conditional
    information by a derivative network shared across Taylor orders:

        g_{k+1} = G([g_k, IR, RGB]) + k * g_k
        output = g_0 + sum(g_k / k!), k=1..order

    The module does not use wavelet decomposition or channel grouping. It
    returns ``[fused, ir_zero]`` at the original spatial resolution so the main
    and auxiliary detection branches can be supervised independently.

    Args:
        c1 (int): Input channels for each modality.
        c2 (int): Output channels of the fused feature.
        scale (str): Feature scale used to configure the optional IR VMamba block.
        order (int): Taylor approximation order. Must be at least 1.
        use_vmamba (bool): Use VMAMBA2Block to refine the IR zero-order term.
    """

    def __init__(self, c1, c2, scale="p2", order=3, use_vmamba=True):
        super().__init__()
        if order < 1:
            raise ValueError(f"TalorNet order must be >= 1, but got {order}")

        self.c1 = c1
        self.c2 = c2
        self.scale = str(scale).lower()
        self.order = int(order)
        self.use_vmamba = bool(use_vmamba)

        self.ir_mapping = self._make_mapping(c1, c2)
        self.ir_context = (
            VMAMBA2Block(hidden_dim=c2, **_make_vmamba_config(c2, self.scale))
            if self.use_vmamba
            else nn.Identity()
        )
        self.rgb_condition = self._make_mapping(c1, c2)
        self.derivative = _TaylorDerivativeBlock(c2)

    @staticmethod
    def _make_mapping(c1, c2):
        return nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def _unfold_taylor(self, xir, xrgb):
        """Build the IR zero-order term and progressively add RGB corrections."""
        ir_condition = self.ir_mapping(xir)
        if self.use_vmamba:
            # VMAMBA2Block consumes NHWC and already contains its own residual.
            ir_condition = self.ir_context(ir_condition.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        rgb_condition = self.rgb_condition(xrgb)

        term = ir_condition  # g_0: the primary representation is IR-only.
        output = term
        factorial = 1
        for k in range(self.order):
            term = self.derivative(term, ir_condition, rgb_condition) + k * term
            factorial *= k + 1
            output = output + term / factorial
        return output, ir_condition

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("TalorNet expects [RGB_features, IR_features]")

        xrgb, xir = x
        if xrgb.shape != xir.shape:
            raise ValueError(
                f"TalorNet requires aligned RGB/IR shapes, got {tuple(xrgb.shape)} and {tuple(xir.shape)}"
            )

        fused, ir_zero = self._unfold_taylor(xir, xrgb)
        return [fused, ir_zero]


class getreturntalorfuse(nn.Module):
    """Select the Taylor-fused feature from ``TalorNet`` output."""

    def __init__(self, c1, c2):
        super().__init__()

    def forward(self, x):
        return x[0]


class getreturntalorir(nn.Module):
    """Select the IR zero-order feature from ``TalorNet`` output."""

    def __init__(self, c1, c2):
        super().__init__()

    def forward(self, x):
        return x[1]



class fusehd_fourth(nn.Module):
    def __init__(self, c1, c2, scale='p2', use_se=False, use_cross_group=True, depth=1):
        super().__init__()
        self.scale = scale.lower()
        self.use_se = use_se
        self.use_cross_group = use_cross_group
        self.depth = max(1, depth)
        
        # 根据scale确定分组数和卷积核大小
        scale_config = {
            'p2': (4, [3, 3, 5, 7]),
            'p3': (2, [3, 5])
        }
        self.num_groups, self.kernel_sizes = scale_config.get(self.scale, (1, [3]))
        
        assert c1 % self.num_groups == 0, f"c1={c1} must be divisible by num_groups={self.num_groups}"
        self.channels_per_group = c1 // self.num_groups
        
        # 创建小波滤波器（按 depth x group）
        self.wt_filters = nn.ParameterList()
        self.iwt_filters = nn.ParameterList()
        for _ in range(self.depth):
            for _g in range(self.num_groups):
                wt_filter, _ = create_wavelet_filter("haar", self.channels_per_group,
                                                     self.channels_per_group, torch.float)
                self.wt_filters.append(nn.Parameter(wt_filter, requires_grad=False))
            _, iwt_filter = create_wavelet_filter("haar", self.channels_per_group,
                                                  self.channels_per_group, torch.float)
            self.iwt_filters.append(nn.Parameter(iwt_filter, requires_grad=False))
        
        # 初始化模块列表
        self.group_convs_rgb = nn.ModuleList()
        self.group_convs_ir = nn.ModuleList()
        self.group_fusion_fc = nn.ModuleList()
        self.low_fuse_proj = nn.ModuleList()
        self.vss_blocks = nn.ModuleList()
        self.attn_high1 = nn.ModuleList()
        self.attn_high2 = nn.ModuleList()
        self.attn_high3 = nn.ModuleList()
        
        if self.use_se:
            self.se_modules = nn.ModuleList()
        else:
            self.eca_modules = nn.ModuleList()
        
        if self.use_cross_group and self.num_groups > 1:
            if self.depth > 1:
                self.cross_group_convs = nn.ModuleList()
                self.cross_group_norms = nn.ModuleList()
                self.cross_group_acts = nn.ModuleList()
        
        # 根据depth初始化模块
        num_levels = 1 if self.depth == 1 else self.depth
        for d in range(num_levels):
            self._init_level_modules(c1)

    def _create_attention_module(self):
        """创建高频注意力模块"""
        return nn.Sequential(
            nn.Conv2d(self.channels_per_group * 2, self.channels_per_group // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels_per_group // 4, 2, 1),
            nn.Softmax(dim=1)
        )
    
    def _create_se_module(self):
        """创建SE通道注意力模块"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channels_per_group, self.channels_per_group // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels_per_group // 4, self.channels_per_group, 1),
            nn.Sigmoid()
        )

    def _create_eca_module(self):
        """创建ECA通道注意力模块（use_se=False时使用）"""
        return ECAAttention(self.channels_per_group)
    
    def _get_vmamba_config(self):
        """获取VMAMBA2Block配置"""
        return _make_vmamba_config(self.channels_per_group, self.scale)
    
    def _init_level_modules(self, c1):
        """初始化单层级模块"""
        # 分组卷积
        rgb_convs, ir_convs, fusion_fcs = [], [], []
        for k in self.kernel_sizes:
            padding = k // 2
            rgb_convs.append(nn.Conv2d(self.channels_per_group, self.channels_per_group,
                                      kernel_size=k, padding=padding, bias=True))
            ir_convs.append(nn.Conv2d(self.channels_per_group, self.channels_per_group,
                                     kernel_size=k, padding=padding, bias=True))
            fusion_fcs.append(nn.Conv2d(self.channels_per_group, self.channels_per_group,
                                       kernel_size=1, bias=True))
        
        if self.depth == 1:
            self.group_convs_rgb = nn.ModuleList(rgb_convs)
            self.group_convs_ir = nn.ModuleList(ir_convs)
            self.group_fusion_fc = nn.ModuleList(fusion_fcs)
            # self.low_fuse_proj = nn.ModuleList([
            #     nn.Conv2d(self.channels_per_group * 2, self.channels_per_group, kernel_size=1, bias=False)
            #     for _ in range(self.num_groups)
            # ])
        else:
            self.group_convs_rgb.append(nn.ModuleList(rgb_convs))
            self.group_convs_ir.append(nn.ModuleList(ir_convs))
            self.group_fusion_fc.append(nn.ModuleList(fusion_fcs))
            # self.low_fuse_proj.append(nn.ModuleList([
            #     nn.Conv2d(self.channels_per_group * 2, self.channels_per_group, kernel_size=1, bias=False)
            #     for _ in range(self.num_groups)
            # ]))
        
        # VMAMBA2Block
        vmamba_config = self._get_vmamba_config()
        vss_blocks = nn.ModuleList([
            VMAMBA2Block(hidden_dim=self.channels_per_group, **vmamba_config)
            for _ in range(self.num_groups)
        ])
        # vss_blocks = nn.ModuleList([
        #     VSSBlock(hidden_dim=self.channels_per_group, drop_path=0.0, d_state=16)
        #     for _ in range(self.num_groups)
        # ])
        if self.depth == 1:
            self.vss_blocks = vss_blocks
            # pass
        else:
            self.vss_blocks.append(vss_blocks)
        
        # 高频注意力模块
        attn_modules = [nn.ModuleList([self._create_attention_module() 
                                       for _ in range(self.num_groups)]) 
                        for _ in range(3)]
        
        if self.depth == 1:
            self.attn_high1, self.attn_high2, self.attn_high3 = attn_modules
        else:
            self.attn_high1.append(attn_modules[0])
            self.attn_high2.append(attn_modules[1])
            self.attn_high3.append(attn_modules[2])
        
        # SE模块
        if self.use_se:
            se_modules = nn.ModuleList([self._create_se_module() for _ in range(self.num_groups)])
            if self.depth == 1:
                self.se_modules = se_modules
            else:
                self.se_modules.append(se_modules)
        else:
            eca_modules = nn.ModuleList([self._create_eca_module() for _ in range(self.num_groups)])
            if self.depth == 1:
                self.eca_modules = eca_modules
            else:
                self.eca_modules.append(eca_modules)
        
        # 跨组交互（使用全通道1x1卷积，实现真正的跨组信息交换）
        if self.use_cross_group and self.num_groups > 1:
            if self.depth == 1:
                self.cross_group_conv = nn.Conv2d(c1, c1, kernel_size=1, bias=False)
                self.cross_group_norm = nn.BatchNorm2d(c1)
                self.cross_group_act = nn.ReLU(inplace=True)
            else:
                self.cross_group_convs.append(nn.Conv2d(c1, c1, kernel_size=1, bias=False))
                self.cross_group_norms.append(nn.BatchNorm2d(c1))
                self.cross_group_acts.append(nn.ReLU(inplace=True))

    def _get_modules(self, depth_idx):
        """获取指定层级的模块"""
        if self.depth == 1:
            return (
                self.group_convs_rgb,
                self.group_convs_ir,
                self.group_fusion_fc,
                self.vss_blocks,
                self.attn_high1,
                self.attn_high2,
                self.attn_high3,
                self.se_modules if self.use_se else None,
                self.eca_modules if not self.use_se else None
            )
        else:
            return (
                self.group_convs_rgb[depth_idx],
                self.group_convs_ir[depth_idx],
                self.group_fusion_fc[depth_idx],
                self.vss_blocks[depth_idx],
                self.attn_high1[depth_idx],
                self.attn_high2[depth_idx],
                self.attn_high3[depth_idx],
                self.se_modules[depth_idx] if self.use_se else None,
                self.eca_modules[depth_idx] if not self.use_se else None
            )

    def _process_single_level(self, xrgb, xir, depth_idx):
        """
        Process a single depth level.
        
        Args:
            xrgb: RGB channel input
            xir: IR channel input
            depth_idx: current depth level index
            
        Returns:
            tuple: (xlow_fuse, xhigh_fuse)
        """
        # 获取当前层级的模块和滤波器
        conv_rgb, conv_ir, fusion_fc, vss_blocks, attn_h1, attn_h2, attn_h3, se_modules, eca_modules = self._get_modules(depth_idx)
        level_idx = depth_idx if self.depth > 1 else 0
        
        # 分组处理
        xrgb_groups = torch.split(xrgb, self.channels_per_group, dim=1)
        xir_groups = torch.split(xir, self.channels_per_group, dim=1)
        
        fused_low, fused_high1, fused_high2, fused_high3 = [], [], [], []
        
        for i in range(self.num_groups):
            # 小波变换
            wt_idx = level_idx * self.num_groups + i
            wt_fn = partial(wavelet_transform, filters=self.wt_filters[wt_idx])
            xrgb_wt = wt_fn(xrgb_groups[i])
            xir_wt = wt_fn(xir_groups[i])
            
            xrgb_low, xrgb_h1, xrgb_h2, xrgb_h3 = [xrgb_wt[:, :, j, :, :] for j in range(4)]
            xir_low, xir_h1, xir_h2, xir_h3 = [xir_wt[:, :, j, :, :] for j in range(4)]
            
            # 低频融合
            xrgb_low_conv, xir_low_conv = conv_rgb[i](xrgb_low), conv_ir[i](xir_low)
            alpha = torch.sigmoid(fusion_fc[i](xrgb_low_conv + xir_low_conv))
            low_fused = xrgb_low_conv * alpha + xir_low_conv * (1 - alpha)
            # low_fused = torch.cat([xrgb_low, xir_low], dim=1)
            # if self.depth == 1:
            #     low_fused = self.low_fuse_proj[i](low_fused)
            # else:
            #     low_fused = self.low_fuse_proj[depth_idx][i](low_fused)
            # SE注意力
            if se_modules is not None:
                low_fused = low_fused * se_modules[i](low_fused)
            elif eca_modules is not None:
                low_fused = eca_modules[i](low_fused)
            
            # VMAMBA2处理 + 残差连接
            low_fused_residual = low_fused
            low_fused = vss_blocks[i](low_fused.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) + low_fused_residual
            fused_low.append(low_fused)
            
            # 高频融合
            for attn_module, rgb_high, ir_high, fused_list in [
                (attn_h1[i], xrgb_h1, xir_h1, fused_high1),
                (attn_h2[i], xrgb_h2, xir_h2, fused_high2),
                (attn_h3[i], xrgb_h3, xir_h3, fused_high3)
            ]:
                concat = torch.cat([rgb_high, ir_high], dim=1)
                weights = attn_module(concat)
                fused_list.append(weights[:, 0:1] * rgb_high + weights[:, 1:2] * ir_high)
        
        # 拼接所有组
        xlow_fuse = torch.cat(fused_low, dim=1)
        
        # 跨组信息交互
        if self.use_cross_group and self.num_groups > 1:
            if self.depth == 1:
                xlow_fuse = self.cross_group_act(self.cross_group_norm(self.cross_group_conv(xlow_fuse)))
            else:
                xlow_fuse = self.cross_group_acts[depth_idx](
                    self.cross_group_norms[depth_idx](self.cross_group_convs[depth_idx](xlow_fuse)))
        
        # 拼接高频组件
        xhigh_fuse = torch.stack([torch.cat(fused_high1, dim=1), 
                                  torch.cat(fused_high2, dim=1),
                                  torch.cat(fused_high3, dim=1)], dim=2)
        
        return xlow_fuse, xhigh_fuse

    def forward(self, x):
        """
        Forward pass with multi-level recursive loop.
        
        Args:
            x: input list [xrgb, xir]
            
        Returns:
            [xlow_fuse, xhigh_fuse] where:
            - xlow_fuse: fused low-frequency features
            - xhigh_fuse: stacked high-frequency components (shape: [B, C, 3, H, W] for last level)
        """
        xrgb, xir = x[0], x[1]
        
        if self.depth == 1:
            return self._process_single_level(xrgb, xir, 0)
        
        # Multi-level processing (depth >= 2)
        for d in range(self.depth):
            xlow_fuse, xhigh_fuse = self._process_single_level(xrgb, xir, d)
            
            if d < self.depth - 1:
                # 非最后一层：使用逆小波变换合并低频和高频
                xfused = torch.cat([xlow_fuse.unsqueeze(2), xhigh_fuse], dim=2)
                
                # 分组处理逆小波变换
                xmerged_groups = []
                for i in range(self.num_groups):
                    start_ch = i * self.channels_per_group
                    end_ch = start_ch + self.channels_per_group
                    
                    xfused_group = xfused[:, start_ch:end_ch, :, :, :]
                    b, c_g, _, h_half, w_half = xfused_group.shape
                    xfused_reshaped = xfused_group.reshape(b, c_g * 4, h_half, w_half)
                    
                    pad = (self.iwt_filters[d].shape[2] // 2 - 1, self.iwt_filters[d].shape[3] // 2 - 1)
                    xmerged = F.conv_transpose2d(xfused_reshaped, self.iwt_filters[d], 
                                                stride=2, groups=c_g, padding=pad)
                    xmerged_groups.append(xmerged)
                
                xmerged = torch.cat(xmerged_groups, dim=1)
                xrgb = xir = xmerged
        
        return [xlow_fuse, xhigh_fuse]



class getreturnir(nn.Module):
      def __init__(self,c1,c2):
        super().__init__()
      def forward(self,x):
        #print(x[0].shape)
        #print("3")
        return x[0]
class getreturnrgb(nn.Module):
      def __init__(self,c1,c2):
        super().__init__()
      def forward(self,x):
        return x[1]
class getreturnfuse(nn.Module):
      def __init__(self,c1,c2):
        super().__init__()
        #self.upnear = nn.Upsample(scale_factor=2)
        #self.conv1=Conv(c1,c2,1,1)
      def forward(self,x):
        #y=torch.cat([x[0],x[1]], 1)
        #y=self.conv1(y)
        #print(y.shape)
        #print(self.upnear(y).shape)
        y=x[0]+x[1]
        return y
class getreturnfhigh(nn.Module):
      def __init__(self,c1,c2):
        super().__init__()
        #self.upnear = nn.Upsample(scale_factor=2)
      def forward(self,x):
        x1=torch.squeeze(x[2][:,:,1,:,:],2)
        x2=torch.squeeze(x[2][:,:,2,:,:],2)
        x3=torch.squeeze(x[2][:,:,3,:,:],2)
        x4=torch.squeeze(x[3][:,:,1,:,:],2)
        x5=torch.squeeze(x[3][:,:,2,:,:],2)
        x6=torch.squeeze(x[3][:,:,3,:,:],2)
        #print(self.upnear(y).shape)
        return [x1,x2,x3,x4,x5,x6]
class getreturnfeature(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.wt_filter, self.iwt_filter = create_wavelet_filter("haar", c1, c1, torch.float)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

    def forward(self, x):
        # x = [xlow_fuse, xhigh_fuse] where xhigh_fuse has 3 subbands on dim=2
        xlow = torch.unsqueeze(x[0], 2)
        xfuse = torch.cat([xlow, x[1]], dim=2)
        return self.iwt_function(xfuse)
class getreturnfeature2(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.wt_filter, self.iwt_filter = create_wavelet_filter("haar", c1, c1, torch.float)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

    def forward(self, x):
        # x = [xlow, fusehd_out] after fusehd returns [xlow_fuse, xhigh_fuse]
        xlow_in = x[0]
        fuse_out = x[1]

        # 兼容上游可能返回的 (tensor, tensor) / [tensor, tensor] 形式
        if isinstance(xlow_in, (tuple, list)):
            xlow_in = xlow_in[0]
        if isinstance(fuse_out, (tuple, list)) and len(fuse_out) > 0 and isinstance(fuse_out[0], (tuple, list)):
            fuse_out = fuse_out[0]

        # 低频：仅使用输入特征（避免重复融合）
        xlow = torch.unsqueeze(xlow_in, 2)
        # 高频：从fusehd提取高频细节信息
        xhigh = fuse_out[1]
        if isinstance(xhigh, (tuple, list)):
            xhigh = xhigh[0]

        xfuse = torch.cat([xlow, xhigh], dim=2)
        return self.iwt_function(xfuse)


class spatial_fuse_ncssd(nn.Module):
    """Spatial-domain RGB-IR fusion with grouped NC-SSD, without DWT/IWT."""

    def __init__(self, c1, c2, scale='p2', use_se=False, use_cross_group=True, depth=1):
        super().__init__()
        self.scale = scale.lower()
        self.use_se = use_se
        self.use_cross_group = use_cross_group
        self.depth = max(1, depth)

        scale_config = {
            'p2': (4, [3, 3, 5, 7]),
            'p3': (2, [3, 5]),
        }
        self.num_groups, self.kernel_sizes = scale_config.get(self.scale, (1, [3]))
        assert c1 % self.num_groups == 0, f"c1={c1} must be divisible by num_groups={self.num_groups}"
        self.channels_per_group = c1 // self.num_groups

        self.group_convs_rgb = nn.ModuleList()
        self.group_convs_ir = nn.ModuleList()
        self.group_fusion_fc = nn.ModuleList()
        self.vss_blocks = nn.ModuleList()

        if self.use_se:
            self.se_modules = nn.ModuleList()
        else:
            self.eca_modules = nn.ModuleList()

        for _ in range(self.depth):
            self._init_level_modules(c1)

        if c1 != c2:
            self.out_proj = nn.Sequential(
                nn.Conv2d(c1, c2, kernel_size=1, bias=False),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
            )
        else:
            self.out_proj = nn.Identity()

    def _create_se_module(self):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channels_per_group, self.channels_per_group // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels_per_group // 4, self.channels_per_group, 1),
            nn.Sigmoid()
        )

    def _create_eca_module(self):
        return ECAAttention(self.channels_per_group)

    def _get_vmamba_config(self):
        expand = 2
        d_inner = expand * self.channels_per_group
        preferred_headdim = min(64, max(16, self.channels_per_group // 2))

        if preferred_headdim >= d_inner:
            headdim = d_inner
        else:
            factors = [i for i in range(1, min(preferred_headdim, d_inner) + 1) if d_inner % i == 0]
            headdim = max(factors) if factors else 1

        cpg = int(self.channels_per_group)
        if cpg <= 32:
            d_state, drop_path = 16, 0.00
        elif cpg <= 64:
            d_state, drop_path = 24, 0.05
        elif cpg <= 128:
            d_state, drop_path = 32, 0.10
        else:
            d_state, drop_path = 48, 0.15

        chunk_size = 128 if self.scale in {"p2"} else 256
        return {
            'drop_path': float(drop_path),
            'd_state': int(d_state),
            'expand': expand,
            'headdim': int(headdim),
            'ngroups': 1,
            'chunk_size': int(chunk_size),
            'use_cpe': True,
        }

    def _init_level_modules(self, c1):
        rgb_convs, ir_convs, fusion_fcs = [], [], []
        for k in self.kernel_sizes:
            padding = k // 2
            rgb_convs.append(nn.Conv2d(self.channels_per_group, self.channels_per_group,
                                      kernel_size=k, padding=padding, bias=True))
            ir_convs.append(nn.Conv2d(self.channels_per_group, self.channels_per_group,
                                     kernel_size=k, padding=padding, bias=True))
            fusion_fcs.append(nn.Conv2d(self.channels_per_group, self.channels_per_group,
                                       kernel_size=1, bias=True))

        self.group_convs_rgb.append(nn.ModuleList(rgb_convs))
        self.group_convs_ir.append(nn.ModuleList(ir_convs))
        self.group_fusion_fc.append(nn.ModuleList(fusion_fcs))

        vmamba_config = self._get_vmamba_config()
        self.vss_blocks.append(nn.ModuleList([
            VMAMBA2Block(hidden_dim=self.channels_per_group, **vmamba_config)
            for _ in range(self.num_groups)
        ]))

        if self.use_se:
            self.se_modules.append(nn.ModuleList([self._create_se_module() for _ in range(self.num_groups)]))
        else:
            self.eca_modules.append(nn.ModuleList([self._create_eca_module() for _ in range(self.num_groups)]))

        if self.use_cross_group and self.num_groups > 1:
            if not hasattr(self, 'cross_group_convs'):
                self.cross_group_convs = nn.ModuleList()
                self.cross_group_norms = nn.ModuleList()
                self.cross_group_acts = nn.ModuleList()
            self.cross_group_convs.append(nn.Conv2d(c1, c1, kernel_size=1, bias=False))
            self.cross_group_norms.append(nn.BatchNorm2d(c1))
            self.cross_group_acts.append(nn.ReLU(inplace=True))

    def _process_single_level(self, xrgb, xir, depth_idx):
        xrgb_groups = torch.split(xrgb, self.channels_per_group, dim=1)
        xir_groups = torch.split(xir, self.channels_per_group, dim=1)

        fused_groups = []
        for i in range(self.num_groups):
            rgb_feat = self.group_convs_rgb[depth_idx][i](xrgb_groups[i])
            ir_feat = self.group_convs_ir[depth_idx][i](xir_groups[i])
            alpha = torch.sigmoid(self.group_fusion_fc[depth_idx][i](rgb_feat + ir_feat))
            fused = rgb_feat * alpha + ir_feat * (1 - alpha)

            if self.use_se:
                fused = fused * self.se_modules[depth_idx][i](fused)
            else:
                fused = self.eca_modules[depth_idx][i](fused)

            residual = fused
            fused = self.vss_blocks[depth_idx][i](fused.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) + residual
            fused_groups.append(fused)

        fused = torch.cat(fused_groups, dim=1)
        if self.use_cross_group and self.num_groups > 1:
            fused = self.cross_group_acts[depth_idx](
                self.cross_group_norms[depth_idx](self.cross_group_convs[depth_idx](fused))
            )
        return fused

    def forward(self, x):
        xrgb, xir = x[0], x[1]
        fused = None
        for d in range(self.depth):
            fused = self._process_single_level(xrgb, xir, d)
            if d < self.depth - 1:
                xrgb = xir = fused
        return self.out_proj(fused)
class SelfAttention(nn.Module):
    """
     Multi-head masked self-attention layer
    """

    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SelfAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj = nn.Linear(d_model, h * self.d_v)  # value projection
        self.out_proj = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''

        b_s, nq = x.shape[:2]
        nk = x.shape[1]
        q = self.que_proj(x).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.key_proj(x).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v = self.val_proj(x).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        # Self-Attention
        #  :math:`(\text(Attention(Q,K,V) = Softmax((Q*K^T)/\sqrt(d_k))`
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        # weight and mask
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        # get attention matrix
        att = torch.softmax(att, -1)
        att = self.attn_drop(att)

        # output
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.resid_drop(self.out_proj(out))  # (b_s, nq, d_model)

        return out


class myTransformerBlock(nn.Module):
    """ Transformer block """

    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)

        """
        super().__init__()
        self.ln_input = nn.LayerNorm(2*d_model)
        self.ln_output = nn.LayerNorm(2*d_model)
        self.sa = SelfAttention(2*d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(2*d_model, block_exp * d_model*2),
            # nn.SiLU(),  # changed from GELU
            nn.GELU(),  # changed from GELU
            nn.Linear(block_exp * d_model*2, d_model*2),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        bs, nx, c = x.size()

        x = x + self.sa(self.ln_input(x))
        x = x + self.mlp(self.ln_output(x))

        return x

class Add(nn.Module):
    #  Add two tensors
    def __init__(self, arg):
        super(Add, self).__init__()
        self.arg = arg

    def forward(self, x):
        #print(x[0].shape,x[1].shape)
        return torch.add(x[0], x[1])


class Add2(nn.Module):
    #  x + transformer[0] or x + transformer[1]
    def __init__(self, c1, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        if self.index == 0:
            return torch.add(x[0], x[1][0])
        elif self.index == 1:
            return torch.add(x[0], x[1][1])
class frequencyfuse(nn.Module):
    def __init__(self,c1,c2):
        super().__init__()
        self.wt_filter, self.iwt_filter = create_wavelet_filter("db3", c1, c1, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.wt_function = partial(wavelet_transform, filters = self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters = self.iwt_filter)
        
        self.fusehigh=fusehigh(c1,c2)
        self.fuselow=fuselow(c1,c2)
        self.fusemiddle=fusemiddle(c1,c2)
    def forward(self,x):
        xir,xrgb=x[0],x[1]
        #print(xir.shape,xrgb.shape)
        xirf = self.wt_function(xir)
        xirflow=xirf[:,:,0,:,:]
        xirfhigh=xirf[:,:,3,:,:]
        xirfmiddle=xirf[:,:,1:3,:,:]
        xrgbf = self.wt_function(xrgb)
        xrgbflow=xrgbf[:,:,0,:,:]
        xrgbfhigh=xrgbf[:,:,3,:,:]
        xrgbfmiddle=xrgbf[:,:,1:3,:,:]
        fusehighrgb=self.fusehigh(xirfhigh,xrgbfhigh)
        fuselowir=self.fuselow(xirflow,xrgbflow)
        fusemiddlecom=self.fusemiddle(xirfmiddle,xrgbfmiddle)
        #print(fusehighrgb.shape)
        #print(fuselowir.shape)
        #print(fusemiddlecom.shape)
        xirflow=torch.unsqueeze(xirflow,2)
        xrgbflow=torch.unsqueeze(xrgbflow,2)
        xirfhigh=torch.unsqueeze(xirfhigh,2)
        xrgbhigh=torch.unsqueeze(xrgbfhigh,2)
        xirfuse=torch.cat([fuselowir,fusemiddlecom,xirfhigh],dim=2)
        xrgbfuse=torch.cat([xrgbflow,fusemiddlecom,fusehighrgb],dim=2)
        xirreturn = self.iwt_function(xirfuse)
        xrgbreturn=self.iwt_function(xrgbfuse)
        return [xirreturn,xrgbreturn]
        
class fusehigh(nn.Module):
    def __init__(self, c1, c2, s=1, e=4):
        super().__init__()
        #print(c1)
        #print(c2)
        self.convrgb=nn.Conv2d(c1, c2, 1, 1, 0, bias=True)
        self.normrgb=nn.BatchNorm2d(c2)
        self.CBAM=CBAM3(c2)
        self.convir=nn.Conv2d(c1, c2, 1, 1, 0, bias=True)
        self.normir=nn.BatchNorm2d(c2)
        #self.normcom=nn.BatchNorm2d(c2)
    def forward(self,xir,xrgb):
        xshape=xir.shape
        #print(xshape)
        #print(xrgb.shape)
        xiruse=torch.squeeze(xir,2)
        xrgbuse=torch.squeeze(xrgb,2)
        #xrgborl=
        xircancha=xrgbuse
        #print(xircancha.shape)
        xrgbuse=self.convrgb(xiruse)
        #if len(xrgbuse.shape)==3:
           #xrgbuse=
        xrgbuse=self.normrgb(xrgbuse)
        xrgbuse=self.CBAM(xrgbuse)
        xiruse=self.convir(xircancha)
        xiruse=self.normir(xiruse)
        xiruse=F.silu(xiruse)
        xfuse=xiruse*xrgbuse
        #print(xfuse.shape)
        #xfuse=self.normcom(xfuse)
        xreturn=xfuse+xircancha
        xreturn=torch.unsqueeze(xreturn,2)
        return xreturn
class fuselow(nn.Module): 
    def __init__(self,c1,c2,**kwargs):
        super().__init__()
        self.normir=nn.BatchNorm2d(c2)
        self.convir=nn.Conv2d(c1, c2, 1, 1, 0, bias=True)
        #self.in_proj1 = nn.Linear(c2, 64, bias=False)
        self.attentionir=Cross_modal_Vmaba_module2(d_model=c2, dropout=0, d_state=16, **kwargs)
        self.convrgb=nn.Conv2d(c1, c2, 1, 1, 0, bias=True)
        self.normrgb=nn.BatchNorm2d(c2)
    def forward(self,xir,xrgb):
        xiruse=torch.squeeze(xir,2)
        xrgbuse=torch.squeeze(xrgb,2)
        xrgbcancha=xiruse
        xiruse=self.convir(xrgbuse)
        xiruse=self.normir(xiruse)
        xiruse=self.attentionir(xiruse)
        xrgbuse=self.convrgb(xrgbcancha)
        xrgbuse=self.normrgb(xrgbuse)
        xrgbuse=F.silu(xrgbuse)
        #print(xiruse.shape)
        #print(xrgbuse.shape)
        xfuse=xiruse*xrgbuse
        xreturn=xfuse+xrgbcancha
        xreturn=torch.unsqueeze(xreturn,2)
        return xreturn
class fusemiddle(nn.Module):
     def __init__(self,c1,c2,h=8, block_exp=4,
                 n_layer=8, vert_anchors=8, horz_anchors=8,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.n_embd = c1
        self.vert_anchors = 2*vert_anchors
        self.horz_anchors = 2*horz_anchors
        d_k = c1
        d_v = c1
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
        self.pos_emb = nn.Parameter(torch.zeros(1, self.vert_anchors * self.horz_anchors, 2*self.n_embd))

        # transformer
        self.trans_blocks = myTransformerBlock(c1, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
        self.ln_f = nn.LayerNorm(2*self.n_embd)

        # regularization
        self.drop = nn.Dropout(embd_pdrop)
        self.conv1=nn.Conv2d(2*c1, c1, 1, 1, 0, bias=True)
        self.conv2=nn.Conv2d(c1, 2*c2, 1, 1, 0, bias=True)
        self.norm1=nn.BatchNorm2d(2*c2)
        self.apply(self._init_weights)
        
     @staticmethod
     def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
     def forward(self,xir,xrgb):
         #xfuse=torch.cat([xir,xrgb],dim=2)
         xfuse=xir+xrgb
         shape=xfuse.shape
         #print(shape)
         xfuse=xfuse.reshape(shape[0], shape[1] * 2, shape[3], shape[4])
         xcancha=xfuse
         bs, c, h, w = xcancha.shape
         xfuse = self.avgpool(xfuse)
         xfuse = xfuse.view(bs, c, -1)
         token_embeddings = xfuse.permute(0, 2, 1).contiguous()
         #print(token_embeddings.shape)
         
         x = self.drop(self.pos_emb + token_embeddings)  # sum positional embedding and token    dim:(B, 2*H*W, C)
         x = self.trans_blocks(x)
         x = self.ln_f(x)
         x = x.view(bs, self.vert_anchors, self.horz_anchors, 2*self.n_embd)
         x = x.permute(0, 3, 1, 2)
         xfuse = x.contiguous().view(bs, 2*self.n_embd, self.vert_anchors, self.horz_anchors)
         xfuse = F.interpolate(xfuse, size=([h, w]), mode='bilinear')
         xfuse=self.conv1(xfuse)
         xfuse=self.conv2(xfuse)
         xfuse=self.norm1(xfuse)
         xfuse=F.silu(xfuse)
         xfuse=xfuse+xcancha
         xfuse=xfuse.reshape(shape[0],shape[1], 2, shape[3], shape[4])
         return xfuse
         
         
         

class Cross_modal_Vmaba_module2(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None, 
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj1 = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        #self.in_proj2 = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d1 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        '''self.conv2d2 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )'''

        self.act1 = nn.SiLU()
        #self.act2 = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        self.selective_scan = selective_scan_fn
        self.forward_core1 = self.forward_corev0
        #self.forward_core2 = self.forward_corev0

        self.out_norm1 = nn.LayerNorm(self.d_inner)
        #self.out_norm2 = nn.LayerNorm(self.d_inner)
        self.out_proj1 = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        #self.out_proj2 = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0. else None
        #self.dropout2 = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            #torch.arange(1, d_state + 1, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    
    #得到可以依据输入变化的参数A、B、C、D并且计算最终的输出
    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, xs, **kwargs):
        x1= xs
        x1=x1.permute(0, 2, 3, 1).contiguous()
        #assert x1.shape == x2.shape, "Inputs must have the same shape."
        B, H, W, C = x1.shape
        
        xz1 = self.in_proj1(x1)
        #xz2 = self.in_proj2(x2)
        x1, z1 = xz1.chunk(2, dim=-1) # (b, h, w, d)
        #x2, z2 = xz2.chunk(2, dim=-1) # (b, h, w, d)
        
        x1 = x1.permute(0, 3, 1, 2).contiguous()
        #x2 = x2.permute(0, 3, 1, 2).contiguous()

        x1 = self.act1(self.conv2d1(x1)) # (b, d, h, w)
        #x2 = self.act1(self.conv2d2(x2)) # (b, d, h, w)

        y1_1, y2_1, y3_1, y4_1 = self.forward_core1(x1)
        #y1_2, y2_2, y3_2, y4_2 = self.forward_core2(x2)
        #assert y1.dtype == torch.float32
        y1 = y1_1 + y2_1 + y3_1 + y4_1
        #y2 = y1_2 + y2_2 + y3_2 + y4_2
        y1 = torch.transpose(y1, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        #y2 = torch.transpose(y2, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y1 = y1.to(x1.dtype)
        #y2 = y2.to(x2.dtype)
        y1 = self.out_norm1(y1)
        #y2 = self.out_norm2(y2)
        out1 = self.out_proj1(y1)
        if self.dropout1 is not None:
            out1 = self.dropout1(out1)
        out1 = out1.permute(0, 3, 1, 2).contiguous()
        return out1     
class ChannelAttention2(nn.Module):
    """Deprecated: Use ChannelAttention instead"""
    def __init__(self, channels: int) -> None:
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class SpatialAttention2(nn.Module):
    """Deprecated: Use SpatialAttention instead"""
    def __init__(self, kernel_size=7):
        super().__init__()
    def forward(self, x):
        pass

class CBAM3(nn.Module):
    """Deprecated: Use CBAM instead"""
    def __init__(self, c1, kernel_size=7):
        super().__init__()
    def forward(self, x):
        pass

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()

        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False, padding_mode='replicate')
        
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class ECAAttention(nn.Module):
    """Efficient Channel Attention for lightweight channel reweighting."""

    def __init__(self, channels: int, k_size: int = 3) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.act(y).transpose(-1, -2).unsqueeze(-1)
        return x * y


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))



class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        # print(x.shape)
        return torch.cat(x, self.d)
