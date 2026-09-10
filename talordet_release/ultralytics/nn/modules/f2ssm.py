import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None

class F2SSM(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
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
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # Shared A and parameters
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)

        # Modality-specific D parameters (learnable fixed)
        self.Ds_r = self.D_init(self.d_inner, copies=4, merge=True)
        self.Ds_i = self.D_init(self.d_inner, copies=4, merge=True)

        # Direction weights (learnable) for 4 scanning directions
        self.dir_weights_r = nn.Parameter(torch.ones(4, **factory_kwargs))
        self.dir_weights_i = nn.Parameter(torch.ones(4, **factory_kwargs))

        # Modality-specific projections (B, C, dt are generated from these)
        # Note: In SS2D code, x_proj generates dt, B, C.
        # "B_r, B_i are dynamically generated ... via learnable linear layers" -> Separate layers
        self.x_proj_r = self._init_x_proj(**factory_kwargs)
        self.x_proj_i = self._init_x_proj(**factory_kwargs)

        self.dt_projs_r = self._init_dt_proj(dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
        self.dt_projs_i = self._init_dt_proj(dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)

        # In-projects and Convs (likely separate for each modality before fusion)
        self.in_proj_r = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_i = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv2d_r = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.conv2d_i = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # --- Alignment Layer (New) ---
        # Lightweight offset predictor for Deformable Convolution or simple resampling
        # Here we implement a simple offset prediction convolution.
        # We will assume a small DCN-like operation or just prepare features for it.
        # Since implementing full DCN requires custom cuda ops or torchvision >= 0.9,
        # we will simulate alignment via "conv-based feature rectifying" or standard Conv.

        # To keep it portable and simple for now: A spatial transformation prediction layer
        # predicting channel-wise shifts could be efficiently implemented via depthwise convs.
        # But DCN is standard. Let's use a standard Conv to "preprocess/align" Xi features
        # based on Xr guidance.

        # Structure: Concatenate(Xr, Xi) -> Conv -> Offsets -> Feature Warp (Simulated by Conv)

        # Simplified: A "Cross-Modality Calibration" convolution
        # Learns to modify Xi using Xr context.
        # Input: 2 * d_inner (both modalities) -> Output: d_inner (aligned Xi)
        self.align_conv = nn.Sequential(
            nn.Conv2d(self.d_inner * 2, self.d_inner, kernel_size=3, padding=1, bias=bias, **factory_kwargs),
            nn.SiLU(),
            nn.Conv2d(self.d_inner, self.d_inner, kernel_size=3, padding=1, bias=bias, **factory_kwargs)
        )
        # -----------------------------

        self.act = nn.SiLU()

        self.out_norm_r = nn.LayerNorm(self.d_inner)
        self.out_norm_i = nn.LayerNorm(self.d_inner)
        self.out_proj_r = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_i = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    def _init_x_proj(self, **factory_kwargs):
        # Initialize x_proj weight similar to SS2D
        # We need K=4 projections
        projs = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(4)
        ]
        return nn.Parameter(torch.stack([t.weight for t in projs], dim=0)) # (K=4, N, inner)

    def _init_dt_proj(self, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs):
        # We need 4 dt projs
        dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(4)
        ]
        weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))
        bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))
        return nn.ParameterDict({'weight': weight, 'bias': bias})

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core_single(self, x, x_proj_weight, dt_projs_weight, dt_projs_bias, Ds, A_logs):
        # Implementation of forward_corev0 for a single modality
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = Ds.float().view(-1)
        As = -torch.exp(A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = dt_projs_bias.float().view(-1)

        out_y = selective_scan_fn(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, xr, xi):
        # Process x_r
        xz_r = self.in_proj_r(xr)
        x_r, z_r = xz_r.chunk(2, dim=-1)
        x_r = x_r.permute(0, 3, 1, 2).contiguous()
        x_r = self.act(self.conv2d_r(x_r))

        # Process x_i
        xz_i = self.in_proj_i(xi)
        x_i, z_i = xz_i.chunk(2, dim=-1)
        x_i = x_i.permute(0, 3, 1, 2).contiguous()
        x_i = self.act(self.conv2d_i(x_i))

        # --- Alignment Step ---
        # Use x_r to guide x_i alignment
        # In a real DCN implementation, we would predict offsets here.
        # Here we usage feature-level calibration:
        # xi_aligned = Align(cat(xr, xi)) + xi
        # We perform this on the (B, D, H, W) features before SSM
        cat_feat = torch.cat([x_r, x_i], dim=1) # (B, 2D, H, W)
        align_offset = self.align_conv(cat_feat) # (B, D, H, W)
        x_i = x_i + align_offset # Residual correction to align/calibrate IR features
        # ----------------------

        # Core SSM
        yr1, yr2, yr3, yr4 = self.forward_core_single(
            x_r, self.x_proj_r, self.dt_projs_r['weight'], self.dt_projs_r['bias'], self.Ds_r, self.A_logs
        )
        yi1, yi2, yi3, yi4 = self.forward_core_single(
            x_i, self.x_proj_i, self.dt_projs_i['weight'], self.dt_projs_i['bias'], self.Ds_i, self.A_logs
        )

        # Fusion: Weighted Sum the outputs (Eq 11 implies F2SSM = SSM(r) + SSM(i))
        # We apply learnable weights to each direction before merging.
        # But for Dual Output, we keep y_r and y_i separate.

        # Normalize weights to prevent overflow/exploding gradients and ensure probabilistic interpretation
        # Scaling by 4 brings the softmax output (approx 0.25) back to magnitude 1.0, preserving gradient flow
        w_r = F.softmax(self.dir_weights_r, dim=0) * 4
        w_i = F.softmax(self.dir_weights_i, dim=0) * 4

        y_r = yr1 * w_r[0] + yr2 * w_r[1] + yr3 * w_r[2] + yr4 * w_r[3]
        y_i = yi1 * w_i[0] + yi2 * w_i[1] + yi3 * w_i[2] + yi4 * w_i[3]

        # Dual Output: we DO NOT sum y_r and y_i into y here.
        # y = y_r + y_i # REMOVED

        B, D, H, W = x_r.shape
        # Output handling for dual streams
        # Reshape both streams
        y_r = torch.transpose(y_r, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y_i = torch.transpose(y_i, dim0=1, dim1=2).contiguous().view(B, H, W, -1)

        # Norm separately
        y_r = self.out_norm_r(y_r)
        y_i = self.out_norm_i(y_i)

        # Dual Gating:
        # Instead of fusing to one y, we allow cross-gating or shared gating but keep two outputs.
        # Option A (Simulating Cross_Block style):
        # out1 = y_r * silu(z_r + z_i)
        # out2 = y_i * silu(z_r + z_i)
        # This allows both streams to benefit from the joint gate.

        # Independent Gating Strategy for Dual Stream Output
        # Each stream focuses on its own highlighted regions.
        gate_r = F.silu(z_r)
        gate_i = F.silu(z_i)

        y_r = y_r * gate_r
        y_i = y_i * gate_i

        # Final projections
        out_r = self.out_proj_r(y_r)
        out_i = self.out_proj_i(y_i)

        if self.dropout is not None:
             out_r = self.dropout(out_r)
             out_i = self.dropout(out_i)

        return out_r, out_i
