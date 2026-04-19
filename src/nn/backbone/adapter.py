from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# LayerNorm2d (NCHW)
# -----------------------------
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / (var + self.eps).sqrt()
        if self.affine:
            w = self.weight[:, None, None]
            b = self.bias[:,   None, None]
            x = x * w + b
        return x


# -----------------------------
# -----------------------------
class MonaOp2d(nn.Module):
    """Depthwise 3/5/7 conv fusion + 1x1 projector, residual."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 7, padding=3, groups=in_channels)
        self.projector = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = (self.conv1(x) + self.conv2(x) + self.conv3(x)) / 3.0 + identity
        return x + self.projector(x)


class SEBlock(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        mid = max(1, in_channels // reduction_ratio)
        self.fc1 = nn.Conv2d(in_channels, mid, 1)
        self.fc2 = nn.Conv2d(mid, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class ECABlock(nn.Module):
    def __init__(self, in_channels: int, k_size: int = 3):
        super().__init__()
        assert k_size % 2 == 1, "ECA kernel size must be odd"
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg(x)                                # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(1, 2)              # (B, 1, C)
        y = self.conv(y)                               # (B, 1, C)
        y = torch.sigmoid(y).transpose(1, 2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y


class ScaleOnlyGate(nn.Module):
    def __init__(self, in_channels: int, use_bias: bool = True, init_scale: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((in_channels, 1, 1), init_scale))
        self.use_bias = use_bias
        if use_bias:
            self.beta = nn.Parameter(torch.zeros(in_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = x.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        gate = self.alpha * m
        if self.use_bias:
            gate = gate + self.beta
        gate = torch.sigmoid(gate)
        return x * gate


# -----------------------------
# Util convs
# -----------------------------
def _conv1x1(in_ch: int, out_ch: int, bias: bool = True) -> nn.Conv2d:
    conv = nn.Conv2d(in_ch, out_ch, 1, bias=bias)
    nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))
    if bias:
        nn.init.zeros_(conv.bias)
    return conv

def _dw3x3(C: int) -> nn.Conv2d:
    conv = nn.Conv2d(C, C, 3, padding=1, groups=C, bias=True)
    nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))
    nn.init.zeros_(conv.bias)
    return conv


class MonaFreq2DAdapter(nn.Module):

    def __init__(self,
                 in_channels: int,
                 bottleneck_dim: int = 128,
                 dropout_p: float = 0.1,
                 cutoff_ratio: float = 0.3,
                 # cutoff_ratio: float = 0.35,
                 scale_init: float = 0.1,
                 se_reduction: int = 16,
                 use_se: bool = True,
                 se_mode: str = "eca",      # "se" | "eca" | "scale"
                 eca_kernel: int = 3,
                 share_se: bool = False,
                 size_adaptive_gamma: bool = False
                 ):

        super().__init__()
        assert 0 < cutoff_ratio < 1, "cutoff_ratio should be in (0,1)"
        C = in_channels
        D = bottleneck_dim

        self.norm = LayerNorm2d(C)
        self.gamma  = nn.Parameter(torch.ones(C) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(C))

        self.size_adaptive_gamma = bool(size_adaptive_gamma)  # [ADD]
        self._ref_hw = None

        # content (Mona)
        self.proj1   = nn.Conv2d(C, D, 1, bias=True)
        self.mona_op = MonaOp2d(D)
        self.drop    = nn.Dropout(dropout_p)
        self.proj2   = nn.Conv2d(D, C, 1, bias=True)
        nn.init.constant_(self.proj2.weight, 0.0)
        if self.proj2.bias is not None:
            nn.init.constant_(self.proj2.bias, 0.0)


        self.cutoff_ratio = float(cutoff_ratio)
        self.freq_dw  = _dw3x3(C)


        self.use_se   = bool(use_se)
        self.se_mode  = se_mode
        self.share_se = bool(share_se)
        if self.use_se:
            if self.se_mode == "eca":
                se_factory = lambda: ECABlock(C, k_size=eca_kernel)
            elif self.se_mode == "scale":
                se_factory = lambda: ScaleOnlyGate(C, use_bias=True, init_scale=1.0)
            else:
                se_factory = lambda: SEBlock(C, se_reduction)

            if self.share_se:
                se_module = se_factory()
                self.se_low  = se_module
                self.se_high = se_module
            else:
                self.se_low  = se_factory()
                self.se_high = se_factory()
        else:
            self.se_low  = nn.Identity()
            self.se_high = nn.Identity()


        self.beta_low     = nn.Parameter(torch.zeros(C, 1, 1))
        self.beta_high    = nn.Parameter(torch.zeros(C, 1, 1))

        # per-branch × per-channel scale
        self.scale_c = nn.Parameter(torch.full((C, 1, 1), float(scale_init)))
        self.scale_l = nn.Parameter(torch.full((C, 1, 1), float(scale_init)))
        self.scale_h = nn.Parameter(torch.full((C, 1, 1), float(scale_init)))


        self.router = _conv1x1(C, 3, bias=True)

        self.act = nn.GELU()


        self.disc_k = 2.0
        self.use_disc = True
        self.detach_disc_input = True

        self.disc_head = nn.Sequential(
            nn.Conv2d(C, max(8, C // 8), kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(max(8, C // 8), 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # cache for engine losses
        self.last_u = None
        self.last_delta = None


    @torch.no_grad()
    def _low_mask(self, H: int, W: int, device) -> torch.Tensor:
        half = int(min(H, W) * self.cutoff_ratio // 2)
        cx, cy = H // 2, W // 2
        x0, x1 = max(cx - half, 0), min(cx + half, H)
        y0, y1 = max(cy - half, 0), min(cy + half, W)
        m = torch.zeros((H, W), device=device, dtype=torch.bool)
        m[x0:x1, y0:y1] = True
        return m

    def _fft_low_high(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        if x.is_cuda:
            import torch.cuda.amp as amp
            with amp.autocast(enabled=False):
                x32 = x.to(torch.float32)
                F2  = torch.fft.fft2(x32, norm='ortho')
                F2s = torch.fft.fftshift(F2, dim=(-2, -1))
                B, C, H, W = F2s.shape
                m2d = self._low_mask(H, W, x32.device)
                low_m  = m2d.view(1,1,H,W).expand(B,C,H,W)
                high_m = ~low_m
                zero = torch.zeros_like(F2s)
                Fl = torch.where(low_m,  F2s, zero)
                Fh = torch.where(high_m, F2s, zero)
                low  = torch.fft.ifft2(torch.fft.ifftshift(Fl, dim=(-2,-1)), norm='ortho').real
                high = torch.fft.ifft2(torch.fft.ifftshift(Fh, dim=(-2,-1)), norm='ortho').real
        else:
            F2  = torch.fft.fft2(x, norm='ortho')
            F2s = torch.fft.fftshift(F2, dim=(-2, -1))
            B, C, H, W = F2s.shape
            m2d = self._low_mask(H, W, x.device)
            low_m  = m2d.view(1,1,H,W).expand(B,C,H,W)
            high_m = ~low_m
            zero = torch.zeros_like(F2s)
            Fl = torch.where(low_m,  F2s, zero)
            Fh = torch.where(high_m, F2s, zero)
            low  = torch.fft.ifft2(torch.fft.ifftshift(Fl, dim=(-2,-1)), norm='ortho').real
            high = torch.fft.ifft2(torch.fft.ifftshift(Fh, dim=(-2,-1)), norm='ortho').real

        if low.dtype != orig_dtype:
            low  = low.to(orig_dtype)
            high = high.to(orig_dtype)
        return low, high

    # ------- 前向 -------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) -> y: (B, C, H, W)
        """
        identity = x

        if self.size_adaptive_gamma and (self._ref_hw is None):
            self._ref_hw = (x.shape[-2], x.shape[-1])  # (H, W)

        if self.size_adaptive_gamma and (self._ref_hw is not None):
            Hr, Wr = self._ref_hw
            H, W = x.shape[-2], x.shape[-1]
            s_gamma = math.sqrt(max(1.0, Hr * Wr) / max(1.0, H * W))
            x_norm = self.norm(x) * (self.gamma.view(1,-1,1,1) * s_gamma) + x * self.gammax.view(1,-1,1,1)
        else:
            x_norm = self.norm(x) * self.gamma.view(1,-1,1,1) + x * self.gammax.view(1,-1,1,1)

        c = self.proj1(x_norm)
        c = self.mona_op(c)
        c = self.act(c)
        c = self.drop(c)
        c = self.proj2(c)
        c = self.scale_c * c              # per-channel scale

        low, high = self._fft_low_high(x)
        yl = self.freq_dw(low)
        yh = self.freq_dw(high)

        dl = self.se_low(yl)  + self.beta_low
        dh = self.se_high(yh) + self.beta_high

        dl = self.scale_l * dl
        dh = self.scale_h * dh


        w = torch.softmax(self.router(x), dim=1)  # (B,3,H,W) -> content/low/high
        delta = w[:, 0:1] * c + w[:, 1:2] * dl + w[:, 2:3] * dh

        if self.use_disc:
            disc_in = x.detach() if self.detach_disc_input else x
            u = self.disc_head(disc_in)
            delta = (1.0 + self.disc_k * u) * delta
            self.last_u = u
        else:
            self.last_u = None

        self.last_delta = delta
        return identity + delta


# =============================
# Mona-style 2D Adapter (content only)
# =============================
class Mona2DAdapter(nn.Module):
    """
    Mona (token) -> Mona2D (feature map)
    - project1: 1x1 conv C -> D (默认 D=64)
    - MonaOp2d on D-ch feature
    - GELU + Dropout
    - project2: 1x1 conv D -> C
    - Pre-norm with learnable gamma/gammax (LN2d风格)
    - Residual add
    """
    def __init__(self, in_channels: int, bottleneck_dim: int = 64, dropout_p: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.bottleneck_dim = bottleneck_dim

        self.norm = LayerNorm2d(in_channels)
        self.gamma  = nn.Parameter(torch.ones(in_channels) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_channels))

        self.project1 = nn.Conv2d(in_channels, bottleneck_dim, kernel_size=1, bias=True)
        self.mona_op  = MonaOp2d(bottleneck_dim)
        self.dropout  = nn.Dropout(p=dropout_p)
        self.project2 = nn.Conv2d(bottleneck_dim, in_channels, kernel_size=1, bias=True)
        self.act = nn.GELU()

        # init
        nn.init.constant_(self.project2.weight, 0.0)
        if self.project2.bias is not None:
            nn.init.constant_(self.project2.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        identity = x
        x = self.norm(x) * self.gamma.view(1, -1, 1, 1) + x * self.gammax.view(1, -1, 1, 1)
        x = self.project1(x)
        x = self.mona_op(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.project2(x)
        return identity + x
