# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmpose.registry import MODELS


def _make_gaussian_kernel(kernel_size: int = 5) -> Tensor:
    if kernel_size == 5:
        k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
    elif kernel_size == 3:
        k = torch.tensor([1.0, 2.0, 1.0])
    else:
        raise ValueError(f'Unsupported kernel_size={kernel_size}, use 3 or 5.')
    k2 = torch.outer(k, k)
    k2 = k2 / k2.sum()
    return k2


@MODELS.register_module()
class LowLightFGE(nn.Module):
    """Luminance-Texture Adapter block for low-light pose estimation.

    The module follows the robust low-light design pattern:
    1) Laplacian decomposition (low/high frequency split)
    2) Dynamic illumination correction on low-frequency base (DIC-like)
    3) Low-rank denoising on high-frequency details (MLD-like)
    4) Multi-scale reconstruction.

    This block is intentionally lightweight and can be plugged before the
    backbone in both top-down and bottom-up estimators.
    """

    def __init__(
        self,
        levels: int = 4,
        kernel_size: int = 5,
        low_mid_channels: int = 24,
        hf_channels: int = 24,
        hf_rank: int = 3,
        local_strength: float = 1.0,
        hf_strength: float = 1.0,
        assume_input_normed: bool = True,
        mean: Sequence[float] = (123.675, 116.28, 103.53),
        std: Sequence[float] = (58.395, 57.12, 57.375),
        input_range: float = 255.0,
        eps: float = 1e-4,
        use_glic: bool = True,
        use_lrbd: bool = True,
        use_dcc: bool = True,
    ):
        super().__init__()
        if levels < 2:
            raise ValueError('levels must be >= 2 for Laplacian decomposition.')
        self.levels = levels
        self.assume_input_normed = assume_input_normed
        self.input_range = float(input_range)
        self.eps = eps
        self.local_strength = float(local_strength)
        # Keep for backward config compatibility.
        self.hf_strength = float(hf_strength)
        self.use_glic = use_glic
        self.use_lrbd = use_lrbd
        self.use_dcc = use_dcc

        mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer('mean', mean_t, persistent=False)
        self.register_buffer('std', std_t, persistent=False)

        kernel = _make_gaussian_kernel(kernel_size=kernel_size)
        self.register_buffer('gauss_kernel', kernel, persistent=False)

        # DIC-like branch on low-frequency component
        self.global_conv = nn.Conv2d(3, low_mid_channels, 3, padding=1)
        self.global_fc = nn.Linear(low_mid_channels, 3)
        self.local_net = nn.Conv2d(3, 3, 3, padding=1)
        # Symmetric residual enhancement network (6 conv layers).
        self.low_refine = nn.Sequential(
            nn.Conv2d(3, low_mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(low_mid_channels, low_mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(low_mid_channels, low_mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(low_mid_channels, low_mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(low_mid_channels, low_mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(low_mid_channels, 3, 3, padding=1),
        )

        # MLD-like branch on each high-frequency component
        self.hf_embed = nn.Conv2d(3, hf_channels, 3, padding=1)
        self.u_proj = nn.Conv2d(hf_channels, hf_rank, 3, padding=1)
        self.f_proj = nn.Conv2d(hf_channels, hf_channels, 3, padding=1)
        self.hf_out = nn.Conv2d(hf_channels, 3, 3, padding=1)
        self.cross_scale_convs = nn.ModuleList(
            [nn.Conv2d(3, 3, 3, padding=1) for _ in range(levels - 1)])

    def _gaussian_blur(self, x: Tensor) -> Tensor:
        c = x.shape[1]
        k = self.gauss_kernel.to(dtype=x.dtype, device=x.device)
        k = k.view(1, 1, *k.shape).repeat(c, 1, 1, 1)
        pad = (k.shape[-1] - 1) // 2
        return F.conv2d(x, k, padding=pad, groups=c)

    def _laplacian_decompose(self, x: Tensor) -> Tuple[List[Tensor], Tensor]:
        highs: List[Tensor] = []
        cur = x
        for _ in range(self.levels - 1):
            blur = self._gaussian_blur(cur)
            down = F.avg_pool2d(blur, kernel_size=2, stride=2)
            up = F.interpolate(
                down, size=cur.shape[-2:], mode='bilinear', align_corners=False)
            highs.append(cur - up)
            cur = down
        return highs, cur

    def _laplacian_reconstruct(self, highs: List[Tensor], base: Tensor) -> Tensor:
        cur = base
        for high in reversed(highs):
            cur = F.interpolate(
                cur, size=high.shape[-2:], mode='bilinear', align_corners=False)
            cur = cur + high
        return cur

    def _dic_low_freq(self, low: Tensor) -> Tensor:
        # Global correction coefficients (per channel)
        g_feat = self.global_conv(low)
        g_pool = F.adaptive_avg_pool2d(g_feat, 1).flatten(1)
        gamma_g = torch.sigmoid(self.global_fc(g_pool)).view(-1, 3, 1, 1)

        # Numerically stable log term
        log_term = torch.tanh(torch.log(low.clamp(min=self.eps)))
        low_g = low * (1.0 + gamma_g * log_term + 0.5 * (gamma_g**2) *
                       (log_term**2))

        # Local correction map
        gamma_l = torch.sigmoid(self.local_net(low_g))
        log_term_l = torch.tanh(torch.log(low_g.clamp(min=self.eps)))
        gamma_l = self.local_strength * gamma_l
        low_l = low_g * (1.0 + gamma_l * log_term_l + 0.5 * (gamma_l**2) *
                         (log_term_l**2))

        low_e = low_l + self.low_refine(low_l)
        return low_e.clamp(0.0, 1.0)

    def _mld_high_freq(self, high: Tensor) -> Tensor:
        # Eq.(6): map to high-dimensional feature space C=24.
        feat = F.relu(self.hf_embed(high), inplace=True)

        # Eq.(7): U in R^{HW x c}, c=3.
        u_map = F.relu(self.u_proj(feat), inplace=True)  # B,c,H,W
        b, c, h, w = u_map.shape
        u = u_map.flatten(2).transpose(1, 2).contiguous()  # B,HW,c

        # Eq.(8): F in R^{HW x C}.
        f_map = F.relu(self.f_proj(feat), inplace=True)  # B,C,H,W
        f_c_hw = f_map.flatten(2).contiguous()  # B,C,HW

        # Eq.(9): V from F and U.
        v = torch.bmm(f_c_hw, u)  # B,C,c

        # Eq.(10): UV -> conv -> reshape.
        uv = torch.bmm(u, v.transpose(1, 2))  # B,HW,C
        uv_map = uv.transpose(1, 2).reshape(b, -1, h, w).contiguous()  # B,C,H,W
        high_e = self.hf_out(uv_map)
        return self.hf_strength * high_e

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            return x

        if self.assume_input_normed:
            x01 = (x * self.std + self.mean) / self.input_range
        else:
            x01 = x
        x01 = x01.clamp(0.0, 1.0)

        highs, low = self._laplacian_decompose(x01)

        # GLIC: low-frequency illumination correction
        low_e = self._dic_low_freq(low) if self.use_glic else low

        # LRBD: high-frequency denoising
        high_enhanced = [self._mld_high_freq(h) for h in highs] \
            if self.use_lrbd else list(highs)

        # DCC: directed cross-scale carry
        if self.use_dcc:
            fused = [None] * len(high_enhanced)
            carry = None
            for idx in range(len(high_enhanced) - 1, -1, -1):
                h = high_enhanced[idx]
                if carry is not None:
                    h = h + F.interpolate(
                        carry, size=h.shape[-2:], mode='bilinear',
                        align_corners=False)
                h = self.cross_scale_convs[idx](h)
                fused[idx] = h
                carry = h
        else:
            fused = high_enhanced

        out01 = self._laplacian_reconstruct(fused, low_e).clamp(0.0, 1.0)

        if self.assume_input_normed:
            out = (out01 * self.input_range - self.mean) / self.std
        else:
            out = out01
        return out
