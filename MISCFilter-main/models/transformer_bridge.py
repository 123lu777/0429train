import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # 如果用户已经把 0427train 的 basicsr 放到 PYTHONPATH，可直接使用
    from basicsr.models.archs import polar_utils as _polar_utils
    _POLAR_AVAILABLE = True
except Exception:
    _POLAR_AVAILABLE = False

from .motion_guidance import compute_orientation_map


def compute_polar_orientation_map(x, mode='auto', **kwargs):
    """
    Returns a 1-channel orientation-like map.
    mode:
      - 'auto' : try to use polar_utils if available, else fallback to simple orientation map
      - 'polar_utils' : require polar_utils (will raise if not available)
      - 'simple' : use Sobel->atan2 implementation
    """
    if mode == 'polar_utils' and not _POLAR_AVAILABLE:
        raise ImportError('polar_utils not available. Please install/put basicsr in PYTHONPATH or use mode="simple".')

    if mode == 'auto' and _POLAR_AVAILABLE:
        mode = 'polar_utils'
    if mode == 'polar_utils':
        # 这里作为占位：如果需要使用 polar_utils 的具体函数（例如更复杂的采样/theta map），
        # 请根据 basicsr/models/archs/polar_utils.py 的接口替换下面的实现。
        try:
            # 示例 fallback：仍然使用简单 orientation 作为占位实现
            return compute_orientation_map(x)
        except Exception:
            return compute_orientation_map(x)
    else:
        return compute_orientation_map(x)


class _PriorEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, prior, target_hw):
        if prior is None:
            return None
        if prior.dim() == 2:
            prior = prior[:, :, None, None].expand(-1, -1, target_hw[0], target_hw[1])
        elif prior.dim() == 4 and prior.shape[2:] != target_hw:
            prior = F.interpolate(prior, size=target_hw, mode='bilinear', align_corners=False)
        if prior.dim() == 4 and prior.size(1) != self.in_channels:
            if prior.size(1) < self.in_channels:
                pad = self.in_channels - prior.size(1)
                prior = torch.cat([prior, prior.new_zeros(prior.size(0), pad, prior.size(2), prior.size(3))], dim=1)
            else:
                prior = prior[:, :self.in_channels, :, :]
        return self.net(prior)


class SimpleTransformerBridge(nn.Module):
    def __init__(self, channels, prior_channels=4):
        super().__init__()
        self.prior_encoder = _PriorEncoder(prior_channels, channels)
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=1),
        )

    def forward(self, x, dist=None):
        if dist is not None:
            prior_feat = self.prior_encoder(dist, x.shape[2:])
            if prior_feat is not None:
                x = x + prior_feat
        return self.body(x)


def _load_pretrained(model, pretrained):
    if not pretrained:
        return
    if not os.path.exists(pretrained):
        warnings.warn(f"transformer pretrained not found: {pretrained}. Skipping load.")
        return
    state = torch.load(pretrained, map_location='cpu')
    state_dict = state.get('state_dict', state)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        warnings.warn(f"transformer pretrained load: missing={len(missing)} unexpected={len(unexpected)}")


def build_transformer(channels, pretrained=None, img_size=None, prior_channels=4):
    """
    Build a lightweight transformer bridge with optional prior (dist) support.
    If external transformer code is unavailable, fallback to a simple residual conv block.
    """
    model = SimpleTransformerBridge(channels=channels, prior_channels=prior_channels)
    _load_pretrained(model, pretrained)
    return model
