import torch

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
