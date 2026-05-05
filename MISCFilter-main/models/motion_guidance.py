import torch
import torch.nn.functional as F

def compute_orientation_map(x, eps=1e-6):
    """
    Compute a 1-channel orientation map normalized to [-1, 1].
    Input:
        x: Tensor (B, C, H, W)
    Output:
        theta_norm: Tensor (B, 1, H, W)
    """
    # convert to gray: simple average
    if x.shape[1] == 1:
        gray = x
    else:
        gray = x.mean(dim=1, keepdim=True)  # (B,1,H,W)

    # Sobel kernels
    device = x.device
    dtype = x.dtype
    kx = torch.tensor([[-1., 0., 1.],
                       [-2., 0., 2.],
                       [-1., 0., 1.]], device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.],
                       [ 0.,  0.,  0.],
                       [ 1.,  2.,  1.]], device=device, dtype=dtype).view(1, 1, 3, 3)

    grad_x = F.conv2d(gray, kx, padding=1)
    grad_y = F.conv2d(gray, ky, padding=1)

    # angle in [-pi, pi]
    theta = torch.atan2(grad_y, grad_x + eps)  # (B,1,H,W)
    # normalize to [-1,1]
    theta_norm = theta / 3.14159265
    return theta_norm
