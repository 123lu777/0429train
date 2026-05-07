"""
MISCFilterNet with Deformable Convolution (extended with optional motion guidance / transformer)
使用可变形卷积的 MISCFilterNet 版本，增加了可选的 motion guidance 与 transformer bridge
"""
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers_Deform import *
import models.MISCKernel_cuda as misckernel

from models.motion_guidance import compute_orientation_map
from models.polar_bridge import compute_polar_orientation_map
from models.transformer_bridge import build_transformer


class EBlock_Deform(nn.Module):
    """使用可变形卷积的编码块"""
    def __init__(self, out_channel, num_res=8, ResBlock=ResBlock_Deform_fft_bench):
        super(EBlock_Deform, self).__init__()
        layers = [ResBlock(out_channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock_Deform(nn.Module):
    """使用可变形卷积的解码块"""
    def __init__(self, channel, num_res=8, ResBlock=ResBlock_Deform_fft_bench):
        super(DBlock_Deform, self).__init__()
        layers = [ResBlock(channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AFF_Deform(nn.Module):
    """使用可变形卷积的特征融合模块"""
    def __init__(self, in_channel, out_channel, BasicConv=BasicConv_Deform):
        super(AFF_Deform, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class SCM_Deform(nn.Module):
    """使用可变形卷积的浅层特征提取模块"""
    def __init__(self, out_plane, BasicConv=BasicConv_Deform, inchannel=3):
        super(SCM_Deform, self).__init__()
        self.main = nn.Sequential(
            BasicConv(inchannel, out_plane // 4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane - inchannel, kernel_size=1, stride=1, relu=True)
        )
        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM_Deform(nn.Module):
    """使用可变形卷积的特征注意力模块"""
    def __init__(self, channel, BasicConv=BasicConv_Deform):
        super(FAM_Deform, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


def CharbonnierFunc(data, epsilon=0.001):
    return torch.mean(torch.sqrt(data ** 2 + epsilon ** 2))


def flow_warp(x, flow, interpolation='bilinear', padding_mode='zeros', align_corners=True):
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    device = flow.device

    if 'indexing' in torch.meshgrid.__code__.co_varnames:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, device=device, dtype=x.dtype),
            torch.arange(0, w, device=device, dtype=x.dtype),
            indexing='ij')
    else:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, device=device, dtype=x.dtype),
            torch.arange(0, w, device=device, dtype=x.dtype))
    grid = torch.stack((grid_x, grid_y), 2)
    grid.requires_grad = False

    grid_flow = grid + flow
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    grid_flow = grid_flow.type(x.type())
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output


class MISCKernelNet_Deform(nn.Module):
    """
    使用可变形卷积的 MISCKernelNet, 增加了 motion guidance 与 transformer 的可选集成
    """
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=32,
                 num_blocks=[12, 12, 12],
                 num_blocks_kernel=[1, 1, 1],
                 kernel_size=7,
                 inference=False,
                 use_deform_in_feat=True,
                 use_deform_in_encoder=True,
                 # 新增参数
                 use_motion_guidance=False,
                 motion_guidance_mode='simple',
                 use_polar_sampling=False,
                 use_transformer=False,
                 transformer_pretrained=None,
                 freeze_transformer=True,
                 transformer_img_size=128,  # 新增：传给 mdt(img_size=...)
                 transformer_dist_mode='dummy',  # dummy: static dist_value | none: no dist | coord: coordinate prior
                 transformer_dist_value=(0.5, 0.5, 0.5, 0.5),
                  ):
        super(MISCKernelNet_Deform, self).__init__()
        self.inference = inference
        self.dim = dim
        self.kernel_size = kernel_size
        self.kernel_pad = int((self.kernel_size - 1) / 2.0)

        # motion/transformer flags 保存
        self.use_motion_guidance = use_motion_guidance
        self.motion_guidance_mode = motion_guidance_mode
        self.use_polar_sampling = use_polar_sampling
        self.use_transformer = use_transformer
        self.transformer_pretrained = transformer_pretrained
        self.freeze_transformer = freeze_transformer
        self.transformer_img_size = transformer_img_size
        self.transformer_dist_mode = transformer_dist_mode
        self.transformer_dist_value = transformer_dist_value
        self._meshgrid_supports_indexing = 'indexing' in inspect.signature(torch.meshgrid).parameters

        # 根据模式选择卷积类型
        if not inference:
            if use_deform_in_feat:
                BasicConv = BasicConv_Deform
            else:
                BasicConv = BasicConv_do

            if use_deform_in_encoder:
                ResBlock = ResBlock_Deform_fft_bench
            else:
                ResBlock = ResBlock_do_fft_bench
        else:
            if use_deform_in_feat:
                BasicConv = BasicConv_Deform_eval
            else:
                BasicConv = BasicConv_do_eval

            if use_deform_in_encoder:
                ResBlock = ResBlock_Deform_fft_bench_eval
            else:
                ResBlock = ResBlock_do_fft_bench_eval

        base_channel = dim

        # 如果启用 motion guidance，创建一个小的投影层（1 -> inp_channels）以便非侵入性融合
        if self.use_motion_guidance:
            self.motion_proj = nn.Sequential(
                nn.Conv2d(1, inp_channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
            )

        # transformer bridge：作用在 res2（通道=base_channel*2）
        if self.use_transformer:
            self.transformer = build_transformer(
                channels=base_channel * 2,
                pretrained=transformer_pretrained,
                img_size=self.transformer_img_size,
                prior_channels=4
            )
            if self.freeze_transformer:
                for p in self.transformer.parameters():
                    p.requires_grad = False
            self._transformer_dist_support = self._detect_transformer_dist_support()
        else:
            self._transformer_dist_support = None

        # ============================================
        # 编码器 - 使用可变形卷积的 ResBlock
        # ============================================
        self.Encoder = nn.ModuleList([
            EBlock_Deform(base_channel, num_blocks[0], ResBlock=ResBlock),
            EBlock_Deform(base_channel * 2, num_blocks[1], ResBlock=ResBlock),
            EBlock_Deform(base_channel * 4, num_blocks[2], ResBlock=ResBlock),
        ])

        # ============================================
        # 特征提取层 - 使用可变形卷积
        # ============================================
        self.feat_extract = nn.ModuleList([
            BasicConv(inp_channels, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 4 * 2, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 2 * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
        ])

        # ============================================
        # 解码器 - 使用可变形卷积的 ResBlock
        # ============================================
        self.Decoder = nn.ModuleList([
            DBlock_Deform(base_channel * 4, num_blocks[2], ResBlock=ResBlock),
            DBlock_Deform(base_channel * 2, num_blocks[1], ResBlock=ResBlock),
            DBlock_Deform(base_channel, num_blocks[0], ResBlock=ResBlock)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.AFFs = nn.ModuleList([
            AFF_Deform(base_channel * 7, base_channel * 1, BasicConv=BasicConv),
            AFF_Deform(base_channel * 7, base_channel * 2, BasicConv=BasicConv)
        ])

        self.FAM1 = FAM_Deform(base_channel * 4, BasicConv=BasicConv)
        self.SCM1 = SCM_Deform(base_channel * 4, BasicConv=BasicConv)
        self.FAM2 = FAM_Deform(base_channel * 2, BasicConv=BasicConv)
        self.SCM2 = SCM_Deform(base_channel * 2, BasicConv=BasicConv)

        self.softmax = nn.Softmax(1)
        self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad])
        self.moduleKernel = misckernel.FunctionKernel.apply

        self.KernelPredictFlow = nn.ModuleList([
            BasicConv(base_channel * 4, 2, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, 2, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel, 2, kernel_size=3, relu=False, stride=1),
        ])
        self.flowup = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.KernelPredictFlowMask = nn.ModuleList([
            BasicConv(base_channel * 4, 1, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, 1, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel, 1, kernel_size=3, relu=False, stride=1),
        ])
        self.sigmoid = nn.Sigmoid()

        self.KernelOutBias = nn.ModuleList([
            BasicConv(base_channel * 4, out_channels, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, out_channels, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel, out_channels, kernel_size=3, relu=False, stride=1),
        ])

        self.KernelOutWeight = nn.ModuleList([
            BasicConv(base_channel * 4 * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2 * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
        ])

        self.KernelOutkernelx = nn.ModuleList([
            BasicConv(base_channel * 4 * 2, kernel_size, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2 * 2, kernel_size, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, kernel_size, kernel_size=3, relu=False, stride=1),
        ])

        self.KernelOutkernely = nn.ModuleList([
            BasicConv(base_channel * 4 * 2, kernel_size, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2 * 2, kernel_size, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, kernel_size, kernel_size=3, relu=False, stride=1),
        ])

        self.KernelOutAlpha = nn.ModuleList([
            BasicConv(base_channel * 4 * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2 * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
        ])

        self.KernelOutBeta = nn.ModuleList([
            BasicConv(base_channel * 4 * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2 * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
        ])

    def _build_transformer_coord_prior(self, feat):
        b, _, h, w = feat.shape
        device = feat.device
        dtype = feat.dtype
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        if self._meshgrid_supports_indexing:
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        else:
            grid_y, grid_x = torch.meshgrid(ys, xs)
        dist = torch.sqrt(grid_x ** 2 + grid_y ** 2)
        bias = torch.ones_like(dist)
        coord = torch.stack([grid_x, grid_y, dist, bias], dim=0).unsqueeze(0).repeat(b, 1, 1, 1)
        return coord

    def _build_transformer_dist(self, feat):
        if self.transformer_dist_mode == 'none':
            return None
        if self.transformer_dist_mode == 'coord':
            return self._build_transformer_coord_prior(feat)
        dist_value = torch.tensor(self.transformer_dist_value, device=feat.device, dtype=feat.dtype)
        dist = dist_value.view(1, -1).repeat(feat.size(0), 1)
        return dist

    def _detect_transformer_dist_support(self):
        try:
            sig = inspect.signature(self.transformer.forward)
            params = list(sig.parameters.values())
            if any(p.kind == p.VAR_KEYWORD for p in params):
                return 'keyword'
            if 'dist' in sig.parameters:
                dist_param = sig.parameters['dist']
                if dist_param.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                    return 'keyword'
                if dist_param.kind == inspect.Parameter.POSITIONAL_ONLY:
                    return 'positional'
            if len(params) >= 2:
                return 'positional'
        except (TypeError, ValueError):
            return 'unknown'
        return 'none'

    def _transformer_forward(self, feat, dist):
        if dist is None or self._transformer_dist_support in (None, 'none'):
            return self.transformer(feat)
        if self._transformer_dist_support == 'keyword':
            return self.transformer(feat, dist=dist)
        if self._transformer_dist_support == 'positional':
            return self.transformer(feat, dist)
        return self.transformer(feat)

    def forward(self, x, dist=None):
        """
        dist: optional prior for transformer. If provided, it overrides the generated prior.
        """
        # optional motion guidance: compute orientation-like map and fuse
        if self.use_motion_guidance:
            try:
                motion_map = compute_polar_orientation_map(x, mode=self.motion_guidance_mode)
            except Exception:
                motion_map = compute_polar_orientation_map(x, mode='simple')
            motion_proj = self.motion_proj(motion_map)  # project to inp_channels
            x = x + motion_proj

        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)

        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs_fil = list()
        outputs = list()
        Kernal_Loss = 0

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        # optional transformer fusion: apply on res2
        if self.use_transformer:
            try:
                # If caller provides dist, it takes priority over generated prior.
                dist_to_use = dist if dist is not None else self._build_transformer_dist(res2)
                res2_t = self._transformer_forward(res2, dist_to_use)
                if res2_t.shape == res2.shape:
                    res2 = res2 + res2_t
                else:
                    print(f"[WARN] transformer output shape {tuple(res2_t.shape)} != res2 {tuple(res2.shape)}; skipping add")
            except Exception as e:
                print("[WARN] transformer forward failed, skipping transformer this iter. Err:", e)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        z = self.Decoder[0](z)

        s3_kernal_flow = self.KernelPredictFlow[0](z)
        s3_kernal_flowmask = self.KernelPredictFlowMask[0](z)
        s3_kernal_flowmask = self.sigmoid(s3_kernal_flowmask)

        zx4 = torch.cat([z, x_4], 1)
        s3_kernal_flowfeat0, x_4_0 = torch.split(flow_warp(zx4, s3_kernal_flow.permute(0, 2, 3, 1)), self.dim * 4,
                                                 dim=1)
        s3_kernal_flowfeat1, x_4_1 = torch.split(flow_warp(zx4, -s3_kernal_flow.permute(0, 2, 3, 1)), self.dim * 4,
                                                 dim=1)
        x_4 = x_4_0 * s3_kernal_flowmask + x_4_1 * (1 - s3_kernal_flowmask)

        s3_kernal_bias = self.KernelOutBias[0](z)

        z = torch.cat([z, s3_kernal_flowfeat0 * s3_kernal_flowmask + s3_kernal_flowfeat1 * (1 - s3_kernal_flowmask)], 1)
        s3_kernal_weight = self.KernelOutWeight[0](z)
        s3_kernal_weight = self.softmax(s3_kernal_weight)
        s3_kernal_alpha = self.KernelOutAlpha[0](z)
        s3_kernal_beta = self.KernelOutBeta[0](z)
        s3_kernal_posx = self.KernelOutkernelx[0](z)
        s3_kernal_posy = self.KernelOutkernely[0](z)
        z = self.feat_extract[3](z)

        out3 = self.moduleKernel(
            self.modulePad(torch.cat([x_4, x_4.new_ones(x_4.size(0), 1, x_4.size(2), x_4.size(3))], 1)), s3_kernal_posx,
            s3_kernal_posy, s3_kernal_alpha, s3_kernal_beta, s3_kernal_weight)
        out3_norm = out3[:, -1:, :, :]
        out3_norm[out3_norm.abs() < 0.01] = 1.0
        out3 = out3[:, :-1, :, :] / out3_norm
        out3 += s3_kernal_bias
        if not self.inference:
            outputs.append(out3)
            outputs_fil.append(x_4)

            s3_Alpha = torch.mean(s3_kernal_weight * s3_kernal_alpha, dim=1, keepdim=True)
            s3_Beta = torch.mean(s3_kernal_weight * s3_kernal_beta, dim=1, keepdim=True)
            loss_s3_Alpha = CharbonnierFunc(s3_Alpha[:, :, :, :-1] - s3_Alpha[:, :, :, 1:]) + CharbonnierFunc(
                s3_Alpha[:, :, :-1, :] - s3_Alpha[:, :, 1:, :])
            loss_s3_Beta = CharbonnierFunc(s3_Beta[:, :, :, :-1] - s3_Beta[:, :, :, 1:]) + CharbonnierFunc(
                s3_Beta[:, :, :-1, :] - s3_Beta[:, :, 1:, :])
            Kernal_Loss += loss_s3_Alpha
            Kernal_Loss += loss_s3_Beta

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)

        s2_kernal_flow = self.KernelPredictFlow[1](z) + self.flowup(s3_kernal_flow) * 2
        s2_kernal_flowmask = self.KernelPredictFlowMask[1](z)
        s2_kernal_flowmask = self.sigmoid(s2_kernal_flowmask)

        zx2 = torch.cat([z, x_2], 1)
        s2_kernal_flowfeat0, x_2_0 = torch.split(flow_warp(zx2, s2_kernal_flow.permute(0, 2, 3, 1)), self.dim * 2,
                                                 dim=1)
        s2_kernal_flowfeat1, x_2_1 = torch.split(flow_warp(zx2, -s2_kernal_flow.permute(0, 2, 3, 1)), self.dim * 2,
                                                 dim=1)
        x_2 = x_2_0 * s2_kernal_flowmask + x_2_1 * (1 - s2_kernal_flowmask)

        s2_kernal_bias = self.KernelOutBias[1](z)

        z = torch.cat([z, s2_kernal_flowfeat0 * s2_kernal_flowmask + s2_kernal_flowfeat1 * (1 - s2_kernal_flowmask)], 1)
        s2_kernal_weight = self.KernelOutWeight[1](z)
        s2_kernal_weight = self.softmax(s2_kernal_weight)
        s2_kernal_alpha = self.KernelOutAlpha[1](z)
        s2_kernal_beta = self.KernelOutBeta[1](z)
        s2_kernal_posx = self.KernelOutkernelx[1](z)
        s2_kernal_posy = self.KernelOutkernely[1](z)
        z = self.feat_extract[4](z)

        out2 = self.moduleKernel(
            self.modulePad(torch.cat([x_2, x_2.new_ones(x_2.size(0), 1, x_2.size(2), x_2.size(3))], 1)), s2_kernal_posx,
            s2_kernal_posy, s2_kernal_alpha, s2_kernal_beta, s2_kernal_weight)
        out2_norm = out2[:, -1:, :, :]
        out2_norm[out2_norm.abs() < 0.01] = 1.0
        out2 = out2[:, :-1, :, :] / out2_norm
        out2 += s2_kernal_bias
        if not self.inference:
            outputs.append(out2)
            outputs_fil.append(x_2)

            s2_Alpha = torch.mean(s2_kernal_weight * s2_kernal_alpha, dim=1, keepdim=True)
            s2_Beta = torch.mean(s2_kernal_weight * s2_kernal_beta, dim=1, keepdim=True)
            loss_s2_Alpha = CharbonnierFunc(s2_Alpha[:, :, :, :-1] - s2_Alpha[:, :, :, 1:]) + CharbonnierFunc(
                s2_Alpha[:, :, :-1, :] - s2_Alpha[:, :, 1:, :])
            loss_s2_Beta = CharbonnierFunc(s2_Beta[:, :, :, :-1] - s2_Beta[:, :, :, 1:]) + CharbonnierFunc(
                s2_Beta[:, :, :-1, :] - s2_Beta[:, :, 1:, :])
            Kernal_Loss += loss_s2_Alpha
            Kernal_Loss += loss_s2_Beta

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)

        z = self.Decoder[2](z)

        s1_kernal_flow = self.KernelPredictFlow[2](z) + self.flowup(s2_kernal_flow) * 2
        s1_kernal_flowmask = self.KernelPredictFlowMask[2](z)
        s1_kernal_flowmask = self.sigmoid(s1_kernal_flowmask)

        zx = torch.cat([z, x], 1)
        s1_kernal_flowfeat0, x_0 = torch.split(flow_warp(zx, s1_kernal_flow.permute(0, 2, 3, 1)), self.dim, dim=1)
        s1_kernal_flowfeat1, x_1 = torch.split(flow_warp(zx, -s1_kernal_flow.permute(0, 2, 3, 1)), self.dim, dim=1)
        x = x_0 * s1_kernal_flowmask + x_1 * (1 - s1_kernal_flowmask)

        s1_kernal_bias = self.KernelOutBias[2](z)
        z = torch.cat([z, s1_kernal_flowfeat0 * s1_kernal_flowmask + s1_kernal_flowfeat1 * (1 - s1_kernal_flowmask)], 1)
        s1_kernal_weight = self.KernelOutWeight[2](z)
        s1_kernal_weight = self.softmax(s1_kernal_weight)
        s1_kernal_alpha = self.KernelOutAlpha[2](z)
        s1_kernal_beta = self.KernelOutBeta[2](z)
        s1_kernal_posx = self.KernelOutkernelx[2](z)
        s1_kernal_posy = self.KernelOutkernely[2](z)

        out = self.moduleKernel(self.modulePad(torch.cat([x, x.new_ones(x.size(0), 1, x.size(2), x.size(3))], 1)),
                                s1_kernal_posx, s1_kernal_posy, s1_kernal_alpha, s1_kernal_beta, s1_kernal_weight)
        out_norm = out[:, -1:, :, :]
        out_norm[out_norm.abs() < 0.01] = 1.0
        out = out[:, :-1, :, :] / out_norm
        out += s1_kernal_bias
        if not self.inference:
            outputs.append(out)
            outputs_fil.append(x)
            return outputs[::-1], outputs_fil[::-1]
        else:
            return out


def build_MISCKernelNet_Deform(inference=False, **kwargs):
    """创建使用可变形卷积的 MISCKernelNet（支持额外 kwargs）"""
    return MISCKernelNet_Deform(inference=inference, **kwargs)
