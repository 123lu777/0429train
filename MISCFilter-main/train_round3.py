# train_GoPro_Deform_phase3.py (your round3 script)
import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# 确保能 import 到 /media/JYJ/新加卷/ZJL/420/basicsr/...
sys.path.insert(0, "/media/JYJ/新加卷/ZJL/420")

import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np
import math
import pickle

import utils
from data.data_RGB import get_training_data, get_validation_data

# 导入可变形卷积版本的模型
from models.MISCFilterNet_Deform import MISCKernelNet_Deform as myNet

from loss import losses
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from tools.get_parameter_number import get_parameter_number
import kornia

# AMP 支持与梯度裁剪
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
start_epoch = 1


class Config:
    # 数据集路径
    train_dir = "./dataset/GOPRO_Large"
    train_meta = "./dataset/GOPRO_Large/GOPRO_train_list.txt"
    val_dir = "./dataset/GOPRO_Large"
    val_meta = "./dataset/GOPRO_Large/GOPRO_test_list.txt"

    # 模型保存路径
    model_save_dir = "./checkpoints_deform"
    dataset = "GoPro"
    session = "MISCFilter_Deform_GoPro_phase3promax"

    # 训练参数
    patch_size = 256
    num_epochs = 3000
    batch_size = 10
    val_epochs = 2
    print_epochs = 1

    # mdt 插在 res2（下采样 2x）上，因此 img_size 应该是 patch_size//2
    transformer_img_size = patch_size // 2  # 256 -> 128

    # 可变形卷积设置（与之前保持一致）
    use_deform_in_feat = True
    use_deform_in_encoder = True

    # 恢复训练 / 预训练设置
    RESUME = False
    Pretrain = True
    model_pre_dir = "/media/JYJ/新加卷/ZJL/420/checkpoints_deform/GoPro/MISCFilter_Deform_GoPro_round2/model_best.pth"

    # ---------- Polar / Transformer 开关 -------------
    use_motion_guidance = True
    motion_guidance_mode = "polar_utils"
    use_polar_sampling = True

    use_transformer = True
    transformer_pretrained = ""  # optional: MDT 的权重（如果你有就填）
    freeze_transformer = False   # 直接训练，不冻结

    # 与主体网络同学习率
    transformer_lr_mult = 1.0

    # provided keywords for transformer param detection (case-insensitive)
    # 增加 feat_to_rgb/rgb_to_feat，确保 bridge 的两个 1x1 conv 也会进入 transformer 组
    transformer_param_keywords = ["transformer", "mdt", "attn", "feat_to_rgb", "rgb_to_feat"]

    # AMP and accumulation
    use_amp = True
    grad_accum_steps = 1

    # DataLoader / IO
    num_workers = 8
    pin_memory = True

    # validation batch size
    val_batch_size = 8


args = Config()

# --------- bookkeeping and paths ----------
dataset = args.dataset
session = args.session
patch_size = args.patch_size

model_dir = os.path.join(args.model_save_dir, dataset, session)
utils.mkdir(model_dir)
log_dir = os.path.join(args.model_save_dir, dataset, session, "log.txt")

train_dir = args.train_dir
val_dir = args.val_dir

train_meta = args.train_meta
val_meta = args.val_meta

num_epochs = args.num_epochs
batch_size = args.batch_size
val_epochs = args.val_epochs

start_lr = 1e-4
end_lr = 1e-6

best_psnr = 0.0
best_epoch = 0

######### Model ###########
model_restoration = myNet(
    use_deform_in_feat=args.use_deform_in_feat,
    use_deform_in_encoder=args.use_deform_in_encoder,
    use_motion_guidance=args.use_motion_guidance,
    motion_guidance_mode=args.motion_guidance_mode,
    use_polar_sampling=args.use_polar_sampling,
    use_transformer=args.use_transformer,
    transformer_pretrained=(args.transformer_pretrained if args.transformer_pretrained else None),
    freeze_transformer=args.freeze_transformer,
    transformer_img_size=args.transformer_img_size,  # ★ 新增：传给 mdt(img_size)
)

total_num, trainable_num = get_parameter_number(model_restoration)
print("=" * 60)
print("Model: MISCKernelNet with Deformable Convolution + Transformer (Phase‑3)")
print("Use Deform in Feature Extraction:", args.use_deform_in_feat)
print("Use Deform in Encoder/Decoder:", args.use_deform_in_encoder)
print("Use motion guidance:", args.use_motion_guidance, "mode:", args.motion_guidance_mode)
print("Use transformer:", args.use_transformer, "freeze_transformer:", args.freeze_transformer)
print("Transformer img_size:", args.transformer_img_size)
print("=" * 60)
print("Total params:  ", total_num)
print("Trainable params: ", trainable_num)

with open(log_dir, "a+") as f:
    f.write("=" * 60 + "\n")
    f.write("Model: MISCKernelNet with Deformable Convolution + Transformer (Phase‑3)\n")
    f.write("Use Deform in Feature Extraction: {}\n".format(args.use_deform_in_feat))
    f.write("Use Deform in Encoder/Decoder: {}\n".format(args.use_deform_in_encoder))
    f.write("Use Motion guidance: {} mode: {}\n".format(args.use_motion_guidance, args.motion_guidance_mode))
    f.write("Use transformer: {} freeze_transformer: {}\n".format(args.use_transformer, args.freeze_transformer))
    f.write("Transformer img_size: {}\n".format(args.transformer_img_size))
    f.write("=" * 60 + "\n")
    f.write("Total: {}\n".format(total_num))
    f.write("Trainable: {}\n".format(trainable_num))


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        msg = str(e)
        is_unpickle = isinstance(e, pickle.UnpicklingError) or "Weights only" in msg or "weights_only" in msg or "Unsupported global" in msg
        if not is_unpickle:
            raise
        print("Warning: torch.load failed due to weights_only/unpickling restriction. Retrying with weights_only=False (only do this for trusted checkpoints).")
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            print("weights_only argument is not supported by this torch build; re-raising original error.")
            raise
        except Exception as e2:
            print("Retry with weights_only=False also failed:", e2)
            raise


def enhanced_load_and_match(model, ckpt_path, map_location="cpu", verbose=True):
    import os
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Pretrain checkpoint not found: {ckpt_path}")
    ckpt = safe_torch_load(ckpt_path, map_location=map_location)
    sd_ckpt = ckpt.get("state_dict", ckpt)

    def normalize_key(k):
        prefixes = ["module.", "model.", "net.", "module.module."]
        kk = k
        for p in prefixes:
            if kk.startswith(p):
                kk = kk[len(p):]
        return kk

    ckpt_norm_map = {}
    for orig_k in sd_ckpt.keys():
        norm = normalize_key(orig_k)
        ckpt_norm_map.setdefault(norm, []).append(orig_k)

    model_sd = model.state_dict()
    new_model_sd = model_sd.copy()
    used_ckpt_keys = set()
    matched = {}

    for m_k in model_sd.keys():
        if m_k in ckpt_norm_map:
            chosen = None
            for candidate in ckpt_norm_map[m_k]:
                try:
                    if sd_ckpt[candidate].shape == model_sd[m_k].shape:
                        chosen = candidate
                        break
                except Exception:
                    continue
            if chosen is not None:
                new_model_sd[m_k] = sd_ckpt[chosen].clone()
                matched[m_k] = chosen
                used_ckpt_keys.add(chosen)

    for m_k in model_sd.keys():
        if m_k in matched:
            continue
        for norm_k, orig_list in ckpt_norm_map.items():
            if norm_k.endswith(m_k):
                for cand in orig_list:
                    if cand in used_ckpt_keys:
                        continue
                    try:
                        if sd_ckpt[cand].shape == model_sd[m_k].shape:
                            new_model_sd[m_k] = sd_ckpt[cand].clone()
                            matched[m_k] = cand
                            used_ckpt_keys.add(cand)
                            break
                    except Exception:
                        continue
            if m_k in matched:
                break

    for m_k in model_sd.keys():
        if m_k in matched:
            continue
        if m_k in sd_ckpt and hasattr(sd_ckpt[m_k], "shape") and sd_ckpt[m_k].shape == model_sd[m_k].shape:
            new_model_sd[m_k] = sd_ckpt[m_k].clone()
            matched[m_k] = m_k
            used_ckpt_keys.add(m_k)

    model.load_state_dict(new_model_sd)
    total = len(model_sd)
    n_matched = len(matched)
    missing_model_keys = sorted([k for k in model_sd.keys() if k not in matched])
    unexpected_ckpt_keys = sorted([k for k in sd_ckpt.keys() if k not in used_ckpt_keys])

    if verbose:
        print(f"Enhanced load: matched {n_matched}/{total} model params.")
        if missing_model_keys:
            print("Missing model keys (first 30):")
            for k in missing_model_keys[:30]:
                print(" ", k)
        if unexpected_ckpt_keys:
            print("Unexpected ckpt keys (first 30):")
            for k in unexpected_ckpt_keys[:30]:
                print(" ", k)
    return n_matched, total, missing_model_keys, unexpected_ckpt_keys


if args.Pretrain and args.model_pre_dir:
    try:
        n_loaded, n_total, missing_keys, unexpected_keys = enhanced_load_and_match(model_restoration, args.model_pre_dir, map_location="cpu", verbose=True)
        with open(log_dir, "a+") as f:
            f.write(f"Pretrained weights load summary: matched {n_loaded}/{n_total}\n")
            if missing_keys:
                f.write("Missing model keys (first 50):\n")
                for k in missing_keys[:50]:
                    f.write(k + "\n")
    except Exception as e:
        print("Pretrain enhanced load failed:", e)
        with open(log_dir, "a+") as f:
            f.write(f"Pretrain enhanced load failed: {e}\n")

model_restoration.cuda()

try:
    print("Actual use_motion_guidance:", model_restoration.use_motion_guidance)
except Exception as e:
    print("Warning: could not read model_restoration.use_motion_guidance:", e)

with open(os.path.join(model_dir, "model_param_names.txt"), "w") as f:
    for i, (n, p) in enumerate(model_restoration.named_parameters()):
        f.write(f"{i:04d}: {n}\t{tuple(p.shape)}\n")
print("Wrote model_param_names.txt to", os.path.join(model_dir, "model_param_names.txt"))

provided_keywords = args.transformer_param_keywords
detected_names = [n for n, p in model_restoration.named_parameters() if any(k.lower() in n.lower() for k in provided_keywords)]

if len(detected_names) == 0:
    broad_candidates = ["attn", "self_attn", "q_proj", "k_proj", "v_proj", "multihead", "transform", "mdt", "ffn", "mlp", "norm", "proj", "linear", "feat_to_rgb", "rgb_to_feat"]
    for n, p in model_restoration.named_parameters():
        low = n.lower()
        for c in broad_candidates:
            if c in low:
                detected_names.append(n)
                break

print(f"Auto-detected transformer-like param count: {len(detected_names)} (sample up to 200):")
for n in detected_names[:200]:
    print("  ", n)

with open(os.path.join(model_dir, "transformer_param_list.txt"), "w") as f:
    for n in detected_names:
        p = dict(model_restoration.named_parameters())[n]
        f.write(f"{n}\t{tuple(p.shape)}\trequires_grad={p.requires_grad}\n")

print("Wrote transformer_param_list.txt to", os.path.join(model_dir, "transformer_param_list.txt"))
with open(log_dir, "a+") as f:
    f.write(f"Auto-detected transformer-like param count: {len(detected_names)}\n")
    for n in detected_names[:200]:
        f.write(n + "\n")

transformer_param_names_set = set()
for n in detected_names:
    key = n
    if key.startswith("module."):
        key = key[len("module."):]
    transformer_param_names_set.add(key)

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
    model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

non_transformer_params = []
transformer_params_reserved = []

for name, p in model_restoration.named_parameters():
    key = name[len("module."):] if name.startswith("module.") else name
    if key in transformer_param_names_set:
        transformer_params_reserved.append(p)
    else:
        if p.requires_grad:
            non_transformer_params.append(p)

param_groups = [
    {"params": non_transformer_params, "lr": start_lr},
    {"params": transformer_params_reserved, "lr": start_lr * args.transformer_lr_mult},
]
optimizer = optim.Adam(param_groups, lr=start_lr, betas=(0.9, 0.999), eps=1e-8)

print("Optimizer param_groups count:", len(optimizer.param_groups))
for idx, pg in enumerate(optimizer.param_groups):
    try:
        n_tensors = len(pg["params"])
        total_param_count = sum(p.numel() for p in pg["params"]) if n_tensors > 0 else 0
    except Exception:
        n_tensors = 0
        total_param_count = 0
    print(f"  group[{idx}]: tensors={n_tensors}, total_param_count={total_param_count}, lr={pg.get('lr',None)}")

with open(log_dir, "a+") as f:
    f.write(f"Optimizer groups: {len(optimizer.param_groups)}\n")
    for idx, pg in enumerate(optimizer.param_groups):
        f.write(f"  group[{idx}]: tensors={len(pg['params'])}, lr={pg.get('lr',None)}\n")

warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs - warmup_epochs, eta_min=end_lr)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

criterion_char = losses.CharbonnierLoss()
criterion_edge = losses.EdgeLoss()
criterion_fft = losses.fftLoss()

train_dataset = get_training_data(train_dir, train_meta, {"patch_size": patch_size})
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=False, pin_memory=args.pin_memory)

val_dataset = get_validation_data(val_dir, val_meta, {"patch_size": patch_size})
val_loader = DataLoader(dataset=val_dataset, batch_size=args.val_batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False, pin_memory=args.pin_memory)

print("===> Start Epoch {} End Epoch {}".format(start_epoch, num_epochs + 1))
with open(log_dir, "a+") as f:
    f.write("===> Start Epoch {} End Epoch {} \n".format(start_epoch, num_epochs + 1))
    f.write("===> Loading datasets\n")

iter_count = 0
scaler = GradScaler(enabled=args.use_amp)


def atomic_save(state, path):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


for epoch in range(start_epoch, num_epochs + 1):
    epoch_start_time = time.time()
    epoch_loss = 0

    model_restoration.train()
    sum_loss_char = sum_loss_fft = sum_loss_edge = sum_loss_inter = 0.0
    n_iters = 0

    for i, data in enumerate(train_loader, 0):
        if (i % args.grad_accum_steps) == 0:
            optimizer.zero_grad()

        target_ = data[0].cuda()
        input_ = data[1].cuda()
        target = kornia.geometry.transform.build_pyramid(target_, 3)

        with autocast(enabled=args.use_amp):
            restored, restored_inter = model_restoration(input_)
            loss_fft = criterion_fft(restored[0], target[0]) + criterion_fft(restored[1], target[1]) + criterion_fft(restored[2], target[2])
            loss_char = criterion_char(restored[0], target[0]) + criterion_char(restored[1], target[1]) + criterion_char(restored[2], target[2])
            loss_edge = criterion_edge(restored[0], target[0]) + criterion_edge(restored[1], target[1]) + criterion_edge(restored[2], target[2])
            loss_char_inter = criterion_char(restored_inter[0], target[0]) + criterion_char(restored_inter[1], target[1]) + criterion_char(restored_inter[2], target[2])
            loss = loss_char + loss_char_inter + 0.01 * loss_fft + 0.05 * loss_edge

        sum_loss_char += loss_char.item()
        sum_loss_fft += loss_fft.item()
        sum_loss_edge += loss_edge.item()
        sum_loss_inter += loss_char_inter.item()
        n_iters += 1

        if args.use_amp:
            scaler.scale(loss).backward()
            if (i % args.grad_accum_steps) == (args.grad_accum_steps - 1):
                scaler.unscale_(optimizer)
                clip_grad_norm_(model_restoration.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            if (i % args.grad_accum_steps) == (args.grad_accum_steps - 1):
                clip_grad_norm_(model_restoration.parameters(), 1.0)
                optimizer.step()

        epoch_loss += loss.item()
        iter_count += 1

        if i % 100 == 0:
            print(f"epoch {epoch} iter {i} loss {loss.item():.4f}")

    if n_iters > 0:
        avg_loss_char = sum_loss_char / n_iters
        avg_loss_fft = sum_loss_fft / n_iters
        avg_loss_edge = sum_loss_edge / n_iters
        avg_loss_inter = sum_loss_inter / n_iters
    else:
        avg_loss_char = avg_loss_fft = avg_loss_edge = avg_loss_inter = 0.0

    if epoch % args.print_epochs == 0:
        print(f"avg loss_char {avg_loss_char:.4f} avg loss_fft {avg_loss_fft:.4f} avg loss_edge {avg_loss_edge:.4f}")

    if epoch % val_epochs == 0:
        model_restoration.eval()
        psnr_val_rgb = []
        with torch.no_grad():
            for ii, data_val in enumerate(val_loader, 0):
                target = data_val[0].cuda()
                input_ = data_val[1].cuda()
                with autocast(enabled=args.use_amp):
                    restored, _ = model_restoration(input_)
                for res, tar in zip(restored[0], target):
                    psnr_val_rgb.append(utils.torchPSNR(res, tar))

        if len(psnr_val_rgb) > 0:
            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
        else:
            psnr_val_rgb = 0.0

        print(f"val/psnr {psnr_val_rgb:.4f} epoch {epoch}")
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            atomic_save({
                "epoch": epoch,
                "state_dict": model_restoration.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_psnr": best_psnr,
                "best_epoch": best_epoch
            }, os.path.join(model_dir, "model_best.pth"))

        with open(log_dir, "a+") as f:
            f.write("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f] \n" % (epoch, psnr_val_rgb, best_epoch, best_psnr))

    scheduler.step()

    current_lr_group0 = scheduler.get_last_lr()[0]
    current_lr_group1 = scheduler.get_last_lr()[1] if len(scheduler.get_last_lr()) > 1 else None
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLR group0: {:.6f}".format(
        epoch, time.time() - epoch_start_time, epoch_loss, current_lr_group0))
    if current_lr_group1 is not None:
        print("Transformer LR: {:.6e}".format(current_lr_group1))
    print("------------------------------------------------------------------")
    with open(log_dir, "a+") as f:
        f.write("------------------------------------------------------------------\n")
        f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLR group0: {:.6f}".format(
            epoch, time.time() - epoch_start_time, epoch_loss, current_lr_group0))
        if current_lr_group1 is not None:
            f.write("\tTransformer LR: {:.6e}".format(current_lr_group1))
        f.write("\n------------------------------------------------------------------\n")

    atomic_save({
        "epoch": epoch,
        "state_dict": model_restoration.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_psnr": best_psnr,
        "best_epoch": best_epoch
    }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))

    atomic_save({
        "epoch": epoch,
        "state_dict": model_restoration.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_psnr": best_psnr,
        "best_epoch": best_epoch
    }, os.path.join(model_dir, "model_latest.pth"))
