import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import torch.optim as optim
import matplotlib.pyplot as plt


plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.titlesize': 14,
    'figure.autolayout': True
})

def normalize_np(x):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def percentile_limits(x, low=1.0, high=99.0):
    vmin = np.percentile(x, low)
    vmax = np.percentile(x, high)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def smooth1d_circular(arr, k=7):
    pad = k // 2
    ext = np.concatenate([arr[-pad:], arr, arr[:pad]], axis=0)
    kernel = np.ones(k, dtype=np.float32) / float(k)
    out = np.convolve(ext, kernel, mode='same')[pad:-pad]
    return out



"角谱传播函数(ASM)"
def propagate_asm(field, z, wavelength, pixel_size):
    """
    field: [B, C, H, W] 复数场
    z:     传播距离 (m)
    """
    B, C, H, W = field.shape
    fx = fft.fftfreq(W, d=pixel_size)
    fy = fft.fftfreq(H, d=pixel_size)
    FX, FY = torch.meshgrid(fx, fy, indexing='xy')
    FX, FY = FX.to(field.device), FY.to(field.device)

    k = 2 * torch.pi / wavelength
    term = 1 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2
    term = torch.clamp(term, min=0)
    phase_shift = k * z * torch.sqrt(term)

    H_transfer = torch.exp(1j * phase_shift)
    field_prop = fft.ifft2(fft.fft2(field) * H_transfer)
    return field_prop


# ==========================================
# 损失函数
def total_variation_loss(img):
    dy = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    dx = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return dx + dy


def weighted_tv_loss(img, weight_map=None):
    dy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    dx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
    if weight_map is None:
        return torch.mean(dx) + torch.mean(dy)

    wy = 0.5 * (weight_map[:, :, 1:, :] + weight_map[:, :, :-1, :])
    wx = 0.5 * (weight_map[:, :, :, 1:] + weight_map[:, :, :, :-1])
    return torch.mean(dx * wx) + torch.mean(dy * wy)


def multi_scale_mse(pred, target):
    loss = F.mse_loss(pred, target)
    pred_2 = F.avg_pool2d(pred, 2, 2)
    tar_2 = F.avg_pool2d(target, 2, 2)
    loss = loss + 0.5 * F.mse_loss(pred_2, tar_2)

    pred_4 = F.avg_pool2d(pred, 4, 4)
    tar_4 = F.avg_pool2d(target, 4, 4)
    loss = loss + 0.25 * F.mse_loss(pred_4, tar_4)
    return loss


def make_valid_mask(H, W, border, device):
    """
    生成有效视场掩膜：
    - 中间可靠区域为 1
    - 四周 border 像素置 0
    用于避免 FFT 周期边界、support 平滑 padding 等带来的边缘伪影参与优化。
    """
    m = torch.zeros((1, 1, H, W), dtype=torch.float32, device=device)
    m[:, :, border:H-border, border:W-border] = 1.0
    return m


def masked_l1(x, mask, eps=1e-8):
    return torch.sum(torch.abs(x) * mask) / (torch.sum(mask) + eps)


def masked_mse(x, y, mask, eps=1e-8):
    return torch.sum(((x - y) ** 2) * mask) / (torch.sum(mask) + eps)


def masked_tv_loss(img, weight_map=None, valid_mask=None, eps=1e-8):
    dy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    dx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])

    if weight_map is None:
        wy = 1.0
        wx = 1.0
    else:
        wy = 0.5 * (weight_map[:, :, 1:, :] + weight_map[:, :, :-1, :])
        wx = 0.5 * (weight_map[:, :, :, 1:] + weight_map[:, :, :, :-1])

    if valid_mask is None:
        my = 1.0
        mx = 1.0
    else:
        my = valid_mask[:, :, 1:, :] * valid_mask[:, :, :-1, :]
        mx = valid_mask[:, :, :, 1:] * valid_mask[:, :, :, :-1]

    dy = dy * wy * my
    dx = dx * wx * mx

    if torch.is_tensor(my):
        denom_y = torch.sum(my) + eps
        denom_x = torch.sum(mx) + eps
    else:
        denom_y = dy.numel()
        denom_x = dx.numel()

    return torch.sum(dx) / denom_x + torch.sum(dy) / denom_y


def multi_scale_masked_mse(pred, target, valid_mask):
    """
    仅在有效视场内计算多尺度物理一致性。
    下采样后的 mask 采用“全有效”判定，避免边缘无效区域泄漏到低分辨率尺度。
    """
    loss = masked_mse(pred, target, valid_mask)

    pred_2 = F.avg_pool2d(pred, 2, 2)
    tar_2 = F.avg_pool2d(target, 2, 2)
    mask_2 = F.avg_pool2d(valid_mask, 2, 2)
    mask_2 = (mask_2 > 0.999).float()
    loss = loss + 0.5 * masked_mse(pred_2, tar_2, mask_2)

    pred_4 = F.avg_pool2d(pred, 4, 4)
    tar_4 = F.avg_pool2d(target, 4, 4)
    mask_4 = F.avg_pool2d(valid_mask, 4, 4)
    mask_4 = (mask_4 > 0.999).float()
    loss = loss + 0.25 * masked_mse(pred_4, tar_4, mask_4)
    return loss


# ==========================================
# FFT 估计条纹主方向
def estimate_dominant_stripe_angles(intensity, topk=2, min_sep_deg=18):
    """
    从输入全息图的频谱中估计条纹主方向对应的“频率法向角度”。
    这里返回的是 carrier 的方向角，可直接用于 cos(2*pi*f*proj)。
    """
    img = normalize_np(intensity)
    img = img - np.mean(img)
    H, W = img.shape

    Fmag = np.abs(np.fft.fftshift(np.fft.fft2(img))) ** 2
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H // 2, W // 2
    y = yy - cy
    x = xx - cx
    r = np.sqrt(x ** 2 + y ** 2)
    r_norm = r / (min(H, W) / 2.0 + 1e-8)

    # 低频太接近 DC，高频太靠近 Nyquist 都不稳定
    mask = (r_norm > 0.08) & (r_norm < 0.45)
    angle = (np.rad2deg(np.arctan2(y, x)) + 180.0) % 180.0

    hist = np.zeros(180, dtype=np.float64)
    for ang in range(180):
        sel = mask & (np.abs(angle - ang) < 0.5)
        if np.any(sel):
            hist[ang] = np.sum(Fmag[sel])
    hist = smooth1d_circular(hist.astype(np.float32), k=9)

    chosen = []
    tmp = hist.copy()
    for _ in range(topk):
        idx = int(np.argmax(tmp))
        chosen.append(float(idx))
        for j in range(180):
            d = min(abs(j - idx), 180 - abs(j - idx))
            if d < min_sep_deg:
                tmp[j] = -1.0

    if len(chosen) == 0:
        chosen = [45.0]
    return chosen


# ==========================================
# 网络
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class ObjectResUNet(nn.Module):
    """
    输出：
    - obj_amp   : 透射振幅，背景应接近 1，物体区域低于 1
    - obj_phase : 物体相位
    """
    def __init__(self, in_channels=1, base_f=24):
        super().__init__()
        self.enc1 = ResidualBlock(in_channels, base_f)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(base_f, base_f * 2)
        self.bottleneck = ResidualBlock(base_f * 2, base_f * 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = ResidualBlock(base_f * 3, base_f)
        self.out_conv = nn.Conv2d(base_f, 2, 3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(e2)
        d1 = self.dec1(torch.cat([self.up(b), e1], dim=1))
        out = self.out_conv(d1)

        amp = torch.sigmoid(out[:, 0:1, :, :])
        phase = torch.tanh(out[:, 1:2, :, :]) * torch.pi
        return amp, phase


class SmoothBackgroundModel(nn.Module):
    """
    平滑背景分支：
    用低阶多项式 / Zernike-like 基函数拟合缓慢变化的照明与低频像差。
    """
    def __init__(self, H, W):
        super().__init__()
        y = torch.linspace(-1, 1, H)
        x = torch.linspace(-1, 1, W)
        Y, X = torch.meshgrid(y, x, indexing='ij')
        R2 = X ** 2 + Y ** 2

        basis_list = [
            torch.ones_like(X),
            X, Y,
            X ** 2, Y ** 2, X * Y,
            X ** 3, Y ** 3,
            X * R2, Y * R2,
            R2, (X ** 2 - Y ** 2),
            X ** 4, Y ** 4, (X ** 2) * (Y ** 2),
            X * Y * R2, R2 ** 2
        ]
        self.num_modes = len(basis_list)
        basis = torch.stack(basis_list, dim=0).unsqueeze(0)
        self.register_buffer('basis', basis)

        self.amp_coeffs = nn.Parameter(torch.zeros(1, self.num_modes, 1, 1))
        self.amp_coeffs.data[0, 0, 0, 0] = 0.8
        self.phase_coeffs = nn.Parameter(torch.zeros(1, self.num_modes, 1, 1))

    def forward(self):
        bg_amp = torch.sum(self.amp_coeffs * self.basis, dim=1, keepdim=True)
        bg_phase = torch.sum(self.phase_coeffs * self.basis, dim=1, keepdim=True)

        bg_amp = F.softplus(bg_amp) + 1e-4
        U_bg = bg_amp * torch.exp(1j * bg_phase)
        return U_bg, bg_amp, bg_phase


class StripeBackgroundModel(nn.Module):
    """
    条纹分支：
    在平滑背景分支之上，再显式加入“多方向 + 多频率 + 低频包络”的条纹模型。

    设计思路：
    1) carrier 负责表示条纹的主方向与主频率；
    2) 低频 envelope 负责表示条纹强度在空间中的缓慢变化；
    3) 输出为一个复调制场 U_stripe，再与平滑背景场相乘。
    """
    def __init__(self, H, W, angles_deg, freq_list=(5.0, 9.0, 13.0), amp_mod_max=0.45, phase_mod_max=0.70):
        super().__init__()
        y = torch.linspace(-1, 1, H)
        x = torch.linspace(-1, 1, W)
        Y, X = torch.meshgrid(y, x, indexing='ij')
        self.register_buffer('X', X.unsqueeze(0).unsqueeze(0))
        self.register_buffer('Y', Y.unsqueeze(0).unsqueeze(0))

        env_basis = [
            torch.ones_like(X),
            X, Y,
            X ** 2, Y ** 2, X * Y,
            (X ** 2 + Y ** 2)
        ]
        env_basis = torch.stack(env_basis, dim=0).unsqueeze(0)   # [1, M, H, W]
        self.register_buffer('env_basis', env_basis)
        self.num_env = env_basis.shape[1]

        carrier_angles = []
        carrier_freqs = []
        for ang in angles_deg:
            for f in freq_list:
                carrier_angles.append(float(ang))
                carrier_freqs.append(float(f))
        self.num_carriers = len(carrier_angles)

        theta = torch.tensor(np.deg2rad(carrier_angles), dtype=torch.float32).view(1, self.num_carriers, 1, 1)
        freqs = torch.tensor(carrier_freqs, dtype=torch.float32).view(1, self.num_carriers, 1, 1)
        self.register_buffer('theta', theta)
        self.register_buffer('freqs', freqs)

        # carrier 全局初相
        self.carrier_phase = nn.Parameter(torch.zeros(1, self.num_carriers, 1, 1))

        # envelope: 振幅调制 与 相位调制 分开学习
        self.amp_env_coeffs = nn.Parameter(torch.zeros(1, self.num_carriers, self.num_env, 1, 1))
        self.phase_env_coeffs = nn.Parameter(torch.zeros(1, self.num_carriers, self.num_env, 1, 1))

        # 给常数 envelope 一个小初值，更容易长出条纹解释能力
        self.amp_env_coeffs.data[:, :, 0, :, :] = 0.02
        self.phase_env_coeffs.data[:, :, 0, :, :] = 0.01

        self.amp_mod_max = amp_mod_max
        self.phase_mod_max = phase_mod_max

    def forward(self, gain=1.0):
        # carrier 投影方向
        proj = torch.cos(self.theta) * self.X + torch.sin(self.theta) * self.Y   # [1, K, H, W]

        amp_env = torch.sum(self.amp_env_coeffs * self.env_basis.unsqueeze(1), dim=2)     # [1, K, H, W]
        phase_env = torch.sum(self.phase_env_coeffs * self.env_basis.unsqueeze(1), dim=2)  # [1, K, H, W]

        carrier_arg = 2.0 * torch.pi * self.freqs * proj + self.carrier_phase
        carrier = torch.cos(carrier_arg)
        carrier_q = torch.sin(carrier_arg)

        # 用 cos / sin 两个正交基来表达条纹的局部移相变化，更灵活但仍受限
        stripe_amp_raw = torch.sum(amp_env * carrier, dim=1, keepdim=True)
        stripe_phase_raw = torch.sum(phase_env * carrier_q, dim=1, keepdim=True)

        # 幅度调制采用 log-amplitude，更稳定且天然为正
        stripe_log_amp = gain * self.amp_mod_max * torch.tanh(stripe_amp_raw)
        stripe_phase = gain * self.phase_mod_max * torch.tanh(stripe_phase_raw)

        stripe_amp = torch.exp(stripe_log_amp)
        U_stripe = stripe_amp * torch.exp(1j * stripe_phase)
        return U_stripe, stripe_log_amp, stripe_phase


class HologramSeparatorV4(nn.Module):
    def __init__(self, H, W, stripe_angles_deg):
        super().__init__()
        self.obj_net = ObjectResUNet()
        self.bg_smooth = SmoothBackgroundModel(H, W)
        self.bg_stripe = StripeBackgroundModel(H, W, stripe_angles_deg)

    def forward(self, noise, z, wavelength, pixel_size, stripe_gain=1.0):
        obj_amp, obj_phase = self.obj_net(noise)
        U_bg_smooth, bg_amp, bg_phase = self.bg_smooth()
        U_stripe, stripe_log_amp, stripe_phase = self.bg_stripe(gain=stripe_gain)

        U_obj = obj_amp * torch.exp(1j * obj_phase)
        U_total = U_bg_smooth * U_stripe * U_obj

        U_sensor = propagate_asm(U_total, z, wavelength, pixel_size)
        I_simulated = torch.abs(U_sensor) ** 2

        return {
            'I_sim': I_simulated,
            'obj_amp': obj_amp,
            'obj_phase': obj_phase,
            'U_bg_smooth': U_bg_smooth,
            'bg_amp': bg_amp,
            'bg_phase': bg_phase,
            'U_stripe': U_stripe,
            'stripe_log_amp': stripe_log_amp,
            'stripe_phase': stripe_phase,
        }


# ==========================================
#support 软掩膜：加入高通抑制，减少条纹被罩进去
def build_soft_support_from_baseline_hp(baseline_amp, device,
                                        percentile=93.0,
                                        hp_kernel=31,
                                        smooth_kernel=7,
                                        smooth_iters=2,
                                        boost=1.5):
    """
    先从 baseline_amp 里提取“暗物体线索”，再减去一个大尺度均值背景，
    使细长条纹更不容易进入 support。
    """
    x = normalize_np(baseline_amp)
    obj_like = 1.0 - x

    t = torch.tensor(obj_like, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    low = F.avg_pool2d(t, kernel_size=hp_kernel, stride=1, padding=hp_kernel // 2)
    hp = torch.relu(t - low)

    hp_np = hp.detach().cpu().squeeze().numpy()
    thr = np.percentile(hp_np, percentile)
    hard_mask = (hp_np > thr).astype(np.float32)

    mask_t = torch.tensor(hard_mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    for _ in range(smooth_iters):
        mask_t = F.avg_pool2d(mask_t, kernel_size=smooth_kernel, stride=1, padding=smooth_kernel // 2)
    mask_t = torch.clamp(mask_t * boost, 0.0, 1.0)
    return mask_t, hp_np


# ==========================================
# 重建循环
def reconstruct(intensity_target, z, wavelength, pixel_size, iters=2200, border=6):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    H, W = intensity_target.shape
    target_I = torch.tensor(intensity_target, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    # 有效视场掩膜，用于避免边界伪影参与优化。
    valid_mask = make_valid_mask(H, W, border=border, device=device)

    # ---------- 传统 ASM baseline ----------
    U_raw = torch.sqrt(torch.clamp(target_I, min=0.0)) + 0j
    U_baseline = propagate_asm(U_raw, -z, wavelength, pixel_size)
    baseline_amp = torch.abs(U_baseline).detach().cpu().squeeze().numpy()
    baseline_phase = torch.angle(U_baseline).detach().cpu().squeeze().numpy()

    # ---------- support ----------
    support_mask, support_seed = build_soft_support_from_baseline_hp(
        baseline_amp,
        device=device,
        percentile=93.8,
        hp_kernel=31,
        smooth_kernel=7,
        smooth_iters=2,
        boost=1.45,
    )

    # 轻量加入有效区域：support 本身也不要贴到边界。
    support_mask = support_mask * valid_mask
    outside_mask = (1.0 - support_mask) * valid_mask
    tv_weight_map = (0.25 + 0.75 * outside_mask) * valid_mask

    # ---------- 自动估计条纹主方向 ----------
    stripe_angles_deg = estimate_dominant_stripe_angles(intensity_target, topk=2, min_sep_deg=18)
    print(f'估计到的条纹 carrier 方向角: {stripe_angles_deg}')

    # ---------- 初始化模型 ----------
    torch.manual_seed(42)
    fixed_noise = torch.rand((1, 1, H, W), device=device)
    net = HologramSeparatorV4(H, W, stripe_angles_deg).to(device)

    optimizer = optim.Adam([
        {'params': net.obj_net.parameters(), 'lr': 7e-4},
        {'params': net.bg_smooth.parameters(), 'lr': 3.5e-3},
        {'params': net.bg_stripe.parameters(), 'lr': 2.0e-3},
    ])

    # ---------- 损失权重 ----------
    w_tv_amp = 0.015
    w_tv_phase = 0.012
    w_sparse_inside = 0.008         # 减小：振幅更容易承接目标
    w_support_out_amp = 0.235
    w_support_out_phase = 0.055

    # amp / phase 平衡项：
    # - 减少“目标全部跑进 phase”的倾向
    # - 让 support 内至少保留一部分由振幅承接的目标对比度
    w_phase_inside = 0.010            #增大：phase 不容易承接全部目标
    w_amp_inside_pull = 0.015        #减小：amp 可以更接近透明，目标更可能跑到 phase

    w_bg_amp_reg = 0.002
    w_bg_phase_reg = 0.002

    # 条纹分支正则：
    # 1) 条纹最好主要留在背景区；
    # 2) 条纹调制不要过强，以免吃掉物体；
    w_stripe_inside_amp = 0.045
    w_stripe_inside_phase = 0.024
    w_stripe_l1_amp = 0.010
    w_stripe_l1_phase = 0.008
    w_stripe_tv_amp = 0.004
    w_stripe_tv_phase = 0.003

    # 条纹分支采用 warm-up，先让平滑背景 + 物体分支站稳，再慢慢开放条纹解释能力
    stripe_start_iter = 200
    stripe_ramp_iters = 500

    print('Start the iterative reconstruction process...')
    for i in range(iters):
        optimizer.zero_grad()

        if i < stripe_start_iter:
            stripe_gain = 0.0
        else:
            stripe_gain = min(1.0, (i - stripe_start_iter) / float(stripe_ramp_iters))

        out = net(
            fixed_noise,
            z,
            wavelength,
            pixel_size,
            stripe_gain=stripe_gain,
        )

        I_simulated = out['I_sim']
        obj_amp = out['obj_amp']
        obj_phase = out['obj_phase']
        bg_amp = out['bg_amp']
        bg_phase = out['bg_phase']
        stripe_log_amp = out['stripe_log_amp']
        stripe_phase = out['stripe_phase']

        # 1) 多尺度物理一致性
        loss_data = multi_scale_masked_mse(I_simulated, target_I, valid_mask)

        # 2) support 外：物体应尽量透明、相位尽量接近 0
        loss_support = (
            masked_l1((1.0 - obj_amp), outside_mask) * w_support_out_amp +
            masked_l1(obj_phase, outside_mask) * w_support_out_phase
        )

        # 3) support 内：降低对 amplitude 的压制，让物体不至于全部退回到 phase
        loss_sparse_inside = (
            masked_l1((1.0 - obj_amp), support_mask) * w_sparse_inside
        )

        # 4) support 内 phase 轻约束：
        #    防止相位单独承接全部目标，鼓励 amp / phase 更均衡地解释物体。
        loss_phase_inside = (
            masked_l1(obj_phase, support_mask) * w_phase_inside
        )

        # 5) support 内弱振幅承接项：
        #    不是强迫振幅很暗，而是要求在 support 内至少出现一点对比度，
        #    避免 obj_amp 整幅图都退回到接近 1。
        desired_contrast = 0.10 * support_mask   #减小：amp 可以更淡，更多交给 phase
        loss_amp_inside_pull = (
            masked_l1(F.relu(desired_contrast - (1.0 - obj_amp)), support_mask) * w_amp_inside_pull
        )

        # 6) 加权 TV：背景区更强，物体区更弱
        loss_tv = (
            masked_tv_loss(obj_amp, weight_map=tv_weight_map, valid_mask=valid_mask) * w_tv_amp +
            masked_tv_loss(obj_phase, weight_map=tv_weight_map, valid_mask=valid_mask) * w_tv_phase
        )

        # 7) 平滑背景正则
        bg_mean = torch.sum(bg_amp * valid_mask) / (torch.sum(valid_mask) + 1e-8)
        loss_bg_reg = (
            torch.sum(((bg_amp - bg_mean) ** 2) * valid_mask) / (torch.sum(valid_mask) + 1e-8) * w_bg_amp_reg +
            masked_tv_loss(bg_phase, valid_mask=valid_mask) * w_bg_phase_reg
        )

        # 8) 条纹分支正则：
        #    - 条纹尽量少侵入 support 内；
        #    - 条纹总能量不要无限长；
        #    - 条纹 envelope 也保持适度平滑。
        loss_stripe_reg = (
            masked_l1(stripe_log_amp, support_mask) * w_stripe_inside_amp +
            masked_l1(stripe_phase, support_mask) * w_stripe_inside_phase +
            masked_l1(stripe_log_amp, valid_mask) * w_stripe_l1_amp +
            masked_l1(stripe_phase, valid_mask) * w_stripe_l1_phase +
            masked_tv_loss(stripe_log_amp, valid_mask=valid_mask) * w_stripe_tv_amp +
            masked_tv_loss(stripe_phase, valid_mask=valid_mask) * w_stripe_tv_phase
        )
        # 总损失
        loss = (
            loss_data +
            loss_support +
            loss_sparse_inside +
            loss_phase_inside +
            loss_amp_inside_pull +
            loss_tv +
            loss_bg_reg +
            loss_stripe_reg
        )
        loss.backward()
        optimizer.step()

        if (i + 1) % 300 == 0:
            print(
                f'Iter {i + 1:04d} | '
                f'Data: {loss_data.item():.5f} | '
                f'Support: {loss_support.item():.5f} | '
                f'Stripe: {loss_stripe_reg.item():.5f} | '
                f'gain={stripe_gain:.2f}'
            )

    # ---------- 输出 ----------
    obj_amp_np = obj_amp.detach().cpu().squeeze().numpy()
    obj_phase_np = obj_phase.detach().cpu().squeeze().numpy()
    bg_amp_np = bg_amp.detach().cpu().squeeze().numpy()
    bg_phase_np = bg_phase.detach().cpu().squeeze().numpy()
    stripe_log_amp_np = stripe_log_amp.detach().cpu().squeeze().numpy()
    stripe_phase_np = stripe_phase.detach().cpu().squeeze().numpy()
    I_sim_np = I_simulated.detach().cpu().squeeze().numpy()
    error_map = np.abs(I_sim_np - intensity_target)
    support_np = support_mask.detach().cpu().squeeze().numpy()
    valid_np = valid_mask.detach().cpu().squeeze().numpy()

    obj_contrast = 1.0 - obj_amp_np
    stripe_amp_vis = np.exp(stripe_log_amp_np)
    total_bg_amp = (bg_amp_np * stripe_amp_vis)

    return {
        'base_amp': baseline_amp,
        'base_phase': baseline_phase,
        'obj_amp': obj_amp_np,
        'obj_phase': obj_phase_np,
        'obj_contrast': obj_contrast,
        'bg_amp': bg_amp_np,
        'bg_phase': bg_phase_np,
        'stripe_log_amp': stripe_log_amp_np,
        'stripe_phase': stripe_phase_np,
        'stripe_amp_vis': stripe_amp_vis,
        'total_bg_amp': total_bg_amp,
        'I_sim': I_sim_np,
        'error_map': error_map,
        'support_mask': support_np,
        'valid_mask': valid_np,
        'support_seed': support_seed,
        'stripe_angles_deg': stripe_angles_deg,
    }


# ==========================================
#  绘图
def draw_results(intensity, results, save_path='reconstruction_amp_balanced.png'):
    fig, axs = plt.subplots(2, 6, figsize=(28, 9))

    obj_amp_vis = results['obj_amp'].copy()
    obj_amp_vis[results['valid_mask'] < 0.5] = 1.0

    obj_contrast_vis = results['obj_contrast'].copy()
    obj_contrast_vis[results['valid_mask'] < 0.5] = 0.0

    obj_phase_vis = results['obj_phase'].copy()
    obj_phase_vis[results['valid_mask'] < 0.5] = 0.0

    error_map_vis = results['error_map'].copy()
    error_map_vis[results['valid_mask'] < 0.5] = 0.0

    stripe_amp_vis = results['stripe_amp_vis'].copy()
    stripe_amp_vis[results['valid_mask'] < 0.5] = 1.0

    # ---- 第一行 ----
    im0 = axs[0, 0].imshow(intensity, cmap='gray')
    axs[0, 0].set_title('(a) Raw Hologram')
    plt.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

    im1 = axs[0, 1].imshow(results['I_sim'], cmap='gray')
    axs[0, 1].set_title('(b) Simulated $I_{sim}$')
    plt.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

    evmin, evmax = percentile_limits(error_map_vis, 1, 99)
    im2 = axs[0, 2].imshow(error_map_vis, cmap='magma', vmin=evmin, vmax=evmax)
    axs[0, 2].set_title('(c) Error Map')
    plt.colorbar(im2, ax=axs[0, 2], fraction=0.046, pad=0.04)

    im3 = axs[0, 3].imshow(results['bg_amp'] ** 2, cmap='viridis')
    axs[0, 3].set_title('(d) Smooth Background')
    plt.colorbar(im3, ax=axs[0, 3], fraction=0.046, pad=0.04)

    svmin, svmax = percentile_limits(stripe_amp_vis, 1, 99)
    im4 = axs[0, 4].imshow(stripe_amp_vis, cmap='cividis', vmin=svmin, vmax=svmax)
    axs[0, 4].set_title('(e) Stripe Modulation')
    plt.colorbar(im4, ax=axs[0, 4], fraction=0.046, pad=0.04)

    im5 = axs[0, 5].imshow(results['support_mask'], cmap='cividis', vmin=0, vmax=1)
    axs[0, 5].set_title('(f) Soft Support Mask')
    plt.colorbar(im5, ax=axs[0, 5], fraction=0.046, pad=0.04)

    # ---- 第二行 ----
    bmin, bmax = percentile_limits(results['base_amp'], 1, 99)
    im6 = axs[1, 0].imshow(results['base_amp'], cmap='gray', vmin=bmin, vmax=bmax)
    axs[1, 0].set_title('(g) Baseline ASM (Amp)')
    plt.colorbar(im6, ax=axs[1, 0], fraction=0.046, pad=0.04)

    im7 = axs[1, 1].imshow(results['base_phase'], cmap='twilight_shifted', vmin=-np.pi, vmax=np.pi)
    axs[1, 1].set_title('(h) Baseline ASM (Phase)')
    plt.colorbar(im7, ax=axs[1, 1], fraction=0.046, pad=0.04)

    amin, amax = percentile_limits(obj_amp_vis, 1, 99.5)
    im8 = axs[1, 2].imshow(obj_amp_vis, cmap='gray', vmin=amin, vmax=amax)
    axs[1, 2].set_title('(i) Clean Amp (gray)')
    plt.colorbar(im8, ax=axs[1, 2], fraction=0.046, pad=0.04)

    cmin, cmax = percentile_limits(obj_contrast_vis, 1, 99.5)
    im9 = axs[1, 3].imshow(obj_contrast_vis, cmap='cividis', vmin=cmin, vmax=cmax)
    axs[1, 3].set_title('(j) Clean Contrast (1-Amp)')
    plt.colorbar(im9, ax=axs[1, 3], fraction=0.046, pad=0.04)

    im10 = axs[1, 4].imshow(obj_phase_vis, cmap='twilight_shifted', vmin=-np.pi, vmax=np.pi)
    axs[1, 4].set_title('(k) Clean Phase')
    plt.colorbar(im10, ax=axs[1, 4], fraction=0.046, pad=0.04)

    tvmin, tvmax = percentile_limits(results['total_bg_amp'], 1, 99)
    im11 = axs[1, 5].imshow(results['total_bg_amp'], cmap='viridis', vmin=tvmin, vmax=tvmax)
    axs[1, 5].set_title('(l) Total BG Amp')
    plt.colorbar(im11, ax=axs[1, 5], fraction=0.046, pad=0.04)

    for ax in axs.flat:
        ax.axis('off')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    WAVELENGTH = 632.8e-9
    PIXEL_SIZE = 6.9e-6
    Z_DISTANCE = 0.02100
    ITERS = 2200
    IMG_SIZE = 256

    if os.path.exists('data\sample_007\patch_0013\patch_0013.npy'):
        intensity = np.load('data\sample_007\patch_0013\patch_0013.npy').astype(np.float32)
    elif os.path.exists('holo.jpg'):
        intensity = np.array(Image.open('holo.jpg').convert('L'), dtype=np.float32)
    else:
        Y, X = np.ogrid[-1:1:IMG_SIZE * 1j, -1:1:IMG_SIZE * 1j]
        intensity = np.exp(-(X ** 2 + Y ** 2)) + np.random.randn(IMG_SIZE, IMG_SIZE) * 0.05

    intensity = intensity[:IMG_SIZE, :IMG_SIZE]
    intensity = normalize_np(intensity)

    results = reconstruct(
        intensity_target=intensity,
        z=Z_DISTANCE,
        wavelength=WAVELENGTH,
        pixel_size=PIXEL_SIZE,
        iters=ITERS,
    )

    draw_results(intensity, results, save_path='PNG/reconstruction_sample_007_patch_0013.png')
