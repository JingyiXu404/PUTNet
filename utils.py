import torch
import torch.nn.functional as F
from torch.nn.functional import pad
import torch.nn as nn

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay*init_lr
        lr=param_group['lr']
    return lr
def roll(psf, kernel_size, reverse=False):
    for axis, axis_size in zip([-2, -1], kernel_size):
        psf = torch.roll(psf,int(axis_size / 2) * (-1 if not reverse else 1),dims=axis)
    return psf

def roll_shift(real_part, imag_part):
    # 对实部进行类似fftshift的操作
    shifted_real_part = torch.roll(real_part, shifts=(real_part.shape[-2] // 2, real_part.shape[-1] // 2), dims=(-2, -1))
    # 对虚部进行类似fftshift的操作
    shifted_imag_part = torch.roll(imag_part, shifts=(imag_part.shape[-2] // 2, imag_part.shape[-1] // 2), dims=(-2, -1))
    return shifted_real_part, shifted_imag_part

def iroll_shift(real_part, imag_part):
    # 对实部进行类似ifftshift的操作
    ifftshifted_real_part = torch.roll(real_part, shifts=(-real_part.shape[-2] // 2, -real_part.shape[-1] // 2), dims=(-2, -1))
    # 对虚部进行类似ifftshift的操作
    ifftshifted_imag_part = torch.roll(imag_part, shifts=(-imag_part.shape[-2] // 2, -imag_part.shape[-1] // 2), dims=(-2, -1))
    return ifftshifted_real_part, ifftshifted_imag_part
def fftshift(x):
    """
    实现类似np.fft.fftshift的功能，将零频率分量移到频谱中心（针对torch张量）。

    参数:
    - x (torch.Tensor): 输入的频域张量，形状通常包含频域维度（如[batch_size, channels, height, width]等且可能有表示实部虚部维度）。

    返回:
    - shifted_x (torch.Tensor): 经过类似fftshift操作后的张量，形状与输入x相同。
    """
    dims = list(range(2, len(x.shape)))  # 获取要进行fftshift操作的维度（通常是频域相关维度）
    for dim in dims:
        shift_amount = x.shape[dim] // 2
        x = torch.roll(x, shifts=shift_amount, dims=dim)
    return x
def ifftshift(x):
    """
    实现类似np.fft.ifftshift的功能，将经过fftshift后的频谱恢复到原始布局（针对torch张量）。

    参数:
    - x (torch.Tensor): 输入的经过类似fftshift操作后的频域张量，形状通常包含频域维度（如[batch_size, channels, height, width]等且可能有表示实部虚部维度）。

    返回:
    - shifted_x (torch.Tensor): 经过类似ifftshift操作后的张量，形状与输入x相同。
    """
    dims = list(range(2, len(x.shape)))  # 获取要进行ifftshift操作的维度（通常是频域相关维度）
    for dim in dims:
        shift_amount = x.shape[dim] // 2
        x = torch.roll(x, shifts=-shift_amount, dims=dim)
    return x

def amp_pha(x):
    xp = torch.rfft(x, 2).unsqueeze(1)
    real_part = xp[..., 0]
    imag_part = xp[..., 1]
    magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
    phase = torch.atan2(imag_part, real_part)
    return magnitude, phase

# def amp_pha(x):
#     xp = torch.rfft(x, 2).unsqueeze(1)
#     real_part = xp[..., 0]
#     imag_part = xp[..., 1]
#     real_part, imag_part = roll_shift(real_part, imag_part)
#     magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
#     phase = torch.atan2(imag_part, real_part)
#     return magnitude, phase

def IFFT_xp(xp, size_x):
    complex_xp = torch.stack([torch.cos(xp),torch.sin(xp)], dim=-1).squeeze(1)
    out_xp = torch.irfft(complex_xp, 2,signal_sizes=list(size_x[-2:]))
    return out_xp

    # xp_real = torch.zeros_like(xp)
    # real_part, imag_part = (xp_real, xp)
    # complex_spectrum = torch.stack([real_part, imag_part], dim=-1).squeeze(1)
    # return torch.irfft(complex_spectrum, 2, signal_sizes=list(size_x[-2:]))

def IFFT_xm(xm, size_x):
    complex_xm = torch.stack([xm,torch.zeros_like(xm)], dim=-1).squeeze(1)
    out_xm = torch.irfft(complex_xm, 2, signal_sizes=list(size_x[-2:]))
    return out_xm
# def p2o(psf, shape):
#     kernel_size = (psf.size(-2), psf.size(-1))
#     psf = F.pad(psf,[0, shape[1] - kernel_size[1], 0, shape[0] - kernel_size[0]])
#     # print('0',psf.shape)
#     psf = roll(psf, kernel_size)
#     # print('1',psf.shape)
#     psf = torch.fft.rfft2(psf, dim=(-1),norm = "ortho")
#     psf = torch.stack((psf.real, psf.imag), -1)
#     # print('2',psf.shape)
#     return psf
def p2o(psf, shape):
    kernel_size = (psf.size(-2), psf.size(-1))
    psf = F.pad(psf,[0, shape[1] - kernel_size[1], 0, shape[0] - kernel_size[0]])
    psf = roll(psf, kernel_size)
    psf = torch.rfft(psf, 2)
    return psf
def reshape_params4(lambda1,z):
    lambda1 = lambda1.reshape(lambda1.size(0), lambda1.size(1), lambda1.size(2), lambda1.size(3))
    return lambda1[:,:,:z.size(2),:z.size(3)]
def reshape_params(lambda1,Z):
    lambda1 = lambda1.unsqueeze(1)/ Z.size(2)
    lambda1 = lambda1.view(lambda1.size(0), lambda1.size(2), lambda1.size(1), lambda1.size(3),lambda1.size(4)// 2, 2)
    return lambda1[:,:,:,:Z.size(3),:,:]
def reshape_params3(lambda1,Z):
    return lambda1[:,:,:Z.size(2),:Z.size(3)]
def cdiv(x, y):
    # complex division
    a, b = x[..., 0], x[..., 1]
    c, d = y[..., 0], y[..., 1]
    cd2 = c**2 + d**2
    return torch.stack([(a * c + b * d) / cd2, (b * c - a * d) / cd2], -1)


def csum(x, y):
    # complex + real
    real = x[..., 0]
    real = real + y[..., 0].expand_as(real)
    img = x[..., 1]
    return torch.stack([real, img.expand_as(real)], -1)


def cabs2(x):
    return x[..., 0]**2 + x[..., 1]**2


def cmul(t1, t2):
    '''complex multiplication

    Args:
        t1: NxCxHxWx2, complex tensor
        t2: NxCxHxWx2

    Returns:
        output: NxCxHxWx2
    '''
    real1, imag1 = t1[..., 0], t1[..., 1]
    real2, imag2 = t2[..., 0], t2[..., 1]
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1)


def cconj(t, inplace=False):
    '''complex's conjugation

    Args:
        t: NxCxHxWx2

    Returns:
        output: NxCxHxWx2
    '''
    c = t.clone() if not inplace else t
    c[..., 1] *= -1
    return c

def conv2d(input, weight, padding=0, sample_wise=False):
    """
        sample_wise=False, normal conv2d:
            input - (N, C_in, H_in, W_in)
            weight - (C_out, C_in, H_k, W_k)
        sample_wise=True, sample-wise conv2d:
            input - (N, C_in, H_in, W_in)
            weight - (N, C_out, C_in, H_k, W_k)
    """
    if isinstance(padding, int):
        padding = [padding] * 4
    if sample_wise:
        # input - (N, C_in, H_in, W_in) -> (1, N * C_in, H_in, W_in)
        input_sw = input.reshape(1,input.size(0) * input.size(1), input.size(2),input.size(3))

        # weight - (N, C_out, C_in, H_k, W_k) -> (N * C_out, C_in, H_k, W_k)
        weight_sw = weight.reshape(weight.size(0) * weight.size(1), weight.size(2), weight.size(3),weight.size(4))

        # group-wise convolution, group_size==batch_size
        out = F.conv2d(pad(input_sw, padding, mode='circular'),weight_sw,groups=input.size(0))
        out = out.reshape(input.size(0), weight.size(1), out.size(2), out.size(3))
    else:
        out = F.conv2d(pad(input, padding, mode='circular'), weight)
    return out