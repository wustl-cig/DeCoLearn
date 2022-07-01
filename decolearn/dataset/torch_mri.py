import torch

import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.fft import ifftshift
import math
import decimal


bernoulli_mask_cache = dict()
def __random_mask(imgSize, fold):
    p_at_edge_dict = {
        4: 0.13,
        6: 0.063,
        8: 0.036,
        10: 0.023,
    }
    p_at_edge = p_at_edge_dict[fold]

    params = {'type': 'bspec', 'p_at_edge': p_at_edge}
    ctype = params['type']
    assert ctype == 'bspec'
    p_at_edge = params['p_at_edge']

    global bernoulli_mask_cache
    if bernoulli_mask_cache.get(p_at_edge) is None:
        h = [s // 2 for s in imgSize]
        r = [np.arange(s, dtype=np.float32) - h for s, h in zip(imgSize, h)]
        r = [x ** 2 for x in r]
        r = (r[0][:, np.newaxis] + r[1][np.newaxis, :]) ** .5
        m = (p_at_edge ** (1. / h[1])) ** r
        bernoulli_mask_cache[p_at_edge] = m

    mask = bernoulli_mask_cache[p_at_edge]
    keep = (np.random.uniform(0.0, 1.0, size=imgSize) ** 2 < mask)
    keep = keep & keep[::-1, ::-1]

    smsk = keep.astype(np.float32)

    re = torch.from_numpy(np.fft.fftshift(smsk, axes=(0, 1)).astype(np.float32))
    re = torch.stack([re, re], -1)
    re = re.unsqueeze_(0)

    return re


def __normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)


def __cartesian_mask(imgSize, fold, centred=False):
    """
    Sampling density estimated from implementation of kt FOCUSS
    shape: tuple - of form (..., nx, ny)
    acc: float - doesn't have to be integer 4, 8, etc..
    """
    N, Nx, Ny = int(np.prod(imgSize[:-2])), imgSize[-2], imgSize[-1]
    pdf_x = __normal_pdf(Nx, 0.5/(Nx/10.)**2)
    lmda = Nx/(2.*fold)
    n_lines  = int(Nx / fold)
    sample_n = int(n_lines / 3)

    # add uniform distribution
    pdf_x += lmda * 1./Nx

    if sample_n:
        pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape(imgSize)

    if not centred:
        mask = ifftshift(mask, axes=(-1, -2))

    mask = mask.astype(np.float32)

    re = torch.from_numpy(mask)
    re = torch.stack([re, re], -1)
    re = re.unsqueeze_(0)

    return re


def __radial_mask(imgSize, fold):
    numLines_dict = {
        10:  31,
        8:   39,
        6:   52,
        4:   80,
    }
    imgSize = np.array(imgSize)

    numLines = numLines_dict[fold]

    if imgSize[0] % 2 != 0 or imgSize[1] % 2 != 0:
        print('image must be even sized! ')
        exit(1)

    center = np.ceil(imgSize / 2) + 1
    freqMax = math.ceil(np.sqrt(np.sum(np.power((imgSize / 2), 2), axis=0)))
    ang = np.linspace(0, math.pi, num=numLines + 1)
    inc = 0 + (math.pi / numLines) * np.random.rand(1, )
    mask = np.zeros(imgSize, dtype=bool)

    for indLine in range(0, numLines):
        ix = np.zeros(2 * freqMax + 1)
        iy = np.zeros(2 * freqMax + 1)
        ind = np.zeros(2 * freqMax + 1, dtype=bool)
        for i in range(2 * freqMax + 1):
            ix[i] = decimal.Decimal(center[1] + (-freqMax + i) * math.cos(ang[indLine] + inc)).quantize(
                0, rounding=decimal.ROUND_HALF_UP)
        for i in range(2 * freqMax + 1):
            iy[i] = decimal.Decimal(center[0] + (-freqMax + i) * math.sin(ang[indLine] + inc)).quantize(
                0, rounding=decimal.ROUND_HALF_UP)

        for k in range(2 * freqMax + 1):
            if (ix[k] >= 1) & (ix[k] <= imgSize[1]) & (iy[k] >= 1) & (iy[k] <= imgSize[0]):
                ind[k] = True
            else:
                ind[k] = False

        ix = ix[ind]
        iy = iy[ind]
        ix = ix.astype(np.int64)
        iy = iy.astype(np.int64)

        for i in range(len(ix)):
            mask[iy[i] - 1][ix[i] - 1] = True

    mask = np.fft.fftshift(mask, axes=(0, 1))

    mask = mask.astype(np.float32)

    re = torch.from_numpy(mask)
    re = torch.stack([re, re], -1)
    re = re.unsqueeze_(0)

    return re


def generate_mask(type_, imgSize, fold):
    __mask_dict = {
        'random': __random_mask,
        'cartesian': __cartesian_mask,
        'radial': __radial_mask
    }

    assert type_ in __mask_dict, 'unsupported value of type_'

    return __mask_dict[type_](imgSize=imgSize, fold=fold)


def addwgn(x: torch.Tensor, input_snr):
    noiseNorm = torch.norm(x.flatten()) * 10 ** (-input_snr / 20)

    noise = torch.randn(x.size())

    noise = noise / torch.norm(noise.flatten()) * noiseNorm

    y = x + noise
    return y, noise


def fmult(x: torch.Tensor, mask: torch.Tensor):
    x_coml = torch.rfft(x, 2, onesided=False)
    x_coml = x_coml * mask

    return x_coml


def ftran(x_coml: torch.Tensor, mask: torch.Tensor):
    x_coml = x_coml * mask
    x_real = torch.irfft(x_coml, 2, onesided=False)

    return x_real


def fmult_nd(
        x: torch.Tensor,
        mask_type=None,
        mask_fold=None,
        is_add_noise=True,
        noise_input_snr=None,
        pre_defined_mask=None,
        pre_defined_noise=None,
):

    if (mask_type is None or mask_fold is None) and (pre_defined_mask is None):
        raise ValueError('mask_type and mask_fold are required or pre_defined_mask is needed.')

    if is_add_noise:
        if (noise_input_snr is None) and (pre_defined_noise is None):
            raise ValueError('noise_input_snr or pre_defined_mask is required.')

    if pre_defined_mask is not None:
        print("found pre_defined_mask: ",  pre_defined_mask.shape)
    if pre_defined_noise is not None:
        print("found pre_defined_noise: ", pre_defined_noise.shape)

    if x.shape.__len__() == 4:
        n_slice, n_channel, n_x, n_y = x.shape

        if n_channel != 1:
            raise NotImplementedError('only channel=1(gray image) is supported.')

        ret_y     = torch.zeros(size=(n_slice, 1, n_x, n_y, 2))
        ret_mask  = torch.zeros(size=(n_slice, 1, n_x, n_y, 2))
        ret_noise = torch.zeros(size=(n_slice, 1, n_x, n_y, 2))

        x = torch.squeeze(x, 1)  # drop channel.
        for i_slice in range(n_slice):
            x_cur = x[i_slice]

            if pre_defined_mask is not None:
                mask_cur = pre_defined_mask[i_slice]
            else:
                mask_cur = generate_mask(type_=mask_type, imgSize=(n_x, n_y), fold=mask_fold)

            y_cur = fmult(x_cur, mask_cur)

            noise_cur = 0
            if is_add_noise:
                if pre_defined_noise is not None:
                    noise_cur = pre_defined_noise[i_slice]
                    y_cur     = y_cur + noise_cur

                else:
                    y_cur, noise_cur = addwgn(y_cur, input_snr=noise_input_snr)

            ret_y[i_slice,     0] = y_cur
            ret_mask[i_slice,  0] = mask_cur
            ret_noise[i_slice, 0] = noise_cur

        return ret_y, ret_mask, ret_noise

    elif x.shape.__len__() == 3:

        n_channel, n_x, n_y = x.shape

        if n_channel != 1:
            raise NotImplementedError('only channel=1(gray image) is supported.')

        if pre_defined_mask is not None:
            ret_mask = pre_defined_mask
        else:
            ret_mask = generate_mask(type_=mask_type, imgSize=(n_x, n_y), fold=mask_fold)

        ret_y = fmult(x, ret_mask)

        ret_noise = 0
        if is_add_noise:
            if pre_defined_noise is not None:
                ret_noise = pre_defined_noise
                ret_y = ret_y + ret_noise

            else:
                ret_y, ret_noise = addwgn(ret_y, input_snr=noise_input_snr)

        return ret_y, ret_mask, ret_noise

    else:
        raise NotImplementedError('only 4D x(batch, channel, height, width) or 3d x(channel, height, width) is supported currently.')


def ftran_nd(
        y,
        mask
):

    if y.shape.__len__() == 5:

        n_slice, n_channel, n_x, n_y, _ = y.shape

        if n_channel != 1:
            raise NotImplementedError('only channel=1(gray image) is supported.')

        y_tran = torch.zeros(size=(n_slice, 1, n_x, n_y))

        for i_slice in range(n_slice):
            y_cur    = y[i_slice].squeeze(0)
            mask_cur = mask[i_slice].squeeze(0)

            y_tran_cur = ftran(y_cur, mask_cur)

            y_tran[i_slice, 0] = y_tran_cur

        return y_tran

    elif y.shape.__len__() == 4:
        n_channel, n_x, n_y, _ = y.shape

        if n_channel != 1:
            raise NotImplementedError('only channel=1(gray image) is supported.')

        y_tran = ftran(y, mask)

        return y_tran

    else:
        raise NotImplementedError('only 5d y(batch, channel, height, width, 2) or 4d y(channel, height, width, 2) is supported currently.')
