from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


class Metrics:
    def __init__(self):
        self._value = defaultdict(list)

    def reset_state(self):
        self._value = defaultdict(list)

    def update_state(self, value: dict):
        for key in value:
            self._value[key].append(value[key])

    def result(self):
        """Computes and returns the metric value tensor.
        Result computation is an idempotent operation that simply calculates the
        metric value using the state variables.
        """
        raise NotImplementedError('Must be implemented in subclasses.')


class Mean(Metrics):
    def __init__(self):
        super().__init__()

    def result(self):
        rec = {}
        for key in self._value:
            rec[key] = np.array(self._value[key]).mean()

        return rec


class Stack(Metrics):
    def __init__(self):
        super().__init__()

    def result(self):
        rec = {}
        for key in self._value:
            rec[key] = np.stack(self._value[key], 0)

        return rec


class Concatenate(Metrics):
    def __init__(self):
        super().__init__()

    def result(self):
        rec = {}
        for key in self._value:
            rec[key] = np.concatenate(self._value[key], 0)

        return rec


def compute_dice(vol1: torch.Tensor, vol2: torch.Tensor, labels=None, nargout=1, size_average=True):

    if not size_average:
        raise NotImplementedError()

    if labels is None:
        labels = torch.unique(torch.cat((vol1, vol2)))
        labels = labels[labels != 0]  # remove background

    dicem = torch.zeros_like(labels)
    for idx, lab in enumerate(labels):
        vol1l = (vol1 == lab)
        vol2l = (vol2 == lab)

        top    = 2 * torch.sum(vol1l & vol2l, dtype=torch.float32)
        bottom = torch.sum(vol1l, dtype=torch.float32) + torch.sum(vol2l, dtype=torch.float32)
        if bottom == 0:
            bottom = 1e-10  # avoid 0 as Denominator.

        dicem[idx] = top / bottom

    if nargout == 1:
        return dicem.mean()

    else:
        raise NotImplementedError()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# noinspection PyArgumentList
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(-1).mean(-1).mean(-1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def compare_ssim(img_test, img_true, size_average=True, window_size=11):
    (_, channel, _, _) = img_test.size()
    window = create_window(window_size, channel)

    if img_test.is_cuda:
        window = window.cuda(img_test.get_device())
    window = window.type_as(img_test)

    return _ssim(img_test, img_true, window, window_size, channel, size_average)


def compare_mse(img_test, img_true, size_average=True):
    img_diff = img_test - img_true
    img_diff = img_diff ** 2

    if size_average:
        img_diff = img_diff.mean()
    else:
        img_diff = img_diff.mean(-1).mean(-1).mean(-1)

    return img_diff


def compare_psnr(img_test, img_true, size_average=True, max_value=1):
    return 10 * torch.log10((max_value ** 2) / compare_mse(img_test, img_true, size_average))


def compare_snr(img_test, img_true, size_average=True):
    if not size_average:
        raise NotImplementedError('size_average must be True')

    return 20 * torch.log10(torch.norm(img_true.flatten()) / torch.norm(img_true.flatten() - img_test.flatten()))


def compare_rsnr(img_test, img_true, size_average=True):
    if not size_average:
        raise NotImplementedError('size_average must be True')

    img_test = torch.squeeze(img_test)
    img_true = torch.squeeze(img_true)

    if img_test.shape.__len__() != 2 or img_true.shape.__len__() != 2:
        raise NotImplementedError('only 2D images are supported')

    img_true_flatten = img_true.flatten()
    img_test_flatten = img_test.flatten()

    A = torch.zeros((2, 2))
    A[0, 0] = torch.sum(img_true_flatten ** 2)
    A[0, 1] = torch.sum(img_true_flatten)
    A[1, 0] = A[0, 1]
    A[1, 1] = img_test.shape[0] * img_test.shape[1]

    b = torch.zeros((2, 1))
    b[0] = torch.sum(img_test_flatten * img_true_flatten)
    b[1] = torch.sum(img_test_flatten)

    c = torch.matmul(torch.inverse(A), b)
    if img_true.is_cuda:
        c = c.cuda()

    rsnr = compare_snr(img_test, c[0] * img_true + c[1], size_average=True)

    return rsnr


def compare_rpsnr(img_test, img_true, size_average=True):
    if not size_average:
        raise NotImplementedError('size_average must be True')

    img_test = torch.squeeze(img_test)
    img_true = torch.squeeze(img_true)

    if img_test.shape.__len__() != 2 or img_true.shape.__len__() != 2:
        raise NotImplementedError('only 2D images are supported')

    img_true_flatten = img_true.flatten()
    img_test_flatten = img_test.flatten()

    A = torch.zeros((2, 2))
    A[0, 0] = torch.sum(img_true_flatten ** 2)
    A[0, 1] = torch.sum(img_true_flatten)
    A[1, 0] = A[0, 1]
    A[1, 1] = img_test.shape[0] * img_test.shape[1]

    b = torch.zeros((2, 1))
    b[0] = torch.sum(img_test_flatten * img_true_flatten)
    b[1] = torch.sum(img_test_flatten)

    c = torch.matmul(torch.inverse(A), b)
    if img_true.is_cuda:
        c = c.cuda()

    rsnr = compare_psnr(img_test, c[0] * img_true + c[1], size_average=True)

    return rsnr


def compare_rssim(img_test, img_true, size_average=True):
    if not size_average:
        raise NotImplementedError('size_average must be True')

    img_test = torch.squeeze(img_test)
    img_true = torch.squeeze(img_true)

    if img_test.shape.__len__() != 2 or img_true.shape.__len__() != 2:
        raise NotImplementedError('only 2D images are supported')

    img_true_flatten = img_true.flatten()
    img_test_flatten = img_test.flatten()

    A = torch.zeros((2, 2))
    A[0, 0] = torch.sum(img_true_flatten ** 2)
    A[0, 1] = torch.sum(img_true_flatten)
    A[1, 0] = A[0, 1]
    A[1, 1] = img_test.shape[0] * img_test.shape[1]

    b = torch.zeros((2, 1))
    b[0] = torch.sum(img_test_flatten * img_true_flatten)
    b[1] = torch.sum(img_test_flatten)

    c = torch.matmul(torch.inverse(A), b)
    if img_true.is_cuda:
        c = c.cuda()

    img_test = img_test.unsqueeze(0).unsqueeze(0)
    img_true = img_true.unsqueeze(0).unsqueeze(0)

    rsnr = compare_ssim(img_test, c[0] * img_true + c[1], size_average=True)

    return rsnr
