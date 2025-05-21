
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skvideo.measure import niqe


class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res
"""
#修改后
def compute_psnr_ssim(recovered, clean):
    assert recovered.shape == clean.shape, "Shapes of recovered and clean images must be the same."
    psnr = 0
    ssim = 0
    num_images = recovered.shape[0]

    for i in range(num_images):
        current_recovered = np.clip(recovered[i].detach().cpu().numpy(), 0, 1)
        current_clean = np.clip(clean[i].detach().cpu().numpy(), 0, 1)

        min_dimension = min(current_recovered.shape[:2])  # Get the smallest dimension from height or width
        win_size = min(7, min_dimension)  # Win_size should be less than the smallest image dimension and 7
        if win_size % 2 == 0:
            win_size -= 1  # Ensure win_size is odd

        # Ensure win_size is at least 3 to avoid ValueError in structural_similarity when images are very small
        if win_size < 3:
            print("Warning: Image too small for standard SSIM calculation. Skipping SSIM for this image.")
            continue  # Skip SSIM calculation if the window size is too small

        psnr += peak_signal_noise_ratio(current_clean, current_recovered, data_range=1)
        ssim += structural_similarity(current_clean, current_recovered, data_range=1, multichannel=True,
                                      win_size=win_size)

    return psnr / num_images, ssim / num_images if num_images > 0 else 0, num_images
"""
#原版

def compute_psnr_ssim(recovered, clean):
    assert recovered.shape == clean.shape
    recovered = np.clip(recovered.detach().cpu().numpy(), 0, 1)
    clean = np.clip(clean.detach().cpu().numpy(), 0, 1)

    recovered = recovered.transpose(0, 2, 3, 1)
    clean = clean.transpose(0, 2, 3, 1)
    psnr = 0
    ssim = 0

    for i in range(recovered.shape[0]):
        # psnr_val += compare_psnr(clean[i], recovered[i])
        # ssim += compare_ssim(clean[i], recovered[i], multichannel=True)
        psnr += peak_signal_noise_ratio(clean[i], recovered[i], data_range=1)
        ssim += structural_similarity(clean[i], recovered[i], channel_axis=-1, data_range=1)

    return psnr / recovered.shape[0], ssim / recovered.shape[0], recovered.shape[0]

def compute_niqe(image):
    image = np.clip(image.detach().cpu().numpy(), 0, 1)
    image = image.transpose(0, 2, 3, 1)
    niqe_val = niqe(image)

    return niqe_val.mean()

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0