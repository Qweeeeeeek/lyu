import numpy as np
import math


def compare_SNR(f_img, l_img):
    f_img = f_img.data.cpu().numpy()
    l_img = l_img.data.cpu().numpy()
    f_img = f_img.squeeze()
    l_img = l_img.squeeze()
    ps = np.mean(np.square(l_img))
    tmp = l_img - f_img
    pn = np.mean(np.square(tmp))
    if ps == 0 or pn == 0:
        s = 999.99
    else:
        s = 10 * math.log(ps / pn, 10)
    return s


def batch_snr(f_data, l_data):
    De_data = f_data.data.cpu().numpy()  # 将数据从GPU中拷贝出来，放入CPU中，并转换为numpy数组
    Clean_data = l_data.data.cpu().numpy()
    SNR = 0
    De = De_data.squeeze()  # 默认压缩所有为1的维度
    Clean = Clean_data.squeeze()
    SNR += compare_SNR(De, Clean)
    return SNR

# --------------------------------------------
# SSIM
# --------------------------------------------


"""
ValueError: win_size exceeds image extent. Either ensure that your images are at least 7x7;
 or pass win_size explicitly in the function call, with an odd value less than or equal 
 不可用
"""
from skimage.metrics import structural_similarity as ssim


def compute_ssim(img1, img2):
    img1 = img1.data.cpu().numpy()  # 将数据从GPU中拷贝出来，放入CPU中，并转换为numpy数组
    img2 = img2.data.cpu().numpy()
    img1 = img1.squeeze()
    img2 = img2.squeeze()
    return ssim(img1, img2)


# --------------------------------------------
# PSNR
# --------------------------------------------


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    else:
        return 10 * np.log10(4 / mse)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.data.cpu().numpy()  # 将数据从GPU中拷贝出来，放入CPU中，并转换为numpy数组
    img2 = img2.data.cpu().numpy()
    img1 = img1.squeeze()
    img2 = img2.squeeze()
    return psnr(img1, img2)


