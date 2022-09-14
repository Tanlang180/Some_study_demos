ROOT_PATH = "Z:/DataSet/test/"
WEST = "Westerner/"
ASIAN = "Asian/"


import numpy as np
import cv2 as cv


def gaussian_kernel_2d_opencv(kernel_size=3, sigma=0):
    kx = cv.getGaussianKernel(kernel_size, sigma)
    ky = cv.getGaussianKernel(kernel_size, sigma)
    return np.multiply(kx, np.transpose(ky))


if __name__ == '__main__':
    s = gaussian_kernel_2d_opencv(3, 1)
    print(s)