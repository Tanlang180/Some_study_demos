import numpy as np
from skimage import morphology, io
from skimage import measure
from sklearn.cluster import KMeans
import cv2 as cv
import matplotlib.pyplot as plt

img_file = r'./imgs/lung.jpg'
img = cv.imread(img_file)
imgs_to_process = np.transpose(img, (2, 0, 1))
print("the shape of imgs_to_process :  ", imgs_to_process.shape)  # 病历离肺结节最近的三个切片


# 数值分布标准化
mean = np.mean(img)
# print(mean)
std = np.std(img)
img = img - mean
img = img / std


image_array = imgs_to_process[0]
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))
ax1.imshow(image_array, cmap='gray')
plt.hist(img.flatten(), bins=200)   # 绘制直方图
plt.show()