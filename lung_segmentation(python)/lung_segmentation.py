import numpy as np
from skimage import morphology, io
from skimage import measure
from sklearn.cluster import KMeans
import cv2 as cv
import matplotlib.pyplot as plt

# 导入图像
img_file = r'./imgs/lung.jpg'
img = cv.imread(img_file)
imgs_to_process = np.transpose(img, (2, 0, 1))
print("the shape of imgs_to_process :  ", imgs_to_process.shape)  # 打印处理图片的维度形状

# 数值分布标准化
mean = np.mean(img)
std = np.std(img)
img = img - mean
img = img / std

# 裁剪出肺部的主要区域，用于KMeans聚类
middle = img[100:1000, 300:9000]
mean = np.mean(middle)

# 将图片最大值和最小值替换为肺部大致均值，使得图像灰度值分布均匀些，让聚类结果更加可靠
max = np.max(img)
min = np.min(img)
# print(mean, min, max)
img[img == max] = mean
img[img == min] = mean

# KMeans聚类，产生两簇,一类是骨骼和血管，一类是肺部区域
kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
centers = sorted(kmeans.cluster_centers_.flatten())
threshold = np.mean(centers)
thresh_img = np.where(img < threshold, 1.0, 0.0)  # 用计算的阈值，对图像进行二值化处理，得到thresh_img
print('kmean centers:', centers)
print('threshold:', threshold)

# 聚类完成后，清晰可见骨骼和血管为一类，肺部区域为另一类。
# image_array = thresh_img
# plt.imshow(image_array, cmap='gray')
# plt.show()


# 腐蚀和膨胀

kernel_erosion = np.ones((5, 5))  # 腐蚀算子
kernel_dilation = np.ones((23, 23))  # 膨胀算子

# opening = cv.morphologyEx(thresh_img, cv.MORPH_OPEN, kernel)

erosion = cv.erode(thresh_img, kernel_erosion, iterations=1)
dilation = cv.dilate(erosion, kernel_dilation, iterations=1)

labels = measure.label(dilation[:, :, 0])  # 对连通区域进行标记
regions = measure.regionprops(labels)  # 获取连通区域
# print(len(regions))

# 根据给的lung图，设置经验值，获取肺部标签
good_labels = []
for prop in regions:
    B = prop.bbox
    # print(B)
    if B[2] - B[0] < 700 and B[3] - B[1] < 500 and B[0] > 200 and B[2] < 1000:
        good_labels.append(prop.label)

#  根据肺部标签获取肺部mask，并再次进行’膨胀‘操作，以填满并扩张肺部区域
mask = np.ndarray([1148, 1144], dtype=np.int8)
mask[:] = 0
for N in good_labels:
    mask = mask + np.where(labels == N, 1, 0)
mask = morphology.dilation(mask, np.ones([10, 10]))  # 膨胀操作

# 通道合并
# maskTemp = np.stack((mask,mask,mask), axis = 2)
# mask_region = img * maskTemp

mask_region = img[:, :, 0] * mask

# 画出分割图像
fig, ax = plt.subplots(2, 2, figsize=[10, 10])
ax[0, 0].imshow(img, cmap='gray')  # CT切片图
ax[0, 0].set_title("CT_source_img")
ax[0, 1].imshow(labels)  # CT切片标签图
ax[0, 1].set_title("CT_labels")
ax[1, 0].imshow(mask, cmap='gray')  # 标注mask，标注区域为1，其他为0
ax[1, 0].set_title("CT_mask")
ax[1, 1].imshow(mask_region, cmap='gray')  # 标注mask区域切片图
ax[1, 1].set_title("CT_mask_region")

plt.savefig("./result/result.jpg")
plt.show()

# 转为int8 格式图片
# 由于转换后图像效果变差，所以我选择不使用
# mask = (mask * 255).astype(np.uint8)
# mask_region = (mask_region * 255).astype(np.uint8)

# 保存分割图像
io.imsave("./result/lung_shape.jpg", mask)
io.imsave("./result/lung_mask_region.jpg", mask_region)
