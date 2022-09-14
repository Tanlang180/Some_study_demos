import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import PIL
import cv2 as cv
from config import *
import logging
import os


def print_package_version():
    print("Matplotlib version：%s" % matplotlib.__version__)
    print("Numpy version：%s" % np.__version__)
    print("Pillow version：%s" % PIL.__version__)
    print("OpenCV version：%s" % cv.__version__)


##################
# 全局变量
src = None
gradient = None
name = None
mask = None
dst_blur = None
dst_blend_skin = None
dst_whiten = None
dst_sharpen = None

blur_param1 = None
blur_param2 = None
blur_param3 = None
sharpen_beta = None
sharpen_base = None


##################


def img_read(name, PATH):
    path = PATH + name
    img = cv.imread(path)
    h, w, c = img.shape
    while h > 1000 or w > 1000:
        h = h // 2
        w = w // 2
    temp = cv.resize(img, (h, w))
    cv.imshow(name, temp)

    return temp


def edge_protect_filtering_blur(src, name, param1, param2, param3):
    if src is None \
            or param1 is None \
            or param2 is None \
            or param3 is None:
        return

    # param3 = cv.getTrackbarPos("blur_param3", "beauty")
    temp = src.copy()

    # 双边滤波
    # cv.bilateralFilter(src, param3, param1, param2, temp)

    # 均值迁移滤波
    # cv.pyrMeanShiftFiltering(src, param1, param2, temp)

    # OpenCV实现的保边滤波
    # cv.edgePreservingFilter(src, temp, cv.NORMCONV_FILTER, param1 * 2, param2 / 100)

    # 引导滤波 何凯明
    # cv.ximgproc.guidedFilter(src, src, param1, param2, temp)
    gradient = get_gradient_sobel(temp)
    temp = guideFilter_optimize(temp / 255, winSize=(param3, param3),
                                eps=param2 / 1000)  # winSize 3根据斑点特征调整, eps 调整磨皮程度，越大磨皮程度越大
    temp = (temp * 255).astype(np.uint8)

    # 表面模糊 ps滤镜
    # thre = 20
    # half_size = 10
    # temp[:, :, 0] = surface_blur(temp[:, :, 0], thre, half_size)
    # temp[:, :, 1] = surface_blur(temp[:, :, 1], thre, half_size)
    # temp[:, :, 2] = surface_blur(temp[:, :, 2], thre, half_size)

    cv.imshow("dst_blur", temp)
    # cv.imwrite("./result/blur_%s" % name, dst)
    return temp


def guideFilter(I, p, winSize, eps, s=1):
    # eps 可以取 0.01，0.001 等，具体的自己测试一下吧
    # 输入图像的高、宽
    h, w = I.shape[:2]

    # 缩小图像
    size = (int(round(w * s)), int(round(h * s)))

    small_I = cv.resize(I, size, interpolation=cv.INTER_CUBIC)
    small_p = cv.resize(I, size, interpolation=cv.INTER_CUBIC)

    # 缩小滑动窗口
    X = winSize[0]
    small_winSize = (int(round(X * s)), int(round(X * s)))

    # I的均值平滑
    mean_small_I = cv.blur(small_I, small_winSize)

    # p的均值平滑
    mean_small_p = cv.blur(small_p, small_winSize)

    # I*I和I*p的均值平滑
    mean_small_II = cv.blur(small_I * small_I, small_winSize)

    mean_small_Ip = cv.blur(small_I * small_p, small_winSize)

    # 方差
    var_small_I = mean_small_II - mean_small_I * mean_small_I  # 方差公式

    # 协方差
    cov_small_Ip = mean_small_Ip - mean_small_I * mean_small_p

    small_a = cov_small_Ip / (var_small_I + eps)
    small_b = mean_small_p - small_a * mean_small_I

    # 对a、b进行均值平滑
    mean_small_a = cv.blur(small_a, small_winSize)
    mean_small_b = cv.blur(small_b, small_winSize)

    # 放大
    size1 = (w, h)
    mean_a = cv.resize(mean_small_a, size1, interpolation=cv.INTER_LINEAR)
    mean_b = cv.resize(mean_small_b, size1, interpolation=cv.INTER_LINEAR)

    q = mean_a * I + mean_b

    return q


def guideFilter_optimize(I, winSize, eps, s=1):
    # eps 可以取 0.01，0.001 等，具体的自己测试一下吧
    # 输入图像的高、宽
    h, w = I.shape[:2]

    # 缩小图像
    size = (int(round(w * s)), int(round(h * s)))

    small_I = cv.resize(I, size, interpolation=cv.INTER_CUBIC)

    # 缩小滑动窗口
    X = winSize[0]
    small_winSize = (int(round(X * s)), int(round(X * s)))

    # I的均值平滑
    mean_small_I = cv.blur(small_I, small_winSize)

    # I*I的均值平滑
    mean_small_II = cv.blur(small_I * small_I, small_winSize)

    # 方差
    var_small_I = mean_small_II - mean_small_I * mean_small_I  # 方差公式

    small_a = var_small_I / (var_small_I + eps)
    small_b = mean_small_I * (1 - small_a)

    # 对a、b进行均值平滑
    mean_small_a = cv.blur(small_a, small_winSize)
    mean_small_b = cv.blur(small_b, small_winSize)

    # 放大
    size1 = (w, h)
    mean_a = cv.resize(mean_small_a, size1, interpolation=cv.INTER_LINEAR)
    mean_b = cv.resize(mean_small_b, size1, interpolation=cv.INTER_LINEAR)

    q = mean_a * I + mean_b

    return q


def surface_blur(I_in, thre, half_size):
    I_out = I_in * 1.0
    row, col = I_in.shape
    w_size = half_size * 2 + 1
    for ii in range(half_size, row - 1 - half_size):
        for jj in range(half_size, col - 1 - half_size):
            aa = I_in[ii - half_size:ii + half_size + 1, jj - half_size:jj + half_size + 1]
            p0 = I_in[ii, jj]
            mask_1 = np.matlib.repmat(p0, w_size, w_size)
            mask_2 = 1 - abs(aa - mask_1) / (2.5 * thre)
            mask_3 = mask_2 * (mask_2 > 0)
            t1 = aa * mask_3
            I_out[ii, jj] = t1.sum() / mask_3.sum()

    return I_out


def gen_curve_list(beta):
    """
    功能：生成美白映射曲线，提升色阶
    paper: A_Two-Stage_Contrast_Enhancement_Algorithm_for_Digital_Images
    """
    val_color = list(range(1, 257))
    res = [(math.log10(x / 255 * (beta - 1) + 1) / math.log10(beta)) * 255 for x in val_color]

    # 绘制曲线
    plt.ion()
    plt.clf()
    plt.plot(val_color, res, "r-")
    plt.title('logarithmic curve')
    plt.xlim(0, 256)
    plt.ylim(0, 256)
    plt.xlabel("src_color")
    plt.ylabel("dst_color")
    plt.legend
    plt.show()
    # plt.pause(1)  # 显示等待关闭
    # plt.close()

    return res


def whiten_skin(src, mask, name):
    temp = None
    if src is not None:
        factor = cv.getTrackbarPos("whiten_factor", "beauty") / 10
        Color_list = gen_curve_list(factor)
        h, w, c = src.shape
        temp = src.copy()
        for i in range(h):
            for j in range(w):
                b = temp[i, j, 0]
                g = temp[i, j, 1]
                r = temp[i, j, 2]
                temp[i, j, 0] = Color_list[b]
                temp[i, j, 1] = Color_list[g]
                temp[i, j, 2] = Color_list[r]

        # cv.imwrite("./result/beta%d_%s" % (beta, name), temp)
        # temp = mask * temp + (1-mask) * src
        cv.imshow("dst_whiten", temp)

    return temp


def check_skin_color(src, name):
    if src is None:
        return
    # mask = check_skin_rgb_algorithm(src)
    mask = check_skin_HSV_algorithm(src)
    cv.imshow("mask", mask[:, :, 0] * 255)
    # cv.imshow("1-mask", (1 - mask[:, :, 0]) * 255)
    # cv.imwrite("./result/skin_mask_%s" % name, mask*255)
    return mask


def check_skin_rgb_algorithm(src):
    # RGB 颜色空间阈值划分 肤色检测
    h, w, c = src.shape
    temp = src.copy()
    mask = np.zeros((h, w, 1), np.uint8)
    for i in range(h):
        for j in range(w):
            b = temp[i, j, 0]
            g = temp[i, j, 1]
            r = temp[i, j, 2]
            max_v = max(b, g, r)
            min_v = min(b, g, r)

            # 日光灯照情况下 和 闪光灯和手电筒照明情况下
            if ((r > 95 and g > 40 and b > 20
                 and max_v - min_v > 15
                 and abs(int(r) - int(g)) > 15
                 and r > g and r > b)
                    or
                    (r > 220 and g > 210 and b > 170
                     and abs(int(r) - int(g)) < 15
                     and r > b and g > b
                    )
            ):
                mask[i, j, 0] = 1

    return mask


def check_skin_YCbCr_algorithm(src):
    h, w, c = src.shape
    temp = src.copy()
    mask = np.zeros((h, w, 1), np.uint8)
    cv.cvtColor(src, cv.COLOR_RGB2YCrCb, temp)
    for i in range(h):
        for j in range(w):
            y = temp[i, j, 0]
            cb = temp[i, j, 1]
            cr = temp[i, j, 2]

            if 90 <= cb <= 127 and 135 <= cr <= 173:
                # 78 <= cb <= 127 and 133 <= cr <= 173:
                mask[i, j, 0] = 1

    return mask


def check_skin_YCRCB_dynamic(src):
    h, w, c = src.shape
    temp = src.copy()
    mask = np.zeros((h, w, 1), np.uint8)
    cv.cvtColor(src, cv.COLOR_RGB2YCrCb, temp)

    theda1 = 0
    theda2 = 0
    theda3 = 0
    theda4 = 0

    for i in range(h):
        for j in range(w):
            y = temp[i, j, 0]
            cb = temp[i, j, 1]
            cr = temp[i, j, 2]

            if y > 128:
                theda1 = -2 + (256 - y) / 16
                theda2 = 20 - (256 - y) / 16
                theda3 = 6
                theda4 = 8
            elif y <= 128:
                theda1 = 6
                theda2 = 12
                theda3 = 2 + y / 32
                theda4 = -16 + y / 16

            if (cr >= -2 * (cb + 24) and
                    cr >= -(cb + 17) and
                    cr >= -4 * (cb + 32) and
                    cr >= 2.5 * (cb + theda1) and
                    cr >= theda3 and
                    0.5 * (theda4 - cb) <= cr <= (220 - cb) / 6 and
                    cr <= 4 / 3 * (theda2 - cb)):
                mask[i, j, 0] = 1

    return mask


def check_skin_HSV_algorithm(src):
    h, w, c = src.shape
    temp = src.copy()
    mask = np.zeros((h, w, 1), np.uint8)
    cv.cvtColor(src, cv.COLOR_RGB2HSV, temp)
    # h 0,180
    # s 0,255
    # v 0,255
    for i in range(h):
        for j in range(w):
            h = temp[i, j, 0]
            s = temp[i, j, 1]
            v = temp[i, j, 2]
            if 90 <= h <= 150 and 10 <= s and 50 <= v:
                # 90 <= h <= 125 and 28 <= s <= 256 and 70 <= v <= 256
                # and 28 <= s <= 256 and 50 <= v <= 256
                # 100 <= h <= 120 and 50 <= v <= 256
                # 边缘 80 < v < 150
                mask[i, j, 0] = 1

    return mask


def check_skin_YCrCb_HSV_fusing(src):
    h, w, c = src.shape
    mask = np.zeros((h, w, 1), np.uint8)
    hsv = cv.cvtColor(src, cv.COLOR_RGB2HSV)
    yuv = cv.cvtColor(src, cv.COLOR_RGB2YCrCb)
    for i in range(h):
        for j in range(w):
            h = hsv[i, j, 0]
            s = hsv[i, j, 1]
            v = hsv[i, j, 2]
            cb = yuv[i, j, 1]
            cr = yuv[i, j, 2]
            if 90 <= h <= 125 and 28 <= s <= 256 and 70 <= v <= 256 \
                    and 90 <= cb <= 127 and 135 <= cr <= 173:
                mask[i, j, 0] = 1

    return mask


def gradient_mode(src):
    gradient = None
    if src is not None:
        sharpen_mode_switch = cv.getTrackbarPos("sharpen_mode_switch", "beauty")
        temp = src.copy()
        # 求解梯度图
        if sharpen_mode_switch == 1:
            gradient = get_gradient_laplace(temp)
        elif sharpen_mode_switch == 2:
            gradient = get_gradient_Robert(temp)
        elif sharpen_mode_switch == 3:
            gradient = get_gradient_prewitt(temp)
        elif sharpen_mode_switch == 4:
            gradient = get_gradient_sobel(temp)

    return gradient


def get_gradient_laplace(src):
    temp = src.copy()
    # temp = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    sharpen_op = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    sharpen_image = cv.filter2D(temp, cv.CV_32F, sharpen_op)
    sharpen_image = cv.convertScaleAbs(sharpen_image)  # 图像转换为 CV_8U
    cv.imshow("laplace", sharpen_image)
    return sharpen_image


def get_gradient_Robert(src):
    temp = src.copy()
    # temp = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    op_x = np.array([[-1, 0], [0, 1]], dtype=int)
    op_y = np.array([[0, -1], [1, 0]], dtype=int)

    x = cv.filter2D(temp, cv.CV_16S, kernel=op_x)
    y = cv.filter2D(temp, cv.CV_16S, kernel=op_y)

    abs_x = cv.convertScaleAbs(x)
    abs_y = cv.convertScaleAbs(y)

    alpha = 0.5
    res = cv.addWeighted(abs_x, alpha, abs_y, 1 - alpha, 0)
    cv.imshow("robert", res)
    return res


def get_gradient_sobel(src):
    temp = src.copy()
    # temp = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    op_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=int)
    op_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=int)

    x = cv.filter2D(temp, cv.CV_16S, kernel=op_x)
    y = cv.filter2D(temp, cv.CV_16S, kernel=op_y)

    abs_x = cv.convertScaleAbs(x)
    abs_y = cv.convertScaleAbs(y)

    alpha = 0.5
    res = cv.addWeighted(abs_x, alpha, abs_y, 1 - alpha, 0)
    cv.imshow("sobel", res)
    return res


def get_gradient_prewitt(src):
    temp = src.copy()
    # temp = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    op_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    op_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=int)

    x = cv.filter2D(temp, cv.CV_16S, kernel=op_x)
    y = cv.filter2D(temp, cv.CV_16S, kernel=op_y)

    abs_x = cv.convertScaleAbs(x)
    abs_y = cv.convertScaleAbs(y)

    alpha = 0.5
    res = cv.addWeighted(abs_x, alpha, abs_y, 1 - alpha, 0)
    cv.imshow("prewitt", res)
    return res


def gen_sharpen_image(src, gradient, base, beta):
    """
    功能： 梯度与图像融合
    """
    if src is None \
            and gradient is None:
        return None

    dst = src.copy()

    # dst = (1 - beta) * gradient + beta * dst
    dst = base * beta * gradient + dst

    dst = cv.convertScaleAbs(dst)
    cv.imshow("dst_beauty", dst)
    return dst


def blend_skin(src, dst, mask, name):
    """
    功能： 1.完成磨皮图像和原图通过求解的皮肤区域mask进行融合
          2.完成磨皮皮肤区域和原图通过融合因子alpha进行融合
    """
    temp = None
    if dst is not None \
            and src is not None \
            and mask is not None:
        alpha = cv.getTrackbarPos("blend_alpha", "beauty") / 100
        temp = (dst * alpha + src * (1 - alpha)) * mask + src * (1 - mask)
        temp = cv.convertScaleAbs(temp)
        cv.imshow("dst_blend_skin", temp)
    return temp


def image_process():
    """
    功能 ： 实现美颜技术方案
    输入： bgr图像
    """
    global src, mask, dst_blur, dst_blend_skin, dst_whiten, \
        dst_sharpen, gradient, name, blur_param1, blur_param2, blur_param3, \
        sharpen_base, sharpen_beta

    dst_blur = edge_protect_filtering_blur(src, name, blur_param1, blur_param2, blur_param3)
    mask = check_skin_color(dst_blur, name)
    dst_blend_skin = blend_skin(src, dst_blur, mask, name)
    dst_whiten = whiten_skin(dst_blend_skin, mask, name)
    gradient = gradient_mode(dst_whiten)
    dst_sharpen = gen_sharpen_image(dst_whiten, gradient, sharpen_base, sharpen_beta)

    return dst_sharpen


def onChangeGradientMode(x):
    global dst_whiten, gradient
    gradient = gradient_mode(dst_whiten)


def onChangeBlurSkinParam1(x):
    global src, dst_blur, name, blur_param1, blur_param2, blur_param3
    blur_param1 = cv.getTrackbarPos("blur_param1", "beauty")
    dst_blur = edge_protect_filtering_blur(src, name, blur_param1, blur_param2, blur_param3)
    mask = check_skin_color(dst_blur, name)


def onChangeBlurSkinParam2(x):
    global src, dst_blur, name, blur_param1, blur_param2, blur_param3
    blur_param2 = cv.getTrackbarPos("blur_param2", "beauty")
    dst_blur = edge_protect_filtering_blur(src, name, blur_param1, blur_param2, blur_param3)
    mask = check_skin_color(dst_blur, name)


def onChangeBlurSkinParam3(x):
    global src, dst_blur, name, blur_param1, blur_param2, blur_param3
    blur_param3 = cv.getTrackbarPos("blur_param3", "beauty")
    dst_blur = edge_protect_filtering_blur(src, name, blur_param1, blur_param2, blur_param3)
    mask = check_skin_color(dst_blur, name)


def onChangeBlendImage(x):
    global src, dst_blur, mask, dst_blend_skin, name
    dst_blend_skin = blend_skin(src, dst_blur, mask, name)


def onChangeWhitenSkin(x):
    global dst_blend_skin, dst_whiten, mask, name
    dst_whiten = whiten_skin(dst_blend_skin, mask, name)


def onChangeGenSharpenBeta(x):
    global src, dst_whiten, dst_sharpen, gradient, sharpen_beta, sharpen_base
    sharpen_beta = cv.getTrackbarPos("sharpen_beta", "beauty") / 100
    dst_sharpen = gen_sharpen_image(dst_whiten, gradient, sharpen_base, sharpen_beta)


def onChangeGenSharpenBase(x):
    global src, dst_whiten, dst_sharpen, gradient, sharpen_beta, sharpen_base
    sharpen_base = cv.getTrackbarPos("sharpen_base", "beauty") / 100
    dst_sharpen = gen_sharpen_image(dst_whiten, gradient, sharpen_base, sharpen_beta)


def onChange(x):
    pass


def test_beauty():
    """
    功能：美颜 demo UI
    调节：
        sigmaColor : 颜色通道方差
        sigmaSpace : 空间距离方差
        radius : 采样半径
    控制：
        w : 上一张图
        s : 下一张图
        空格 ：滤波处理
        q : 退出
    """
    global src, gradient

    # 参数初始化
    path = ROOT_PATH + ASIAN
    images = os.listdir(path)
    index = 0
    num_img = len(images) - 1
    name = images[index]
    # name = "no9_face.png"

    blur_param1 = 40
    blur_param2 = 4
    blur_param3 = 15
    blend_alpha = 75
    whiten_factor = 20
    sharpen_mode_switch = 4
    sharpen_beta = 50
    sharpen_base = 15

    # 读入图像
    src = img_read(name, path)

    # 创建UI界面
    cv.namedWindow("beauty", cv.WINDOW_NORMAL)
    cv.resizeWindow("beauty", 400, 800)

    cv.createTrackbar("blur_param1", "beauty", blur_param1, 100, onChangeBlurSkinParam1)
    cv.createTrackbar("blur_param2", "beauty", blur_param2, 100, onChangeBlurSkinParam2)
    cv.createTrackbar("blur_param3", "beauty", blur_param3, 100, onChangeBlurSkinParam3)
    cv.createTrackbar("blend_alpha", "beauty", blend_alpha, 100, onChangeBlendImage)
    cv.createTrackbar("whiten_factor", "beauty", whiten_factor, 100, onChangeWhitenSkin)
    cv.createTrackbar("sharpen_mode_switch", "beauty", sharpen_mode_switch, 4, onChangeGradientMode)
    cv.createTrackbar("sharpen_beta", "beauty", sharpen_beta, 100, onChangeGenSharpenBeta)
    cv.createTrackbar("sharpen_base", "beauty", sharpen_base, 100, onChangeGenSharpenBase)
    cv.setTrackbarMin("whiten_factor", "beauty", 1)
    cv.setTrackbarMin("sharpen_mode_switch", "beauty", 1)

    while True:
        dst_beauty = image_process()
        cv.imshow("dst_beauty", dst_beauty)
        k = cv.waitKey(0)
        if k == ord('q'):
            cv.destroyAllWindows()
            break
        elif k == ord('w'):
            cv.destroyWindow(name)
            # 读入上张图像w
            index = (index - 1) % num_img
            name = images[index]
            src = img_read(name, path)
        elif k == ord('s'):
            cv.destroyWindow(name)
            # 读入下张图像
            index = (index + 1) % num_img
            name = images[index]
            src = img_read(name, path)


def test_solo():
    """
        测试需要修改onchange 函数 的窗口名字
    """
    global src, gradient, dst_sharpen

    # 参数初始化
    path = ROOT_PATH + ASIAN
    images = os.listdir(path)
    index = 0
    num_img = len(images) - 1
    name = images[index]

    # 读入图像
    src = img_read(name, path)

    sharpen_beta = 20
    sharpen_base = 10

    # 创建UI界面
    cv.imshow(name, src)
    # cv.namedWindow("test_solo", cv.WINDOW_NORMAL)
    # cv.resizeWindow("test_solo", 400, 400)
    # cv.createTrackbar("sharpen_beta", "test_solo", sharpen_beta, 100, onChangeGenSharpenBeta)
    # cv.createTrackbar("sharpen_base", "test_solo", sharpen_base, 100, onChangeGenSharpenBase)

    while True:
        #####################################
        # start to compute the gradient of image
        #####################################
        # gradient = get_gradient_laplace(src)
        # gradient = get_gradient_Robert(src)
        # gradient = get_gradient_sobel(src)
        # gradient = get_gradient_prewitt(src)

        # dst_sharpen = gen_sharpen_image(src, gradient, sharpen_base, sharpen_beta)
        # cv.imshow("dst_beauty", dst_sharpen)
        #####################################
        # end to compute the gradient of image
        #####################################

        ##############################################
        # 肤色检测
        # mask_rgb = check_skin_rgb_algorithm(src)
        # mask_yuv = check_skin_YCbCr_algorithm(src)
        mask_hsv = check_skin_HSV_algorithm(src)
        # mask_yuv_dynamic = check_skin_YCRCB_dynamic(src)
        # mask_fusing = mask_yuv * mask_hsv
        # mask_fusing = check_skin_YCrCb_HSV_fusing(src)

        # cv.imshow("mask_rgb", mask_rgb[:, :, 0] * 255)
        # cv.imshow("mask_YCbCr", mask_yuv[:, :, 0] * 255)
        cv.imshow("mask_hsv", mask_hsv[:, :, 0] * 255)
        # cv.imshow("mask_fusing", mask_fusing[:, :, 0] * 255)

        # cv.imshow("mask_YCrCb_dynamic", mask_yuv_dynamic[:, :, 0] * 255)
        ##################################################

        k = cv.waitKey(0)
        if k == ord('q'):
            cv.destroyAllWindows()
            break
        elif k == ord('w'):
            cv.destroyWindow(name)
            # 读入上张图像w
            index = (index - 1) % num_img
            name = images[index]
            src = img_read(name, path)
        elif k == ord('s'):
            cv.destroyWindow(name)
            # 读入下张图像
            index = (index + 1) % num_img
            name = images[index]
            src = img_read(name, path)


def test():
    # 参数初始化
    path = ROOT_PATH + WEST
    images = os.listdir(path)
    index = 0
    # name = images[index]
    name = "30844800_1.jpg"
    src = img_read(name, path)

    # mask_rgb = check_skin_rgb_algorithm(src)
    # mask_yuv = check_skin_YCbCr_algorithm(src)
    mask_hsv = check_skin_HSV_algorithm(src)

    # cv.imshow("mask_rgb", mask_rgb[:, :, 0] * 255)
    # cv.imshow("mask_yCbCr", mask_yuv[:, :, 0] * 255)
    cv.imshow("mask_hsv", mask_hsv[:, :, 0] * 255)

    cv.waitKey()
    cv.destroyAllWindows()


def gaussian_kernel_2d_opencv(kernel_size=3, sigma=0):
    kx = cv.getGaussianKernel(kernel_size, sigma)
    ky = cv.getGaussianKernel(kernel_size, sigma)
    return np.multiply(kx, np.transpose(ky))


if __name__ == '__main__':
    # s = gaussian_kernel_2d_opencv(3, 1)
    # print(s)
    # print_package_version()
    test_beauty()
    # test_solo()
    # test()


