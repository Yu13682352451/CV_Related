# -*- coding:utf-8 -*-
# @FileName :img_process.py
# @Time     :2023/4/7 16:35
# @Author   :YJ
import cv2
import numpy as np
import stereoConfig


# 图片预处理，将彩色图变为灰度图并进行直方图均衡操作，以此减少光照变化对操作过程的影响
def preprocess(img_l, img_r):
    if img_l.ndim == 3:
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    if img_r.ndim == 3:
        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    # 直方图均衡
    img_l = cv2.equalizeHist(img_l)
    img_r = cv2.equalizeHist(img_r)

    return img_l, img_r


# 消除畸变
def unDistortion(image, camera_matrix, dist_coeff):
    unDistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)

    return unDistortion_image


# 获取畸变校正和立体校正的映射变换矩阵,重投影矩阵
# config是一个类,存储了双目相机的参数:config = stereoConfig.stereoCamera()
def getRectifyTransform(height, width, config):
    # 读取相机的内外参数
    K1 = config.cam_matrix_left
    K2 = config.cam_matrix_right
    D1 = config.distortion_l
    D2 = config.distortion_r
    R, T = config.R, config.T

    # 计算校正变换
    height = int(height)
    width = int(width)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, (width, height), R, T, alpha=0)

    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (width, height), cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, Q


# 畸变校正和立体校正
def rectifyImage(img_l, img_r, map1x, map1y, map2x, map2y):
    rec_img_l = cv2.remap(img_l, map1x, map1y, cv2.INTER_AREA)
    rec_img_r = cv2.remap(img_r, map2x, map2y, cv2.INTER_AREA)

    return rec_img_l, rec_img_r


# 立体校正检验----画线
def draw_line(image1, image2):
    # 建立输出图像
    # [0]是高度, [1]是宽度
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]

    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2

    # 绘制等间距平行线
    line_interval = 50  # 直线间隔：50
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0), thickness=2,
                 lineType=cv2.LINE_AA)

    return output


# 视差计算
def stereoMatchSGBM(img_l, img_r, down_scale=False, WLS_Filter=True):
    # SGBM匹配参数设置
    if img_l.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3

    blockSize = 3
    param_l = {
        'minDisparity': -1,
        'numDisparities': 6*16,
        'blockSize': blockSize,
        'P1': 8 * img_channels * blockSize ** 2,
        'P2': 32 * img_channels * blockSize ** 2,
        'disp12MaxDiff': 1,
        'preFilterCap': 32,
        'uniquenessRatio': 10,
        'speckleWindowSize': 100,
        'speckleRange': 32,
        'mode': cv2.STEREO_SGBM_MODE_HH
    }

    # 构建SGBM对象
    matcher_l = cv2.StereoSGBM_create(**param_l)
    matcher_r = cv2.ximgproc.createRightMatcher(matcher_l)

    # 计算视差图
    size = (img_l.shape[1], img_l.shape[0])
    if not down_scale:
        disparity_l = matcher_l.compute(img_l, img_r)
        disparity_r = matcher_r.compute(img_r, img_l)
        disparity_l = np.int16(disparity_l)
        disparity_r = np.int16(disparity_r)
    else:
        img_l_down = cv2.pyrDown(img_l)
        img_r_down = cv2.pyrDown(img_r)
        factor = img_l.shape[1] / img_l_down.shape[1]
        disparity_l_half = matcher_l.compute(img_l_down, img_r_down)
        disparity_r_half = matcher_r.compute(img_r_down, img_l_down)
        disparity_l = cv2.resize(disparity_l_half, size, interpolation=cv2.INTER_AREA)
        disparity_r = cv2.resize(disparity_r_half, size, interpolation=cv2.INTER_AREA)
        disparity_l = factor * disparity_l
        disparity_r = factor * disparity_r
        disparity_l = np.int16(disparity_l)
        disparity_r = np.int16(disparity_r)

    dis = disparity_l

    if WLS_Filter:
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=matcher_l)
        wls_filter.setLambda(80000)
        wls_filter.setSigmaColor(1.3)

        # disparity_l = np.int16(disparity_l)
        # disparity_r = np.int16(disparity_r)

        dis = wls_filter.filter(disparity_l, img_l, None, disparity_r)
        # dis.astype(np.uint8)

    dis = cv2.normalize(src=dis, dst=dis, beta=0, alpha=65535, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    # # 真实视差（SGBM算法得到的视差是真实视差x16的）
    trueDisp = np.divide(dis.astype(np.float32), 16.)
    # trueDisp.astype(np.uint16)
    # new_disp = trueDisp * 2

    return trueDisp


# 根据公式计算深度
def getDepth(disparity, Q, scale=1.0, method=False):
    """
    reprojectImageTo3D(disparity, Q), 输入的Q，单位必须是毫米（mm）
    :param disparity: 视差图
    :param Q: 重投影矩阵 Q=[[1,0,    0,        -cx]
                          [0,1,    0,        -cy]
                          [0,0,    0,          f]
                          [1,0,-1/Tx,(cx-cx`)/Tx]]
              其中f为焦距，Tx相当于平移向量T的第一个参数
    :param scale:单位变化尺度，默认scale=1.0，即将单位转换为毫米
    :param method:是否使用cv2.reprojectImageTo3D函数计算深度，默认为否
    :return: depth：ndarray(np.uint16)，返回深度图
    """
    if method:
        points_3d = cv2.reprojectImageTo3D(disparity, Q)
        x, y, depth = cv2.split(points_3d)
    else:
        baseline = 1 / Q[3, 2]
        fx = abs(Q[2, 3])
        depth = (fx * baseline) / (disparity + 1e-6)
        depth = depth * scale
    return depth


def convert2uint16(depth_img):
    depth_imgu16 = depth_img.astype(np.uint16)
    depth_imgu16 = ((depth_imgu16 / 254) * 65535).astype(np.uint16)
    return depth_imgu16


average_depth = 0
drawing_rec = True


if __name__ == "__main__":
    # real_depth = 700
    result = stereoConfig.readCameraCfg("../Pictures/myntresult.yaml")
    config = stereoConfig.stereoCamera(result)

    img_l = cv2.imread("/data/P05/left/249202344.jpg")
    img_r = cv2.imread("/data/P05/right/249202344.jpg")

    # 立体校正
    height, width = img_l.shape[0:2]
    map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)
    print(Q)
    # img_l_rectified, img_r_rectified = rectifyImage(img_l, img_r, map1x, map1y, map2x, map2y)
    # # 立体匹配
    # img_l_, img_r_ = preprocess(img_l_rectified, img_r_rectified)
    # # cv2.imwrite('../data/rec_left_img_xiao.png', img_l_)
    # # cv2.imwrite('../data/rec_right_img_xiao.png', img_r_)
    # disp = stereoMatchSGBM(img_l_, img_r_, down_scale=True, WLS_Filter=True)
    # cv2.imshow('disp', disp)
    # cv2.waitKey(0)
    # cv2.imwrite('../data/xiao_disp_div11.bmp', disp)
    # depth_img = getDepth(disp, Q, method=False)
    #
    # cv2.imwrite('../data/xiao_depth_div11.bmp', depth_img)
    #
    # cv2.namedWindow('depth', 1)
    # cv2.imshow('depth', depth_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    #
    # drawing_rec = True
    # cv2.namedWindow('depth_1', 1)
    # cv2.imshow('depth_1', depth_img)
    # cv2.waitKey(0)
