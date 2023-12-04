# -*- coding:utf-8 -*-
# @FileName :get_pose.py
# @Time     :2023/4/13 17:38
# @Author   :YJ
import glob
from pyquaternion import Quaternion
import numpy as np
import pandas as pd
import cv2
import stereoConfig
import Camera


def get_ORBFeature(img_l, img_r):
    """
    :param img_l:相机拍摄的相邻帧
    :param img_r:相机拍摄的相邻帧
    :return:相邻帧对应的ORB匹配点
    """
    orb = cv2.ORB_create()
    # 特征求取
    keypoint_l = orb.detect(img_l)
    keypoint_r = orb.detect(img_r)
    keypoint_l, des_l = orb.compute(img_l, keypoint_l)
    keypoint_r, des_r = orb.compute(img_r, keypoint_r)
    # 特征匹配
    bf = cv2.DescriptorMatcher_create('BruteForce-Hamming')
    matches = bf.match(des_l, des_r)
    # 初筛
    min_distance = 10000
    max_distance = 0
    for x in matches:
        if x.distance < min_distance:
            min_distance = x.distance
        if x.distance > max_distance:
            max_distance = x.distance
    good_matches = []
    for x in matches:
        if x.distance <= max(2 * min_distance, 30):
            good_matches.append(x)
    # print("匹配数： %d" % len(good_matches))

    # 提取配准点
    points_l = []
    points_r = []
    for i in good_matches:
        points_l.append(list(keypoint_l[i.queryIdx].pt))
        points_r.append(list(keypoint_r[i.trainIdx].pt))
    points_l = np.array(points_l).astype(int)
    points_r = np.array(points_r).astype(int)
    return points_l, points_r


def solve_PnP(points_l, points_r, depthMap, config):
    """
    PnP算法的实现
    :param points_l: get_ORBFeature的返回值
    :param points_r: get_ORBFeature的返回值
    :param depthMap: 深度地图
    :param config: 相机参数
    :return: 旋转矩阵，平移矩阵
    """
    K = config.cam_matrix_left
    points_l_3D = []
    points_r_2D = []
    for i in range(points_l.shape[0]):
        p = points_l[i]
        d = depthMap[p[1], p[0]] / 5000.0  # 深度信息 以5000为一个单位
        if d == 0:
            continue
        p = (p - (K[0][2], K[1][2])) / (K[0][0], K[1][1]) * d  # 归一化坐标，根据深度转为实际坐标
        points_l_3D.append([p[0], p[1], d])
        points_r_2D.append(points_r[i])

    # print('最终匹配数： %d' % len(points_l_3D))
    points_l_3D = np.array(points_l_3D).astype(np.float64)
    points_r_2D = np.array(points_r_2D).astype(np.float64)

    flag, R, t = cv2.solvePnP(points_l_3D, points_r_2D, K, None)
    R, Jacobian = cv2.Rodrigues(R)
    return R, t


def save_pose(poses, len):
    """
    将位姿保存成csv文件
    :param poses: 表示位姿的矩阵， 其中位姿的格式为[x, y, z, qx, qy, qz, qw], 其中后四位使用四元数表示的相机的旋转向量
    :param len: 需要保存到长度
    :return: None
    """
    index = []
    for i in range(1, len+1):
        index.append(i)
    data = pd.DataFrame(poses, index=index)
    data.round(8)
    print(data)
    data.to_csv('../pose.txt', index=False, float_format='%.8f', sep=' ')
    print("位姿保存成功！")


if __name__ == "__main__":
    # 读取图像以及相机参数
    rgbl_imgs = sorted(glob.glob(r"/home/dq/CLionProjects/Dev/pointcloud/data/rgb/*.jpg"))
    rgbr_imgs = sorted(glob.glob(r"/data/ORB_SLAM_series/orb_slam3_agv_latest/P05/right/*.jpg"))
    depth_imgs = sorted(glob.glob(r"/home/dq/CLionProjects/Dev/pointcloud/data/depth/*.png"))
    pfm_imgs = sorted(glob.glob(r"/home/dq/CLionProjects/Dev/pointcloud/data/depth/*.pfm"))
    result = Camera.readCameraCfg("../Pictures/result.yaml")
    config = stereoConfig.stereoCamera(result)

    poses = []

    for i in range(len(rgbl_imgs)):
        # 特征提取以及PnP算法实现
        print("计算第{}张照片的位姿......".format(i))
        imgl = cv2.imread(rgbl_imgs[i])
        imgr = cv2.imread(rgbr_imgs[i])
        depth_img = cv2.imread(depth_imgs[i], cv2.IMREAD_GRAYSCALE)
        key_points1, key_points2 = get_ORBFeature(imgl, imgr)
        # 计算当前帧相机位姿
        R, t = solve_PnP(key_points1, key_points2, depth_img, config)
        print(type(R))
        # 旋转矩阵转换为四元数
        q = Quaternion(matrix=R)
        # 如果只想要位姿
        # p = [t[0][0], t[1][0], t[2][0], q.x, q.y, q.z, q.w]
        # 把位姿对应的图片路径加进来
        a = [rgbl_imgs[i], pfm_imgs[i], t[0][0], t[1][0], t[2][0], q.x, q.y, q.z, q.w]
        poses.append(a)

    print('')
    save_pose(poses, len(rgbl_imgs))
