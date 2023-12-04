# -*- coding:utf-8 -*-
# @FileName :main.py
# @Time     :2023/4/6 9:27
# @Author   :YJ
# -*- coding: utf-8 -*-
import os.path
import sys
import numpy as np
import cv2
import glob
import Camera
import img_process
import stereoConfig


if __name__ == "__main__":
    # 读取相机内参和外参
    result = Camera.readCameraCfg('../Pictures/myntresult.yaml')
    config = stereoConfig.stereoCamera(result)
    path = "/data/OwnDataset/P06"
    # # 读取图片
    imgls_path = sorted(glob.glob(f"{path}/left/*.png"))
    imgrs_path = sorted(glob.glob(f'{path}/right/*.png'))
    Q = []
    for i in range(0, len(imgls_path)):
        print("---------------- 处理第 %d 对照片中 ----------------" % i)
        img_l = cv2.imread(imgls_path[i], cv2.IMREAD_UNCHANGED)
        img_r = cv2.imread(imgrs_path[i], cv2.IMREAD_UNCHANGED)
        if(img_l is None) or (img_r is None):
            print("Error: Can`t read images, please check your images` path!")
            sys.exit(0)
        height, width = img_l.shape[0:2]
        # 立体校正
        map1x, map1y, map2x, map2y, Q = img_process.getRectifyTransform(height, width, config)
        img_l_rectified, img_r_rectified = img_process.rectifyImage(img_l, img_r, map1x, map1y, map2x, map2y)
        if not os.path.exists(f"{path}/left_rec"):
            os.makedirs(f"{path}/left_rec")
        if not os.path.exists(f"{path}/right_rec"):
            os.makedirs(f"{path}/right_rec")
        cv2.imwrite(f"{path}/left_rec/{i:08d}.png", img_l_rectified)
        cv2.imwrite(f"{path}/right_rec/{i:08d}.png", img_r_rectified)
    print(Q)
        # img_l_rectified = img_l
        # img_r_rectified = img_r
        # #绘制等间距的平行线，检验立体校正的效果
        # line = img_process.draw_line(img_l_rectified, img_r_rectified)
        # cv2.imwrite('../data/check_rectification' + str(i) + '.png', line)
        #
        # #立体匹配
        # img_l_, img_r_ = img_process.preprocess(img_l_rectified, img_r_rectified)
        # disp = img_process.stereoMatchSGBM(img_l, img_r, down_scale=True, WLS_Filter=True)
        # cv2.imshow('disp', disp)
        # cv2.waitKey(0)
        # cv2.imwrite('../disparity' + str([i]) + '.png', disp * 4)
        #
        # #计算深度图
        # depthMap = img_process.getDepth(disparity=disp*4, Q=Q, method=True)
        # depthMap.astype(np.uint16)
        # cv2.namedWindow('depth', 0)
        # cv2.imshow('depth', depthMap)
        # cv2.waitKey(0)
        # cv2.imwrite(f'../data/depth/{i:04d}.png', depthMap)
