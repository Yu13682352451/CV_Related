# -*- coding:utf-8 -*-
# @FileName :distance_cal.py
# @Time     :2023/4/12 21:03
# @Author   :YJ
import numpy as np
import cv2
import math
import img_process
import Camera
import stereoConfig


img_l = cv2.imread("/home/dq/PycharmProjects/xiaomi/Pictures/0_left.jpg", 1)
img_r = cv2.imread("/home/dq/PycharmProjects/xiaomi//Pictures/0_right.jpg", 1)
global point1
global point2
global img


# 鼠标回调函数
def draw_circle(event, u, v, flags, param):
    global point1
    global point2
    global img
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        point1 = (u, v)
        cv2.circle(img_1, point1, 1, (255, 255, 255), -1)
        print(point1)
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
        point2 = (u, v)
        cv2.circle(img_1, point2, 1, (0, 0, 0), -1)
        cv2.line(img_1, point1, point2, (255, 255, 255), 1)
        print(point2)


def distance(x1, y1, z1, x2, y2, z2):
    x = x1 - x2
    y = y1 - y2
    z = z1 - z2
    return math.sqrt(x ** 2 + y ** 2 + z ** 2)


def convert2world(point, dis, Q):
    u = point[0]
    v = point[1]
    cx = -Q[0, 3]
    cy = -Q[1, 3]
    baseline = -1 / Q[3, 2]
    fx = abs(Q[2, 3])

    x = (u - cx) * baseline / dis[v][u]
    y = (v - cy) * baseline / dis[v][u]
    depth = (fx * baseline) / dis[v][u]
    return x, y, depth
    # points_3d = cv2.reprojectImageTo3D(disp, Q)
    # x, y, depth = cv2.split(points_3d)
    # return x[v][u], y[v][u], depth[v][u]


if __name__ == "__main__":
    # 读取相机内参和外参
    result = Camera.readCameraCfg('../Pictures/result.yaml')
    config = stereoConfig.stereoCamera(result)
    # 立体校正
    height, width = img_l.shape[0:2]
    map1x, map1y, map2x, map2y, Q = img_process.getRectifyTransform(height, width, config)
    img_l_rectified, img_r_rectified = img_process.rectifyImage(img_l, img_r, map1x, map1y, map2x, map2y)
    # 立体匹配
    img_l_, img_r_ = img_process.preprocess(img_l_rectified, img_r_rectified)
    disp = img_process.stereoMatchSGBM(img_l_, img_r_, down_scale=True, WLS_Filter=True)

    img_1 = img_l_rectified.copy()
    # 鼠标回调，图像中取点
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", draw_circle)
    while 1:
        cv2.imshow("image", img_1)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    x1, y1, z1 = convert2world(point1, disp, Q)
    x2, y2, z2 = convert2world(point2, disp, Q)
    Point1 = [x1, y1, z1]
    print(Point1)
    Point2 = [x2, y2, z2]
    print(Point2)
    distance = distance(x1, y1, z1, x2, y2, z2)
    print(distance)

    cv2.destroyAllWindows()
