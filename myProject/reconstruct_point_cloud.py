# -*- coding :utf-8 -*-
# @FileName  :reconstruct_point_cloud.py
# @Time      :2023/11/21 下午5:20
# @Author    :YJ
import glob
import numpy as np
import plotly.graph_objects as go
import open3d as o3d
from imageio.v3 import imread
import re

import disp2depth
import stereoConfig


def transform_points(points, pose):
    """
    将每幅图像的点转换到第一张图像的坐标系上
    :param points: 利用深度转点云的公式计算出来的点坐标
    :param pose: 该幅图像的位姿
    :return: 转换后的点云坐标
    """
    # 通过相机位姿对点云进行变换
    transformed_points = np.dot(pose[:3, :3], points.T) + pose[:3, 3:4]
    return transformed_points.T


def reconstruct_point_cloud(images, disps, poses, config, threshold=10, NUM_POINTS_TO_DRAW=5000000):
    """
    构造点云
    :param images: 需要构造点云的彩色图像序列路径
    :param disps: 需要构造点云的视差图像序列路径
    :param poses: 需要构造点云的图像序列位姿路径
    :param config: 相机的配置文件
    :param threshold: 对点云简单筛选的阈值
    :return: 点云坐标，每个点对应的颜色信息
    """
    cx0 = config.cam_matrix_left[0][2]
    cy0 = config.cam_matrix_left[1][2]
    fx = config.cam_matrix_left[0][0]
    fy = config.cam_matrix_left[1][1]

    num_frames = len(images)

    all_points = []
    all_colors = []

    for i in range(num_frames):
        img = imread(images[i])
        # depth = disp2depth.disp2depth(np.load(disps[i]), config, method="CRE")
        depth = imread(disps[i])
        pose = read_pose(poses[i])

        H, W = depth.shape
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        points_grid = np.stack(((xx-cx0)/fx, -(yy-cy0)/fy, -np.ones_like(xx)), axis=0) * depth
        mask = np.ones((H, W), dtype=bool)

        # Remove flying points
        if i > 0:
            mask[1:][np.abs(depth[1:] - depth[i-1][:-1]) > threshold] = False
            mask[:, 1:][np.abs(depth[:, 1:] - depth[i-1][:,:-1]) > threshold] = False

        points = points_grid.transpose(1,2,0)[mask]
        colors = img[mask].astype(np.float64) / 200

        # Apply pose transformation to points
        transposed_points = transform_points(points, pose)
        # transposed_colors = transposed_points(colors, pose)

        all_points.append(transposed_points)
        all_colors.append(colors)

        # Combine points and colors from all frames
        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)

        # filter

        subset = np.random.choice(points.shape[0], size=(NUM_POINTS_TO_DRAW,), replace=True)
        points_subset = all_points[subset]
        colors_subset = all_colors[subset]
        return points_subset, colors_subset


def read_pose(pose_path):
    """
    从txt中读取位姿
    :param pose_path: 存储位姿信息的txt文件
    :return: 4✖4的位姿矩阵
    """
    pose = []
    with open(pose_path, "r") as f:
        for line in f:
            numbers = re.findall(r"[-+]?\d*\.\d+e[+-]?\d+", line.strip())
            # 将提取的数字转换为浮点数
            float_numbers = [float(number) for number in numbers]
            pose.append(float_numbers)
    return np.array(pose)


if __name__ == "__main__":
    result = stereoConfig.readCameraCfg("/home/lbyj/PycharmProjects/xiaomi/Pictures/mynt_mean.yaml")
    config = stereoConfig.stereoCamera(result)

    image_files = sorted(glob.glob("/data/OwnDataset/617_nvbloxFormat/seq-01/*.color.png"))
    depth_files = sorted(glob.glob("/data/OwnDataset/617_nvbloxFormat/seq-01/*.depth.png"))
    pose_files = sorted(glob.glob("/data/OwnDataset/617_nvbloxFormat/seq-01/*.pose.txt"))

    points, colors = reconstruct_point_cloud(image_files, depth_files, pose_files, config)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud("../617pointcloud.ply", point_cloud)

    # print("""
    # Controls:
    # ---------
    # Zoom:      Scroll Wheel
    # Translate: Right-Click + Drag
    # Rotate:    Left-Click + Drag
    # """)
    #
    # x, y, z = points.T
    #
    # fig = go.Figure(
    #     data=[
    #         go.Scatter3d(
    #             x=x, y=z, z=y,  # flipped to make visualization nicer
    #             mode='markers',
    #             marker=dict(size=1, color=colors)
    #         )
    #     ],
    #     layout=dict(
    #         scene=dict(
    #             xaxis=dict(visible=True),
    #             yaxis=dict(visible=True),
    #             zaxis=dict(visible=True),
    #         )
    #     )
    # )
    # fig.show()
