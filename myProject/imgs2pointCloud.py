# -*- coding:utf-8 -*-
# @FileName :point_cloud.py
# @Time     :2023/4/8 21:49
# @Author   :YJ
import cv2
import numpy as np
import open3d as o3d
import time


class point_cloud_generator:
    def __init__(self, rgb_file, depth_file, save_ply, camera_intrinsics=[0.0, 0.0, 0.0, 0.0]):
        """
        :param rgb_file: 彩色图路径
        :param depth_file: 深度图路径
        :param save_ply: ply保存路径
        :param camera_intrinsics: 相机本质参数[fx, fy, cx, cy]
        """
        self.rgb_file = rgb_file
        self.depth_file = depth_file
        self.save_ply = save_ply

        self.rgb = cv2.imread(rgb_file)
        self.depth = cv2.imread(depth_file, -1)

        print("your depth image shape is:", self.depth.shape)

        self.width = self.rgb.shape[1]
        self.height = self.rgb.shape[0]

        self.camera_intrinsics = camera_intrinsics
        self.depth_scale = 1000

    def compute(self):
        t1 = time.time()

        depth = np.asarray(self.depth, dtype=np.uint16).T
        self.Z = depth / self.depth_scale
        fx, fy, cx, cy = self.camera_intrinsics

        X = np.zeros((self.width, self.height))
        Y = np.zeros((self.width, self.height))
        for i in range(self.width):
            X[i, :] = np.full(X.shape[1], i)
        self.X = ((X - cx / 2) * self.Z) / fx

        for i in range(self.height):
            Y[:, i] = np.full(Y.shape[0], i)
        self.Y = ((Y - cy / 2) * self.Z) / fy

        data_ply = np.zeros((6, self.width * self.height))
        data_ply[0] = self.X.T.reshape(-1)
        data_ply[1] = -self.Y.T.reshape(-1)
        data_ply[2] = -self.Z.T.reshape(-1)
        img = np.array(self.rgb, dtype=np.uint8)
        data_ply[3] = img[:, :, 0:1].reshape(-1)
        data_ply[4] = img[:, :, 1:2].reshape(-1)
        data_ply[5] = img[:, :, 2:3].reshape(-1)
        self.data_ply = data_ply
        t2 = time.time()
        print('calcualte 3d point cloud Done.', t2 - t1)

    def write_ply(self):
        start = time.time()
        float_formatter = lambda x: "%.4f" % x
        points = []
        for i in self.data_ply.T:
            points.append("{} {} {} {} {} {} 0\n".format
                          (float_formatter(i[0]), float_formatter(i[1]), float_formatter(i[2]),
                           int(i[3]), int(i[4]), int(i[5])))

        file = open(self.save_ply, "w")
        file.write('''ply
        format ascii 1.0
        element vertex %d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        property uchar alpha
        end_header
        %s
        ''' % (len(points), "".join(points)))
        file.close()

        end = time.time()
        print("Write into .ply file Done.", end - start)

    def show_point_cloud(self):
        pcd = o3d.io.read_point_cloud(self.save_ply)
        o3d.visualization.draw_geometries([pcd])


def TQ2Matrix(t, Q):
    """
    将平移向量以及四元数转换成相机位姿矩阵
    :param t: 平移向量，[x, y, z]
    :param Q: 四元数，[qx, qy, qz, qw]
    :return:相机位姿 -> np.array
    """
    # 创建一个四元数,格式为[qw, qx, qy, qz]
    q = np.array([Q[3], Q[0], Q[1], Q[2]])

    # 创建一个3D变换矩阵
    M = np.eye(4)  # 创建一个单位矩阵作为初始的变换矩阵

    # 使用四元数进行旋转
    R = np.eye(3)  # 创建一个单位矩阵作为旋转矩阵
    qw, qx, qy, qz = q
    R[0, 0] = 1 - 2 * (qy ** 2 + qz ** 2)
    R[0, 1] = 2 * (qx * qy - qz * qw)
    R[0, 2] = 2 * (qx * qz + qy * qw)
    R[1, 0] = 2 * (qx * qy + qz * qw)
    R[1, 1] = 1 - 2 * (qx ** 2 + qz ** 2)
    R[1, 2] = 2 * (qy * qz - qx * qw)
    R[2, 0] = 2 * (qx * qz - qy * qw)
    R[2, 1] = 2 * (qy * qz + qx * qw)
    R[2, 2] = 1 - 2 * (qx ** 2 + qy ** 2)

    M[:3, :3] = R  # 将旋转矩阵赋给变换矩阵

    # 使用平移向量进行平移
    M[:3, 3] = [t[0], t[1], t[2]]

    return M


def generatePC(camera_intrinsics, img, depth, M):
    width = img.shape[1]
    height = img.shape[0]
    depth_scale = 1000
    img = np.array(img, dtype=np.uint8)
    depth = np.asarray(depth, dtype=np.uint16).T
    t1 = time.time()

    Z = depth / depth_scale
    fx, fy, cx, cy = camera_intrinsics

    X = np.zeros((width, height))
    Y = np.zeros((width, height))

    for i in range(width):
        X[i, :] = np.full(X.shape[1], i)
    X = ((X - cx / 2) * Z) / fx

    for i in range(height):
        Y[:, i] = np.full(Y.shape[0], i)
    Y = ((Y - cy / 2) * Z) / fy

    points = np.zeros((3, width * height))
    points[0] = X.T.reshape(-1)
    points[1] = - Y.T.reshape(-1)
    points[2] = - Z.T.reshape(-1)
    points = np.dot(M[:3, :3], points)

    data_ply = np.zeros((6, width * height))
    data_ply[0:3] = points
    data_ply[3] = img[:, :, 0:1].reshape(-1)
    data_ply[4] = img[:, :, 1:2].reshape(-1)
    data_ply[5] = img[:, :, 2:3].reshape(-1)
    t2 = time.time()
    print('calcualte 3d point cloud Done.', t2 - t1)


if __name__ == '__main__':
    file_path = "/home/dq/CLionProjects/Dev/pointcloud/data/CRE_depth_pose.txt"
    camera_intrinsics = [707.58898925781250000, 708.73779296875000000, 629.33026123046875000, 389.08087158203125000]
    imgs = []
    depths = []
    t = []
    Q = []
    M = []
    with open(file_path, 'r') as file:
        for line in file:
            ele = line.split(' ')
            imgs.append(cv2.imread(ele[0], cv2.IMREAD_UNCHANGED))
            depths.append(cv2.imread(ele[1], cv2.IMREAD_UNCHANGED))
            t.append([float(ele[2]), float(ele[3]), float(ele[4])])
            Q.append([float(ele[5]), float(ele[6]), float(ele[7]), float(ele[8])])
            M.append(TQ2Matrix(t[-1], Q[-1]))
    generatePC(camera_intrinsics, imgs[0], depths[0], M[0])

    # rgb_file = "/home/dq/CLionProjects/Dev/pointcloud/data/rgb/249202344.jpg"
    # depth_file = "/home/dq/CLionProjects/Dev/pointcloud/data/depth_minusadd/249202344-dpt_swin2_large_384.png"
    # save_ply = "../pointcloud/data.ply"
    # a = point_cloud_generator(rgb_file=rgb_file,
    #                           depth_file=depth_file,
    #                           save_ply=save_ply,
    #                           camera_intrinsics=camera_intrinsics
    #                           )
    # a.compute()
    # a.write_ply()
    # a.show_point_cloud()
