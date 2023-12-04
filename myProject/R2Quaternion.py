import numpy as np
from pyquaternion import Quaternion
import glob
import pandas as pd


def read_pose_txt(txtpath):
    # 保存R，t矩阵的数组
    R_list = []
    t_list = []
    # 打开文件
    with open(txtpath, 'r') as file:
        lines_per_batch = 5  # 每次读取的行数
        while True:
            try:
                Rt = []
                R = []
                t = []
                # 读取五行数据
                frame_data = [next(file) for _ in range(lines_per_batch)]
                # 检查是否读取到文件末尾
                # 处理每一行的内容，例如打印
                for i, line in enumerate(frame_data):
                    if 0 < i < 4:  # 跳过第一行和最后一行
                        line = line.replace("[", "").replace("]", "").replace(";", "")
                        row = [float(x) for x in line.strip().split(',')]
                        Rt.append(row)

                R = [row[:3] for row in Rt]
                t = [row[3] for row in Rt]
                R_list.append(R)
                t_list.append(t)

            except StopIteration:
                # 文件读取完毕，跳出循环
                break

    return R_list, t_list


def read_orbslam_pose(CameraTrajectory):
    """
       数据集跑orbslam3得到的CameraTrajectory.txt，里面有每幅图像的位姿向量，将其转换为位姿矩阵并保存成txt
       :param CameraTrajectory:数据集跑orbslam3得到的CameraTrajectory.txt所在路径
       :return:R_list, t_list,旋转向量（四元数）和平移向量
       """
    R_list = []
    t_list = []
    with open(CameraTrajectory) as f:
        for line in f:
            pose_vector = [float(v) for v in line.split()]
            # 首先提取平移向量和四元数
            t = pose_vector[1:4]
            q = pose_vector[4:]
            R_list.append(q)
            t_list.append(t)
    print(R_list)
    print(t_list)

    return R_list, t_list


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
    data.to_csv('../617_depth_pose.txt', index=False, float_format='%.8f', sep=' ')
    print("位姿保存成功！")


if __name__ == "__main__":
    rgbl_imgs = sorted(glob.glob(r"/data/OwnDataset/617_nvbloxFormat/seq-01/*.color.png"))
    depth_imgs = sorted(glob.glob(r"/data/OwnDataset/617_nvbloxFormat/seq-01/*.depth.png"))
    poses = []

    R_list, t_list = read_orbslam_pose('/data/OwnDataset/617/CameraTrajectory.txt')
    for R, t, rgb, depth in zip(R_list, t_list, rgbl_imgs, depth_imgs):
        a = [rgb, depth, t[0], t[1], t[2], R[0], R[1], R[2], R[3]]
        poses.append(a)

    save_pose(poses, len(depth_imgs))
