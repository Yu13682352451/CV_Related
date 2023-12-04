import glob
import os
import cv2
import shutil
import numpy as np
from scipy.spatial.transform import Rotation

import disp2depth
import stereoConfig


def color_frame(images_file, output):
    """
    读取彩色照片
    :param images_file:彩色照片所在的路径（数组）
    :param output: 想要输出的路径
    :return: None
    """
    for i, image_file in enumerate(images_file):
        print(f"Handling {i + 1} / {len(images_file)} ...")
        shutil.copy(image_file, f"{output}/frame-{i:06d}.color.png")


def depth_frame(disps_file, config, output):
    """
    读取视差图，将其转换为深度图并按一定命名格式保存到制定路径
    :param disps_file: 视差图所在的路径（数组）
    :param config: 相机参数
    :param output: 想要输出的路径
    :return: None
    """
    for i, disp_file in enumerate(disps_file):
        print(f"Handling {i + 1} / {len(disps_file)} ...")
        disp = np.load(disp_file)
        depth = disp2depth.disp2depth(disp, config, "CRE")
        # depth_gray = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"{output}/frame-{i:06d}.depth.png", depth)


def pose_arr(pose_text, output):
    """
    读取位姿文件，并将其以类3dmatch的文件格式进行保存
    :param pose_text:位姿txt文件
    :param output:想要输出的路径
    :return:None
    """
    # 打开文件
    with open(pose_text, 'r') as file:
        lines_per_batch = 5  # 每次读取的行数
        iterator = 0
        while True:
            try:
                Rt = []
                # 读取五行数据
                frame_data = [next(file) for _ in range(lines_per_batch)]
                # 处理每一行的内容
                for i, line in enumerate(frame_data):
                    if 0 < i < 5:  # 跳过第一行
                        line = line.replace("[", "").replace("]", "").replace(";", "")
                        row = [float(x) for x in line.strip().split(',')]
                        Rt.append(row)
                formatted_str = ""
                for row in Rt:
                    for number in row:
                        formatted_str += f"{number:+.8e}\t "
                    formatted_str += "\n"
                with open(f'{output}/frame-{iterator:06d}.pose.txt', 'w') as f:
                    f.write(formatted_str)
                iterator += 1

            except StopIteration:
                # 文件读取完毕，跳出循环
                break


def handle_orbslam_pose(CameraTrajectory, output):
    """
    数据集跑orbslam3得到的CameraTrajectory.txt，里面有每幅图像的位姿向量，将其转换为位姿矩阵并保存成txt
    :param CameraTrajectory:数据集跑orbslam3得到的CameraTrajectory.txt所在路径
    :param output:想要输出的路径
    :return:Frame_List 帧名字列表（不带后缀）
    """
    with open(CameraTrajectory) as f:
        frame = []
        for line in f:
            data_vector = [float(v) for v in line.split()]
            # 提取帧名字
            name = int(data_vector[0])
            frame.append(name)
            # 提取平移向量和四元数
            t = data_vector[1:4]
            qw = data_vector[7]
            qx = data_vector[4]
            qy = data_vector[5]
            qz = data_vector[6]

            # 将四元数转换为旋转矩阵
            rotation_matrix = np.array([
                [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)]
            ])

            # 构建4x4位姿矩阵
            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = rotation_matrix
            pose_matrix[:3, 3] = np.array(t)

            # 保存到txt文件中
            formatted_str = ""
            for row in pose_matrix:
                for number in row:
                    formatted_str += f"{number:+.8e}\t"
                formatted_str += "\n"
            with open(f"{output}/frame-{name:06d}.pose.txt", "w") as out:
                out.write(formatted_str)

    return frame


if __name__ == "__main__":
    # 变量初始化
    images_file = []
    disps_file = []
    images_input_path = "/data/OwnDataset/P06/left_rec/"
    disps_input_path = "/data/OwnDataset/P06/CRE_arr/"
    output_path = "/data/OwnDataset/P06KF_nvblox/seq-01"
    txt = "/data/OwnDataset/P06_EuRocformat/KeyFrameTrajectory_P06.txt"

    # 读取小觅相机的参数
    result = stereoConfig.readCameraCfg("/home/lbyj/PycharmProjects/xiaomi/Pictures/mynt_Q.yaml")
    config = stereoConfig.stereoCamera(result)

    # 判断文件是否存在，如果不存在则创建
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    frame_list = handle_orbslam_pose(txt, output_path)

    for frame in frame_list:
        images_file.append(images_input_path + f"{frame:08d}.png")
        disps_file.append(disps_input_path + f"{frame:08d}.npy")

    # color_frame(images_file, output_path)
    depth_frame(disps_file, config, output_path)
    # pose_arr("/home/lbyj/PycharmProjects/xiaomi/P005_pose_data.txt", output_path)

