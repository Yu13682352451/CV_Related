import cv2
import glob
import stereoConfig
import numpy as np
from PIL import Image


def disp2depth(disp, baseline, fx, cx0, cx1):
    """
    利用相机参数将视差图转换为深度图，其公式为 （fx * baseline） / |baseline + (cx1 - cx0)|
    :param disp: 视差图
    :param baseline:相机基线长度 单位：mm
    :param fx:相机焦距长度，一般左右目差别不大，任取一个即可
    :param cx0:左相机的cx值
    :param cx1:右相机的cx值
    :return:深度图
    """
    depth = (fx * baseline) / (-disp + (cx1 - cx0))
    return depth.astype(np.uint16)


def disp2depth(disp, config, method="CRE"):
    """
    利用相机参数将视差图转换为深度图，其公式为 （fx * baseline） / |baseline + (cx1 - cx0)|
    :param disp:视差图
    :param config:使用streroConfig读取的相机配置数据
    :param method:使用什么立体匹配方法得到的误差图
    :return:深度图
    """
    # 将计算视差时用到的参数拿出来
    baseline = config.T[0]
    fx = config.cam_matrix_left[0][0]
    cx0 = config.cam_matrix_left[0][2]
    cx1 = config.cam_matrix_right[0][2]
    # # 使用Q矩阵中的参数
    # baseline = 119.8959
    # fx = 655.3433
    # # 使用左右相机的平均参数
    # baseline = config.T[0]
    # fx = (config.cam_matrix_left[0][0] + config.cam_matrix_right[0][0]) / 2

    if method == "RAFT":
        depth = (fx * baseline) / (-disp + (cx1 - cx0))
        # depth = (fx * baseline) / (-disp)
        # print("RAFT", depth)
    elif method == "CRE":
        depth = (fx * baseline) / (-disp + (cx1 - cx0))
        depth = depth / 4
        # depth = (fx * baseline) / (disp)
        # depth = depth.astype(np.uint16)
        # print("CRE", depth)
    return depth.astype(np.uint16)


# def disp2depth(disp, Q, method="CRE"):
#     fx = Q[2][3]
#     baseline = 1 / Q[3][2]
#     if method == "RAFT":
#         depth = (fx * baseline) / (-disp)
#         # print("RAFT", depth)
#     elif method == "CRE":
#         depth = (fx * baseline) / (disp)
#         # depth = abs(depth)
#         # depth = 65535 * (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
#         # depth = depth.astype(np.uint16)
#         # print("CRE", depth)
#     return depth


if __name__ == "__main__":
    # 读取小觅相机的参数
    result = stereoConfig.readCameraCfg("../Pictures/myntresult.yaml")
    config = stereoConfig.stereoCamera(result)
    disp_files = sorted(glob.glob("/home/lbyj/GitHubProjects/CREStereo/arr_office-rec/00000000.npy"))

    for i, disp_file in enumerate(disp_files):
        print(f"Handling {i+1}/{len(disp_files)} Picture ...")
        # frame_name = disp_file.split("/")[-1].split(".")[0].split("arr")[-1]
        disp = np.load(disp_file)
        depth = disp2depth(disp, config, "CRE")
        cv2.imwrite("../depth.png", depth)

