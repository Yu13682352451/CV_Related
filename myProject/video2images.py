# -*- coding :utf-8 -*-
# @FileName  :video2images.py
# @Time      :2023/11/18 上午9:23
# @Author    :YJ

import cv2
import os


if __name__ == "__main__":
    # 视频文件路径
    video_path = '/data/OwnDataset/P06/right1.avi'
    # 输出帧保存的文件夹
    output_folder = '/data/OwnDataset/P06/right'
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    # 获取视频帧率
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # 获取视频宽高
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 计数器，用于为输出的图像文件命名
    frame_count = 0
    # 设置视频的开始时间（毫秒）
    start_time_ms = 180000  # 180秒（3分钟）
    # 设置视频的结束时间（毫秒）
    end_time_ms = 460000  # 460秒（7分40秒)

    # 设置视频的当前位置
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time_ms)

    # 读取一帧以确保视频的当前位置被正确设置
    ret, frame = cap.read()

    # 循环读取视频帧直到达到指定的结束时间
    while cap.get(cv2.CAP_PROP_POS_MSEC) <= end_time_ms:
        ret, frame = cap.read()

        if not ret:
            break

        # 构建输出图像文件名
        output_filename = f"{output_folder}/{frame_count:08d}.png"
        # 保存帧为图像文件
        cv2.imwrite(output_filename, frame)
        print(f"已保存{frame_count}帧")
        # 帧计数器递增
        frame_count += 1

    # 释放视频捕获对象
    cap.release()

    print(f"帧提取完成，共提取 {frame_count} 帧。")

