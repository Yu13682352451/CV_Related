import os
import shutil

# 设置输入数据集文件夹和输出文件夹
input_dataset_path = "/data/OwnDataset/stairs"
output_dataset_path = "/data/OwnDataset/stairs_EuRocformat"

# 创建输出文件夹结构
if not os.path.exists(output_dataset_path):
    os.makedirs(output_dataset_path)

mav0_path = os.path.join(output_dataset_path, "MH01", "mav0")
os.makedirs(mav0_path)
cam0_path = os.path.join(mav0_path, "cam0")
os.makedirs(cam0_path)
cam1_path = os.path.join(mav0_path, "cam1")
os.makedirs(cam1_path)
data0_path = os.path.join(cam0_path, "data")
os.makedirs(data0_path)
data1_path = os.path.join(cam1_path, "data")
os.makedirs(data1_path)

# 获取左右相机图像文件列表
left_images = sorted(os.listdir(os.path.join(input_dataset_path, "left_rec")))
right_images = sorted(os.listdir(os.path.join(input_dataset_path, "right_rec")))

# 将图像复制到相应的文件夹并生成时间戳信息
with open(os.path.join(output_dataset_path, "times.txt"), "w") as times_file:
    for i, (left_image, right_image) in enumerate(zip(left_images, right_images)):
        left_image_path = os.path.join(input_dataset_path, "left_rec", left_image)
        right_image_path = os.path.join(input_dataset_path, "right_rec", right_image)

        # 使用图像名作为时间戳
        timestamp = left_image.split(".")[0]

        # 复制左右图像到cam0和cam1文件夹
        left_output_path = os.path.join(data0_path, f"{timestamp}.png")
        right_output_path = os.path.join(data1_path, f"{timestamp}.png")
        shutil.copy(left_image_path, left_output_path)
        shutil.copy(right_image_path, right_output_path)

        # 将时间戳写入times.txt
        times_file.write(f"{timestamp}\n")

print("Dataset conversion completed.")
