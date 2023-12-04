import cv2
import numpy as np

# 读取深度图像
depth_image = cv2.imread('/home/dq/PycharmProjects/xiaomi/1311877890.075198.png', cv2.IMREAD_GRAYSCALE)


# 定义填充函数
def fill_depth_zeros(depth_image):
    # 首先创建一个深度图像的副本
    filled_depth_image = depth_image.copy()

    # 获取深度图像的高度和宽度
    height, width = depth_image.shape

    # 迭代遍历深度图像中的每个像素
    for y in range(height-1):
        for x in range(width-1):
            if depth_image[y, x] == 0:
                # 找到最近的非零深度值
                nearest_depth = find_nearest_nonzero_depth(depth_image, x, y)
                # # 双线性插值
                # nearest_depth = bilinear_interpolation(depth_image, x, y)

                # 使用最近的深度值填充当前像素
                filled_depth_image[y, x] = nearest_depth

    return filled_depth_image


# 定义最近邻插值函数来查找最近的非零深度值
def find_nearest_nonzero_depth(depth_image, x, y):
    max_distance = max(depth_image.shape)
    for distance in range(1, max_distance):
        for dx in range(-distance, distance + 1):
            for dy in range(-distance, distance + 1):
                if 0 <= x + dx < depth_image.shape[1] and 0 <= y + dy < depth_image.shape[0]:
                    if depth_image[y + dy, x + dx] > 0:
                        return depth_image[y + dy, x + dx]
    return 0


# 定义双线性插值函数
def bilinear_interpolation(depth_image, x, y):
    x1 = int(x)
    x2 = x1 + 1
    y1 = int(y)
    y2 = y1 + 1

    # 四个相邻像素的深度值
    q11 = depth_image[y1, x1]
    q12 = depth_image[y2, x1]
    q21 = depth_image[y1, x2]
    q22 = depth_image[y2, x2]

    # 权重
    dx = x - x1
    dy = y - y1

    # 使用双线性插值计算深度值
    depth = (1 - dx) * (1 - dy) * q11 + dx * (1 - dy) * q21 + (1 - dx) * dy * q12 + dx * dy * q22

    return depth


# # 使用填充函数填充深度图像
# filled_depth_image = fill_depth_zeros(depth_image)

# 使用opencv自带的滤波方法填充深度图像
filled_depth_image = cv2.bilateralFilter(depth_image, 9, 75, 75)

# 保存填充后的深度图像
cv2.imwrite('../filled_depth_image_cvbila.png', filled_depth_image)
