import time

import cv2
import numpy as np

global depth_image, average_depth


# 定义鼠标回调函数
def draw_retangle(event, x, y, flags, param):
    global depth_image, average_depth

    # 鼠标左键按下时记录起始坐标
    if event == cv2.EVENT_LBUTTONDOWN:
        param['start'] = (x, y)

    # 鼠标左键释放时记录结束坐标，并绘制矩形框
    elif event == cv2.EVENT_LBUTTONUP:
        start = param['start']
        end = (x, y)
        print(start, end)
        cv2.rectangle(depth_image, start, end, (0, 255, 0), 2)
        cv2.imshow('Depth Image', depth_image)
        # 提取矩形框中的深度值
        roi = depth_image[start[1]:end[1], start[0]:end[0]]
        average_depth = np.mean(roi)

        print("平均深度:", average_depth)


def attention_color(path_dimg, path_img):
    global depth_image, average_depth
    depth_image = cv2.imread(path_dimg, cv2.IMREAD_ANYDEPTH)
    img = cv2.imread(path_img, cv2.IMREAD_UNCHANGED)

    print(depth_image.shape, img.shape)

    if depth_image is None:
        print("Failed to load depth image!")
    else:
        cv2.namedWindow('Depth Image')

        cv2.setMouseCallback('Depth Image', draw_retangle, {'start': (-1, -1)})

        while 1:
            cv2.imshow("Depth Image", depth_image)
            k = cv2.waitKey(1)
            if k == ord('q'):
                break

        # 设置深度范围
        min_depth = average_depth - 30
        max_depth = average_depth + 30

        # 创建一个与深度图像大小相同的掩码，初值为0
        mask = np.zeros_like(img, dtype=np.uint8)

        # 根据深度范围创建掩码
        T1 = time.clock()
        mask[(depth_image >= min_depth) & (depth_image <= max_depth)] = [255, 255, 255]

        # 应用掩码
        result_image = cv2.bitwise_and(img, mask)
        T2 = time.clock()
        cv2.imshow('Result Image', result_image)
        cv2.waitKey(0)
        print("总耗时{}s".format((T2 - T1)))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    path_dimg = r'/data/dataset/same_clothes_1/depth/depth00000332.jpg'
    path_img = r'/data/dataset/same_clothes_1/left/left00000332.jpg'
    attention_color(path_dimg, path_img)
