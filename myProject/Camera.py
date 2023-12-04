# -*- coding:utf-8 -*-
# @FileName :Camera.py
# @Time     :2023/4/6 22:12
# @Author   :YJ
import cv2
import yaml


class Camera:
    def __init__(self):
        self.cap = None
        self.i = 0

    # 打开摄像头，需要输入的参数为想要打开的摄像头的序号，一般来说电脑自带的摄像头为零，用usb接口接入的摄像头依次为1，2，3
    # 其实应该把分辨率也设为输入参数，这样子普适性更强
    def open(self, numCamera):
        self.cap = cv2.VideoCapture(numCamera)
        # 小觅相机的分辨率设为了1280*720，由于是双目相机，其拍到的照片并排放置，因此设置宽度为1280*2=2560
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 在电脑上展示摄像头的当前帧
    def show(self):
        ret, frame = self.cap.read()
        cv2.namedWindow('left-frame', 0)
        cv2.namedWindow('right-frame', 0)
        right_frame = frame[:, :1280, :]
        left_frame = frame[:, 1280:, :]
        cv2.imshow('left-frame', left_frame)
        cv2.imshow('right-frame', right_frame)

    # 保存摄像头的当前帧
    def shot(self):
        filename = "../Pictures/"
        str_left = "left1/"+str(self.i) + "_left.jpg"
        str_right = "right1/"+str(self.i) + "_right.jpg"
        self.i += 1
        print("Successfully shot!")

        ret, frame = self.cap.read()
        left_frame = frame[:, :1280, :]
        right_frame = frame[:, 1280:, :]
        cv2.imwrite(filename + str_left, left_frame)
        cv2.imwrite(filename + str_right, right_frame)

    # 释放资源，常规操作
    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()

def readCameraCfg(yamlpath):
    with open(yamlpath, 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    return result


if __name__ == "__main__":
    cam = Camera()
    cam.open(0)
    while True:
        cam.show()
        k = cv2.waitKey(1)
        if k == ord('s'):
            cam.shot()
        elif k == ord('q'):
            break
    cam.close()
