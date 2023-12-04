import cv2


# 打开第一个摄像头（通常是编号为0的摄像头）
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 设置视频编解码器和创建两个VideoWriter对象以保存两个视频
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用XVID编解码器，您可以根据需要更改
out1 = cv2.VideoWriter('left.avi', fourcc, 10.0, (1280, 720))  # 更改文件名和参数
out2 = cv2.VideoWriter('right.avi', fourcc, 10.0, (1280, 720))  # 更改文件名和参数

while True:
    # 读取第一个摄像头捕获的帧
    ret, frame = cap.read()

    if not ret:
        print("无法捕获帧")
        break

    # 在窗口中显示两个摄像头捕获的帧
    cv2.imshow('Frame 1', frame[:, :1280])
    cv2.imshow('Frame 2', frame[:, 1280:])

    # 将两个帧分别写入两个输出视频文件
    out1.write(frame[:, :1280])
    out2.write(frame[:, 1280:])

    # 如果按下'q'键，退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和VideoWriter对象，关闭所有窗口
cap.release()
out1.release()
out2.release()
cv2.destroyAllWindows()
