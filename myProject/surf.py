import cv2
import numpy as np
import stereoConfig
import img_process


if __name__ == "__main__":
    # 读取左右目相机图像
    left_image = cv2.imread('/data/ORB_SLAM_series/orb_slam3_agv_latest/P05/left/249202344.jpg', cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread('/data/ORB_SLAM_series/orb_slam3_agv_latest/P05/right/249202344.jpg', cv2.IMREAD_GRAYSCALE)
    # depth_image = cv2.imread('/home/dq/PycharmProjects/xiaomi/data/depth/0000.png', cv2.IMREAD_UNCHANGED)

    # 相机参数读取
    result = stereoConfig.readCameraCfg("../Pictures/myntresult.yaml")
    config = stereoConfig.stereoCamera(result)
    _, _, _, _, Q = img_process.getRectifyTransform(left_image.shape[1], left_image.shape[0], config)

    # 计算视差图
    disp = img_process.stereoMatchSGBM(left_image, right_image, down_scale=True, WLS_Filter=True)
    # 计算深度图
    depthMap = img_process.getDepth(disparity=disp, Q=Q, method=True)

    # 初始化Surf特征检测器
    surf = cv2.xfeatures2d_SURF.create(hessianThreshold=5000)

    # 检测特征点和特征描述符
    keypoints_left, descriptors_left = surf.detectAndCompute(left_image, None)
    keypoints_right, descriptors_right = surf.detectAndCompute(right_image, None)
    # 初始化FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE,
                        trees=5)
    search_params = dict(check=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # 使用KNN匹配特征点
    matches = flann.knnMatch(descriptors_left, descriptors_right, k=2)

    # RANSAC
    # 进行比值测试以获得良好的匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

    # 提取匹配点的坐标
    left_pts = np.float32([keypoints_left[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    right_pts = np.float32([keypoints_right[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
    # 使用RANSAC进行匹配
    F, mask = cv2.findFundamentalMat(left_pts, right_pts, cv2.FM_RANSAC,ransacReprojThreshold=1.0)
    # 提取匹配点
    matched_left_pts = left_pts[mask.ravel() == 1]
    matched_right_pts = right_pts[mask.ravel() == 1]

    # 绘制匹配点
    matches_image = cv2.drawMatches(left_image, keypoints_left, right_image, keypoints_right, good_matches, None)

    for i, (pt1, pt2) in enumerate(zip(left_pts, right_pts)):
        pt1 = (int(pt1[0]), int(pt1[1]))
        pt2 = (int(pt2[0]), int(pt2[1]))
        depth = img_process.getDepth(pt2[1]-pt1[1], Q)

        cv2.circle(left_image, pt1, 5, (255, 255, 255), -1)
        text = f"{i+1}"
        cv2.putText(left_image, text, (pt1[0], pt1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        print(text + f" depth_for = {depth}")
        print(text + f" depth_map = {depthMap[pt1[1], pt1[0]]}")

    # cv2.namedWindow('Image', 0)
    # cv2.imshow('Image', left_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("../surf08_featurePoints.png", left_image)

    # 显示匹配结果
    cv2.namedWindow('Matches', 0)
    cv2.imshow('Matches', matches_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('../surf08_matches.png', matches_image)