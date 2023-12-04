import cv2
import numpy as np
import img_process
import stereoConfig
import glob


if __name__ == "__main__":
    # 相机参数读取
    # result = stereoConfig.readCameraCfg("../Pictures/myntresult.yaml")
    # config = stereoConfig.stereoCamera(result)
    imgl_path_array = sorted(glob.glob("../SaveImage/left_rec/000000.png"))
    imgr_path_array = sorted(glob.glob("../SaveImage/left_rec/000001.png"))
    for i, (imgl_path, imgr_path) in enumerate(zip(imgl_path_array, imgr_path_array)):
        print(f"正在处理第{i+1}对图像，共{len(imgl_path_array)}对")
        imgl = cv2.imread(imgl_path, cv2.IMREAD_GRAYSCALE)
        imgr = cv2.imread(imgr_path, cv2.IMREAD_GRAYSCALE)

        # imgl = cv2.imread(imgl_path)
        # imgr = cv2.imread(imgr_path)
        # _, _, _, _, Q = img_process.getRectifyTransform(imgl.shape[1], imgl.shape[0], config)
        #
        # # 计算视差图
        # disp = img_process.stereoMatchSGBM(imgl, imgr, down_scale=True, WLS_Filter=True)
        # # 计算深度图
        # depthMap = img_process.getDepth(disparity=disp, Q=Q, method=False)
        # 使用orb算子进行特征提取
        orb = cv2.ORB_create(nfeatures=10000, scaleFactor=1.2, nlevels=8, edgeThreshold=15, patchSize=15)
        keypoints_left, descriptors_left = orb.detectAndCompute(imgl, None)
        keypoints_right, descriptors_right = orb.detectAndCompute(imgr, None)
        # 初始化FLANN匹配器
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,key_size=12,
                            multi_probe_level=1)
        search_params = dict(check=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        # 使用FLANN匹配器进行筛选
        matches = flann.knnMatch(descriptors_left, descriptors_right, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_matches.append(m)
        # 提取匹配点的坐标
        left_pts = np.float32([keypoints_left[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        right_pts = np.float32([keypoints_right[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
        # 使用RANSAC进行匹配
        F, mask = cv2.findFundamentalMat(left_pts, right_pts, cv2.FM_RANSAC,ransacReprojThreshold=1.0)
        # # 提取好的匹配点
        # matched_left_pts = left_pts[mask.ravel() == 1]
        # matched_right_pts = right_pts[mask.ravel() == 1]
        # # 将特征点进行保存
        # np.save(f"/home/dq/PycharmProjects/xiaomi/Pictures/Imgs_left_kp/{i:06d}.npy", matched_left_pts)
        # np.save(f"/home/dq/PycharmProjects/xiaomi/Pictures/Imgs_right_kp/{i:06d}.npy", matched_right_pts)
    # # 将亚像素点变成像素点
    # matched_left_pts = matched_left_pts.astype(np.int)
    # matched_right_pts = matched_right_pts.astype(np.int)
    # 绘制匹配点
        matches_image = cv2.drawMatches(imgl, keypoints_left, imgr, keypoints_right, good_matches, None)

    # for i, (pt1, pt2) in enumerate(zip(matched_left_pts, matched_right_pts)):
        # pt1 = tuple(pt1)
        # pt2 = tuple(pt2)
        # pt1 = (int(pt1[0]), int(pt1[1]))
        # pt2 = (int(pt2[0]), int(pt2[1]))
        # depth = img_process.getDepth(abs(pt2[0] - pt1[0]), Q)

        # cv2.circle(imgl, pt1, 5, (255, 255, 255), -1)
        # text = f"{i+1}"
        # cv2.putText(imgl, text, (pt1[0], pt1[1]-10.0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        # print(text + f" depth_get = {depth}")
        # print(text + f" depth_map = {depthMap[pt1[1], pt1[0]]}")

        # 显示匹配结果
        cv2.namedWindow('Matches', 0)
        cv2.imshow('Matches', matches_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("../orb_matches.png", matches_image)

    # # 查看特征点
    # cv2.namedWindow('Image', 0)
    # cv2.imshow('Image', imgl)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite("../orb_featurePoints_l.png", imgl)
