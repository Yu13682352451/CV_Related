# -*- coding:utf-8 -*-
# @FileName :stereoConfig.py
# @Time     :2023/4/7 16:17
# @Author   :YJ
import numpy as np
import yaml


# 读取双目相机参数
class stereoCamera(object):
    def __init__(self, result):
        self.result = result
        # 左相机内参
        self.cam_matrix_left = np.array(self.result['K1'])
        # 右相机内参
        self.cam_matrix_right = np.array(self.result['K2'])

        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([self.result['D1']])
        self.distortion_r = np.array([self.result['D2']])

        # 旋转矩阵
        self.R = np.array(self.result['rot'])

        # 平移矩阵
        self.T = np.array(self.result['trans'])

        # 主点列坐标的差
        self.doffs = 0

        # 指示上述内外参是否为经过立体校正后的结果
        self.isRectified = False


# 首先用该函数读取保存了相机参数的yaml文件，然后将result作为参数传递给stereoCamera类
def readCameraCfg(yamlpath):
    with open(yamlpath, 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    return result


# 双目相机的标定
# class StereoCalibration:
#     def __init__(self, filepath):
#
#         self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#
#         # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
#         self.objp = np.zeros((9 * 6, 3), np.float32)
#         self.objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
#
#         # Arrays to store object points and image points from all the images.
#         self.objpoints = []  # 3d point in real world space
#         self.imgpoints_l = []  # 2d points in image plane.
#         self.imgpoints_r = []  # 2d points in image plane.
#
#         self.filepath = filepath
#         self.read_images()
#
#     def read_images(self):
#         images_right = glob(self.filepath + '/right/*.jpg')
#         images_left = glob(self.filepath + '/left/*.jpg')
#
#         for i, fname in enumerate(images_right):
#             img_l = cv2.imread(images_left[i])
#             img_r = cv2.imread(images_right[i])
#
#             gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
#             gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
#
#             # Find the chess board corners
#             ret_l, corners_l = cv2.findChessboardCorners(gray_l, (9, 6), None)
#             ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 6), None)
#
#             # If found, add object points, image points (after refining them)
#
#             if ret_l and ret_r:
#
#                 self.objpoints.append(self.objp)
#
#                 rt = cv2.cornerSubPix(gray_l, corners_l, (5, 5), (-1, -1), self.criteria)
#                 self.imgpoints_l.append(corners_l)
#
#                 rt = cv2.cornerSubPix(gray_r, corners_r, (5, 5), (-1, -1), self.criteria)
#                 self.imgpoints_r.append(corners_r)
#             else:
#                 print('Couldn\'t be found')
#
#             img_shape = gray_l.shape[::-1]
#
#         rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(self.objpoints, self.imgpoints_l, img_shape, None,
#                                                                      None)
#         rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(self.objpoints, self.imgpoints_r, img_shape, None,
#                                                                      None)
#
#         self.camera_model = self.stereo_calibrate(img_shape)
#
#     def stereo_calibrate(self, dims):
#
#         h, w = dims
#
#         flags = 0
#         # 如果该标志被设置，那么就会固定输入的cameraMatrix和distCoeffs不变，只求解R,T,E,F.
#         flags |= cv2.CALIB_FIX_INTRINSIC
#         # 根据用户提供的cameraMatrix和distCoeffs为初始值开始迭代
#         flags |= cv2.CALIB_USE_INTRINSIC_GUESS
#         # 迭代过程中不会改变焦距
#         flags |= cv2.CALIB_FIX_FOCAL_LENGTH
#         # 切向畸变保持为零
#         flags |= cv2.CALIB_ZERO_TANGENT_DIST
#
#         stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
#         ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
#             self.objpoints,
#             self.imgpoints_l,
#             self.imgpoints_r,
#             self.M1,
#             self.d1,
#             self.M2,
#             self.d2,
#             dims,
#             criteria=stereocalib_criteria,
#             flags=flags
#         )
#
#         camera_model = dict([
#             ('size', (h, w)),
#             ('K1', M1),
#             ('D1', d1),
#             ('K2', M2),
#             ('D2', d2),
#             # ('rvecs1', self.r1),
#             # ('rvecs2', self.r2),
#             ('T', T),
#             ('R', R),
#             # ('E', E),
#             # ('F', F),
#         ])
#
#         return camera_model
#
#     def rectify(self, camera_model):
#
#         M1, d1, M2, d2, R, T = camera_model.get('M1'), camera_model.get('d1'), camera_model.get('M2'), camera_model.get(
#             'd2'), camera_model.get('R'), camera_model.get('T')
#
#         # 双目矫正 alpha=-1, 0, 0.9
#         R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(M1, d1, M2, d2, camera_model.get('dim'), R, T,
#                                                                           alpha=-1)
#         print('stereo rectify done...')
#
#         # 得到映射变换
#         stereo_left_mapx, stereo_left_mapy = cv2.initUndistortRectifyMap(M1, d1, R1, P1, camera_model.get('dim'),
#                                                                          cv2.CV_32FC1)
#         stereo_right_mapx, stereo_right_mapy = cv2.initUndistortRectifyMap(M2, d2, R2, P2, camera_model.get('dim'),
#                                                                            cv2.CV_32FC1)
#         print('initUndistortRectifyMap done...')
#
#         rectify_model = dict([
#             ('R1', R1),
#             ('R2', R2),
#             ('P1', P1),
#             ('P2', P2),
#             ('Q', Q),
#             ('stereo_left_mapx', stereo_left_mapx),
#             ('stereo_left_mapy', stereo_left_mapy),
#             ('stereo_right_mapx', stereo_right_mapx),
#             ('stereo_right_mapy', stereo_right_mapy)
#         ])
#
#         return rectify_model
