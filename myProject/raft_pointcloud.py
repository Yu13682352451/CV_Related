import cv2
import plotly.graph_objects as go
import numpy as np
from imageio.v3 import imread
import time

import stereoConfig
import disp2depth
import img_process

result = stereoConfig.readCameraCfg("/home/lbyj/PycharmProjects/xiaomi/Pictures/myntresult.yaml")
config = stereoConfig.stereoCamera(result)

disp = np.load("/home/lbyj/GitHubProjects/CREStereo/arr_office-rec/00000000.npy")
# disp1 = np.load("/home/lbyj/GitHubProjects/RAFT-Stereo/raft_office/00000000.npy")
img = imread("/data/OwnDataset/office_voxbloxFormat/seq-01/frame-000000.color.png")

cx0 = config.cam_matrix_left[0][2]
cx1 = config.cam_matrix_right[0][2]
cy0 = config.cam_matrix_left[1][2]
fx = config.cam_matrix_left[0][0]
fy = config.cam_matrix_left[1][1]

# # # 使用Q矩阵中的参数
# height, width = img.shape[0:2]
# # 立体校正
# map1x, map1y, map2x, map2y, Q = img_process.getRectifyTransform(height, width, config)
# fx = Q[2][3]
# fy = Q[2][3]
# cx0 = -Q[0][3]
# cy0 = -Q[1][3]

# # 使用左右目参数的平均值
# fx = (config.cam_matrix_left[0][0] + config.cam_matrix_right[0][0]) / 2
# fy = (config.cam_matrix_left[1][1] + config.cam_matrix_right[1][1]) / 2
# cx0 = config.cam_matrix_left[0][2]
# cy0 = config.cam_matrix_left[1][2]

# inverse-project
t1 = time.perf_counter()
depth = disp2depth.disp2depth(disp, config, method="CRE")
t2 = time.perf_counter()
# depth = disp2depth.disp2depth(disp, config, method="RAFT")
depth = cv2.imread("/home/lbyj/PycharmProjects/xiaomi/depth.png", cv2.IMREAD_UNCHANGED)
H, W = depth.shape
xx, yy = np.meshgrid(np.arange(W), np.arange(H))
points_grid = np.stack(((xx-cx0)/fx, (yy-cy0)/fy, np.ones_like(xx)), axis=0) * depth

mask = np.ones((H, W), dtype=bool)

# Remove flying points
mask[1:][np.abs(depth[1:] - depth[:-1]) > 1000] = False
mask[:, 1:][np.abs(depth[:, 1:] - depth[:, :-1]) > 1000] = False

points = points_grid.transpose(1, 2, 0)[mask]
colors = img[mask].astype(np.float64) / 255


NUM_POINTS_TO_DRAW = 100000

subset = np.random.choice(points.shape[0], size=(NUM_POINTS_TO_DRAW,), replace=True)
points_subset = points[subset]
colors_subset = colors[subset]
t3 = time.perf_counter()

print(f"视差计算深度耗时{t2 - t1}")
print(f"深度转换为点云耗时{t3 - t2}")
print(f"总耗时{t3 - t1}")

print("""
Controls:
---------
Zoom:      Scroll Wheel
Translate: Right-Click + Drag
Rotate:    Left-Click + Drag
""")

x, y, z = points_subset.T

fig = go.Figure(
    data=[
        go.Scatter3d(
            x=x, y=z, z=-y, # flipped to make visualization nicer
            mode='markers',
            marker=dict(size=1, color=colors_subset)
        )
    ],
    layout=dict(
        scene=dict(
            xaxis=dict(visible=True),
            yaxis=dict(visible=True),
            zaxis=dict(visible=True),
        )
    )
)
fig.show()
