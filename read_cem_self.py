import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.io import loadmat
from scipy.ndimage import label
import os
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d

# 读取背景图像
data_img_path = "/home/fastdvdnet-master-unsupervised/fastdvdnet-master/test_seq/SUM_cem_1010.tif"
img = cv2.imread(data_img_path, cv2.IMREAD_UNCHANGED)
img = img.astype(np.float32) / (512.0*76)  # 归一化
img1 = (img-img.min())/(img.max()-img.min())
img1 = img1[0:1058,65:1535]
H, W = img1.shape
save_dir = "/home/fastdvdnet-master-unsupervised/fastdvdnet-master/cem_out/Self_sig"
# ROI 区域坐标（中间区域）
#y1, y2, x1, x2 = 256, 256 + 160, 756, 756 + 160
y1, y2, x1, x2 = 256, 256 + 160, 960, 960 + 160

th = 0.2
binary = (img1 > th).astype(np.uint8)
mask_roi = binary[y1-50:y2+50, x1-50:x2+50]
kernel = np.ones((3, 3), np.uint8)

# --- ① 形态学梯度 ---
edge_roi = cv2.morphologyEx(
    mask_roi,
    cv2.MORPH_GRADIENT,
    kernel
)

# --- ② 只保留最大连通边缘 ---
edge_bin = (edge_roi > 0).astype(np.uint8)
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    edge_bin, connectivity=8
)
if num_labels > 1:
    max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    edge_roi = (labels == max_label).astype(np.uint8)

# =========================
# ★★★ ③ 从 ROI 边界 → 拟合倾斜直线并延长 ★★★
# =========================

edge_bool = edge_roi.astype(bool)

# ROI 内边界点坐标
ys_roi, xs_roi = np.where(edge_bool)

# ROI → 全图坐标
y0 = y1 - 50
x0 = x1 - 50
ys = ys_roi + y0
xs = xs_roi + x0

# --- 拟合直线：x = a*y + b ---
# （用 y 做自变量更稳定，适合“近似竖直”的线）
a, b = np.polyfit(ys, xs, 1)

# --- 在整张图上延长 ---
H, W = img1.shape
edge_full = np.zeros_like(binary, dtype=np.uint8)

for y in range(H):
    x = int(a * y + b)
    if 0 <= x < W:
        edge_full[y, x] = 1

plt.figure(figsize=(6, 6))
plt.imshow(img1, cmap='gray')
# ROI 中真实边界点（蓝）
plt.scatter(xs, ys, s=3, c='cyan', label='Detected edge (ROI)')
# 拟合并延长后的直线（红）
ys_line = np.arange(H)
xs_line = a * ys_line + b
plt.plot(xs_line, ys_line, 'r', linewidth=2, label='Extended fitted line')
plt.legend()
plt.axis('off')
# plt.show()
# ############################# noise_U #########################################
# mat_path_U = "/mnt/home/yuhuizhen/project/URetinex-Net-main/fastDVDnet_final_evaluate_map_U_HalfSigArea.mat"
# data_U = loadmat(mat_path_U)['data']  # (5,34,50)
# data_U = np.repeat(data_U, repeats=5, axis=0)
# # a = data_U[0]
# # b = data_U[1]
# # c = data_U[2]
# # d = data_U[3]
# # e = data_U[4]
# thresholds = [82, 82, 82, 82, 82]#[37, 38, 33.8, 36, 39]
# data_thresholded_U = np.zeros_like(data_U)
# for i in range(5):
#     data_thresholded_U[i] = np.where(data_U[i] < thresholds[i], data_U[i], 0)
# nonzero_ratios_U = []
# nonzero_ratios_left_U = []
# nonzero_ratios_right_U = []
# m = [0,1,2,3,4]
# for i in m:
#     heatmap_U = data_thresholded_U[i]
#     binary_mask_U = heatmap_U > 0
#     labeled_U, num_features_U = label(binary_mask_U)
#     if num_features_U > 0:
#         sizes_U = np.bincount(labeled_U.ravel())
#         sizes_U[0] = 0
#         largest_label_U = sizes_U.argmax()
#         mask_largest_U = (labeled_U == largest_label_U)
#         heatmap_U = heatmap_U * mask_largest_U
#     else:
#         print(f"第{i+1}张无连通区域")
#         heatmap_U[:] = 0
#
#     # ② 放大到原图大小
#     heatmap_resized_U = cv2.resize(heatmap_U,(img.shape[1], img.shape[0]),interpolation=cv2.INTER_CUBIC)
#     eps = 1  # 或者 0.005 / 0.02，按你数据调
#     heatmap_masked_U = np.ma.masked_where(heatmap_resized_U <= eps,heatmap_resized_U)
#     heatmap_masked_U = heatmap_masked_U[0:1058,65:1535]
#     heatmap_masked_U = (heatmap_masked_U-heatmap_masked_U.min())/(heatmap_masked_U.max()-heatmap_masked_U.min())
#
#     ##图二原图 + 热图叠加
#     H, W = img1.shape[:2]
#     white_red = LinearSegmentedColormap.from_list('white_red', ['white', 'red'])
#     fig2 = plt.figure(frameon=False)
#     ax2 = plt.Axes(fig2, [0., 0., 1., 1.])
#     fig2.add_axes(ax2)
#     ax2.imshow(img1,cmap='gray',extent=[0, W, H, 0] )  # ← 关键)
#     ax2.imshow(heatmap_masked_U,cmap=white_red,alpha=0.5,extent=[0, W, H, 0])   # ← 必须一致)
#     # ROI 区域标注
#     x0 = 65
#     y0 = 0
#     rect = plt.Rectangle((x1 - x0, y1 - y0), x2 - x1, y2 - y1,
#                          linewidth=0, edgecolor='none',
#                          facecolor='lime', alpha=0.5)
#     ax2.add_patch(rect)
#     ax2.text(x1, y1 - 10, 'ROI', color='lime', fontsize=12, weight='bold')
#     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
#     plt.show()
#     ax2.set_xlim(0, W)
#     ax2.set_ylim(H, 0)
#     ax2.axis('off')
#     fig2.tight_layout(pad=0)
#     fig2.savefig(f"{save_dir}/frame_{i + 1}_cem_overlay_u.tiff", dpi=600, bbox_inches='tight', pad_inches=0)
#     plt.show()
#     plt.close(fig2)
#
#     # ③ 统计非零和零数量
#     nonzero_count = np.count_nonzero(heatmap_masked_U)
#     zero_count = heatmap_masked_U.size - nonzero_count
#     nonzero_ratios_U.append(100 * (nonzero_count / heatmap_masked_U.size))
#     print(
#         f"Frame {i + 1}: 非零占比={nonzero_count / heatmap_masked_U.size:.2%}, 零值占比={zero_count / heatmap_masked_U.size:.2%}")
#     # 统计边界左右两侧非零值数目
#     # valid_mask = ~heatmap_masked.mask
#     left_mask = np.zeros((H, W), dtype=bool)
#     right_mask = np.zeros((H, W), dtype=bool)
#     for y in range(H):
#         x_line = int(a * y + b)
#         if 0 <= x_line < W:
#             left_mask[y, :x_line] = True
#             right_mask[y, x_line:] = True
#     nonzero_left = np.count_nonzero((heatmap_masked_U.data != 0) & left_mask)
#     nonzero_right = np.count_nonzero((heatmap_masked_U.data != 0) & right_mask)
#     total_left = np.sum(left_mask)
#     total_right = np.sum(right_mask)
#     nonzero_ratios_left_U.append(100 * (nonzero_left / heatmap_masked_U.size))
#     nonzero_ratios_right_U.append(100 * (nonzero_right / heatmap_masked_U.size))
#     print(
#         f"左侧非零数目: {nonzero_left} / {heatmap_masked_U.size} "
#         f"({nonzero_left / heatmap_masked_U.size:.2%})"
#     )
#     print(
#         f"右侧非零数目: {nonzero_right} / {heatmap_masked_U.size} "
#         f"({nonzero_right / heatmap_masked_U.size:.2%})"
#     )
#
#     # =====================
#     # ④ 绘制红白饼状图（透明背景）
#     # =====================
#     fig_pie, ax_pie = plt.subplots(figsize=(4, 4), facecolor='none')
#     ax_pie.pie(
#         [nonzero_count, zero_count],
#         colors=['red', 'white'],
#         labels=None,
#         autopct=None,
#         startangle=90,
#         wedgeprops={'edgecolor': 'none'}
#     )
#     ax_pie.axis('equal')
#     ax_pie.set_facecolor('none')
#     fig_pie.patch.set_alpha(0.0)
#     plt.show()
#     # 保存
#     fig_pie.savefig(os.path.join(save_dir, f"pie_frame_u_{i + 1}.tiff"), dpi=600, bbox_inches='tight', pad_inches=0,
#                     transparent=True)
#     plt.close(fig_pie)
############################# noise_N #########################################
mat_path_N = "/home/fastdvdnet-master-unsupervised/fastdvdnet-master/fastDVDnet_final_evaluate_map_HalfSigArea_nlcl_1.mat"
data_N = loadmat(mat_path_N)['data']  # (5,34,50)
# a = data_N[0]
# b = data_N[1]
c = data_N[2]
# d = data_N[3]
e = data_N[4]
thresholds = [42, 39, 40, 40, 50]#[37, 38, 33.8, 36, 39]
data_thresholded_N = np.zeros_like(data_N)
for i in range(5):
    data_thresholded_N[i] = np.where(data_N[i] < thresholds[i], data_N[i], 0)
nonzero_ratios_N = []
nonzero_ratios_left_N = []
nonzero_ratios_right_N = []
m = [0,1,2,3,4]
from scipy.ndimage import binary_dilation
y1, y2, x1, x2 = 256, 256 + 160, 960, 960 + 160
dilate_iter = 2
visualize = True

for i in m:
    heatmap_N = data_thresholded_N[i]

    # ===== 坐标映射 =====
    scale_y = heatmap_N.shape[0] / img.shape[0]
    scale_x = heatmap_N.shape[1] / img.shape[1]
    hy1 = max(int(y1 * scale_y), 0)
    hy2 = min(int(y2 * scale_y), heatmap_N.shape[0])
    hx1 = max(int(x1 * scale_x), 0)
    hx2 = min(int(x2 * scale_x), heatmap_N.shape[1])

    # ===== ROI 提取 =====
    roi = heatmap_N[hy1:hy2, hx1:hx2]

    # ===== 在原 heatmap 尺寸创建全局 mask =====
    full_mask = np.zeros_like(heatmap_N, dtype=bool)
    full_mask[hy1:hy2, hx1:hx2] = roi > 0

    # ===== 膨胀 =====
    structure = np.ones((2 * dilate_iter + 1, 2 * dilate_iter + 1), dtype=bool)
    full_mask_dilated = binary_dilation(full_mask, structure=structure)

    # ===== 生成膨胀后热图 =====
    heatmap_processed = np.zeros_like(heatmap_N, dtype=heatmap_N.dtype)
    # 只把膨胀后 mask 内的灰度值取自原阈值分割后的数据
    heatmap_processed[full_mask_dilated] = heatmap_N[full_mask_dilated]

    # ===== 写回 heatmap =====
    heatmap_N[:] = heatmap_processed


    # ② 放大到原图大小
    heatmap_resized_N = cv2.resize(heatmap_N,(img.shape[1], img.shape[0]),interpolation=cv2.INTER_CUBIC)
    eps = 1  # 或者 0.005 / 0.02，按你数据调
    heatmap_masked_N = np.ma.masked_where(heatmap_resized_N <= eps,heatmap_resized_N)
    heatmap_masked_N = heatmap_masked_N[0:1058,65:1535]
    heatmap_masked_N = (heatmap_masked_N-heatmap_masked_N.min())/(heatmap_masked_N.max()-heatmap_masked_N.min())

    ##图二原图 + 热图叠加
    H, W = img1.shape[:2]
    white_red = LinearSegmentedColormap.from_list('white_red', ['white', 'red'])
    fig2 = plt.figure(frameon=False)
    ax2 = plt.Axes(fig2, [0., 0., 1., 1.])
    fig2.add_axes(ax2)
    ax2.imshow(img1,cmap='gray',extent=[0, W, H, 0] )  # ← 关键)
    ax2.imshow(heatmap_masked_N,cmap=white_red,alpha=0.5,extent=[0, W, H, 0])   # ← 必须一致)
    # ROI 区域标注
    x0 = 65
    y0 = 0
    rect = plt.Rectangle((x1 - x0, y1 - y0), x2 - x1, y2 - y1,
                         linewidth=0, edgecolor='none',
                         facecolor='lime', alpha=0.5)
    ax2.add_patch(rect)
    ax2.text(x1, y1 - 10, 'ROI', color='lime', fontsize=12, weight='bold')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax2.set_xlim(0, W)
    ax2.set_ylim(H, 0)
    ax2.axis('off')
    fig2.tight_layout(pad=0)
    fig2.savefig(f"{save_dir}/frame_{i + 1}_cem_overlay_nlcl.tiff", dpi=600, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close(fig2)

    # ③ 统计非零和零数量
    nonzero_count = np.count_nonzero(heatmap_masked_N)
    zero_count = heatmap_masked_N.size - nonzero_count
    nonzero_ratios_N.append(100 * (nonzero_count / heatmap_masked_N.size))
    print(
        f"Frame {i + 1}: 非零占比={nonzero_count / heatmap_masked_N.size:.2%}, 零值占比={zero_count / heatmap_masked_N.size:.2%}")

    # 统计边界左右两侧非零值数目
    #valid_mask = ~heatmap_masked.mask
    left_mask = np.zeros((H, W), dtype=bool)
    right_mask = np.zeros((H, W), dtype=bool)
    for y in range(H):
        x_line = int(a * y + b)
        if 0 <= x_line < W:
            left_mask[y, :x_line] = True
            right_mask[y, x_line:] = True
    nonzero_left = np.count_nonzero((heatmap_masked_N.data != 0) & left_mask)
    nonzero_right = np.count_nonzero((heatmap_masked_N.data != 0) & right_mask)
    total_left = np.sum(left_mask)
    total_right = np.sum(right_mask)
    nonzero_ratios_left_N.append(100 * (nonzero_left / heatmap_masked_N.size))
    nonzero_ratios_right_N.append(100 * (nonzero_right / heatmap_masked_N.size))
    print(
        f"左侧非零数目: {nonzero_left} / {heatmap_masked_N.size} "
        f"({nonzero_left / heatmap_masked_N.size:.2%})"
    )
    print(
        f"右侧非零数目: {nonzero_right} / {heatmap_masked_N.size} "
        f"({nonzero_right / heatmap_masked_N.size:.2%})"
    )
    # =====================
    # ④ 绘制红白饼状图（透明背景）
    # =====================
    fig_pie, ax_pie = plt.subplots(figsize=(4, 4), facecolor='none')
    ax_pie.pie(
        [nonzero_count, zero_count],
        colors=['red', 'white'],
        labels=None,
        autopct=None,
        startangle=90,
        wedgeprops={'edgecolor': 'none'}
    )
    ax_pie.axis('equal')
    ax_pie.set_facecolor('none')
    fig_pie.patch.set_alpha(0.0)
    # 保存
    fig_pie.savefig(os.path.join(save_dir, f"pie_frame_nlcl_{i + 1}.tiff"), dpi=600, bbox_inches='tight', pad_inches=0,
                    transparent=True)
    plt.show()
    plt.close(fig_pie)

############################# noise_F #########################################
mat_path_F = "/home/fastdvdnet-master-unsupervised/fastdvdnet-master/fastDVDnet_final_evaluate_map_HalfSigArea_fastdvd_1.mat"
data_F = loadmat(mat_path_F)['data']  # (5,34,50)
# a = data_F[0]
# b = data_F[1]
c = data_F[2]
# d = data_F[3]
e = data_F[4]
thresholds = [42, 39, 37, 38, 41]#[37, 38, 33.8, 36, 39]
data_thresholded_F = np.zeros_like(data_F)
for i in range(5):
    data_thresholded_F[i] = np.where(data_F[i] < thresholds[i], data_F[i], 0)
nonzero_ratios_F = []
nonzero_ratios_left_F = []
nonzero_ratios_right_F = []
m = [0,1,2,3,4]
for i in m:
    heatmap_F = data_thresholded_F[i]
    binary_mask_F = heatmap_F > 0
    labeled_F, num_features_F = label(binary_mask_F)
    if num_features_F > 0:
        sizes_F = np.bincount(labeled_F.ravel())
        sizes_F[0] = 0
        largest_label_F = sizes_F.argmax()
        mask_largest_F = (labeled_F == largest_label_F)
        heatmap_F = heatmap_F * mask_largest_F
    else:
        print(f"第{i+1}张无连通区域")
        heatmap_F[:] = 0

    # ② 放大到原图大小
    heatmap_resized_F = cv2.resize(heatmap_F,(img.shape[1], img.shape[0]),interpolation=cv2.INTER_CUBIC)
    eps = 1  # 或者 0.005 / 0.02，按你数据调
    heatmap_masked_F = np.ma.masked_where(heatmap_resized_F <= eps,heatmap_resized_F)
    heatmap_masked_F = heatmap_masked_F[0:1058,65:1535]
    heatmap_masked_F = (heatmap_masked_F-heatmap_masked_F.min())/(heatmap_masked_F.max()-heatmap_masked_F.min())

    ##图二原图 + 热图叠加
    H, W = img1.shape[:2]
    white_red = LinearSegmentedColormap.from_list('white_red', ['white', 'red'])
    fig2 = plt.figure(frameon=False)
    ax2 = plt.Axes(fig2, [0., 0., 1., 1.])
    fig2.add_axes(ax2)
    ax2.imshow(img1,cmap='gray',extent=[0, W, H, 0] )  # ← 关键)
    ax2.imshow(heatmap_masked_F,cmap=white_red,alpha=0.5,extent=[0, W, H, 0])   # ← 必须一致)
    # ROI 区域标注
    x0 = 65
    y0 = 0
    rect = plt.Rectangle((x1 - x0, y1 - y0), x2 - x1, y2 - y1,
                         linewidth=0, edgecolor='none',
                         facecolor='lime', alpha=0.5)
    ax2.add_patch(rect)
    ax2.text(x1, y1 - 10, 'ROI', color='lime', fontsize=12, weight='bold')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax2.set_xlim(0, W)
    ax2.set_ylim(H, 0)
    ax2.axis('off')
    fig2.tight_layout(pad=0)
    fig2.savefig(f"{save_dir}/frame_{i + 1}_cem_overlay_fast.tiff", dpi=600, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close(fig2)

    # ③ 统计非零和零数量
    nonzero_count = np.count_nonzero(heatmap_masked_F)
    zero_count = heatmap_masked_F.size - nonzero_count
    nonzero_ratios_F.append(100 * (nonzero_count / heatmap_masked_F.size))
    print(
        f"Frame {i + 1}: 非零占比={nonzero_count / heatmap_masked_F.size:.2%}, 零值占比={zero_count / heatmap_masked_F.size:.2%}")

    # 统计边界左右两侧非零值数目
    #valid_mask = ~heatmap_masked.mask
    left_mask = np.zeros((H, W), dtype=bool)
    right_mask = np.zeros((H, W), dtype=bool)
    for y in range(H):
        x_line = int(a * y + b)
        if 0 <= x_line < W:
            left_mask[y, :x_line] = True
            right_mask[y, x_line:] = True
    nonzero_left = np.count_nonzero((heatmap_masked_F.data != 0) & left_mask)
    nonzero_right = np.count_nonzero((heatmap_masked_F.data != 0) & right_mask)
    total_left = np.sum(left_mask)
    total_right = np.sum(right_mask)
    nonzero_ratios_left_F.append(100 * (nonzero_left / heatmap_masked_F.size))
    nonzero_ratios_right_F.append(100 * (nonzero_right / heatmap_masked_F.size))
    print(
        f"左侧非零数目: {nonzero_left} / {heatmap_masked_F.size} "
        f"({nonzero_left / heatmap_masked_F.size:.2%})"
    )
    print(
        f"右侧非零数目: {nonzero_right} / {heatmap_masked_F.size} "
        f"({nonzero_right / heatmap_masked_F.size:.2%})"
    )
    # =====================
    # ④ 绘制红白饼状图（透明背景）
    # =====================
    fig_pie, ax_pie = plt.subplots(figsize=(4, 4), facecolor='none')
    ax_pie.pie(
        [nonzero_count, zero_count],
        colors=['red', 'white'],
        labels=None,
        autopct=None,
        startangle=90,
        wedgeprops={'edgecolor': 'none'}
    )
    ax_pie.axis('equal')
    ax_pie.set_facecolor('none')
    fig_pie.patch.set_alpha(0.0)
    # 保存
    fig_pie.savefig(os.path.join(save_dir, f"pie_frame_fast_{i + 1}.tiff"), dpi=600, bbox_inches='tight', pad_inches=0,
                    transparent=True)
    plt.show()
    plt.close(fig_pie)
############################# proposed #########################################
mat_path = "/home/fastdvdnet-master-unsupervised/fastdvdnet-master/fastDVDnet_final_evaluate_map_HalfSigArea_1.mat"
data = loadmat(mat_path)['data']  # (5,34,50)
# a = data[0]
# b = data[1]
# c = data[2]
# d = data[3]
# e = data[4]
#thresholds = [55, 43, 39, 43, 60]
thresholds = [55, 44, 41.3, 45, 62] #half_sigArea
#thresholds = [100, 100, 100, 100, 100] #ALL
data_thresholded = np.zeros_like(data)
for i in range(5):
    data_thresholded[i] = np.where(data[i] < thresholds[i], data[i], 0)

nonzero_ratios = []
nonzero_ratios_left = []
nonzero_ratios_right = []
m = [0,1,2,3,4]
#ours
for i in m:
    heatmap = data_thresholded[i]
    binary_mask = heatmap > 0
    labeled, num_features = label(binary_mask)
    if num_features > 0:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        largest_label = sizes.argmax()
        mask_largest = (labeled == largest_label)
        heatmap = heatmap * mask_largest
    else:
        print(f"第{i+1}张无连通区域")
        heatmap[:] = 0

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    eps = 1  # 或者 0.005 / 0.02，按你数据调
    heatmap_masked = np.ma.masked_where(heatmap_resized <= eps, heatmap_resized)
    heatmap_masked = heatmap_masked[0:1058,65:1535]
    heatmap_masked = (heatmap_masked-heatmap_masked.min())/(heatmap_masked.max()-heatmap_masked.min())
    ##图二原图 + 热图叠加
    H, W = img1.shape[:2]
    white_red = LinearSegmentedColormap.from_list('white_red', ['white', 'red'])
    # fig2, ax2 = plt.subplots(figsize=(8, 6))
    fig2 = plt.figure(frameon=False)
    ax2 = plt.Axes(fig2, [0., 0., 1., 1.])
    fig2.add_axes(ax2)
    ax2.imshow(img1, cmap='gray', extent=[0, W, H, 0])  # ← 关键)
    ax2.imshow(heatmap_masked, cmap=white_red, alpha=0.5, extent=[0, W, H, 0])  # ← 必须一致)
    # ROI 区域标注
    x0 = 65
    y0 = 0
    rect = plt.Rectangle((x1-x0, y1-y0), x2 - x1, y2 - y1,
                         linewidth=0, edgecolor='none',
                         facecolor='lime', alpha=0.5)
    ax2.add_patch(rect)
    ax2.text(x1, y1 - 10, 'ROI', color='lime', fontsize=12, weight='bold')
    # ====== 额外画倾斜边界线 ======
    # y_vals = np.arange(0, H)
    # x_vals = a * y_vals + b
    # valid = (x_vals >= 0) & (x_vals < W)
    # ax2.plot(
    #     x_vals[valid],
    #     y_vals[valid],
    #     color='cyan',
    #     linewidth=2.5
    # )
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax2.set_xlim(0, W)
    ax2.set_ylim(H, 0)
    ax2.axis('off')
    fig2.tight_layout(pad=0)
    fig2.savefig(
        f"{save_dir}/frame_{i + 1}_cem_overlay_ours.tiff",
        dpi=600,
        bbox_inches='tight',
        pad_inches=0
    )
    plt.show()
    plt.close(fig2)

    # ③ 统计非零和零数量
    nonzero_count = np.count_nonzero(heatmap_masked)
    zero_count = heatmap_masked.size - nonzero_count
    nonzero_ratios.append(100 * (nonzero_count / heatmap_masked.size))
    print(
        f"Frame {i + 1}: 非零占比={nonzero_count / heatmap_masked.size:.2%}, 零值占比={zero_count / heatmap_masked.size:.2%}")
    # 统计边界左右两侧非零值数目
    #valid_mask = ~heatmap_masked.mask
    left_mask = np.zeros((H, W), dtype=bool)
    right_mask = np.zeros((H, W), dtype=bool)

    for y in range(H):
        x_line = int(a * y + b)
        if 0 <= x_line < W:
            left_mask[y, :x_line] = True
            right_mask[y, x_line:] = True
    nonzero_left = np.count_nonzero((heatmap_masked.data != 0) & left_mask)
    nonzero_right = np.count_nonzero((heatmap_masked.data != 0) & right_mask)
    total_left = np.sum(left_mask)
    total_right = np.sum(right_mask)
    nonzero_ratios_left.append(100 * (nonzero_left / heatmap_masked.size))
    nonzero_ratios_right.append(100 * (nonzero_right / heatmap_masked.size))
    print(
        f"左侧非零数目: {nonzero_left} / {heatmap_masked.size} "
        f"({nonzero_left / heatmap_masked.size:.2%})"
    )
    print(
        f"右侧非零数目: {nonzero_right} / {heatmap_masked.size} "
        f"({nonzero_right / heatmap_masked.size:.2%})"
    )
    # plt.figure(figsize=(6, 6))
    # plt.imshow(img1, cmap='gray')
    # ys = np.arange(H)
    # xs = a * ys + b
    # plt.plot(xs, ys, 'r', linewidth=2)
    # plt.axis('off')
    # plt.show()
    # =====================
    # ④ 绘制红白饼状图（透明背景）
    # =====================
    fig_pie, ax_pie = plt.subplots(figsize=(4, 4), facecolor='none')
    ax_pie.pie(
        [nonzero_count, zero_count],
        colors=['red', 'white'],
        labels=None,
        autopct=None,
        startangle=90,
        wedgeprops={'edgecolor': 'none'}
    )
    ax_pie.axis('equal')
    ax_pie.set_facecolor('none')
    fig_pie.patch.set_alpha(0.0)
    # 保存
    fig_pie.savefig(os.path.join(save_dir, f"pie_frame_ours{i + 1}.tiff"), dpi=600, bbox_inches='tight', pad_inches=0,
                    transparent=True)
    plt.show()
    plt.close(fig_pie)






time_labels = [
    r'$\mathit{t}$-2',
    r'$\mathit{t}$-1',
    r'$\mathit{t}$',
    r'$\mathit{t}$+1',
    r'$\mathit{t}$+2'
]
nonzero_ratios_left    = np.array(nonzero_ratios_left )
nonzero_ratios_left_F  = np.array(nonzero_ratios_left_F)
nonzero_ratios_left_U  = np.array(nonzero_ratios_left_N)
nonzero_ratios_right    = np.array(nonzero_ratios_right)
nonzero_ratios_right_F  = np.array(nonzero_ratios_right_F)
nonzero_ratios_right_U  = np.array(nonzero_ratios_right_N)
x = np.arange(len(time_labels))
width = 0.2

from matplotlib import rcParams
# ===== 全局字体设置 =====
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = [
    'Times New Roman',
    'Nimbus Roman',
    'Liberation Serif',
    'DejaVu Serif'
]
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(7, 5))

# 第一组（蓝色）
plt.bar(
    x - width,
    nonzero_ratios_left,
    width=width,
    color='lightblue',
    label='Proposed Left'
)
plt.bar(
    x - width,
    nonzero_ratios_right,
    width=width,
    bottom=nonzero_ratios_left,
    color='blue',
    label='Proposed Right'
)

# 第二组（绿色）
plt.bar(
    x,
    nonzero_ratios_left_F,
    width=width,
    color='lightgreen',
    label='FastDVDnet Left'
)
plt.bar(
    x,
    nonzero_ratios_right_F,
    width=width,
    bottom=nonzero_ratios_left_F,
    color='green',
    label='FastDVDnet Right'
)

# 第三组（红色）
plt.bar(
    x + width,
    nonzero_ratios_left_N,
    width=width,
    color='pink',
    label='URetinex-Net Left'
)
plt.bar(
    x + width,
    nonzero_ratios_right_N,
    width=width,
    bottom=nonzero_ratios_left_N,
    color='red',
    label='NLCL'
)

# ===== 坐标轴 =====
plt.xticks(x, time_labels, fontsize=28)
plt.yticks([0, 2, 4, 6, 8, 10, 12], fontsize=28)
plt.ylabel('Utilization (%)', fontsize=28)
plt.ylim(0, 12)

# ===== 刻度线长度和宽度 =====
plt.tick_params(
    axis='both',
    which='major',
    length=6,
    width=1.2
)
# ===== 图例 =====
#plt.legend(fontsize=24, frameon=False)

plt.tight_layout()

# ===== 保存 =====
fig_path = os.path.join(save_dir, "all_stacked.tiff")
plt.savefig(fig_path, dpi=600, bbox_inches='tight', pad_inches=0, transparent=False)
plt.show()
plt.close()














