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
save_dir = "/home/fastdvdnet-master-unsupervised/fastdvdnet-master/cem_out/ALL_sig"
# ROI 区域坐标（中间区域）
y1, y2, x1, x2 = 256, 256 + 160, 756, 756 + 160
#y1, y2, x1, x2 = 256, 256 + 160, 960, 960 + 160

############################# proposed #########################################
mat_path = "/home/fastdvdnet-master-unsupervised/fastdvdnet-master/fastDVDnet_final_evaluate_map_SigArea_1.mat"
data = loadmat(mat_path)['data']  # (5,34,50)
a = data[0]
b = data[1]
c = data[2]
d = data[3]
e = data[4]
thresholds = [55, 43, 39, 43, 60] #all_sigArea [55, 44, 39, 43, 60]
#thresholds = [55, 44, 41, 45, 62] #half_sigArea
#thresholds = [100, 100, 100, 100, 100] #ALL
data_thresholded = np.zeros_like(data)
for i in range(5):
    data_thresholded[i] = np.where(data[i] < thresholds[i], data[i], 0)
nonzero_ratios = []
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
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax2.set_xlim(0, W)
    ax2.set_ylim(H, 0)
    ax2.axis('off')
    fig2.tight_layout(pad=0)
    fig2.savefig(f"{save_dir}/frame_{i + 1}_cem_overlay_ours.tiff", dpi=600, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close(fig2)

    # ③ 统计非零和零数量
    nonzero_count = np.count_nonzero(heatmap_resized)
    zero_count = heatmap_resized.size - nonzero_count
    nonzero_ratios.append(100 * (nonzero_count / heatmap_resized.size))
    print(
        f"Frame {i + 1}: 非零占比={nonzero_count / heatmap_resized.size:.2%}, 零值占比={zero_count / heatmap_resized.size:.2%}")

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


############################# noise_F #########################################
mat_path_F = "/home/fastdvdnet-master-unsupervised/fastdvdnet-master/fastDVDnet_final_evaluate_map_SigArea_fastdvd_1.mat"
data_F = loadmat(mat_path_F)['data']  # (5,34,50)
a = data_F[0]
b = data_F[1]
c = data_F[2]
d = data_F[3]
e = data_F[4]
thresholds = [38, 38, 33, 35, 38]#[37, 38, 33.8, 36, 39]
data_thresholded_F = np.zeros_like(data_F)
for i in range(5):
    data_thresholded_F[i] = np.where(data_F[i] < thresholds[i], data_F[i], 0)
nonzero_ratios_F = []
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
    nonzero_count = np.count_nonzero(heatmap_resized_F)
    zero_count = heatmap_resized_F.size - nonzero_count
    nonzero_ratios_F.append(100 * (nonzero_count / heatmap_resized_F.size))
    print(
        f"Frame {i + 1}: 非零占比={nonzero_count / heatmap_resized_F.size:.2%}, 零值占比={zero_count / heatmap_resized_F.size:.2%}")

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

############################# noise_n #########################################
mat_path_N = "fastDVDnet_final_evaluate_map_SigArea_nlcl_1.mat"
data_N = loadmat(mat_path_N)['data']  # (5,34,50)
a = data_N[0]
b = data_N[1]
c = data_N[2]
d = data_N[3]
e = data_N[4]
thresholds = [41, 40, 48, 39, 44]#[37, 38, 33.8, 36, 39]
thresholds_0 = [0, 0, 0, 0, 0]
data_thresholded_N = np.zeros_like(data_N)
for i in range(5):
    data_thresholded_N[i] = np.where((data_N[i] > thresholds_0[i]) & (data_N[i] < thresholds[i]),data_N[i], 0)
nonzero_ratios_N = []
m = [0,1,2,3,4]
from scipy.ndimage import binary_dilation
y1, y2, x1, x2 = 256, 256 + 160, 756, 756 + 160
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

    # ===== 可视化 =====
    # if visualize:
    #     plt.figure(figsize=(12, 4))
    #
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(roi, cmap='hot')
    #     plt.title(f'Original ROI (img {i + 1})')
    #     plt.colorbar()
    #
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(full_mask_dilated, cmap='gray')
    #     plt.title(f'Dilated ROI mask (img {i + 1})')
    #
    #     plt.subplot(1, 3, 3)
    #     plt.imshow(heatmap_N, cmap='hot')
    #     plt.title(f'Heatmap with dilated ROI overlay (img {i + 1})')
    #     plt.colorbar()
    #
    #     plt.tight_layout()
    #     plt.show()



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
    nonzero_count = np.count_nonzero(heatmap_resized_N)
    zero_count = heatmap_resized_N.size - nonzero_count
    nonzero_ratios_N.append(100 * (nonzero_count / heatmap_resized_N.size))
    print(
        f"Frame {i + 1}: 非零占比={nonzero_count / heatmap_resized_N.size:.2%}, 零值占比={zero_count / heatmap_resized_N.size:.2%}")

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

# ############################# noise_U #########################################
# mat_path_U = "/mnt/home/yuhuizhen/project/URetinex-Net-main/fastDVDnet_final_evaluate_map_U_SigArea.mat"
# data_U = loadmat(mat_path_U)['data']  # (5,34,50)
# data_U = np.repeat(data_U, repeats=5, axis=0)
# a = data_U[0]
# b = data_U[1]
# c = data_U[2]
# d = data_U[3]
# e = data_U[4]
# thresholds = [82, 82, 82, 82, 82]#[37, 38, 33.8, 36, 39]
# data_thresholded_U = np.zeros_like(data_U)
# for i in range(5):
#     data_thresholded_U[i] = np.where(data_U[i] < thresholds[i], data_U[i], 0)
# nonzero_ratios_U = []
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
#     nonzero_count = np.count_nonzero(heatmap_resized_U)
#     zero_count = heatmap_resized_U.size - nonzero_count
#     nonzero_ratios_U.append(100 * (nonzero_count / heatmap_resized_U.size))
#     print(
#         f"Frame {i + 1}: 非零占比={nonzero_count / heatmap_resized_U.size:.2%}, 零值占比={zero_count / heatmap_resized_U.size:.2%}")
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




time_labels = [
    r'$\mathit{t}$-2',
    r'$\mathit{t}$-1',
    r'$\mathit{t}$',
    r'$\mathit{t}$+1',
    r'$\mathit{t}$+2'
]
nonzero_ratios    = np.array(nonzero_ratios)
nonzero_ratios_F  = np.array(nonzero_ratios_F)
nonzero_ratios_N  = np.array(nonzero_ratios_N)
x = np.arange(len(time_labels))
width = 0.2
from matplotlib import rcParams
# ===== 全局字体：Times New Roman =====
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = [
    'Times New Roman',
    'Nimbus Roman',
    'Liberation Serif',
    'DejaVu Serif'
]
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(7, 5))
# 第一组（蓝）
plt.bar(
    x - width/2,
    nonzero_ratios,
    width=width,
    color='tab:blue',
    alpha=0.7,
    label='Proposed'
)
# 第二组（绿）
plt.bar(
    x + width/2,
    nonzero_ratios_F,
    width=width,
    color='tab:green',
    alpha=0.7,
    label='FastDVDnet'
)
#第三组
plt.bar(
    x + width*1.5,
    nonzero_ratios_N,
    width=width,
    color='tab:red',
    alpha=0.7,
    label='URetinex-Net'
)
# 可选：折线（如果你想保留趋势）
plt.plot(x - width/2, nonzero_ratios,   color='tab:blue',  marker='o')
plt.plot(x + width/2, nonzero_ratios_F, color='tab:green', marker='o')
plt.plot(x + width*1.5, nonzero_ratios_N, color='tab:red', marker='o')

# ===== 坐标轴刻度 =====
plt.xticks(x + width/2, time_labels, fontsize=32)
plt.yticks([0, 2, 4, 6, 8, 10, 12], fontsize=32)

# ===== 坐标轴范围与标签 =====
plt.ylabel('Utilization (%)', fontsize=28)
plt.ylim(0, 12)

# ===== 统一刻度字号 =====
plt.tick_params(
    axis='both',
    which='major',
    length=6,
    width=1.2
)

# # ===== 强制刻度字体为 Times New Roman（防止混字体）=====
# ax = plt.gca()
# for label in ax.get_xticklabels() + ax.get_yticklabels():
#     label.set_fontname('Times New Roman')

plt.tight_layout()

# ===== 保存 =====
fig.savefig(
    os.path.join(save_dir, "all.tiff"),
    dpi=600,
    bbox_inches='tight',
    pad_inches=0,
    transparent=False
)

plt.show()
plt.close(fig)















