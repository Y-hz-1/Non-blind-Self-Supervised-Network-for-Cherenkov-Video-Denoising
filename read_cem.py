import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.io import loadmat
from scipy.ndimage import label
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d

# 读取背景图像
data_img_path = "/home/fastdvdnet-master-unsupervised/fastdvdnet-master/test_seq/SUM_cem_1010.tif"
img = cv2.imread(data_img_path, cv2.IMREAD_UNCHANGED)
img = img.astype(np.float32) / (512.0*76)  # 归一化

# 读取 .mat 数据
mat_path = "/home/fastdvdnet-master-unsupervised/fastdvdnet-master/fastDVDnet_final_evaluate_map_Global_1.mat"
data = 100-loadmat(mat_path)['data']  # (5,34,50)
a = data[0]
b = data[1]
c = data[2]
d = data[3]
e = data[4]
#thresholds = [55, 44, 39, 43, 60] #all_sigArea
#thresholds = [55, 44, 41, 45, 62] #half_sigArea
thresholds = [100, 100, 100, 100, 100] #ALL
data_thresholded = np.zeros_like(data)
for i in range(5):
    data_thresholded[i] = np.where(data[i] < thresholds[i], data[i], 0)

# 阈值操作
# data_thresholded = np.where(data < 67, data, 0)  # <50 ALL sig ；

# ROI 区域坐标（中间区域）
y1, y2, x1, x2 = 256, 256 + 160, 756, 756 + 160
#y1, y2, x1, x2 = 256, 256 + 160, 960, 960 + 160
# =====================
# 逐帧处理
# =====================
nonzero_ratios = []
m = [2,4]
#ours
heatmap = data_thresholded[2]
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
heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
save_dir = "/home/fastdvdnet-master-unsupervised/fastdvdnet-master/cem_out"

# #URetinex-Net-main########################################################################################
mat_path_U = "/mnt/home/yuhuizhen/project/URetinex-Net-main/fastDVDnet_final_evaluate_map_U_Global.mat"
data_U = loadmat(mat_path_U)['data']  # (5,34,50)
thresholds = 100 #ALL
data_thresholded_U = np.zeros_like(data_U)
for i in range(5):
    data_thresholded_U = np.where(data_U < thresholds, data_U, 0)
nonzero_ratios = []
m = [2,4]
heatmap_U = data_thresholded_U.squeeze(0)
# ① 找最大连通区域
binary_mask_U = heatmap_U > 0
labeled_U, num_features_U = label(binary_mask_U)
if num_features_U > 0:
    sizes_U = np.bincount(labeled_U.ravel())
    sizes_U[0] = 0
    largest_label_U = sizes_U.argmax()
    mask_largest_U = (labeled_U == largest_label_U)
    heatmap_U = heatmap_U * mask_largest_U
else:
    print(f"第{i+1}张无连通区域")
    heatmap_U[:] = 0

# ② 放大到原图大小
heatmap_resized_U = cv2.resize(heatmap_U, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
save_dir = "/home/fastdvdnet-master-unsupervised/fastdvdnet-master/cem_out"
# ============================
# 在 CEM 叠加图上画横线
# ============================
img1 = (img-img.min())/(img.max()-img.min())
img = img1[0:1058,65:1535]
# # ax2.imshow(img, cmap='gray')
# heatmap_masked_U = np.ma.masked_where(heatmap_resized_U == 0, heatmap_resized_U)
# heatmap_masked_U = heatmap_masked_U[0:1058,65:1535]
# heatmap_masked_U = (heatmap_masked_U-heatmap_masked_U.min())/(heatmap_masked_U.max()-heatmap_masked_U.min())
#
# # 中线强度曲线
# # ========= 中线强度曲线 =========
# mid_y_U = heatmap_masked_U.shape[0] // 2
# intensity_profile_U = heatmap_masked_U[mid_y_U, :]
# intensity_profile_filled_U = intensity_profile_U.filled(0)
# sigma = 40
# intensity_smooth_U = gaussian_filter1d(intensity_profile_filled_U, sigma=sigma)
# fig1, ax1 = plt.subplots(figsize=(10, 4))
# ax1.plot(
#     intensity_smooth_U,
#     linewidth=2,
#     color='tab:blue',
#     label='URetinex'
# )
# ax1.set_xlabel("Pixel", fontsize=12)
# ax1.set_ylabel("Intensity", fontsize=12)
# ax1.set_title("Mid-line Intensity Profile", fontsize=13)
# ax1.legend()
# ax1.grid(alpha=0.3)
# fig1.tight_layout()
# fig1.savefig(f"{save_dir}/frame_{i + 1}_intensity_profile.tiff",dpi=600)
# plt.show()
# plt.close(fig1)
# ##图二原图 + 热图叠加
# H, W = img.shape[:2]
# white_red = LinearSegmentedColormap.from_list('white_red', ['white', 'red'])
# #fig2, ax2 = plt.subplots(figsize=(8, 6))
# fig2 = plt.figure(frameon=False)
# ax2 = plt.Axes(fig2, [0., 0., 1., 1.])
# fig2.add_axes(ax2)
# ax2.imshow(img,cmap='gray',extent=[0, W, H, 0] )  # ← 关键)
# ax2.imshow(heatmap_masked_U,cmap=white_red,alpha=0.5,extent=[0, W, H, 0])   # ← 必须一致)
# ax2.plot([0, W], [mid_y_U, mid_y_U], color='yellow', linewidth=2)
# ax2.set_xlim(0, W)
# ax2.set_ylim(H, 0)
# ax2.axis('off')
# fig2.tight_layout(pad=0)
# fig2.savefig(f"{save_dir}/frame_{i + 1}_cem_overlay_U.tiff",dpi=600,bbox_inches='tight',pad_inches=0)
# plt.show()
# plt.close(fig2)

############################# noise_N #########################################
img1_uint8 = (img1 * 255).astype(np.uint8)
#_, mask = cv2.threshold(img1_uint8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
p = 78   # 只取最亮的 15%
thr = np.percentile(img1, p)
mask = (img1 >= thr).astype(np.uint8)
# plt.figure(figsize=(6, 4))
# plt.imshow(mask, cmap='gray')
# plt.title("Binary mask (Otsu)")
# plt.axis('off')
# plt.colorbar()
# plt.show()
mat_path_N = "/home/fastdvdnet-master-unsupervised/fastdvdnet-master/fastDVDnet_final_evaluate_map_Global_nlcl_1.mat"
data_N = loadmat(mat_path_N)['data']  # (5,34,50)
thresholds = 100 #ALL
data_thresholded_N = np.zeros_like(data_N)
for i in range(5):
    data_thresholded_N = np.where(data_N < thresholds, data_N, 0)
nonzero_ratios = []
m = [2,4]
heatmap_N = data_thresholded_N[2]
binary_mask_N = heatmap_N > 0
labeled_N, num_features_N = label(binary_mask_N)
if num_features_N > 0:
    sizes_N = np.bincount(labeled_N.ravel())
    sizes_N[0] = 0
    largest_label_N = sizes_N.argmax()
    mask_largest_N = (labeled_N == largest_label_N)
    heatmap_N = heatmap_N * mask_largest_N
else:
    print(f"第{i+1}张无连通区域")
    heatmap_N[:] = 0

# ② 放大到原图大小
heatmap_resized_N = cv2.resize(heatmap_N, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_NEAREST)
# heatmap_resized_N = heatmap_resized_N * (mask + 0.8 * (1 - mask))
mask_blur = cv2.GaussianBlur(mask.astype(np.float32), (51, 51), 0)
alpha_min = 0.8
alpha = alpha_min + (1 - alpha_min) * mask_blur
heatmap_resized_N = heatmap_resized_N * alpha
save_dir = "/home/fastdvdnet-master-unsupervised/fastdvdnet-master/cem_out"
heatmap_masked_N = np.ma.masked_where(heatmap_resized_N == 0, heatmap_resized_N)
heatmap_masked_N = heatmap_masked_N[0:1058,65:1535]
heatmap_masked_N = (heatmap_masked_N-heatmap_masked_N.min())/(heatmap_masked_N.max()-heatmap_masked_N.min())

# 中线强度曲线
# ========= 中线强度曲线 =========
mid_y_N = heatmap_masked_N.shape[0] // 2
intensity_profile_N = heatmap_masked_N[mid_y_N, :]
intensity_profile_filled_N = intensity_profile_N.filled(0)
sigma = 40
intensity_smooth_N = gaussian_filter1d(intensity_profile_filled_N, sigma=sigma)
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(
    intensity_smooth_N,
    linewidth=2,
    color='tab:blue',
    label='URetinex'
)
ax1.set_xlabel("Pixel", fontsize=12)
ax1.set_ylabel("Intensity", fontsize=12)
ax1.set_title("Mid-line Intensity Profile", fontsize=13)
ax1.legend()
ax1.grid(alpha=0.3)
fig1.tight_layout()
fig1.savefig(f"{save_dir}/frame_{i + 1}_intensity_profile_N.tiff",dpi=600)
plt.show()
plt.close(fig1)
##图二原图 + 热图叠加
H, W = img.shape[:2]
white_red = LinearSegmentedColormap.from_list('white_red', ['white', 'red'])
#fig2, ax2 = plt.subplots(figsize=(8, 6))
fig2 = plt.figure(frameon=False)
ax2 = plt.Axes(fig2, [0., 0., 1., 1.])
fig2.add_axes(ax2)
ax2.imshow(img,cmap='gray',extent=[0, W, H, 0] )  # ← 关键)
ax2.imshow(heatmap_masked_N,cmap=white_red,alpha=0.5,extent=[0, W, H, 0])   # ← 必须一致)
ax2.plot([0, W], [mid_y_N, mid_y_N], color='yellow', linewidth=2)
ax2.set_xlim(0, W)
ax2.set_ylim(H, 0)
ax2.axis('off')
fig2.tight_layout(pad=0)
fig2.savefig(f"{save_dir}/frame_{i + 1}_cem_overlay_N.tiff",dpi=600,bbox_inches='tight',pad_inches=0)
plt.show()
plt.close(fig2)

############################# noise_F #########################################
mat_path_F = "/home/fastdvdnet-master-unsupervised/fastdvdnet-master/fastDVDnet_final_evaluate_map_Global_fastdvd_1.mat"
data_F = loadmat(mat_path_F)['data']  # (5,34,50)
thresholds = 100 #ALL
data_thresholded_F = np.zeros_like(data_F)
for i in range(5):
    data_thresholded_F = np.where(data_F < thresholds, data_F, 0)
nonzero_ratios = []
m = [2,4]
heatmap_F = data_thresholded_F[2]
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
heatmap_resized_F = cv2.resize(heatmap_F, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_NEAREST)
save_dir = "/home/fastdvdnet-master-unsupervised/fastdvdnet-master/cem_out"
heatmap_masked_F = np.ma.masked_where(heatmap_resized_F == 0, heatmap_resized_F)
heatmap_masked_F = heatmap_masked_F[0:1058,65:1535]
heatmap_masked_F = (heatmap_masked_F-heatmap_masked_F.min())/(heatmap_masked_F.max()-heatmap_masked_F.min())

# 中线强度曲线
# ========= 中线强度曲线 =========
mid_y_F = heatmap_masked_F.shape[0] // 2
intensity_profile_F = heatmap_masked_F[mid_y_F, :]
intensity_profile_filled_F = intensity_profile_F.filled(0)
sigma = 40
intensity_smooth_F = gaussian_filter1d(intensity_profile_filled_F, sigma=sigma)
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(
    intensity_smooth_F,
    linewidth=2,
    color='tab:blue',
    label='URetinex'
)
ax1.set_xlabel("Pixel", fontsize=12)
ax1.set_ylabel("Intensity", fontsize=12)
ax1.set_title("Mid-line Intensity Profile", fontsize=13)
ax1.legend()
ax1.grid(alpha=0.3)
fig1.tight_layout()
fig1.savefig(f"{save_dir}/frame_{i + 1}_intensity_profile_F.tiff",dpi=600)
plt.show()
plt.close(fig1)
##图二原图 + 热图叠加
H, W = img.shape[:2]
white_red = LinearSegmentedColormap.from_list('white_red', ['white', 'red'])
#fig2, ax2 = plt.subplots(figsize=(8, 6))
fig2 = plt.figure(frameon=False)
ax2 = plt.Axes(fig2, [0., 0., 1., 1.])
fig2.add_axes(ax2)
ax2.imshow(img,cmap='gray',extent=[0, W, H, 0] )  # ← 关键)
ax2.imshow(heatmap_masked_F,cmap=white_red,alpha=0.5,extent=[0, W, H, 0])   # ← 必须一致)
ax2.plot([0, W], [mid_y_F, mid_y_F], color='yellow', linewidth=2)
ax2.set_xlim(0, W)
ax2.set_ylim(H, 0)
ax2.axis('off')
fig2.tight_layout(pad=0)
fig2.savefig(f"{save_dir}/frame_{i + 1}_cem_overlay_F.tiff",dpi=600,bbox_inches='tight',pad_inches=0)
plt.show()
plt.close(fig2)



# ============================
# 在 CEM 叠加图上画横线   propossed
# ============================
fig2, ax2 = plt.subplots(figsize=(8, 6), facecolor='none')
# img = img[0:1058,65:1535]
# img = (img-img.min())/(img.max()-img.min())
ax2.imshow(img, cmap='gray')
heatmap_masked = np.ma.masked_where(heatmap_resized == 0, heatmap_resized)
heatmap_masked = heatmap_masked[0:1058,65:1535]
heatmap_masked = (heatmap_masked-heatmap_masked.min())/(heatmap_masked.max()-heatmap_masked.min())
# ========= 中线强度曲线 =========
mid_y = heatmap_masked.shape[0] // 2
intensity_profile = heatmap_masked[mid_y, :]
intensity_profile_filled = intensity_profile.filled(0)
sigma = 40
intensity_smooth = gaussian_filter1d(intensity_profile_filled, sigma=sigma)
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(
    intensity_smooth,
    linewidth=2,
    color='tab:blue',
    label='Ours')
ax1.set_xlabel("Pixel", fontsize=12)
ax1.set_ylabel("Intensity", fontsize=12)
ax1.set_title("Mid-line Intensity Profile", fontsize=13)
ax1.legend()
ax1.grid(alpha=0.3)
fig1.tight_layout()
fig1.savefig(f"{save_dir}/frame_{i + 1}_intensity_profile_ours.tiff", dpi=600)
plt.show()
plt.close(fig1)
##图二原图 + 热图叠加
H, W = img.shape[:2]
white_red = LinearSegmentedColormap.from_list('white_red', ['white', 'red'])
# fig2, ax2 = plt.subplots(figsize=(8, 6))
fig2 = plt.figure(frameon=False)
ax2 = plt.Axes(fig2, [0., 0., 1., 1.])
fig2.add_axes(ax2)

ax2.imshow(img, cmap='gray', extent=[0, W, H, 0])  # ← 关键)
ax2.imshow(heatmap_masked, cmap=white_red, alpha=0.5, extent=[0, W, H, 0])  # ← 必须一致)
ax2.plot([0, W], [mid_y, mid_y], color='yellow', linewidth=2)
ax2.set_xlim(0, W)
ax2.set_ylim(H, 0)
ax2.axis('off')
fig2.tight_layout(pad=0)
fig2.savefig(f"{save_dir}/frame_{i + 1}_cem_overlay_ours.tiff", dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()
plt.close(fig2)



###黄线强度曲线
from scipy.ndimage import gaussian_filter1d
sigma = 40
def normalize_01(x, eps=1e-8):
    return (x - x.min()) / (x.max() - x.min() + eps)
# --- Base ---
mid_y_B = img.shape[0]//2
half_width = 100
y_start = max(0, mid_y_B - half_width)
y_end   = min(img.shape[0], mid_y_B + half_width + 1)
intensity_region_B = img[y_start:y_end, :]
intensity_B = img.mean(axis=0)
intensity_smooth_B = gaussian_filter1d(intensity_B, sigma=sigma)
intensity_smooth_B = normalize_01(intensity_smooth_B)

# --- URetinex ---
mid_y_N = heatmap_masked_N.shape[0] // 2
half_width = 100
y_start = max(0, mid_y_N - half_width)
y_end   = min(heatmap_masked_N.shape[0], mid_y_N + half_width + 1)
intensity_region_N = heatmap_masked_N[y_start:y_end, :].filled(0)
intensity_N = intensity_region_N.mean(axis=0)
intensity_smooth_N = gaussian_filter1d(intensity_N, sigma=sigma)
intensity_smooth_N = normalize_01(intensity_smooth_N)

# --- f ---
mid_y_F = heatmap_masked_F.shape[0] // 2
half_width = 100
y_start = max(0, mid_y_F - half_width)
y_end   = min(heatmap_masked_F.shape[0], mid_y_F + half_width + 1)
intensity_region_F = heatmap_masked_F[y_start:y_end, :].filled(0)
intensity_F = intensity_region_F.mean(axis=0)
intensity_smooth_F = gaussian_filter1d(intensity_F, sigma=sigma)
intensity_smooth_F = normalize_01(intensity_smooth_F)

# --- Ours ---
mid_y = heatmap_masked.shape[0] // 2
half_width = 100
y_start = max(0, mid_y - half_width)
y_end   = min(heatmap_masked.shape[0], mid_y + half_width + 1)
intensity_region = heatmap_masked[y_start:y_end, :].filled(0)
intensity = intensity_region.mean(axis=0)
intensity_smooth = gaussian_filter1d(intensity, sigma=sigma)
intensity_smooth = normalize_01(intensity_smooth)

# ========= 构造横坐标 =========
n = len(intensity_smooth_B[80:])
x = np.linspace(-15, 15, n)

# ========= 绘图 =========
fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(
    x, intensity_smooth_B[80:],
    linewidth=2,
    color='black',
    label='Reference'
)

ax.plot(
    x, intensity_smooth_N[80:],
    linewidth=2,
    linestyle='--',
    color='tab:red',
    label='NLCL'
)

ax.plot(
    x, intensity_smooth_F[80:],
    linewidth=2,
    linestyle='--',
    color='tab:green',
    label='FastDVDnet'
)

ax.plot(
    x, intensity_smooth[80:],
    linewidth=2,
    linestyle='--',
    color='tab:blue',
    label='Proposed'
)

font_size = 16
ax.set_xlabel("Pixel", fontsize=font_size)
ax.set_ylabel("Utilization (%)", fontsize=font_size)
ax.set_xlim(-15, 15)
ax.tick_params(axis='both', labelsize=font_size)
ax.legend(fontsize=16, frameon=False)

fig.tight_layout()
fig.savefig(
    f"{save_dir}/frame_{i + 1}_intensity_profile_comparison.tiff",
    dpi=600
)
plt.show()
plt.close(fig)


ax2.imshow(heatmap_masked, cmap=white_red, alpha=0.5)
ax2.set_xlim(0, img.shape[1])
ax2.set_ylim(img.shape[0], 0)
ax2.plot([0, 1480], [mid_y, mid_y], color='yellow', linewidth=2)
#ax2.text(10, mid_y - 10, 'Profile Line', color='yellow', fontsize=12)
ax2.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
fig2.savefig(
    f"{save_dir}/frame_{i + 1}_cem_overlay.png",
    dpi=600,
    bbox_inches='tight',
    pad_inches=0,
    transparent=False
)
plt.close(fig2)

# =====================
# ④ graph
# =====================
window = 11  # 平滑窗口大小，可调整
kernel = np.ones(window) / window
intensity_smooth = np.convolve(intensity_profile, kernel, mode='same')
from scipy.interpolate import make_interp_spline
from scipy import interpolate
a = intensity_profile.shape[0]
x = np.linspace(0,a-1,a)
x_new = np.linspace(0, a, 100*len(intensity_profile))
tck = interpolate.splrep(x, intensity_profile-1, s=0)
intensity_smooth = interpolate.splev(x_new, tck, der=0)
#intensity_smooth = np.linspace(intensity_profile.min(), intensity_profile.max(), 300)
plt.figure(figsize=(10, 5))
#plt.plot(intensity_profile, label='Original', linewidth=1)
plt.plot(intensity_smooth, label='Smoothed', linewidth=2)
plt.xlabel("Index")
plt.ylabel("Intensity")
plt.title("Intensity Profile (Original vs Smoothed)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =====================
# ④ colorbar
# =====================
fig_cb, ax_cb = plt.subplots(figsize=(2, 6))
dummy = ax_cb.imshow(np.linspace(0, 1, 100).reshape(100, 1), cmap=white_red)
cbar = fig_cb.colorbar(dummy, ax=ax_cb, fraction=0.5)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["0", "1"])
ax_cb.remove()  # 移除 dummy axes
fig_cb.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig_cb.savefig(
    f"{save_dir}/frame_{i + 1}_colorbar.png",
    dpi=600,
    bbox_inches='tight',
    pad_inches=0,
    transparent=False
)
plt.show()
plt.close(fig_cb)

# ③ 统计非零和零数量
nonzero_count = np.count_nonzero(heatmap_resized)
zero_count = heatmap_resized.size - nonzero_count
nonzero_ratios.append(100*(nonzero_count / heatmap_resized.size))
print(f"Frame {i+1}: 非零占比={nonzero_count / heatmap_resized.size:.2%}, 零值占比={zero_count / heatmap_resized.size:.2%}")

# =====================
# ④ 绘制红白饼状图（透明背景）
# =====================
save_dir = "/home/fastdvdnet-master-unsupervised/fastdvdnet-master/CEM_out/ALL_Sig"
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
plt.show()
plt.close(fig_pie)

#保存（如需要）
import os
fig_pie.savefig(os.path.join(save_dir, f"pie_frame_{i+1}.tiff"),dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
plt.close(fig_pie)

# =====================
# ⑤ 绘制CEM叠加图（无白边 + ROI 标注）
# =====================
white_red = LinearSegmentedColormap.from_list('white_red', ['white', 'red'])
fig, ax = plt.subplots(figsize=(8, 6), facecolor='none')

ax.imshow(img, cmap='gray')
heatmap_masked = np.ma.masked_where(heatmap_resized == 0, heatmap_resized)
ax.imshow(heatmap_masked, cmap=white_red, alpha=0.5)

# ROI 区域标注
rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                     linewidth=0, edgecolor='none',
                     facecolor='lime', alpha=0.5)
ax.add_patch(rect)
ax.text(x1, y1 - 10, 'ROI', color='lime', fontsize=12, weight='bold')

ax.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()

# ⑦ 保存结果
fig.savefig(
    os.path.join(save_dir, f"all_sig_cem_frame_{i + 1}.tiff"),
    dpi=300,
    bbox_inches='tight',
    pad_inches=0,
    transparent=True
)
plt.close()

####
time_labels = ['t-2', 't-1', 't', 't+1', 't+2']
plt.figure(figsize=(8,5))
plt.bar(time_labels, nonzero_ratios, color='skyblue', alpha=0.7, width=0.4)
plt.plot(time_labels, nonzero_ratios, color='skyblue', marker='o', linestyle='-', alpha=0.7)
plt.ylabel('Utilization (%)')
plt.ylim(0, 100)

#plt.show()

print("已完成：每张仅保留最大连通区域，并分别保存为带ROI标注的热力图")
