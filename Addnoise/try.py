import numpy as np
import matplotlib.pyplot as plt
# 设置图像大小
image_shape = (1, 1088, 1600)
# 生成泊松噪声图像
lam = 30  # 控制泊松噪声的强度
poisson_noise_image = np.random.poisson(lam=lam, size=image_shape)
# 生成稀疏掩码
sparsity = 0.05  # 控制稀疏度，0.01表示1%的像素位置有噪声
mask = np.random.choice([0, 1], size=image_shape, p=[1-sparsity, sparsity])
# 应用稀疏掩码
sparse_poisson_noise_image = poisson_noise_image * mask
# 去掉单通道维度用于可视化
sparse_poisson_noise_image_visual = sparse_poisson_noise_image[0]
# 可视化图像
plt.imshow(sparse_poisson_noise_image_visual, cmap='gray')
plt.title('Sparse Poisson Noise Image')
plt.axis('off')  # 关闭坐标轴
plt.show()
