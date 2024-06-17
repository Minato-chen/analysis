import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import deltaE_cie76

# 从CSV文件中读取数据
df = pd.read_csv('updated_averaged_data.csv')

# 提取数据
inducer_colors = df[['L_inducer', 'a_inducer', 'b_inducer']].values
labeled_colors = df[['L_label', 'a_label', 'b_label']].values
bar_colors = df[['R_inducer', 'G_inducer', 'B_inducer']].values / 255  # 归一化到0-1之间

# 固定的A色Lab值
L_fixed = 80
A_color = np.array([L_fixed, -90, 90])  # A色的原始Lab值

# 计算ΔE值
delta_E_values = np.array(
    [deltaE_cie76(A_color, obs_A_color) for obs_A_color in labeled_colors]
)

# 绘制ΔE值图表
plt.figure(figsize=(10, 5))
plt.bar(range(1, len(delta_E_values) + 1), delta_E_values, color=bar_colors)
plt.xlabel("Inducer C NO.")
plt.ylabel("ΔE Value")
plt.title("The effect to T value by different inducer C")
plt.xticks(range(1, len(delta_E_values) + 1), labels=[f"({L},{a},{b})" for L, a, b in inducer_colors], rotation=45)
plt.tight_layout()
plt.show()

# 计算方向变化
delta_a = labeled_colors[:, 1] - A_color[1]
delta_b = labeled_colors[:, 2] - A_color[2]
angles = np.arctan2(delta_b, delta_a)  # 计算角度
angles_degrees = np.degrees(angles)  # 转换为度数

# 绘制a和b值的变化方向图
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
norm = plt.Normalize(delta_E_values.min(), delta_E_values.max())
cmap = plt.cm.Reds
c = ax.scatter(angles, delta_E_values, c=delta_E_values, cmap=cmap, norm=norm, alpha=0.75)
ax.set_title("Direction and ΔE")
ax.set_yticklabels([])  # 隐藏径向标签
plt.colorbar(c, ax=ax, label="ΔE")
plt.show()

# 对比A色和观测A色的L, a, b值
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# 绘制L值对比
axs[0].plot(range(1, len(delta_E_values) + 1), [A_color[0]] * len(delta_E_values), label="T:L value", linestyle="--", color="red")
axs[0].plot(range(1, len(delta_E_values) + 1), labeled_colors[:, 0], label="Observed T(R):L value", marker="o")
axs[0].plot(range(1, len(delta_E_values) + 1), inducer_colors[:, 0], label="Inducer C:L value", marker="x")
axs[0].set_xlabel("Inducer C NO.")
axs[0].set_ylabel("L Value")
axs[0].set_title("The effect to T's L value by different inducer C")
axs[0].legend()
axs[0].set_xticks(range(1, len(delta_E_values) + 1))
axs[0].set_xticklabels([f"({L},{a},{b})" for L, a, b in inducer_colors], rotation=45)

# 绘制a值对比
axs[1].plot(range(1, len(delta_E_values) + 1), [A_color[1]] * len(delta_E_values), label="T:a value", linestyle="--", color="red")
axs[1].plot(range(1, len(delta_E_values) + 1), labeled_colors[:, 1], label="Observed T(R):a value", marker="o")
axs[1].plot(range(1, len(delta_E_values) + 1), inducer_colors[:, 1], label="Inducer C:a value", marker="x")
axs[1].set_xlabel("Inducer C NO.")
axs[1].set_ylabel("a Value")
axs[1].set_title("The effect to T's a value by different inducer C")
axs[1].legend()
axs[1].set_xticks(range(1, len(delta_E_values) + 1))
axs[1].set_xticklabels([f"({L},{a},{b})" for L, a, b in inducer_colors], rotation=45)

# 绘制b值对比
axs[2].plot(range(1, len(delta_E_values) + 1), [A_color[2]] * len(delta_E_values), label="T:b value", linestyle="--", color="red")
axs[2].plot(range(1, len(delta_E_values) + 1), labeled_colors[:, 2], label="Observed T(R):b value", marker="o")
axs[2].plot(range(1, len(delta_E_values) + 1), inducer_colors[:, 2], label="Inducer C:b value", marker="x")
axs[2].set_xlabel("Inducer C NO.")
axs[2].set_ylabel("b Value")
axs[2].set_title("The effect to T's b value by different inducer C")
axs[2].legend()
axs[2].set_xticks(range(1, len(delta_E_values) + 1))
axs[2].set_xticklabels([f"({L},{a},{b})" for L, a, b in inducer_colors], rotation=45)

plt.tight_layout()
plt.show()

# 计算ΔL, Δa, Δb值
delta_L = labeled_colors[:, 0] - A_color[0]
delta_a = labeled_colors[:, 1] - A_color[1]
delta_b = labeled_colors[:, 2] - A_color[2]

# 绘制ΔL, Δa, Δb值图表
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# ΔL值
axs[0].bar(range(1, len(delta_L) + 1), delta_L, color=bar_colors)
axs[0].set_xlabel("Inducer C NO.")
axs[0].set_ylabel("ΔL Value")
axs[0].set_title("ΔL values by different inducer C")
axs[0].set_xticks(range(1, len(delta_L) + 1))
axs[0].set_xticklabels([f"({L},{a},{b})" for L, a, b in inducer_colors], rotation=45)

# Δa值
axs[1].bar(range(1, len(delta_a) + 1), delta_a, color=bar_colors)
axs[1].set_xlabel("Inducer C NO.")
axs[1].set_ylabel("Δa Value")
