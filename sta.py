import numpy as np
import matplotlib.pyplot as plt
from skimage.color import deltaE_cie76

# 假设这些是你的数据
L_fixed = 80
A_color = np.array([L_fixed, -90, 90])  # A色的原始Lab值
B_colors = np.array(
    [  # 24种不同的诱导色B的Lab值
        [L_fixed, 20, 30],
        [L_fixed, 40, 50],
        [L_fixed, 41, 50],
        [L_fixed, 42, 40],
        [L_fixed, 25, 52],
        [L_fixed, 26, 80],
        [L_fixed, 30, 60],
        [L_fixed, 35, 30],
        [L_fixed, 38, 55],
        [L_fixed, -40, 44],
        [L_fixed, 80, 13],
        [L_fixed, 60, -5],
        [L_fixed, 5, 25],
        [L_fixed, 45, 40],
        [L_fixed, 41, 88],
        [L_fixed, -42, 3],
        [L_fixed, 25, -52],
        [L_fixed, 26, 11],
        [L_fixed, 7, -36],
        [L_fixed, -35, 30],
        # ... (其他20种颜色)
    ]
)
observed_A_colors = np.array(
    [  # 实验者调整后的A色的观测Lab值（三次平均值）
        [L_fixed, 2, 6],
        [L_fixed, 38, 52],
        [L_fixed, 25, 50],
        [L_fixed, 40, -50],
        [L_fixed, 45, 55],
        [L_fixed, 38, 36],
        [L_fixed, -40, 70],
        [L_fixed, 40, 7],
        [L_fixed, 10, 30],
        [L_fixed, 5, 80],
        [L_fixed, 46, 5],
        [L_fixed, -30, 6],
        [L_fixed, 92, 5],
        [L_fixed, 38, 56],
        [L_fixed, -25, 36],
        [L_fixed, 70, -50],
        [L_fixed, -5, -55],
        [L_fixed, 38, 36],
        [L_fixed, -25, -66],
        [L_fixed, 40, 7],
        # ... (其他20种颜色)
    ]
)

# 计算ΔE值
delta_E_values = np.array(
    [deltaE_cie76(A_color, obs_A_color) for obs_A_color in observed_A_colors]
)

# 绘制ΔE值图表
plt.figure(figsize=(10, 5))
plt.bar(range(1, 25), delta_E_values, color="skyblue")
plt.xlabel("inducer_B NO.")
plt.ylabel("ΔE value")
plt.title("The effect to A value by different inducer B")
plt.xticks(range(1, 25), labels=[f"B{i+1}" for i in range(24)], rotation=45)
plt.tight_layout()
plt.show()

# 计算方向变化
delta_a = observed_A_colors[:, 1] - A_color[1]
delta_b = observed_A_colors[:, 2] - A_color[2]
angles = np.arctan2(delta_b, delta_a)  # 计算角度
angles_degrees = np.degrees(angles)  # 转换为度数

# 绘制a和b值的变化方向图
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
c = ax.scatter(angles, delta_E_values, c=delta_E_values, cmap="hsv", alpha=0.75)
ax.set_title("direction and ΔE")
ax.set_yticklabels([])  # 隐藏径向标签
plt.colorbar(c, ax=ax, label="ΔE")
plt.show()


# 如果你想对A色和观测A色的a和b值分别进行对比
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# 绘制a值对比S
axs[0].plot(
    range(1, 25), [A_color[1]] * 24, label="A:a value", linestyle="--", color="red"
)
axs[0].plot(
    range(1, 25), observed_A_colors[:, 1], label="observe_A:a value", marker="o"
)
axs[0].set_xlabel("inducer_B NO.")
axs[0].set_ylabel("a value")
axs[0].set_title("The effect to A's a value by different inducer B")
axs[0].legend()
axs[0].set_xticks(range(1, 25))
axs[0].set_xticklabels([f"B{i+1}" for i in range(24)], rotation=45)

# 绘制b值对比
axs[1].plot(
    range(1, 25), [A_color[2]] * 24, label="A:b value", linestyle="--", color="red"
)
axs[1].plot(
    range(1, 25), observed_A_colors[:, 2], label="observe_A:b value", marker="o"
)
axs[1].set_xlabel("inducer_B NO.")
axs[1].set_ylabel("b value")
axs[1].set_title("The effect to A's b value by different inducer B")
axs[1].legend()
axs[1].set_xticks(range(1, 25))
axs[1].set_xticklabels([f"B{i+1}" for i in range(24)], rotation=45)

plt.tight_layout()
plt.show()
