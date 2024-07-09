import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 创建保存图表的文件夹
output_folder = "angle"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取CSV文件，假设数据格式和之前一样，这里只需读取需要的数据列
df = pd.read_csv("../updated_averaged_data_012_sta.csv",
                 usecols=["a_label_mean", "b_label_mean", "a_inducer", "b_inducer", "a_test", "b_test", "size",
                          "hue_inducer"])


def calculate_angle(a1, b1, a2, b2):
    vector1 = np.array([a1, b1])  # TC
    vector2 = np.array([a2, b2])  # TR
    cos_theta = np.dot(vector1, vector2) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle)


def determine_sign(a1, b1, a2, b2):
    cross_product = a1 * b2 - b1 * a2
    return -1 if cross_product > 0 else 1


def rotation_matrix(angle_deg):
    angle_rad = np.radians(angle_deg)
    return np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])


# 计算每个样本的旋转角度和矩阵
angles = []
rotation_matrices = []

for index, row in df.iterrows():
    a_label_mean = row["a_label_mean"]
    b_label_mean = row["b_label_mean"]
    a_inducer = row["a_inducer"]
    b_inducer = row["b_inducer"]
    a_test = row["a_test"]
    b_test = row["b_test"]

    RTC_angle = calculate_angle(a_label_mean - a_test, b_label_mean - b_test, a_inducer - a_test, b_inducer - b_test)
    sign = determine_sign(
        a_label_mean - a_test,
        b_label_mean - b_test,
        a_inducer - a_test,
        b_inducer - b_test,
    )
    RTC_angle *= sign
    angles.append(RTC_angle)

    rot_matrix = rotation_matrix(RTC_angle)
    rotation_matrices.append(rot_matrix)

df["RTC_angle"] = angles

# 确定要展示的 size 和 hue 列的值
sizes = df["size"].unique()
hues = df["hue_inducer"].unique()

# 创建图表
fig, axes = plt.subplots(len(sizes), len(hues), figsize=(18, 15))

for i, size in enumerate(sizes):
    for j, hue in enumerate(hues):
        sub_df = df[(df["size"] == size) & (df["hue_inducer"] == hue)]
        if not sub_df.empty:
            rot_matrix = rotation_matrices[sub_df.index[0]]  # 取第一个匹配的旋转矩阵
            ax = axes[i, j]
            sns.heatmap(rot_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=False, ax=ax, square=True, center=0,
                        vmin=-1, vmax=1)
            if i == 0:
                ax.set_title(f'{hue}', fontsize=12)
            if j == 0:
                ax.set_ylabel(f'{size}', fontsize=10)
            ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                           labelbottom=False, labelleft=False)
        else:
            ax = axes[i, j]
            ax.axis('off')  # 如果没有匹配的数据，关闭该子图

plt.tight_layout(pad=1.5)
plt.savefig(os.path.join(output_folder, 'rotation_matrices_heatmap_custom.png'), bbox_inches='tight')
plt.show()
