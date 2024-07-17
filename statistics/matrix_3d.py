import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 创建保存图表的文件夹
output_folder = "angle"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

df = pd.read_csv("./updated_averaged_data_with_RTC_angle.csv",
                 usecols=["L_label_mean", "a_label_mean", "b_label_mean", "L_inducer", "a_inducer", "b_inducer",
                          "L_test", "a_test", "b_test", "size", "hue_inducer"])


def calculate_angle_3d(vector1, vector2):
    cos_theta = np.dot(vector1, vector2) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle)


def determine_sign_3d(vector1, vector2):
    cross_product = np.cross(vector1, vector2)
    return 1 if np.linalg.norm(cross_product) > 0 else -1


def rotation_matrix_3d(angle_deg, axis):
    angle_rad = np.radians(angle_deg)
    axis = axis / np.linalg.norm(axis)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    ux, uy, uz = axis

    return np.array([
        [cos_theta + ux ** 2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta,
         ux * uz * (1 - cos_theta) + uy * sin_theta],
        [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy ** 2 * (1 - cos_theta),
         uy * uz * (1 - cos_theta) - ux * sin_theta],
        [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta,
         cos_theta + uz ** 2 * (1 - cos_theta)]
    ])


# 计算每个样本的旋转角度和矩阵
angles = []
rotation_matrices = []

for index, row in df.iterrows():
    L_label_mean = row["L_label_mean"]
    a_label_mean = row["a_label_mean"]
    b_label_mean = row["b_label_mean"]
    L_inducer = row["L_inducer"]
    a_inducer = row["a_inducer"]
    b_inducer = row["b_inducer"]
    L_test = row["L_test"]
    a_test = row["a_test"]
    b_test = row["b_test"]

    vector_test = np.array([L_inducer - L_test, a_inducer - a_test, b_inducer - b_test])
    vector_label = np.array([L_label_mean - L_test, a_label_mean - a_test, b_label_mean - b_test])

    angle_3d = calculate_angle_3d(vector_test, vector_label)  # 这里对调没有问题
    sign_3d = determine_sign_3d(vector_test, vector_label)  # 这里注意顺序
    angle_3d *= sign_3d

    axis = np.cross(vector_test, vector_label)
    rot_matrix_3d = rotation_matrix_3d(angle_3d, axis)
    angles.append(angle_3d)
    rotation_matrices.append(rot_matrix_3d)

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
plt.savefig(os.path.join(output_folder, 'rotation_matrices_heatmap_3d_custom.png'), bbox_inches='tight')
plt.show()
