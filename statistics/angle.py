import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 创建保存图表的文件夹
output_folder = "angle"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取CSV文件
df = pd.read_csv("../updated_averaged_data_012_sta.csv")


# 计算夹角
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


# cross_product > 0逆时针，为-号，cross_product < 0顺时针，为+号

angles = []
for index, row in df.iterrows():
    a_label_mean = row["a_label_mean"]
    b_label_mean = row["b_label_mean"]
    a_inducer = row["a_inducer"]
    b_inducer = row["b_inducer"]
    a_test = row["a_test"]
    b_test = row["b_test"]

    RTC_angle = calculate_angle(a_label_mean - a_test, b_label_mean - b_test, a_inducer - a_test, b_inducer - b_test)
    # TC_angle = calculate_angle(a_inducer-a_test, b_inducer-b_test, 1, 0)
    # RTC_angle = TR_angle - TC_angle

    sign = determine_sign(
        a_label_mean - a_test,
        b_label_mean - b_test,
        a_inducer - a_test,
        b_inducer - b_test,
    )
    RTC_angle *= sign
    angles.append(RTC_angle)

df["RTC_angle"] = angles

# 获取唯一的 hue_inducer 值
hue_inducers = df["hue_inducer"].unique()

# 为每个 hue_inducer 作图
for hue_inducer in hue_inducers:
    subset = df[df["hue_inducer"] == hue_inducer]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)

    sizes = subset["size"]
    norm = plt.Normalize(sizes.min(), sizes.max())
    cmap = plt.get_cmap("viridis")

    for index, row in subset.iterrows():
        a_test = row["a_test"]
        b_test = row["b_test"]
        a_inducer = row["a_inducer"]
        b_inducer = row["b_inducer"]
        a_label_mean = row["a_label_mean"]
        b_label_mean = row["b_label_mean"]
        delta_E = row["delta_E"]
        size = row["size"]

        # 计算TC的极坐标角度和距离
        TC_angle = np.arctan2(b_inducer - b_test, a_inducer - a_test)
        TC_distance = np.linalg.norm([a_inducer - a_test, b_inducer - b_test])

        # 计算R点的极坐标角度和距离
        R_angle = np.arctan2(b_label_mean - b_test, a_label_mean - a_test)
        R_distance = np.linalg.norm([a_label_mean - a_test, b_label_mean - b_test])

        # 绘制TC方向的线
        ax.plot(
            [0, TC_angle],
            [0, TC_distance],
            color="grey",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )

        # 绘制R点的散点图，颜色表示size大小
        sc = ax.scatter(R_angle, R_distance, color=cmap(norm(size)), alpha=0.6, s=50)

    # 设置图例和标签
    ax.set_theta_zero_location("E")  # 将0度设为东（右侧水平线）
    ax.set_theta_direction(1)  # 顺时针方向

    # 放大坐标轴刻度
    max_distance = max(
        df[
            [
                "a_label_mean",
                "b_label_mean",
                "a_inducer",
                "b_inducer",
                "a_test",
                "b_test",
            ]
        ]
        .max()
        .max(),
        2,
    )
    ax.set_ylim(0, max_distance)

    # 创建颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label="size")

    plt.title(f"Angles for Inducer {hue_inducer}", fontsize=20)

    # 保存图表
    plt.savefig(f"{output_folder}/angles_for_inducer_{hue_inducer}.png")

    # 显示图表
    # plt.show()
