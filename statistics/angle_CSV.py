import pandas as pd
import numpy as np
import os

# 创建保存图表的文件夹
output_folder = "angle"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取CSV文件
df = pd.read_csv("../updated_averaged_data_012_sta.csv")


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

# 将计算出来的RTC_angle保存到CSV文件
output_csv = os.path.join(output_folder, "updated_averaged_data_with_RTC_angle.csv")
df.to_csv(output_csv, index=False)

print(f"RTC angles have been calculated and saved to {output_csv}")
