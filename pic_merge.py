import os
import pandas as pd
from PIL import Image
import numpy as np

# 读取合并后的CSV文件
df_grouped = pd.read_csv('averaged_data.csv')

# 创建输出文件夹
output_folder = 'image_output'
os.makedirs(output_folder, exist_ok=True)

# 定义输入文件夹
input_folder = 'image_data'


# 将数值转换为整数
def convert_to_int(value):
    return int(round(value))


# 遍历分组数据并处理对应的图片
for _, row in df_grouped.iterrows():
    size = row['size']  # 保留size为浮点数
    R_fore = convert_to_int(row['R_fore'])
    G_fore = convert_to_int(row['G_fore'])
    B_fore = convert_to_int(row['B_fore'])
    R_back = convert_to_int(row['R_back'])
    G_back = convert_to_int(row['G_back'])
    B_back = convert_to_int(row['B_back'])
    R_label = convert_to_int(row['R_label'])
    G_label = convert_to_int(row['G_label'])
    B_label = convert_to_int(row['B_label'])

    # 构建对应的图片文件名前缀
    image_prefix = f"{size}_{R_fore}_{G_fore}_{B_fore}_{R_back}_{G_back}_{B_back}_"

    print(f"Processing: {image_prefix}")

    # 查找匹配的图片
    matching_files = [f for f in os.listdir(input_folder) if f.startswith(image_prefix)]

    if not matching_files:
        print(f"No matching files found for prefix: {image_prefix}")
        continue

    for image_file in matching_files:
        image_path = os.path.join(input_folder, image_file)

        print(f"Found image: {image_path}")

        try:
            # 读取图片
            img = Image.open(image_path)

            # 构建新的图片文件名
            new_image_filename = f"{size}_{R_fore}_{G_fore}_{B_fore}_{R_back}_{G_back}_{B_back}_{R_label}_{G_label}_{B_label}.png"
            new_image_path = os.path.join(output_folder, new_image_filename)

            print(f"Saving new image: {new_image_path}")

            # 保存图片
            img.save(new_image_path)

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

print("数据和图片处理完成")
