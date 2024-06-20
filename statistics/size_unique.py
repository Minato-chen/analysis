import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import deltaE_cie76

# 从CSV文件中读取数据
df = pd.read_csv('../updated_averaged_data_012.csv')

# 提取数据
test_color = np.array([80, -90, 90])
inducer_colors = df[['L_inducer', 'a_inducer', 'b_inducer']].values
labeled_colors = df[['L_label_mean', 'a_label_mean', 'b_label_mean']].values
bar_colors = df[['R_inducer', 'G_inducer', 'B_inducer']].values / 255  # 归一化到0-1之间
hue_inducer = df['hue_inducer'].values
test_rgb = np.array([0, 235, 0])/255

# 计算ΔE值
delta_E_values = np.array(
    [deltaE_cie76(test_color, result_color) for result_color in labeled_colors]
)

# 提取独特的sizes
unique_inducers = df[['L_inducer', 'a_inducer', 'b_inducer']].drop_duplicates().values
unique_sizes = df['size'].drop_duplicates().values
os.makedirs('size_unique_plots', exist_ok=True)

for size in unique_sizes:
    mask = (df['size'] == size)
    plt.figure(figsize=(10, 5))

    # 绘制inducer颜色，用bar_color标记
    plt.scatter(df['a_inducer'][mask], df['b_inducer'][mask], c=bar_colors[mask], label='Inducer Colors')

    # 在inducer颜色旁边标记hue_inducer值
    for i, hue in enumerate(hue_inducer[mask]):
        plt.annotate(f'{int(hue)}°', (df['a_inducer'][mask].values[i], df['b_inducer'][mask].values[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='black')

    # 绘制label颜色，并用对应bar_color的箭头指向label颜色
    for i in range(len(df[mask])):
        plt.scatter(df['a_label_mean'][mask].values[i], df['b_label_mean'][mask].values[i], c='black', marker='x')
        plt.arrow(test_color[1], test_color[2],
                  df['a_label_mean'][mask].values[i] - test_color[1],
                  df['b_label_mean'][mask].values[i] - test_color[2],
                  color=bar_colors[mask][i], head_width=1, head_length=2, length_includes_head=True)

    # 绘制test_color
    plt.scatter(test_color[1], test_color[2], color=test_rgb, marker='o'
                                                                     '', label='Test Color')

    plt.xlabel("a")
    plt.ylabel("b")
    plt.title(f"For size {size} with different inducers")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'size_unique_plots/size_{size}.png')
    plt.close()
