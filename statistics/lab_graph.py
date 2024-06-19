import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import colorsys

# 从CSV文件中读取数据
df = pd.read_csv('updated_averaged_data_0.csv')

# 提取数据
test_color = np.array([80, -90, 90])
inducer_colors = df[['L_inducer', 'a_inducer', 'b_inducer']].values
labeled_colors = df[['L_label', 'a_label', 'b_label']].values
bar_colors = df[['R_inducer', 'G_inducer', 'B_inducer']].values / 255  # 归一化到0-1之间

# 提取独特的sizes和hues
unique_sizes = np.sort(df['size'].unique())
unique_hues = np.sort(df['hue_inducer'].unique())


# 将hue_inducer转换为RGB颜色
def hue_to_rgb(hue):
    return colorsys.hsv_to_rgb(hue / 360.0, 1.0, 1.0)


# 创建绘图函数
def plot_labels(label_column, y_label, file_name):
    plt.figure(figsize=(20, 10))
    x_ticks = []
    x_positions = []

    for size_index, size in enumerate(unique_sizes):
        mask = (df['size'] == size)
        sorted_df = df[mask].sort_values(by='hue_inducer')

        sorted_labels = sorted_df[label_column].values
        sorted_hue_inducers = sorted_df['hue_inducer'].values

        for i, hue in enumerate(sorted_hue_inducers):
            color = hue_to_rgb(hue)
            position = len(x_ticks) + 1
            x_ticks.append(f'{hue}')
            x_positions.append(position)
            plt.bar(position, sorted_labels[i], color=color)

        x_ticks.append(f'(Size {size})')
        x_positions.append(len(x_ticks) + 0.5)

    plt.xticks(ticks=np.arange(1, len(x_ticks) + 1), labels=x_ticks, rotation=90)
    plt.xlabel("Hue (and Size in brackets)")
    plt.ylabel(y_label)
    plt.title(f"{y_label} for different Sizes and Hues")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'size_unique_plots/{file_name}')
    plt.show()


# 绘制 b_label 图表
plot_labels('b_label', 'b_label', 'b_label_vs_size_and_hue.png')

# 绘制 L_label 图表
plot_labels('L_label', 'L_label', 'L_label_vs_size_and_hue.png')

# 绘制 a_label 图表
plot_labels('a_label', 'a_label', 'a_label_vs_size_and_hue.png')
