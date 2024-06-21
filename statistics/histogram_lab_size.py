import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import colorsys

# 从CSV文件中读取数据
df = pd.read_csv('../updated_averaged_data_012.csv')

# 提取数据
test_color = np.array([80, -90, 90])
inducer_colors = df[['L_inducer', 'a_inducer', 'b_inducer']].values
labeled_colors = df[['L_label_mean', 'a_label_mean', 'b_label_mean']].values
bar_colors = df[['R_inducer', 'G_inducer', 'B_inducer']].values / 255  # 归一化到0-1之间

# 提取独特的sizes和hues
unique_sizes = np.sort(df['size'].unique())
unique_hues = np.sort(df['hue_inducer'].unique())


# 将hue_inducer转换为RGB颜色
def hue_to_rgb(hue):
    return colorsys.hsv_to_rgb(hue / 360.0, 1.0, 1.0)


# 创建绘图函数
def plot_labels(label_column, sem_column, y_label, file_name, test_value):
    plt.figure(figsize=(20, 10))
    x_ticks = []
    x_positions = []

    # 准备数据存储每个hue的点位置和对应的值
    hue_positions = {size: [] for size in unique_sizes}
    hue_values = {size: [] for size in unique_sizes}

    for size_index, size in enumerate(unique_sizes):
        mask = (df['size'] == size)
        sorted_df = df[mask].sort_values(by='hue_inducer')

        sorted_labels = sorted_df[label_column].values
        sorted_sems = sorted_df[sem_column].values
        sorted_hue_inducers = sorted_df['hue_inducer'].values

        positions = []
        for i, hue in enumerate(sorted_hue_inducers):
            color = hue_to_rgb(hue)
            position = len(x_ticks) + 1
            x_ticks.append(f'{hue}')
            x_positions.append(position)
            positions.append(position)
            plt.bar(position, sorted_labels[i], color=color, yerr=sorted_sems[i], capsize=5)

            # 存储size对应的位置和值
            hue_positions[size].append(position)
            hue_values[size].append(sorted_labels[i])

        x_ticks.append(f'(Size {size})')
        x_positions.append(len(x_ticks) + 0.5)

    # 绘制线条连接相同size的点，并用黑点标记
    for size in unique_sizes:
        plt.plot(hue_positions[size], hue_values[size], color='black', marker='o', linestyle='-', markersize=5)

    # 绘制test_color对应的虚线
    plt.axhline(y=test_value, color='grey', linestyle='--', linewidth=2)

    plt.xticks(ticks=np.arange(1, len(x_ticks) + 1), labels=x_ticks, rotation=90)
    plt.xlabel("Hue (and Size in brackets)")
    plt.ylabel(y_label)
    plt.title(f"{y_label} for different Sizes and Hues")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'size_unique_plots/{file_name}')
    plt.close()


# 创建保存目录
if not os.path.exists('size_unique_plots'):
    os.makedirs('size_unique_plots')

# 绘制 L_label 图表
plot_labels('L_label_mean', 'L_label_sem', 'L_label_mean', 'L_label_vs_diff_size.png', test_color[0])

# 绘制 a_label 图表
plot_labels('a_label_mean', 'a_label_sem', 'a_label_mean', 'a_label_vs_diff_size.png', test_color[1])

# 绘制 b_label 图表
plot_labels('b_label_mean', 'b_label_sem', 'b_label_mean', 'b_label_vs_diff_size.png', test_color[2])
