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

    for hue in unique_hues:
        mask = (df['hue_inducer'] == hue)
        sorted_df = df[mask].sort_values(by='size')

        sorted_labels = sorted_df[label_column].values
        sorted_sems = sorted_df[sem_column].values
        sorted_sizes = sorted_df['size'].values

        positions = []
        for i, size in enumerate(sorted_sizes):
            color = hue_to_rgb(hue)
            position = len(x_ticks) + 1
            x_ticks.append(f'{size}')
            x_positions.append(position)
            positions.append(position)
            plt.bar(position, sorted_labels[i], color=color, yerr=sorted_sems[i], capsize=5)

        # 绘制线条连接相同颜色的点
        plt.plot(positions, sorted_labels, color='black', marker='o', markersize=5, markerfacecolor='black')

        x_ticks.append(f'(Hue {hue})')
        x_positions.append(len(x_ticks) + 0.5)

    # 绘制test_color对应的虚线
    plt.axhline(y=test_value, color='grey', linestyle='--', linewidth=2)

    plt.xticks(ticks=np.arange(1, len(x_ticks) + 1), labels=x_ticks, rotation=90)
    plt.xlabel("Size (and Hue in brackets)")
    plt.ylabel(y_label)
    plt.title(f"{y_label} for different Hues and Sizes")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'inducer_unique_plots/{file_name}')
    plt.close()


# 创建保存目录
if not os.path.exists('inducer_unique_plots'):
    os.makedirs('inducer_unique_plots')

# 绘制 L_label 图表
plot_labels('L_label_mean', 'L_label_sem', 'L_label_mean', 'L_label_vs_diff_hue.png', test_color[0])

# 绘制 a_label 图表
plot_labels('a_label_mean', 'a_label_sem', 'a_label_mean', 'a_label_vs_diff_hue.png', test_color[1])

# 绘制 b_label 图表
plot_labels('b_label_mean', 'b_label_sem', 'b_label_mean', 'b_label_vs_diff_hue.png', test_color[2])
