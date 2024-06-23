import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import colorsys

# 从CSV文件中读取数据
df = pd.read_csv('../../updated_averaged_data_012.csv')

# 确保数据范围正确
print("Data range check:")
print(df[['L_inducer', 'a_inducer', 'b_inducer', 'L_label_mean', 'a_label_mean', 'b_label_mean']].describe())

# 提取数据
test_color = np.array([80, -90, 90])
inducer_colors = df[['L_inducer', 'a_inducer', 'b_inducer']].values
labeled_colors = df[['L_label_mean', 'a_label_mean', 'b_label_mean']].values
bar_colors = df[['R_inducer', 'G_inducer', 'B_inducer']].values / 255  # 归一化到0-1之间

# 检查数据加载是否正确
print("Data loaded successfully:")
print(df.head())

# 提取独特的sizes和hues
unique_sizes = np.sort(df['size'].unique())
unique_hues = np.sort(df['hue_inducer'].unique())


# 将hue_inducer转换为RGB颜色
def hue_to_rgb(hue):
    return colorsys.hsv_to_rgb(hue / 360.0, 1.0, 1.0)


# 计算ΔL, Δa, Δb
df['ΔL'] = df['L_label_mean'] - test_color[0]
df['Δa'] = df['a_label_mean'] - test_color[1]
df['Δb'] = df['b_label_mean'] - test_color[2]

# 计算ΔL, Δa, Δb的标准误差
df['ΔL_sem'] = df['L_label_sem']
df['Δa_sem'] = df['a_label_sem']
df['Δb_sem'] = df['b_label_sem']

# 计算ΔE
df['ΔE'] = np.sqrt(df['ΔL'] ** 2 + df['Δa'] ** 2 + df['Δb'] ** 2)

# 计算ΔE的标准误差
df['ΔE_sem'] = np.sqrt(
    (df['ΔL'] / df['ΔE'] * df['ΔL_sem']) ** 2 +
    (df['Δa'] / df['ΔE'] * df['Δa_sem']) ** 2 +
    (df['Δb'] / df['ΔE'] * df['Δb_sem']) ** 2
)

# 检查新列是否正确添加
print("ΔL, Δa, Δb, ΔE columns added:")
print(df[['ΔL', 'Δa', 'Δb', 'ΔE', 'ΔL_sem', 'Δa_sem', 'Δb_sem', 'ΔE_sem']].head())


# 创建绘图函数
def plot_labels(label_column, sem_column, y_label, file_name, test_value=None):
    plt.figure(figsize=(20, 10))
    x_ticks = []
    x_positions = []

    for hue in unique_hues:
        mask = (df['hue_inducer'] == hue)
        sorted_df = df[mask].sort_values(by='size')

        sorted_labels = sorted_df[label_column].values
        sorted_sems = sorted_df[sem_column].values if sem_column in sorted_df.columns else np.zeros_like(sorted_labels)
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
    if test_value is not None:
        plt.axhline(y=test_value, color='grey', linestyle='--', linewidth=2)

    plt.xticks(ticks=np.arange(1, len(x_ticks) + 1), labels=x_ticks, rotation=90)
    plt.xlabel("Size (and Hue in brackets)")
    plt.ylabel(y_label)
    plt.title(f"{y_label} for different Hues and Sizes")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'delta_inducer/{file_name}')
    plt.close()


# 创建绘图函数用于不同尺寸
def plot_labels_by_size(label_column, sem_column, y_label, file_name, test_value=None):
    plt.figure(figsize=(20, 10))
    x_ticks = []
    x_positions = []

    for size in unique_sizes:
        mask = (df['size'] == size)
        sorted_df = df[mask].sort_values(by='hue_inducer')

        sorted_labels = sorted_df[label_column].values
        sorted_sems = sorted_df[sem_column].values if sem_column in sorted_df.columns else np.zeros_like(sorted_labels)
        sorted_hues = sorted_df['hue_inducer'].values

        positions = []
        for i, hue in enumerate(sorted_hues):
            color = hue_to_rgb(hue)
            position = len(x_ticks) + 1
            x_ticks.append(f'{hue}')
            x_positions.append(position)
            positions.append(position)
            plt.bar(position, sorted_labels[i], color=color, yerr=sorted_sems[i], capsize=5)

        # 绘制线条连接相同尺寸的点
        plt.plot(positions, sorted_labels, color='black', marker='o', markersize=5, markerfacecolor='black')

        x_ticks.append(f'(Size {size})')
        x_positions.append(len(x_ticks) + 0.5)

    # 绘制test_color对应的虚线
    if test_value is not None:
        plt.axhline(y=test_value, color='grey', linestyle='--', linewidth=2)

    plt.xticks(ticks=np.arange(1, len(x_ticks) + 1), labels=x_ticks, rotation=90)
    plt.xlabel("Hue (and Size in brackets)")
    plt.ylabel(y_label)
    plt.title(f"{y_label} for different Sizes and Hues")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'delta_size/{file_name}')
    plt.close()


# 创建保存目录
if not os.path.exists('delta_inducer'):
    os.makedirs('delta_inducer')
if not os.path.exists('delta_size'):
    os.makedirs('delta_size')

# 绘制 ΔL, Δa, Δb 和 ΔE 图表 (按hue)
plot_labels('ΔL', 'ΔL_sem', 'ΔL', 'ΔL_vs_diff_hue.png')
plot_labels('Δa', 'Δa_sem', 'Δa', 'Δa_vs_diff_hue.png')
plot_labels('Δb', 'Δb_sem', 'Δb', 'Δb_vs_diff_hue.png')
plot_labels('ΔE', 'ΔE_sem', 'ΔE', 'ΔE_vs_diff_hue.png')

# 绘制 ΔL, Δa, Δb 和 ΔE 图表 (按size)
plot_labels_by_size('ΔL', 'ΔL_sem', 'ΔL', 'ΔL_vs_diff_size.png')
plot_labels_by_size('Δa', 'Δa_sem', 'Δa', 'Δa_vs_diff_size.png')
plot_labels_by_size('Δb', 'Δb_sem', 'Δb', 'Δb_vs_diff_size.png')
plot_labels_by_size('ΔE', 'ΔE_sem', 'ΔE', 'ΔE_vs_diff_size.png')

print("Plots generated successfully.")
