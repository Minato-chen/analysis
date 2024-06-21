import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import deltaE_cie76
from matplotlib.lines import Line2D

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
    plt.figure(figsize=(7, 7), dpi=600)  # 设置图形为正方形，并提高解析度

    # 绘制inducer颜色，用bar_color标记，缩小点的大小
    plt.scatter(df['a_inducer'][mask], df['b_inducer'][mask], c=bar_colors[mask], label='Inducer Colors', s=15)

    # 在inducer颜色旁边标记hue_inducer值
    for i, hue in enumerate(hue_inducer[mask]):
        plt.annotate(f'{int(hue)}°', (df['a_inducer'][mask].values[i], df['b_inducer'][mask].values[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='black')

    # 绘制label颜色，并用bar_color标记，使用偏移区分重合点
    for i in range(len(df[mask])):
        offset = i * 0.1  # 添加偏移
        plt.scatter(df['a_label_mean'][mask].values[i] + offset, df['b_label_mean'][mask].values[i] + offset,
                    facecolors='none', edgecolors=[bar_colors[mask][i]], marker='o', s=15, linewidth=0.25)
        plt.scatter(df['a_label_mean'][mask].values[i] + offset, df['b_label_mean'][mask].values[i] + offset,
                    c=[bar_colors[mask][i]], marker='o', s=3)

        # 连接test_color和label color，改为虚线，调大间隔
        plt.plot([test_color[1], df['a_label_mean'][mask].values[i] + offset],
                 [test_color[2], df['b_label_mean'][mask].values[i] + offset],
                 color=bar_colors[mask][i], linestyle=(0, (5, 5)), linewidth=0.5)

        # 将label color与对应的inducer color连接起来，改为虚线，调大间隔
        plt.plot([df['a_inducer'][mask].values[i], df['a_label_mean'][mask].values[i] + offset],
                 [df['b_inducer'][mask].values[i], df['b_label_mean'][mask].values[i] + offset],
                 color=bar_colors[mask][i], linestyle=(0, (5, 5)), linewidth=0.5)

    # 绘制test_color，并缩小点的大小
    plt.scatter(test_color[1], test_color[2], color=test_rgb, marker='o', label='Test Color 135°', s=15)

    # 设置横纵坐标比例相同
    plt.gca().set_aspect('equal', adjustable='box')

    # 创建图例元素
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Test Color', markerfacecolor=test_rgb, markersize=5),
    ]

    # 根据hue_inducer添加Inducer Colors和Label Colors的图例
    unique_hues = np.unique(hue_inducer[mask])
    for hue in unique_hues:
        hue_mask = (hue_inducer == hue)
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'Inducer Color {int(hue)}°', markerfacecolor=bar_colors[hue_mask][0], markersize=5))
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'Label Color {int(hue)}°', markerfacecolor='none', markeredgecolor=bar_colors[hue_mask][0], markersize=5))

    # 添加连接线的图例
    legend_elements.append(Line2D([0], [0], linestyle=(0, (5, 5)), color='k', label='Test to Label Color', linewidth=0.5))
    legend_elements.append(Line2D([0], [0], linestyle=(0, (5, 5)), color='k', label='Label to Inducer Color', linewidth=0.5))

    # 显示图例，并放置在图外
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xlabel("a")
    plt.ylabel("b")
    plt.title(f"For size {size} with different inducers")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'size_unique_plots/size_{size}.png', bbox_inches='tight')
    plt.close()
