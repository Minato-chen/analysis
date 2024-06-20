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

# 提取独特的inducer颜色
unique_inducers = df[['L_inducer', 'a_inducer', 'b_inducer']].drop_duplicates().values
unique_sizes = df['size'].drop_duplicates().values
os.makedirs('inducer_unique_plots', exist_ok=True)

for inducer in unique_inducers:
    mask = (df[['L_inducer', 'a_inducer', 'b_inducer']] == inducer).all(axis=1)
    size_mask = df['size'] == df[mask]['size'].values[0]
    bar_colors_masked = bar_colors[mask & size_mask]
    plt.figure(figsize=(10, 5))

    # 绘制label颜色
    plt.scatter(df['a_label_mean'][mask], df['b_label_mean'][mask], c='black', marker='x', label='Labeled Colors')

    # 绘制inducer颜色和test_color
    plt.scatter(inducer[1], inducer[2], c=bar_colors_masked, label='Inducer Color')

    # 绘制test_color,参数单个颜色用color，多个颜色用c
    plt.scatter(test_color[1], test_color[2], color=test_rgb, marker='o', label='Test Color')

    # 标注size
    for i, size in enumerate(df['size'][mask]):
        plt.text(df['a_label_mean'][mask].values[i], df['b_label_mean'][mask].values[i], f'{int(size)}mm', fontsize=8, ha='right')
    plt.xlabel("a")
    plt.ylabel("b")
    inducer_int = tuple(int(x) for x in inducer)
    plt.title(f"For inducer {inducer_int}(or {int(hue_inducer[mask][0])}°) with different sizes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'inducer_unique_plots/inducer_{inducer[0]}_{inducer[1]}_{inducer[2]}.png')
    plt.close()
