import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import deltaE_cie76
from adjustText import adjust_text

# 从CSV文件中读取数据
df = pd.read_csv('../updated_averaged_data_012.csv')

# 提取数据
test_color = np.array([80, -90, 90])
inducer_colors = df[['L_inducer', 'a_inducer', 'b_inducer']].values
labeled_colors = df[['L_label_mean', 'a_label_mean', 'b_label_mean']].values
bar_colors = df[['R_inducer', 'G_inducer', 'B_inducer']].values / 255  # 归一化到0-1之间
hue_inducer = df['hue_inducer'].values
test_rgb = np.array([0, 235, 0]) / 255
# 计算ΔE值
delta_E_values = np.array(
    [deltaE_cie76(test_color, result_color) for result_color in labeled_colors]
)

# 提取独特的inducer颜色
unique_inducers = df[['L_inducer', 'a_inducer', 'b_inducer']].drop_duplicates().values
unique_sizes = df['size'].drop_duplicates().values
os.makedirs('inducer_unique_plots', exist_ok=True)


# 定义随机偏移量函数
def add_random_jitter(arr, jitter_strength=0.5):
    return arr + np.random.normal(scale=jitter_strength, size=arr.shape)


for inducer in unique_inducers:
    mask = (df[['L_inducer', 'a_inducer', 'b_inducer']] == inducer).all(axis=1)
    size_mask = df['size'] == df[mask]['size'].values[0]
    bar_colors_masked = bar_colors[mask & size_mask]
    plt.figure(figsize=(10, 5))

    # 绘制label颜色，添加随机偏移量
    a_label_jittered = add_random_jitter(df['a_label_mean'][mask])
    b_label_jittered = add_random_jitter(df['b_label_mean'][mask])
    plt.scatter(a_label_jittered, b_label_jittered, c='black', marker='o', label='Labeled Colors', linewidth=0.5,
                alpha=0.7)

    # 绘制inducer颜色
    plt.scatter(inducer[1], inducer[2], c=bar_colors_masked, marker='x', s=70, label='Inducer Color')

    # 绘制test_color,参数单个颜色用color，多个颜色用c
    plt.scatter(test_color[1], test_color[2], color=test_rgb, marker='o', s=70, label='Test Color')

    # 添加虚线（直线）将test color和inducer color连起来
    plt.plot([inducer[1], test_color[1]], [inducer[2], test_color[2]], linestyle='--', color='gray')

    # 标注size并优化文本标签位置避免重叠
    texts = []
    for i, (a, b, size) in enumerate(zip(a_label_jittered, b_label_jittered, df['size'][mask])):
        if '.' in str(size):
            label = f'{size:.1f}'
        else:
            label = f'{int(size)}'
        texts.append(plt.text(a, b, label, fontsize=8, ha='right', va='bottom'))

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray'))

    plt.xlabel("a")
    plt.ylabel("b")
    plt.axis('equal')  # 确保横纵坐标刻度宽度相同
    inducer_int = tuple(int(x) for x in inducer)
    plt.title(f"For inducer {inducer_int}(or {int(hue_inducer[mask][0])}°) with different sizes")
    plt.legend()

    # 保留网格线
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(
        f'inducer_unique_plots/inducer_{inducer[0]}_{inducer[1]}_{inducer[2]}_hue_{int(hue_inducer[mask][0])}.png')
    plt.close()
