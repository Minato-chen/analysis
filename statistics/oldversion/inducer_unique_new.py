import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import deltaE_cie76
from adjustText import adjust_text
import matplotlib.colors as mcolors

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

# 创建一个颜色映射
cmap = mcolors.LinearSegmentedColormap.from_list("", ["blue", "gray", "red"])

# 获取size的最大值和最小值
min_size = df['size'].min()
max_size = df['size'].max()


# 定义随机偏移量函数
def add_random_jitter(arr, jitter_strength=0.5):
    return arr + np.random.normal(scale=jitter_strength, size=arr.shape)


for inducer in unique_inducers:
    mask = (df[['L_inducer', 'a_inducer', 'b_inducer']] == inducer).all(axis=1)
    size_mask = df['size'] == df[mask]['size'].values[0]
    bar_colors_masked = bar_colors[mask & size_mask]
    plt.figure(figsize=(8, 8))  # 设置图形为正方形

    # 绘制label颜色，添加随机偏移量
    sizes = df['size'][mask]
    normalized_sizes = (sizes - min_size) / (max_size - min_size)  # 归一化大小
    colors = cmap(normalized_sizes)  # 将归一化的大小映射到颜色
    label_a = add_random_jitter(df['a_label_mean'][mask] - test_color[1])
    label_b = add_random_jitter(df['b_label_mean'][mask] - test_color[2])
    scatter = plt.scatter(label_a, label_b, c=colors, marker='o', linewidth=0.5, alpha=0.7, label='Label Colors')

    # 绘制test_color，放在图的正中间
    plt.scatter(0, 0, color=test_rgb, marker='o', s=70, label='Test Color')

    # 添加虚线（直线）将test color和inducer color连起来
    plt.plot([0, inducer[1] - test_color[1]], [0, inducer[2] - test_color[2]], linestyle='--', color='gray')

    # 标注size并优化文本标签位置避免重叠
    texts = []
    for i, (a, b, size) in enumerate(zip(label_a, label_b, df['size'][mask])):
        if '.' in str(size):
            label = f'{size:.1f}'
        else:
            label = f'{int(size)}'
        texts.append(
            plt.text(a, b, label, fontsize=12, ha='right', va='bottom', color='gray', alpha=0.7))  # 设置文本颜色为灰色，透明度为0.7

    plt.xlabel("a", fontsize=18)  # 字体大小调到18
    plt.ylabel("b", fontsize=18)  # 字体大小调到18

    # 设置坐标轴范围
    all_a = np.append(label_a, [0, inducer[1] - test_color[1]])
    all_b = np.append(label_b, [0, inducer[2] - test_color[2]])
    buffer = 10  # 给坐标轴添加一些缓冲区域
    plt.xlim(all_a.min() - buffer, all_a.max() + buffer)
    plt.ylim(all_b.min() - buffer, all_b.max() + buffer)

    plt.axis('equal')  # 确保横纵坐标刻度宽度相同
    inducer_int = tuple(int(x) for x in inducer)
    plt.title(f"For inducer {inducer_int}(or {int(hue_inducer[mask][0])}°) with different sizes",
              fontsize=18)  # 标题字体大小调到18
    plt.legend(fontsize=16)  # 图例字体大小调到16

    # 保留网格线
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(
        f'inducer_unique_plots/inducer_{inducer[0]}_{inducer[1]}_{inducer[2]}_hue_{int(hue_inducer[mask][0])}.png')
    plt.close()
