import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import deltaE_cie76

# 从CSV文件中读取数据
df = pd.read_csv('updated_averaged_data_test.csv')

# 提取数据
inducer_colors = df[['L_inducer', 'a_inducer', 'b_inducer']].values
labeled_colors = df[['L_label', 'a_label', 'b_label']].values
bar_colors = df[['R_inducer', 'G_inducer', 'B_inducer']].values / 255  # 归一化到0-1之间

# 固定的A色Lab值
test_color = np.array([80, -90, 90])  # A色的原始Lab值

# 计算ΔE值
delta_E_values = np.array(
    [deltaE_cie76(test_color, result_color) for result_color in labeled_colors]
)

# 按不同size和不同inducer绘制ΔE值图表
unique_sizes = df['size'].unique()
unique_inducers = df[['L_inducer', 'a_inducer', 'b_inducer']].drop_duplicates().values

for size in unique_sizes:
    mask = df['size'] == size
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(delta_E_values[mask]) + 1), delta_E_values[mask], color=bar_colors[mask])
    plt.xlabel("Inducer Color")
    plt.ylabel("ΔE")
    plt.title(f"ΔE for size {size} with different inducers")
    plt.xticks(range(1, len(delta_E_values[mask]) + 1), labels=[f"({L},{a},{b})" for L, a, b in inducer_colors[mask]], rotation=45)
    plt.tight_layout()
    plt.show()

for inducer in unique_inducers:
    mask = (df[['L_inducer', 'a_inducer', 'b_inducer']] == inducer).all(axis=1)
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(delta_E_values[mask]) + 1), delta_E_values[mask], color=bar_colors[mask])
    plt.xlabel("Size")
    plt.ylabel("ΔE")
    plt.title(f"ΔE for inducer {tuple(inducer)} with different sizes")
    plt.xticks(range(1, len(delta_E_values[mask]) + 1), labels=[f"Size {s}" for s in df['size'][mask]], rotation=45)
    plt.tight_layout()
    plt.show()

# 计算方向变化
delta_a = labeled_colors[:, 1] - test_color[1]
delta_b = labeled_colors[:, 2] - test_color[2]
angles = np.arctan2(delta_b, delta_a)  # 计算角度
angles_degrees = np.degrees(angles)  # 转换为度数

# 按不同size和不同inducer绘制方向变化图
for size in unique_sizes:
    mask = df['size'] == size
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    norm = plt.Normalize(delta_E_values[mask].min(), delta_E_values[mask].max())
    cmap = plt.cm.Reds
    c = ax.scatter(angles[mask], delta_E_values[mask], c=delta_E_values[mask], cmap=cmap, norm=norm, alpha=0.75)
    ax.set_title(f"Direction and ΔE for size {size}")
    ax.set_yticklabels([])  # 隐藏径向标签
    plt.colorbar(c, ax=ax, label="ΔE")
    plt.show()

for inducer in unique_inducers:
    mask = (df[['L_inducer', 'a_inducer', 'b_inducer']] == inducer).all(axis=1)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    norm = plt.Normalize(delta_E_values[mask].min(), delta_E_values[mask].max())
    cmap = plt.cm.Reds
    c = ax.scatter(angles[mask], delta_E_values[mask], c=delta_E_values[mask], cmap=cmap, norm=norm, alpha=0.75)
    ax.set_title(f"Direction and ΔE for inducer {tuple(inducer)}")
    ax.set_yticklabels([])  # 隐藏径向标签
    plt.colorbar(c, ax=ax, label="ΔE")
    plt.show()

# 按不同size和不同inducer绘制L值对比图
for size in unique_sizes:
    mask = df['size'] == size
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(delta_E_values[mask]) + 1), [test_color[0]] * len(delta_E_values[mask]), label="T:L value", linestyle="--", color="red")
    plt.plot(range(1, len(delta_E_values[mask]) + 1), labeled_colors[mask][:, 0], label="Observed T(R):L value", marker="o")
    plt.plot(range(1, len(delta_E_values[mask]) + 1), inducer_colors[mask][:, 0], label="Inducer:L value", marker="x")
    plt.xlabel("Inducer Color")
    plt.ylabel("L*")
    plt.title(f"L* value comparison for size {size} with different inducers")
    plt.legend()
    plt.xticks(range(1, len(delta_E_values[mask]) + 1), labels=[f"({L},{a},{b})" for L, a, b in inducer_colors[mask]], rotation=45)
    plt.tight_layout()
    plt.show()

for inducer in unique_inducers:
    mask = (df[['L_inducer', 'a_inducer', 'b_inducer']] == inducer).all(axis=1)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(delta_E_values[mask]) + 1), [test_color[0]] * len(delta_E_values[mask]), label="T:L value", linestyle="--", color="red")
    plt.plot(range(1, len(delta_E_values[mask]) + 1), labeled_colors[mask][:, 0], label="Observed T(R):L value", marker="o")
    plt.plot(range(1, len(delta_E_values[mask]) + 1), inducer_colors[mask][:, 0], label="Inducer:L value", marker="x")
    plt.xlabel("Size")
    plt.ylabel("L*")
    plt.title(f"L* value comparison for inducer {tuple(inducer)} with different sizes")
    plt.legend()
    plt.xticks(range(1, len(delta_E_values[mask]) + 1), labels=[f"Size {s}" for s in df['size'][mask]], rotation=45)
    plt.tight_layout()
    plt.show()

# 按不同size和不同inducer绘制a值对比图
for size in unique_sizes:
    mask = df['size'] == size
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(delta_E_values[mask]) + 1), [test_color[1]] * len(delta_E_values[mask]), label="T:a*", linestyle="--", color="red")
    plt.plot(range(1, len(delta_E_values[mask]) + 1), labeled_colors[mask][:, 1], label="Observed T(R):a*", marker="o")
    plt.plot(range(1, len(delta_E_values[mask]) + 1), inducer_colors[mask][:, 1], label="Inducer:a*", marker="x")
    plt.xlabel("Inducer Color")
    plt.ylabel("a*")
    plt.title(f"a* value comparison for size {size} with different inducers")
    plt.legend()
    plt.xticks(range(1, len(delta_E_values[mask]) + 1), labels=[f"({L},{a},{b})" for L, a, b in inducer_colors[mask]], rotation=45)
    plt.tight_layout()
    plt.show()

for inducer in unique_inducers:
    mask = (df[['L_inducer', 'a_inducer', 'b_inducer']] == inducer).all(axis=1)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(delta_E_values[mask]) + 1), [test_color[1]] * len(delta_E_values[mask]), label="T:a*", linestyle="--", color="red")
    plt.plot(range(1, len(delta_E_values[mask]) + 1), labeled_colors[mask][:, 1], label="Observed T(R):a*", marker="o")
    plt.plot(range(1, len(delta_E_values[mask]) + 1), inducer_colors[mask][:, 1], label="Inducer:a*", marker="x")
    plt.xlabel("Size")
    plt.ylabel("a*")
    plt.title(f"a* value comparison for inducer {tuple(inducer)} with different sizes")
    plt.legend()
    plt.xticks(range(1, len(delta_E_values[mask]) + 1), labels=[f"Size {s}" for s in df['size'][mask]], rotation=45)
    plt.tight_layout()
    plt.show()

# 按不同size和不同inducer绘制b值对比图
for size in unique_sizes:
    mask = df['size'] == size
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(delta_E_values[mask]) + 1), [test_color[2]] * len(delta_E_values[mask]), label="T:b*", linestyle="--", color="red")
    plt.plot(range(1, len(delta_E_values[mask]) + 1), labeled_colors[mask][:, 2], label="Observed T(R):b*", marker="o")
    plt.plot(range(1, len(delta_E_values[mask]) + 1), inducer_colors[mask][:, 2], label="Inducer:b*", marker="x")
    plt.xlabel("Inducer Color")
    plt.ylabel("b*")
    plt.title(f"b* value comparison for size {size} with different inducers")
    plt.legend()
    plt.xticks(range(1, len(delta_E_values[mask]) + 1), labels=[f"({L},{a},{b})" for L, a, b in inducer_colors[mask]], rotation=45)
    plt.tight_layout()
    plt.show()

for inducer in unique_inducers:
    mask = (df[['L_inducer', 'a_inducer', 'b_inducer']] == inducer).all(axis=1)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(delta_E_values[mask]) + 1), [test_color[2]] * len(delta_E_values[mask]), label="T:b*", linestyle="--", color="red")
    plt.plot(range(1, len(delta_E_values[mask]) + 1), labeled_colors[mask][:, 2], label="Observed T(R):b*", marker="o")
    plt.plot(range(1, len(delta_E_values[mask]) + 1), inducer_colors[mask][:, 2], label="Inducer:b*", marker="x")
    plt.xlabel("Size")
    plt.ylabel("b*")
    plt.title(f"b* value comparison for inducer {tuple(inducer)} with different sizes")
    plt.legend()
    plt.xticks(range(1, len(delta_E_values[mask]) + 1), labels=[f"Size {s}" for s in df['size'][mask]], rotation=45)
    plt.tight_layout()
    plt.show()

# 计算ΔL, Δa, Δb值
delta_L = labeled_colors[:, 0] - test_color[0]
delta_a = labeled_colors[:, 1] - test_color[1]
delta_b = labeled_colors[:, 2] - test_color[2]

# 按不同size和不同inducer绘制ΔL值图表
for size in unique_sizes:
    mask = df['size'] == size
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(delta_L[mask]) + 1), delta_L[mask], color=bar_colors[mask])
    plt.xlabel("Inducer Color")
    plt.ylabel("ΔL")
    plt.title(f"ΔL for size {size} with different inducers")
    plt.xticks(range(1, len(delta_L[mask]) + 1), labels=[f"({L},{a},{b})" for L, a, b in inducer_colors[mask]], rotation=45)
    plt.tight_layout()
    plt.show()

for inducer in unique_inducers:
    mask = (df[['L_inducer', 'a_inducer', 'b_inducer']] == inducer).all(axis=1)
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(delta_L[mask]) + 1), delta_L[mask], color=bar_colors[mask])
    plt.xlabel("Size")
    plt.ylabel("ΔL")
    plt.title(f"ΔL for inducer {tuple(inducer)} with different sizes")
    plt.xticks(range(1, len(delta_L[mask]) + 1), labels=[f"Size {s}" for s in df['size'][mask]], rotation=45)
    plt.tight_layout()
    plt.show()

# 按不同size和不同inducer绘制Δa值图表
for size in unique_sizes:
    mask = df['size'] == size
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(delta_a[mask]) + 1), delta_a[mask], color=bar_colors[mask])
    plt.xlabel("Inducer Color")
    plt.ylabel("Δa")
    plt.title(f"Δa for size {size} with different inducers")
    plt.xticks(range(1, len(delta_a[mask]) + 1), labels=[f"({L},{a},{b})" for L, a, b in inducer_colors[mask]], rotation=45)
    plt.tight_layout()
    plt.show()

for inducer in unique_inducers:
    mask = (df[['L_inducer', 'a_inducer', 'b_inducer']] == inducer).all(axis=1)
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(delta_a[mask]) + 1), delta_a[mask], color=bar_colors[mask])
    plt.xlabel("Size")
    plt.ylabel("Δa")
    plt.title(f"Δa for inducer {tuple(inducer)} with different sizes")
    plt.xticks(range(1, len(delta_a[mask]) + 1), labels=[f"Size {s}" for s in df['size'][mask]], rotation=45)
    plt.tight_layout()
    plt.show()

# 按不同size和不同inducer绘制Δb值图表
for size in unique_sizes:
    mask = df['size'] == size
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(delta_b[mask]) + 1), delta_b[mask], color=bar_colors[mask])
    plt.xlabel("Inducer Color")
    plt.ylabel("Δb")
    plt.title(f"Δb for size {size} with different inducers")
    plt.xticks(range(1, len(delta_b[mask]) + 1), labels=[f"({L},{a},{b})" for L, a, b in inducer_colors[mask]], rotation=45)
    plt.tight_layout()
    plt.show()

for inducer in unique_inducers:
    mask = (df[['L_inducer', 'a_inducer', 'b_inducer']] == inducer).all(axis=1)
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(delta_b[mask]) + 1), delta_b[mask], color=bar_colors[mask])
    plt.xlabel("Size")
    plt.ylabel("Δb")
    plt.title(f"Δb for inducer {tuple(inducer)} with different sizes")
    plt.xticks(range(1, len(delta_b[mask]) + 1), labels=[f"Size {s}" for s in df['size'][mask]], rotation=45)
    plt.tight_layout()
    plt.show()
