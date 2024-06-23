import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 从CSV文件中读取数据
df = pd.read_csv('../updated_averaged_data_012.csv')

# 提取数据
test_color = np.array([80, -90, 90])
size = df['size']
inducer_colors = df[['L_inducer', 'a_inducer', 'b_inducer']].values
labeled_colors = df[['L_label_mean', 'a_label_mean', 'b_label_mean']].values
bar_colors = df[['R_inducer', 'G_inducer', 'B_inducer']].values / 255  # 归一化到0-1之间
hue_inducer = df['hue_inducer'].values
test_rgb = np.array([0, 235, 0]) / 255
# 示例数据（假设已有实验数据）
data = pd.DataFrame({
    'size': [5, 10, 15, 20, 25],
    'L_diff': [2, 5, 10, 15, 20],
    'a_diff': [3, 6, 12, 18, 24],
    'b_diff': [1, 4, 8, 12, 16]
})

# 绘制LAB差异与格子大小的关系图
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
sns.lineplot(data=data, x='size', y='L_diff')
plt.title('L_diff vs Grid Size')
plt.xlabel('Grid Size')
plt.ylabel('L_diff')

plt.subplot(1, 3, 2)
sns.lineplot(data=data, x='size', y='a_diff')
plt.title('a_diff vs Grid Size')
plt.xlabel('Grid Size')
plt.ylabel('a_diff')

plt.subplot(1, 3, 3)
sns.lineplot(data=data, x='size', y='b_diff')
plt.title('b_diff vs Grid Size')
plt.xlabel('Grid Size')
plt.ylabel('b_diff')

plt.tight_layout()
plt.show()

# 观察不同格子大小下的LAB值变化
print(data)
