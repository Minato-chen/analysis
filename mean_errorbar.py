import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('./csv/exp_data.csv')

# 定义需要分组的列
group_columns = ['size', 'R_test', 'G_test', 'B_test', 'R_inducer', 'G_inducer', 'B_inducer']

# 定义需要取平均值和标准误差的列
mean_columns = ['R_label', 'G_label', 'B_label', 'L_label', 'a_label', 'b_label']

# 对这些列进行分组并计算平均值和标准误差
grouped = df.groupby(group_columns)[mean_columns].agg(['mean', 'sem']).reset_index()

# 展平多层列索引
grouped.columns = ['_'.join(col).strip() if col[1] else col[0] for col in grouped.columns.values]

# 重新排序列，确保mean在一起，sem在一起
ordered_columns = group_columns + \
    [f'{col}_mean' for col in mean_columns] + \
    [f'{col}_sem' for col in mean_columns]

# 确保只保留需要的列
grouped = grouped[ordered_columns]

# 对RGB相关的列进行四舍五入，并限制在[0,255]之间
rgb_columns_mean = ['R_label_mean', 'G_label_mean', 'B_label_mean']
rgb_columns_sem = ['R_label_sem', 'G_label_sem', 'B_label_sem']

grouped[rgb_columns_mean] = grouped[rgb_columns_mean].round().clip(0, 255)
grouped[rgb_columns_sem] = grouped[rgb_columns_sem].round().clip(0, 255)

# 对Lab相关的列保留两位小数
lab_columns_mean = ['L_label_mean', 'a_label_mean', 'b_label_mean']
lab_columns_sem = ['L_label_sem', 'a_label_sem', 'b_label_sem']

grouped[lab_columns_mean] = grouped[lab_columns_mean].round(2)
grouped[lab_columns_sem] = grouped[lab_columns_sem].round(2)

# 保存结果到新的CSV文件
grouped.to_csv('averaged_data_with_error_bars.csv', index=False)

print("数据处理完成，结果已保存到averaged_data_with_error_bars.csv")

