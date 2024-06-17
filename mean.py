import pandas as pd

# 读取CSV文件
df = pd.read_csv('./csv/exp_data.csv')

# 定义需要分组的列
group_columns = ['size', 'R_test', 'G_test', 'B_test', 'R_inducer', 'G_inducer', 'B_inducer']

# 定义需要取平均值的列
mean_columns = ['R_label', 'G_label', 'B_label', 'L_label', 'a_label', 'b_label']

# 对这些列进行分组并计算平均值
df_grouped = df.groupby(group_columns)[mean_columns].mean().reset_index()

# 将结果保存到新的CSV文件
df_grouped.to_csv('averaged_data.csv', index=False)

print("数据处理完成，结果已保存到averaged_data.csv")