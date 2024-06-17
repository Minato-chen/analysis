import pandas as pd

# 读取两个CSV文件
averaged_data = pd.read_csv('averaged_data.csv')
inducer_color_reference = pd.read_csv('inducer_color_reference.csv')

# 创建一个新的DataFrame来存储结果
result_data = averaged_data.copy()

# 遍历inducer_color_reference中的每一行，找到averaged_data中的匹配行并更新L_inducer, a_inducer, b_inducer
for index, row in inducer_color_reference.iterrows():
    R = row['R_inducer']
    G = row['G_inducer']
    B = row['B_inducer']

    # 找到averaged_data中对应的行
    matched_rows = averaged_data[(averaged_data['R_inducer'] == R) &
                                 (averaged_data['G_inducer'] == G) &
                                 (averaged_data['B_inducer'] == B)]

    if not matched_rows.empty:
        # 更新对应行的L_inducer, a_inducer, b_inducer
        averaged_data.loc[matched_rows.index, 'L_inducer'] = row['L_inducer']
        averaged_data.loc[matched_rows.index, 'a_inducer'] = row['a_inducer']
        averaged_data.loc[matched_rows.index, 'b_inducer'] = row['b_inducer']

# 保存更新后的averaged_data到新的CSV文件
averaged_data.to_csv('updated_averaged_data.csv', index=False)
