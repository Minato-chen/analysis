import pandas as pd

# 读取两个CSV文件
add_data = pd.read_csv('training_test.csv')
inducer_color_reference = pd.read_csv('inducer_color_reference.csv')

# 创建一个新的DataFrame来存储结果
result_data = add_data.copy()

# 遍历inducer_color_reference中的每一行，找到add_data中的匹配行并更新相应的列
for index, row in inducer_color_reference.iterrows():
    R = row['R_inducer']
    G = row['G_inducer']
    B = row['B_inducer']
    hue = row['hue_inducer']

    # 找到add_data中对应的行
    matched_rows = add_data[(add_data['R_inducer'] == R) &
                            (add_data['G_inducer'] == G) &
                            (add_data['B_inducer'] == B)]

    if not matched_rows.empty:
        # 更新对应行的L_inducer, a_inducer, b_inducer, hue_inducer, L_test, a_test, b_test
        add_data.loc[matched_rows.index, 'L_inducer'] = row['L_inducer']
        add_data.loc[matched_rows.index, 'a_inducer'] = row['a_inducer']
        add_data.loc[matched_rows.index, 'b_inducer'] = row['b_inducer']
        add_data.loc[matched_rows.index, 'hue_inducer'] = row['hue_inducer']
        add_data.loc[matched_rows.index, 'L_test'] = 80
        add_data.loc[matched_rows.index, 'a_test'] = -90
        add_data.loc[matched_rows.index, 'b_test'] = 90

# 保存更新后的add_data到新的CSV文件
# add_data.to_csv('updated_averaged_data_2.csv', index=False)
add_data.to_csv('training_data.csv', index=False)

print("数据处理完成，结果已保存")
