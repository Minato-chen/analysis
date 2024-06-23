import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 读取数据
df = pd.read_csv('../updated_averaged_data_012.csv')

# 提取自变量
X = df[['size', 'L_inducer', 'a_inducer', 'b_inducer']]

# 添加截距项
X = sm.add_constant(X)

# 计算VIF
vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# 输出VIF结果
print("VIF Results:")
print(vif_data)

# 分别对L_label_mean, a_label_mean, b_label_mean进行回归分析
y_L = df['L_label_mean']
y_a = df['a_label_mean']
y_b = df['b_label_mean']

# 创建回归模型
model_L = sm.OLS(y_L, X).fit()
model_a = sm.OLS(y_a, X).fit()
model_b = sm.OLS(y_b, X).fit()

# 打印回归结果
print("\nRegression Results for L_label_mean:")
print(model_L.summary())

print("\nRegression Results for a_label_mean:")
print(model_a.summary())

print("\nRegression Results for b_label_mean:")
print(model_b.summary())
