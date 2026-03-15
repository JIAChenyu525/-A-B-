import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 100000

user_ids = np.arange(1, n_samples + 1)

groups = np.random.choice(['A', 'B'], size=n_samples, p=[0.5, 0.5])

impressions = np.ones(n_samples, dtype=int)

clicks = np.where(
    groups == 'A',
    np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
    np.random.choice([0, 1], size=n_samples, p=[0.94, 0.06])
)

purchases = np.where(
    clicks == 1,
    np.random.choice([0, 1], size=n_samples, p=[0.80, 0.20]),
    0
)

df = pd.DataFrame({
    'user_id': user_ids,
    'group': groups,
    'impression': impressions,
    'click': clicks,
    'purchase': purchases
})

df.to_csv('experiment_data.csv', index=False)

print("数据生成完成！")
print(f"总记录数: {len(df)}")
print(f"\n各组样本数量:")
print(df['group'].value_counts())
print(f"\n各组点击率:")
print(df.groupby('group')['click'].mean())
print(f"\n各组购买转化率（点击后）:")
print(df[df['click'] == 1].groupby('group')['purchase'].mean())
print(f"\n数据前5行:")
print(df.head())
