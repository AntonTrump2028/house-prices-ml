import pandas as pd

train = pd.read_csv(r"D:\Proj\data\train.csv")

missing = train.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

print("Колонок с пропусками:", len(missing))
print(missing.head(20))
print("\nДоля пропусков (%):")
print((missing / len(train) * 100).round(1).head(20))