import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv(r"D:\Proj\data\train_clean.csv")

plt.figure(figsize=(8, 5))
sns.histplot(train["SalePrice"], bins=40)
plt.title("Распределение SalePrice")
plt.xlabel("Цена")
plt.ylabel("Число домов")
plt.tight_layout()
plt.savefig(r"D:\Proj\plots\saleprice_hist.png")
plt.close()
print("Сохранён: plots/saleprice_hist.png")


num = train.select_dtypes(include=["number"])
corr = num.corr()["SalePrice"].sort_values(ascending=False)
print("\nСвязь с SalePrice (топ-10):")
print(corr.head(10))
print("\nСлабее всего / отрицательно (низ-5):")
print(corr.tail(5))


top_cols = corr.head(10).index
plt.figure(figsize=(10, 8))
sns.heatmap(train[top_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Корреляции: SalePrice и близкие признаки")
plt.tight_layout()
plt.savefig(r"D:\Proj\plots\corr_heatmap.png")
plt.close()
print("Сохранён: plots/corr_heatmap.png")

for col, name in [("GrLivArea", "area_vs_price.png"), ("OverallQual", "qual_vs_price.png")]:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=train, x=col, y="SalePrice", alpha=0.4)
    plt.title(f"SalePrice vs {col}")
    plt.tight_layout()
    plt.savefig(fr"D:\Proj\plots\{name}")
    plt.close()
    print(f"Сохранён: plots/{name}")