from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

train = pd.read_csv(DATA / "train.csv")

drop_cols = ["PoolQC", "MiscFeature", "Alley", "Fence", "Id"]
train = train.drop(columns=drop_cols)

# Use Missing (not None): pandas read_csv treats token None as NaN on CSV round-trip.
none_cols = [
    "FireplaceQu", "MasVnrType",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtExposure", "BsmtFinType2", "BsmtQual", "BsmtCond", "BsmtFinType1",
]
for col in none_cols:
    train[col] = train[col].fillna("Missing")

# LotFrontage median on full train before model split = mild imputation leakage.
train["GarageYrBlt"] = train["GarageYrBlt"].fillna(0)
train["MasVnrArea"] = train["MasVnrArea"].fillna(0)
train["LotFrontage"] = train["LotFrontage"].fillna(train["LotFrontage"].median())

train = train.dropna(subset=["Electrical"])

left = int(train.isna().sum().sum())
print("rows:", len(train))
print("cols:", train.shape[1])
print("missing_left:", left)
if left != 0:
    print(train.isna().sum()[train.isna().sum() > 0].sort_values(ascending=False))
    raise SystemExit("clean failed: still have NaNs")

out = DATA / "train_clean.csv"
train.to_csv(out, index=False)
print("saved:", out)
