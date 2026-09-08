from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

train = pd.read_csv(ROOT / "data" / "train_clean.csv")

X = train.select_dtypes(include=["number"]).drop(columns=["SalePrice"])
y = train["SalePrice"]

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y)

out = MODELS / "rf_house_prices.joblib"
joblib.dump({"model": model, "features": X.columns.tolist()}, out)
print("saved:", out)
print("features:", len(X.columns))
