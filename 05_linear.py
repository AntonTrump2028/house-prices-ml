from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
train = pd.read_csv(ROOT / "data" / "train_clean.csv")

X = train.select_dtypes(include=["number"]).drop(columns=["SalePrice"])
y = train["SalePrice"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_valid)

mae = mean_absolute_error(y_valid, pred)
rmse = np.sqrt(mean_squared_error(y_valid, pred))
r2 = r2_score(y_valid, pred)

print("Linear Regression:")
print(f"MAE:  {mae:,.0f}")
print(f"RMSE: {rmse:,.0f}")
print(f"R2:   {r2:.3f}")
print("\nfact vs pred (first 5):")
print(pd.DataFrame({"fact": y_valid.values[:5], "pred": pred[:5].round(0)}))
