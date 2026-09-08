from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

train = pd.read_csv(DATA / "train_clean.csv")
X = train.select_dtypes(include=["number"]).drop(columns=["SalePrice"])
y = train["SalePrice"]

# 60% train / 20% valid (select params) / 20% test (report once)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
)

candidates = [
    {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
    {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
    {"n_estimators": 200, "max_depth": 15, "min_samples_leaf": 1},
    {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 2},
    {"n_estimators": 300, "max_depth": 20, "min_samples_leaf": 2},
]

rows = []
best_params = None
best_valid_rmse = float("inf")

print("Tuning on VALID only (test is held out)...\n")

for i, params in enumerate(candidates, start=1):
    model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)

    mae = mean_absolute_error(y_valid, pred)
    rmse = float(np.sqrt(mean_squared_error(y_valid, pred)))
    r2 = r2_score(y_valid, pred)

    rows.append({**params, "valid_MAE": round(mae), "valid_RMSE": round(rmse), "valid_R2": round(r2, 3)})
    print(f"[{i}/{len(candidates)}] {params} -> valid MAE={mae:,.0f} RMSE={rmse:,.0f} R2={r2:.3f}")

    if rmse < best_valid_rmse:
        best_valid_rmse = rmse
        best_params = params.copy()

print("\nValid leaderboard:")
print(pd.DataFrame(rows).sort_values("valid_RMSE"))
print("\nSelected by valid RMSE:", best_params)

# Final reported score: fit on train+valid, evaluate once on untouched test
X_tv = pd.concat([X_train, X_valid])
y_tv = pd.concat([y_train, y_valid])
selected = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params)
selected.fit(X_tv, y_tv)
test_pred = selected.predict(X_test)

test_mae = float(mean_absolute_error(y_test, test_pred))
test_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
test_r2 = float(r2_score(y_test, test_pred))

print("\nTEST (never used for selection):")
print(f"MAE:  {test_mae:,.0f}")
print(f"RMSE: {test_rmse:,.0f}")
print(f"R2:   {test_r2:.3f}")

# Production artifact: fit on all rows with selected params
final = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params)
final.fit(X, y)

meta = {
    "model": final,
    "features": X.columns.tolist(),
    "params": best_params,
    "selection": "min valid RMSE",
    "test_metrics": {"MAE": test_mae, "RMSE": test_rmse, "R2": test_r2},
    "split": {"train": 0.6, "valid": 0.2, "test": 0.2, "random_state": 42},
}
out = MODELS / "rf_house_prices_best.joblib"
joblib.dump(meta, out)
print("\nsaved:", out)
