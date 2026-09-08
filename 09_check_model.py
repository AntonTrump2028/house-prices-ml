from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
PATH = ROOT / "models" / "rf_house_prices_best.joblib"

bundle = joblib.load(PATH)
model = bundle["model"]
features = bundle["features"]
params = bundle["params"]
stored = bundle.get("test_metrics")

train = pd.read_csv(ROOT / "data" / "train_clean.csv")
missing = int(train.isna().sum().sum())
if missing:
    raise SystemExit(f"FAIL: train_clean still has {missing} NaNs — re-run 03_clean.py")

expected_features = train.select_dtypes(include=["number"]).drop(columns=["SalePrice"]).columns.tolist()
if list(features) != expected_features:
    raise SystemExit("FAIL: saved feature list != current numeric columns")

# Artifact hyperparams must match metadata
gp = model.get_params()
for key in ("n_estimators", "max_depth", "min_samples_leaf"):
    if gp.get(key) != params.get(key):
        raise SystemExit(f"FAIL: model.{key}={gp.get(key)} != params.{key}={params.get(key)}")

X = train[features]
y = train["SalePrice"]

sample = X.head(5)
smoke = model.predict(sample)
if len(smoke) != 5 or not np.isfinite(smoke).all():
    raise SystemExit("FAIL: smoke predict invalid")
print("artifact:", PATH)
print("params:", params)
print("smoke_predict_ok:", [float(x) for x in np.round(smoke, 0)])
if stored:
    print("stored_test_metrics:", stored)

# Honest re-check: same 60/20/20 recipe as 08
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)
X_tv = pd.concat([X_train, X_valid])
y_tv = y.loc[X_tv.index]

honest = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
honest.fit(X_tv, y_tv)
pred = honest.predict(X_test)

mae = float(mean_absolute_error(y_test, pred))
rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
r2 = float(r2_score(y_test, pred))

print("\nhonest TEST retrain:")
print(f"MAE:  {mae:,.0f}")
print(f"RMSE: {rmse:,.0f}")
print(f"R2:   {r2:.3f}")

if not stored:
    raise SystemExit("FAIL: bundle missing test_metrics — re-run 08_tune_forest.py")

if abs(mae - stored["MAE"]) > 1.0 or abs(rmse - stored["RMSE"]) > 1.0 or abs(r2 - stored["R2"]) > 1e-4:
    raise SystemExit(
        f"FAIL: honest metrics != stored_test_metrics "
        f"(honest={mae}/{rmse}/{r2}, stored={stored})"
    )
print("ASSERT OK: honest metrics match stored_test_metrics")

_, X_hold, _, y_hold = train_test_split(X, y, test_size=0.2, random_state=42)
l_mae = float(mean_absolute_error(y_hold, model.predict(X_hold)))
l_r2 = float(r2_score(y_hold, model.predict(X_hold)))
print("\nmisleading full-artifact-on-holdout:")
print(f"MAE:  {l_mae:,.0f} | R2: {l_r2:.3f}  <- do NOT report this")
print("\nCHECK PASSED")
