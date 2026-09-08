"""Offline ML pipeline: clean, eda, train, tune, check."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
PLOTS = ROOT / "plots"

DROP_COLS = ["PoolQC", "MiscFeature", "Alley", "Fence", "Id"]
NONE_COLS = [
    "FireplaceQu", "MasVnrType",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtExposure", "BsmtFinType2", "BsmtQual", "BsmtCond", "BsmtFinType1",
]


def step_clean() -> None:
    train = pd.read_csv(DATA / "train.csv")
    print("train:", train.shape)
    test = pd.read_csv(DATA / "test.csv")
    print("test:", test.shape)

    missing = train.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print("cols with missing:", len(missing))
    print(missing.head(20))
    print("\nmissing %:")
    print((missing / len(train) * 100).round(1).head(20))

    train = train.drop(columns=DROP_COLS)
    for col in NONE_COLS:
        train[col] = train[col].fillna("Missing")
    train["GarageYrBlt"] = train["GarageYrBlt"].fillna(0)
    train["MasVnrArea"] = train["MasVnrArea"].fillna(0)
    train["LotFrontage"] = train["LotFrontage"].fillna(train["LotFrontage"].median())
    train = train.dropna(subset=["Electrical"])

    left = int(train.isna().sum().sum())
    print("rows:", len(train), "cols:", train.shape[1], "missing_left:", left)
    if left != 0:
        print(train.isna().sum()[train.isna().sum() > 0].sort_values(ascending=False))
        raise SystemExit("clean failed: still have NaNs")

    out = DATA / "train_clean.csv"
    train.to_csv(out, index=False)
    print("saved:", out)


def step_eda() -> None:
    PLOTS.mkdir(exist_ok=True)
    train = pd.read_csv(DATA / "train_clean.csv")

    plt.figure(figsize=(8, 5))
    sns.histplot(train["SalePrice"], bins=40)
    plt.title("SalePrice distribution")
    plt.xlabel("price")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(PLOTS / "saleprice_hist.png")
    plt.close()
    print("saved:", PLOTS / "saleprice_hist.png")

    num = train.select_dtypes(include=["number"])
    corr = num.corr()["SalePrice"].sort_values(ascending=False)
    print("\ncorr with SalePrice (top-10):")
    print(corr.head(10))
    print("\nweak / negative (bottom-5):")
    print(corr.tail(5))

    top_cols = corr.head(10).index
    plt.figure(figsize=(10, 8))
    sns.heatmap(train[top_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation: SalePrice and top features")
    plt.tight_layout()
    plt.savefig(PLOTS / "corr_heatmap.png")
    plt.close()
    print("saved:", PLOTS / "corr_heatmap.png")

    for col, name in [("GrLivArea", "area_vs_price.png"), ("OverallQual", "qual_vs_price.png")]:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=train, x=col, y="SalePrice", alpha=0.4)
        plt.title(f"SalePrice vs {col}")
        plt.tight_layout()
        plt.savefig(PLOTS / name)
        plt.close()
        print("saved:", PLOTS / name)


def _xy(train: pd.DataFrame):
    X = train.select_dtypes(include=["number"]).drop(columns=["SalePrice"])
    y = train["SalePrice"]
    return X, y


def step_train() -> None:
    MODELS.mkdir(exist_ok=True)
    train = pd.read_csv(DATA / "train_clean.csv")
    X, y = _xy(train)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lin = LinearRegression()
    lin.fit(X_train, y_train)
    pred = lin.predict(X_valid)
    mae = mean_absolute_error(y_valid, pred)
    rmse = float(np.sqrt(mean_squared_error(y_valid, pred)))
    r2 = r2_score(y_valid, pred)
    print("Linear Regression:")
    print(f"MAE:  {mae:,.0f}")
    print(f"RMSE: {rmse:,.0f}")
    print(f"R2:   {r2:.3f}")
    print("\nfact vs pred (first 5):")
    print(pd.DataFrame({"fact": y_valid.values[:5], "pred": pred[:5].round(0)}))

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_valid)
    mae = mean_absolute_error(y_valid, pred)
    rmse = float(np.sqrt(mean_squared_error(y_valid, pred)))
    r2 = r2_score(y_valid, pred)
    print("\nRandom Forest:")
    print(f"MAE:  {mae:,.0f}")
    print(f"RMSE: {rmse:,.0f}")
    print(f"R2:   {r2:.3f}")

    full = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    full.fit(X, y)
    out = MODELS / "rf_house_prices.joblib"
    joblib.dump({"model": full, "features": X.columns.tolist()}, out)
    print("\nsaved:", out)
    print("features:", len(X.columns))


def step_tune() -> None:
    MODELS.mkdir(exist_ok=True)
    train = pd.read_csv(DATA / "train_clean.csv")
    X, y = _xy(train)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
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


def step_check() -> None:
    path = MODELS / "rf_house_prices_best.joblib"
    bundle = joblib.load(path)
    model = bundle["model"]
    features = bundle["features"]
    params = bundle["params"]
    stored = bundle.get("test_metrics")

    train = pd.read_csv(DATA / "train_clean.csv")
    missing = int(train.isna().sum().sum())
    if missing:
        raise SystemExit(f"FAIL: train_clean still has {missing} NaNs - re-run clean")

    expected = train.select_dtypes(include=["number"]).drop(columns=["SalePrice"]).columns.tolist()
    if list(features) != expected:
        raise SystemExit("FAIL: saved feature list != current numeric columns")

    gp = model.get_params()
    for key in ("n_estimators", "max_depth", "min_samples_leaf"):
        if gp.get(key) != params.get(key):
            raise SystemExit(f"FAIL: model.{key}={gp.get(key)} != params.{key}={params.get(key)}")

    X = train[features]
    y = train["SalePrice"]
    smoke = model.predict(X.head(5))
    if len(smoke) != 5 or not np.isfinite(smoke).all():
        raise SystemExit("FAIL: smoke predict invalid")
    print("artifact:", path)
    print("params:", params)
    print("smoke_predict_ok:", [float(x) for x in np.round(smoke, 0)])
    if stored:
        print("stored_test_metrics:", stored)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
        raise SystemExit("FAIL: bundle missing test_metrics - re-run tune")
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


STEPS = {
    "clean": step_clean,
    "eda": step_eda,
    "train": step_train,
    "tune": step_tune,
    "check": step_check,
}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="House-prices ML pipeline")
    p.add_argument(
        "step",
        nargs="?",
        default="all",
        choices=[*STEPS.keys(), "all"],
        help="clean | eda | train | tune | check | all",
    )
    args = p.parse_args(argv)
    order = list(STEPS) if args.step == "all" else [args.step]
    for name in order:
        print(f"\n=== {name} ===")
        STEPS[name]()


if __name__ == "__main__":
    main(sys.argv[1:])