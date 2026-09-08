"""House price calculator API — demo estimates, not live bank/comps."""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "app"
MODEL_PATH = ROOT / "models" / "rf_house_prices_best.joblib"
TRAIN_PATH = ROOT / "data" / "train_clean.csv"
GEO_PATH = APP_DIR / "geo_coeffs.json"
RATES_PATH = APP_DIR / "rates.json"
SITE_DIR = ROOT / "site"

app = Flask(__name__, static_folder=None)

@app.after_request
def _cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp


_geo = json.loads(GEO_PATH.read_text(encoding="utf-8"))
_rates = json.loads(RATES_PATH.read_text(encoding="utf-8"))

_bundle = None
_medians = None
_features = None


def _load_or_train():
    global _bundle, _medians, _features
    if _bundle is not None:
        return

    train = pd.read_csv(TRAIN_PATH)
    nums = train.select_dtypes(include=["number"])
    feat_cols = [c for c in nums.columns if c != "SalePrice"]
    _medians = nums[feat_cols].median().to_dict()

    if MODEL_PATH.exists():
        raw = joblib.load(MODEL_PATH)
        if isinstance(raw, dict) and "model" in raw:
            _bundle = raw
            _features = list(raw["features"])
        else:
            _bundle = {"model": raw, "features": feat_cols}
            _features = feat_cols
        print("loaded model:", MODEL_PATH)
    else:
        print("no joblib — training quick RF on", TRAIN_PATH)
        X = train[feat_cols]
        y = train["SalePrice"]
        rf = RandomForestRegressor(
            n_estimators=80, max_depth=20, random_state=42, n_jobs=-1
        )
        rf.fit(X, y)
        _bundle = {"model": rf, "features": feat_cols}
        _features = feat_cols
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(_bundle, MODEL_PATH)
        print("saved quick model:", MODEL_PATH)


def _qual_heuristic(year_built: int, living_area: float) -> int:
    # map age + size into Ames-like OverallQual 1-10
    age = max(0, date.today().year - int(year_built))
    q = 5
    if age < 10:
        q += 2
    elif age < 25:
        q += 1
    elif age > 60:
        q -= 1
    elif age > 90:
        q -= 2
    if living_area >= 2200:
        q += 2
    elif living_area >= 1600:
        q += 1
    elif living_area < 900:
        q -= 1
    return int(np.clip(q, 1, 10))


def _row_from_form(body: dict) -> pd.DataFrame:
    _load_or_train()
    living = float(body.get("living_area") or 1200)
    beds = int(body.get("bedrooms") or 3)
    year = int(body.get("year_built") or 1995)
    qual = _qual_heuristic(year, living)

    row = {f: float(_medians.get(f, 0.0)) for f in _features}

    mapped = {
        "OverallQual": qual,
        "GrLivArea": living,
        "BedroomAbvGr": beds,
        "YearBuilt": year,
        "YearRemodAdd": max(year, int(_medians.get("YearRemodAdd", year))),
        "GarageYrBlt": year,
        "1stFlrSF": living * 0.7,
        "2ndFlrSF": living * 0.3 if beds >= 3 else 0.0,
        "TotalBsmtSF": living * 0.55,
        "FullBath": max(1, min(beds - 1, 3)),
        "TotRmsAbvGrd": max(beds + 2, 4),
        "GarageCars": 2 if living >= 1100 else 1,
        "GarageArea": 480 if living >= 1100 else 280,
        "LotArea": max(living * 6.5, 4000),
    }
    for k, v in mapped.items():
        if k in row:
            row[k] = float(v)

    return pd.DataFrame([[row[f] for f in _features]], columns=_features)


def _geo_mult(country: str, city: str) -> float:
    c = (country or "US").upper()
    base = float(_geo["countries"].get(c, _geo["countries"]["US"]).get("base", 1.0))
    cities = _geo["cities"].get(c, {})
    city_key = (city or "").strip()
    city_m = float(cities.get(city_key, cities.get("default", 1.0)))
    return base * city_m


def _inflate(buy_date_str: str | None) -> float:
    # 0.25%/month from buy_date toward today, capped at +/- 30%
    if not buy_date_str:
        return 1.0
    try:
        bd = datetime.strptime(buy_date_str[:10], "%Y-%m-%d").date()
    except ValueError:
        return 1.0
    today = date.today()
    months = (today.year - bd.year) * 12 + (today.month - bd.month)
    # if buy_date in future, slight deflation of estimate for "today dollars"
    factor = 1.0 + 0.0025 * months
    return float(np.clip(factor, 0.7, 1.3))


def _mortgage(price: float, country: str, years: int, down_pct: float, rate_override=None):
    c = (country or "US").upper()
    info = _rates.get(c, _rates["US"])
    annual = float(rate_override if rate_override is not None else info["annual_rate_pct"])
    down = price * (down_pct / 100.0)
    principal = max(price - down, 0.0)
    n = max(int(years), 1) * 12
    r = annual / 100.0 / 12.0
    if r <= 0:
        monthly = principal / n
    else:
        monthly = principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    return {
        "down_payment": round(down, 2),
        "loan_amount": round(principal, 2),
        "monthly_payment": round(monthly, 2),
        "annual_rate_pct": annual,
        "loan_years": int(years),
        "currency": info.get("currency", "USD"),
        "symbol": info.get("symbol", "$"),
    }


@app.get("/api/meta")
def meta():
    _load_or_train()
    countries = []
    for code, meta_c in _geo["countries"].items():
        cities = sorted(
            [k for k in _geo["cities"].get(code, {}) if k != "default"]
        )
        rate = _rates.get(code, {})
        countries.append(
            {
                "code": code,
                "label": meta_c["label"],
                "currency": meta_c.get("currency", rate.get("currency", "USD")),
                "symbol": rate.get("symbol", "$"),
                "default_rate_pct": rate.get("annual_rate_pct"),
                "cities": cities,
            }
        )
    return jsonify(
        {
            "countries": countries,
            "default_country": "US",
            "cookie_suggestion": {"name": "hp_country", "max_age_days": 365},
            "disclaimer": "Estimates use a trained model plus static geo/rate tables — not live bank quotes or address comps.",
        }
    )


@app.post("/api/estimate")
def estimate():
    _load_or_train()
    body = request.get_json(force=True, silent=True) or {}

    country = (body.get("country") or "US").upper()
    city = body.get("city") or ""
    address = body.get("address") or ""
    buy_date = body.get("buy_date")
    living = float(body.get("living_area") or 0)
    beds = int(body.get("bedrooms") or 0)
    year = int(body.get("year_built") or 0)
    use_mortgage = bool(body.get("mortgage"))
    loan_years = int(body.get("loan_years") or 30)
    down_pct = float(body.get("down_payment_pct") or 20)
    rate_override = body.get("rate_pct")

    if living <= 0 or beds <= 0 or year < 1800:
        return jsonify({"error": "living_area, bedrooms, year_built required"}), 400

    X = _row_from_form(body)
    raw_pred = float(_bundle["model"].predict(X)[0])
    g = _geo_mult(country, city)
    inf = _inflate(buy_date)
    price = raw_pred * g * inf

    out = {
        "estimated_price": round(price, 2),
        "base_model_price": round(raw_pred, 2),
        "geo_multiplier": round(g, 4),
        "inflation_factor": round(inf, 4),
        "country": country,
        "city": city,
        "address": address,
        "inputs": {
            "living_area": living,
            "bedrooms": beds,
            "year_built": year,
            "buy_date": buy_date,
            "overall_qual_heuristic": _qual_heuristic(year, living),
        },
        "disclaimer": "Demo estimate from config tables + Ames RF model — not a live appraisal or bank offer.",
    }

    if use_mortgage:
        out["mortgage"] = _mortgage(
            price, country, loan_years, down_pct, rate_override
        )
    else:
        out["mortgage"] = None

    return jsonify(out)


@app.get("/")
def index():
    return send_from_directory(SITE_DIR, "index.html")


@app.get("/site/<path:path>")
def site_files(path):
    return send_from_directory(SITE_DIR, path)


@app.get("/<path:path>")
def root_static(path):
    # serve site assets when opened via Flask
    target = SITE_DIR / path
    if target.is_file():
        return send_from_directory(SITE_DIR, path)
    return jsonify({"error": "not found"}), 404


if __name__ == "__main__":
    _load_or_train()
    print("API http://127.0.0.1:5000  |  site via same origin")
    app.run(host="127.0.0.1", port=5000, debug=False)


