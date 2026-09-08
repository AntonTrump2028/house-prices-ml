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

# UI uses m2; Ames GrLivArea is sq ft
M2_TO_SQFT = 10.7639

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


def _norm_country(raw: str | None) -> str:
    if not raw:
        return "USA"
    s = str(raw).strip()
    aliases = {
        "US": "USA",
        "UNITED STATES": "USA",
        "PL": "Poland",
        "DE": "Germany",
        "UA": "Ukraine",
        "GB": "Other",
        "UK": "Other",
    }
    up = s.upper()
    if up in aliases:
        return aliases[up]
    # exact keys in rates
    for k in _rates:
        if k.lower() == s.lower():
            return k
    return s if s in _rates else "Other"


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


def _qual_heuristic(year_built: int, living_sqft: float) -> int:
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
    if living_sqft >= 2200:
        q += 2
    elif living_sqft >= 1600:
        q += 1
    elif living_sqft < 900:
        q -= 1
    return int(np.clip(q, 1, 10))


def _row_from_form(living_sqft: float, beds: int, year: int) -> pd.DataFrame:
    _load_or_train()
    qual = _qual_heuristic(year, living_sqft)
    row = {f: float(_medians.get(f, 0.0)) for f in _features}
    mapped = {
        "OverallQual": qual,
        "GrLivArea": living_sqft,
        "BedroomAbvGr": beds,
        "YearBuilt": year,
        "YearRemodAdd": max(year, int(_medians.get("YearRemodAdd", year))),
        "GarageYrBlt": year,
        "1stFlrSF": living_sqft * 0.7,
        "2ndFlrSF": living_sqft * 0.3 if beds >= 3 else 0.0,
        "TotalBsmtSF": living_sqft * 0.55,
        "FullBath": max(1, min(beds - 1, 3)),
        "TotRmsAbvGrd": max(beds + 2, 4),
        "GarageCars": 2 if living_sqft >= 1100 else 1,
        "GarageArea": 480 if living_sqft >= 1100 else 280,
        "LotArea": max(living_sqft * 6.5, 4000),
    }
    for k, v in mapped.items():
        if k in row:
            row[k] = float(v)
    return pd.DataFrame([[row[f] for f in _features]], columns=_features)


def _geo_mult(country: str, city: str) -> float:
    meta_c = _geo["countries"].get(country, _geo["countries"].get("USA", {}))
    base = float(meta_c.get("base", 1.0))
    cities = _geo["cities"].get(country, {})
    city_key = (city or "").strip()
    city_m = float(cities.get(city_key, cities.get("default", 1.0)))
    return base * city_m


def _inflate(buy_date_str: str | None) -> float:
    if not buy_date_str:
        return 1.0
    try:
        bd = datetime.strptime(str(buy_date_str)[:10], "%Y-%m-%d").date()
    except ValueError:
        return 1.0
    today = date.today()
    months = (today.year - bd.year) * 12 + (today.month - bd.month)
    factor = 1.0 + 0.0025 * months
    return float(np.clip(factor, 0.7, 1.3))


def _mortgage(price: float, country: str, years: int, down_pct: float, annual_rate: float):
    info = _rates.get(country, _rates["USA"])
    down = price * (down_pct / 100.0)
    principal = max(price - down, 0.0)
    n = max(int(years), 1) * 12
    r = float(annual_rate) / 100.0 / 12.0
    if r <= 0 or n <= 0:
        monthly = principal / max(n, 1)
    else:
        monthly = principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    return {
        "down_payment": round(down, 2),
        "loan_amount": round(principal, 2),
        "monthly_payment": round(monthly, 2),
        "annual_rate_pct": float(annual_rate),
        "loan_years": int(years),
        "currency": info.get("currency", "USD"),
        "symbol": info.get("symbol", "$"),
    }


@app.route("/api/meta", methods=["GET", "OPTIONS"])
def meta():
    if request.method == "OPTIONS":
        return ("", 204)
    _load_or_train()
    rates_flat = {k: float(v["annual_rate_pct"]) for k, v in _rates.items()}
    countries = []
    for code, meta_c in _geo["countries"].items():
        cities = sorted([k for k in _geo["cities"].get(code, {}) if k != "default"])
        rate = _rates.get(code, {})
        countries.append(
            {
                "code": code,
                "name": code,
                "label": meta_c.get("label", code),
                "currency": meta_c.get("currency", rate.get("currency", "USD")),
                "symbol": rate.get("symbol", "$"),
                "default_rate_pct": rate.get("annual_rate_pct"),
                "cities": cities,
            }
        )
    return jsonify(
        {
            "countries": countries,
            "rates": rates_flat,
            "default_country": "USA",
            "cookie_suggestion": {"name": "hp_country", "max_age_days": 365},
            "area_unit": "m2",
            "disclaimer": (
                "Estimates use a trained model plus static geo/rate tables — "
                "not live bank quotes or address comps."
            ),
        }
    )


@app.route("/api/estimate", methods=["POST", "OPTIONS"])
def estimate():
    if request.method == "OPTIONS":
        return ("", 204)
    _load_or_train()
    body = request.get_json(force=True, silent=True) or {}

    country = _norm_country(body.get("country"))
    city = (body.get("city") or "").strip()
    address = (body.get("address") or "").strip()
    buy_date = body.get("buy_date")
    living_m2 = float(body.get("living_area") or 0)
    beds = int(body.get("bedrooms") or 0)
    year = int(body.get("year_built") or 0)
    use_mortgage = bool(body.get("mortgage"))
    loan_years = int(body.get("loan_years") or 30)
    down_pct = float(body.get("down_payment_pct") or 20)

    # rate: interest_rate from UI, else rate_pct, else country default
    rate_info = _rates.get(country, _rates["USA"])
    annual = float(rate_info["annual_rate_pct"])
    if body.get("interest_rate") is not None and body.get("interest_rate") != "":
        try:
            annual = float(body["interest_rate"])
        except (TypeError, ValueError):
            pass
    elif body.get("rate_pct") is not None and body.get("rate_pct") != "":
        try:
            annual = float(body["rate_pct"])
        except (TypeError, ValueError):
            pass

    if living_m2 <= 0 or beds <= 0 or year < 1800:
        return jsonify({"error": "living_area, bedrooms, year_built required"}), 400

    living_sqft = living_m2 * M2_TO_SQFT
    X = _row_from_form(living_sqft, beds, year)
    raw_pred = float(_bundle["model"].predict(X)[0])
    g = _geo_mult(country, city)
    inf = _inflate(buy_date)
    price = raw_pred * g * inf

    currency = rate_info.get("currency", "USD")

    out = {
        # UI picks estimate / price / predicted_price / prediction
        "estimate": round(price, 2),
        "price": round(price, 2),
        "estimated_price": round(price, 2),
        "currency": currency,
        "base_model_price": round(raw_pred, 2),
        "geo_multiplier": round(g, 4),
        "inflation_factor": round(inf, 4),
        "country": country,
        "city": city,
        "address": address,
        "inputs": {
            "living_area_m2": living_m2,
            "living_area_sqft": round(living_sqft, 1),
            "bedrooms": beds,
            "year_built": year,
            "buy_date": buy_date,
            "overall_qual_heuristic": _qual_heuristic(year, living_sqft),
        },
        "disclaimer": (
            "Demo estimate from config tables + Ames RF model — "
            "not a live appraisal or bank offer."
        ),
    }

    if use_mortgage:
        m = _mortgage(price, country, loan_years, down_pct, annual)
        out["mortgage"] = m
        # top-level aliases for the shipped site UI
        out["down_payment"] = m["down_payment"]
        out["monthly_payment"] = m["monthly_payment"]
    else:
        out["mortgage"] = None

    return jsonify(out)


@app.get("/")
def index():
    return send_from_directory(SITE_DIR, "index.html")


@app.get("/<path:path>")
def root_static(path):
    if path.startswith("api/"):
        return jsonify({"error": "not found"}), 404
    target = SITE_DIR / path
    if target.is_file():
        return send_from_directory(SITE_DIR, path)
    return jsonify({"error": "not found"}), 404


if __name__ == "__main__":
    _load_or_train()
    print("API http://127.0.0.1:5000  |  site via same origin")
    app.run(host="127.0.0.1", port=5000, debug=False)