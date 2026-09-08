"""House price calculator API — OSM/Nominatim + Ames RF demo (not live comps)."""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
MODEL_PATH = ROOT / "models" / "rf_house_prices_best.joblib"
TRAIN_PATH = ROOT / "data" / "train_clean.csv"
SITE_DIR = ROOT / "site"

M2_TO_SQFT = 10.7639
NOMINATIM_UA = "house-prices-ml/1.0 (contact: github.com/AntonTrump2028/house-prices-ml)"
NOMINATIM = "https://nominatim.openstreetmap.org"
OVERPASS = "https://overpass-api.de/api/interpreter"
INFLATION_BASE_YEAR = 2010
INFLATION_ANNUAL = 0.025  # ~2.5%/yr Ames-era → buy_year

app = Flask(__name__, static_folder=None)

_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
_geo = _config["geo_coeffs"]
_rates = _config["rates"]
_fx_static = _config["fx_rates"]

_bundle = None
_medians = None
_features = None
_fx_live: dict | None = None
_fx_live_ts = 0.0

_nominatim_lock = threading.Lock()
_nominatim_last = 0.0


@app.after_request
def _cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp


def _norm_country(raw: str | None) -> str:
    if not raw:
        return "USA"
    s = str(raw).strip()
    aliases = {
        "US": "USA",
        "UNITED STATES": "USA",
        "PL": "Poland",
        "POLSKA": "Poland",
        "DE": "Germany",
        "DEUTSCHLAND": "Germany",
        "UA": "Ukraine",
        "GB": "Other",
        "UK": "Other",
    }
    up = s.upper()
    if up in aliases:
        return aliases[up]
    for k in _rates:
        if k.lower() == s.lower():
            return k
    return s if s in _rates else "Other"


def _http_json(url: str, *, data: bytes | None = None, headers: dict | None = None, timeout: float = 20.0):
    hdrs = {"Accept": "application/json", "User-Agent": NOMINATIM_UA}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _nominatim_get(path: str, params: dict):
    """Respect Nominatim 1 req/s."""
    global _nominatim_last
    with _nominatim_lock:
        now = time.monotonic()
        wait = 1.05 - (now - _nominatim_last)
        if wait > 0:
            time.sleep(wait)
        q = urllib.parse.urlencode(params)
        url = f"{NOMINATIM}{path}?{q}"
        try:
            out = _http_json(url, timeout=25.0)
        finally:
            _nominatim_last = time.monotonic()
        return out


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


def _qual_heuristic(living_sqft: float) -> int:
    """Quality from size only — YearBuilt must NOT influence price."""
    q = 5
    if living_sqft >= 2200:
        q += 2
    elif living_sqft >= 1600:
        q += 1
    elif living_sqft < 900:
        q -= 1
    return int(np.clip(q, 1, 10))


def _row_from_form(living_sqft: float, beds: int) -> pd.DataFrame:
    """Map form inputs; YearBuilt/YearRemodAdd/GarageYrBlt stay at train medians."""
    _load_or_train()
    qual = _qual_heuristic(living_sqft)
    row = {f: float(_medians.get(f, 0.0)) for f in _features}
    mapped = {
        "OverallQual": qual,
        "GrLivArea": living_sqft,
        "BedroomAbvGr": beds,
        # YearBuilt intentionally omitted — median retained
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


def _inflate_buy_year(buy_year: int) -> float:
    """Inflation from Ames-ish 2010 baseline to purchase year."""
    try:
        y = int(buy_year)
    except (TypeError, ValueError):
        return 1.0
    years = y - INFLATION_BASE_YEAR
    factor = (1.0 + INFLATION_ANNUAL) ** years
    return float(np.clip(factor, 0.5, 2.5))


def _refresh_fx_live() -> dict:
    """Try frankfurter / open.er-api; fall back to static fx_rates.json."""
    global _fx_live, _fx_live_ts
    now = time.time()
    if _fx_live is not None and (now - _fx_live_ts) < 3600:
        return _fx_live

    static = dict(_fx_static.get("to_local", {}))
    live = dict(static)

    # frankfurter.app: USD base → EUR, PLN, etc.
    try:
        data = _http_json(
            "https://api.frankfurter.app/latest?from=USD&to=EUR,PLN",
            timeout=8.0,
        )
        rates = data.get("rates") or {}
        for k, v in rates.items():
            live[k] = float(v)
    except Exception as e:
        print("frankfurter FX failed:", e)

    try:
        data = _http_json("https://open.er-api.com/v6/latest/USD", timeout=8.0)
        rates = data.get("rates") or {}
        for cur in ("EUR", "PLN", "UAH", "USD"):
            if cur in rates:
                live[cur] = float(rates[cur])
    except Exception as e:
        print("open.er-api FX failed:", e)

    _fx_live = live
    _fx_live_ts = now
    return live


def _fx_to_local(currency: str) -> float:
    live = _refresh_fx_live()
    return float(live.get(currency, _fx_static.get("to_local", {}).get(currency, 1.0)))


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
        "down_payment_pct": float(down_pct),
        "currency": info.get("currency", "USD"),
        "symbol": info.get("symbol", "$"),
    }


def _polygon_area_m2(coords: list) -> float | None:
    """Shoelace on lon/lat rings approximated via equirectangular meters."""
    if not coords or len(coords) < 3:
        return None
    # Overpass geometry: [{lat, lon}, ...]
    lats = [float(p["lat"]) for p in coords]
    lons = [float(p["lon"]) for p in coords]
    lat0 = sum(lats) / len(lats)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat0))
    xs = [(lon - lons[0]) * m_per_deg_lon for lon in lons]
    ys = [(lat - lats[0]) * m_per_deg_lat for lat in lats]
    if xs[0] != xs[-1] or ys[0] != ys[-1]:
        xs.append(xs[0])
        ys.append(ys[0])
    area = 0.0
    for i in range(len(xs) - 1):
        area += xs[i] * ys[i + 1] - xs[i + 1] * ys[i]
    return abs(area) / 2.0


def _enrich_overpass(lat: float, lon: float, osm_type: str | None = None, osm_id: int | None = None) -> dict:
    """Fetch nearby/selected building tags + footprint area estimate."""
    out: dict = {
        "building_area_m2": None,
        "building_levels": None,
        "building_type": None,
        "living_area_estimate_m2": None,
        "bedrooms_heuristic": None,
        "tags": {},
        "source": None,
    }
    try:
        if osm_type and osm_id and osm_type in ("way", "relation"):
            q = f'[out:json][timeout:25];{osm_type}({int(osm_id)});out tags geom;'
        else:
            # nearest building around point
            q = (
                f"[out:json][timeout:25];"
                f"way(around:40,{lat},{lon})[building];"
                f"out tags geom 1;"
            )
        body = urllib.parse.urlencode({"data": q}).encode("utf-8")
        data = _http_json(
            OVERPASS,
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": NOMINATIM_UA},
            timeout=30.0,
        )
        elements = data.get("elements") or []
        if not elements:
            out["source"] = "overpass-empty"
            return out
        el = elements[0]
        tags = el.get("tags") or {}
        out["tags"] = tags
        out["building_type"] = tags.get("building")
        levels = tags.get("building:levels") or tags.get("levels")
        if levels is not None:
            try:
                out["building_levels"] = float(str(levels).replace(",", "."))
            except ValueError:
                pass
        geom = el.get("geometry")
        area = _polygon_area_m2(geom) if geom else None
        if area and area > 5:
            out["building_area_m2"] = round(area, 1)
            levels_f = float(out["building_levels"] or 1.0)
            # rough living area: footprint * min(levels, 3) * 0.85
            living = area * min(levels_f, 3.0) * 0.85
            out["living_area_estimate_m2"] = round(living, 1)
            # bedrooms heuristic from levels + area
            beds = max(1, min(8, int(round(living / 28.0))))
            if out["building_levels"]:
                beds = max(beds, int(round(float(out["building_levels"]))))
            out["bedrooms_heuristic"] = beds
        out["source"] = "overpass"
    except Exception as e:
        out["source"] = f"overpass-error:{e}"
    return out


def _country_from_address(addr: dict | None) -> str:
    if not addr:
        return "Other"
    cc = (addr.get("country_code") or "").upper()
    name = addr.get("country") or ""
    mapped = {
        "PL": "Poland",
        "DE": "Germany",
        "US": "USA",
        "UA": "Ukraine",
    }
    if cc in mapped:
        return mapped[cc]
    return _norm_country(name)


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
            "map": {"tiles": "OpenStreetMap", "geocoder": "Nominatim"},
            "disclaimer": (
                "OSM/Nominatim locate the property only — OSM has no sale prices. "
                "Estimate = Ames RF (USD) + inflation + geo coeffs + FX demo tables."
            ),
        }
    )


@app.route("/api/search", methods=["GET", "OPTIONS"])
def search():
    if request.method == "OPTIONS":
        return ("", 204)
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify({"error": "q required", "results": []}), 400
    try:
        raw = _nominatim_get(
            "/search",
            {
                "q": q,
                "format": "json",
                "addressdetails": 1,
                "limit": 5,
            },
        )
    except Exception as e:
        return jsonify({"error": f"nominatim: {e}", "results": []}), 502

    results = []
    for item in raw or []:
        addr = item.get("address") or {}
        results.append(
            {
                "display_name": item.get("display_name"),
                "lat": float(item["lat"]),
                "lon": float(item["lon"]),
                "osm_type": item.get("osm_type"),
                "osm_id": item.get("osm_id"),
                "place_id": item.get("place_id"),
                "address": addr,
                "city": addr.get("city")
                or addr.get("town")
                or addr.get("village")
                or addr.get("municipality")
                or "",
                "country": _country_from_address(addr),
                "road": addr.get("road") or "",
                "house_number": addr.get("house_number") or "",
            }
        )
    return jsonify({"results": results})


@app.route("/api/reverse", methods=["GET", "OPTIONS"])
def reverse():
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"error": "lat and lon required"}), 400
    try:
        item = _nominatim_get(
            "/reverse",
            {
                "lat": lat,
                "lon": lon,
                "format": "json",
                "addressdetails": 1,
            },
        )
    except Exception as e:
        return jsonify({"error": f"nominatim: {e}"}), 502
    addr = (item or {}).get("address") or {}
    return jsonify(
        {
            "display_name": item.get("display_name"),
            "lat": float(item.get("lat", lat)),
            "lon": float(item.get("lon", lon)),
            "osm_type": item.get("osm_type"),
            "osm_id": item.get("osm_id"),
            "address": addr,
            "city": addr.get("city")
            or addr.get("town")
            or addr.get("village")
            or addr.get("municipality")
            or "",
            "country": _country_from_address(addr),
            "road": addr.get("road") or "",
            "house_number": addr.get("house_number") or "",
        }
    )


@app.route("/api/place", methods=["GET", "POST", "OPTIONS"])
def place():
    if request.method == "OPTIONS":
        return ("", 204)

    osm_type = None
    osm_id = None
    lat = None
    lon = None

    if request.method == "POST":
        body = request.get_json(force=True, silent=True) or {}
        osm_type = body.get("osm_type") or request.args.get("osm_type")
        osm_id = body.get("osm_id") or request.args.get("osm_id")
        try:
            lat = float(body.get("lat") if body.get("lat") is not None else request.args.get("lat"))
            lon = float(body.get("lon") if body.get("lon") is not None else request.args.get("lon"))
        except (TypeError, ValueError):
            lat = lon = None
    else:
        osm_type = request.args.get("osm_type")
        osm_id = request.args.get("osm_id")
        try:
            lat = float(request.args["lat"]) if "lat" in request.args else None
            lon = float(request.args["lon"]) if "lon" in request.args else None
        except (TypeError, ValueError, KeyError):
            lat = lon = None

    try:
        osm_id_int = int(osm_id) if osm_id is not None else None
    except (TypeError, ValueError):
        osm_id_int = None

    if lat is None or lon is None:
        # try to resolve from OSM id via Nominatim lookup
        if osm_type and osm_id_int:
            try:
                # Nominatim lookup
                prefix = {"node": "N", "way": "W", "relation": "R"}.get(str(osm_type).lower())
                if prefix:
                    raw = _nominatim_get(
                        "/lookup",
                        {"osm_ids": f"{prefix}{osm_id_int}", "format": "json", "addressdetails": 1},
                    )
                    if raw:
                        lat = float(raw[0]["lat"])
                        lon = float(raw[0]["lon"])
            except Exception as e:
                return jsonify({"error": f"resolve place: {e}"}), 502
        else:
            return jsonify({"error": "lat/lon or osm_type+osm_id required"}), 400

    enrich = _enrich_overpass(lat, lon, str(osm_type).lower() if osm_type else None, osm_id_int)
    enrich["lat"] = lat
    enrich["lon"] = lon
    enrich["osm_type"] = osm_type
    enrich["osm_id"] = osm_id_int
    return jsonify(enrich)


@app.route("/api/estimate", methods=["POST", "OPTIONS"])
def estimate():
    if request.method == "OPTIONS":
        return ("", 204)
    _load_or_train()
    body = request.get_json(force=True, silent=True) or {}

    country = _norm_country(body.get("country"))
    city = (body.get("city") or "").strip()
    address = (body.get("address") or "").strip()

    # buy_year preferred; accept legacy buy_date YYYY-...
    buy_year = body.get("buy_year")
    if buy_year is None or buy_year == "":
        bd = body.get("buy_date")
        if bd:
            try:
                buy_year = int(str(bd)[:4])
            except ValueError:
                buy_year = date.today().year
        else:
            buy_year = date.today().year
    try:
        buy_year = int(buy_year)
    except (TypeError, ValueError):
        return jsonify({"error": "buy_year must be an integer year"}), 400

    living_m2 = float(body.get("living_area") or 0)
    beds = int(body.get("bedrooms") or 0)
    use_mortgage = bool(body.get("mortgage"))
    loan_years = int(body.get("loan_years") or 30)
    down_pct = float(body.get("down_payment_pct") if body.get("down_payment_pct") not in (None, "") else 20)

    lat = body.get("lat")
    lon = body.get("lon")
    try:
        lat_f = float(lat) if lat is not None and lat != "" else None
        lon_f = float(lon) if lon is not None and lon != "" else None
    except (TypeError, ValueError):
        lat_f = lon_f = None

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

    if living_m2 <= 0 or beds <= 0:
        return jsonify({"error": "living_area and bedrooms required"}), 400
    if buy_year < 1900 or buy_year > 2100:
        return jsonify({"error": "buy_year out of range"}), 400

    living_sqft = living_m2 * M2_TO_SQFT
    X = _row_from_form(living_sqft, beds)
    raw_pred_usd = float(_bundle["model"].predict(X)[0])
    g = _geo_mult(country, city)
    inf = _inflate_buy_year(buy_year)
    currency = rate_info.get("currency", "USD")
    fx = _fx_to_local(currency)
    price_usd_adj = raw_pred_usd * g * inf
    price = price_usd_adj * fx

    median_year = int(_medians.get("YearBuilt", 1970)) if _medians else 1970

    out = {
        "estimate": round(price, 2),
        "price": round(price, 2),
        "estimated_price": round(price, 2),
        "currency": currency,
        "base_model_price_usd": round(raw_pred_usd, 2),
        "fx_to_local": fx,
        "geo_multiplier": round(g, 4),
        "model_currency": "USD",
        "inflation_factor": round(inf, 4),
        "inflation_base_year": INFLATION_BASE_YEAR,
        "country": country,
        "city": city,
        "address": address,
        "lat": lat_f,
        "lon": lon_f,
        "inputs": {
            "living_area_m2": living_m2,
            "living_area_sqft": round(living_sqft, 1),
            "bedrooms": beds,
            "buy_year": buy_year,
            "year_built_used": median_year,
            "year_built_note": "median YearBuilt kept; form year_built ignored",
            "overall_qual_heuristic": _qual_heuristic(living_sqft),
        },
        "disclaimer": (
            "OSM has no sale prices. Demo estimate from Ames RF + inflation/geo/FX — "
            "not a live appraisal or bank offer. Map is for locating the property only."
        ),
    }

    if use_mortgage:
        m = _mortgage(price, country, loan_years, down_pct, annual)
        out["mortgage"] = m
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
    print("API http://127.0.0.1:5000  |  OSM Nominatim + Leaflet site")
    app.run(host="127.0.0.1", port=5000, debug=False)
