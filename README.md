# House Price Calculator

ML-powered demo that estimates a home price from basic property details, then optionally shows down payment and monthly mortgage payment.

Not a live bank API and not live address comps — geo multipliers and interest rates come from static config files (`app/geo_coeffs.json`, `app/rates.json`). The core price comes from a Random Forest trained on the Ames Housing dataset.

## Features

- Web calculator UI (`site/`) — white / blue (`#2B7BBF`)
- `POST /api/estimate` — price + optional mortgage
- `GET /api/meta` — countries, cities, default rates, cookie hint (`hp_country`)
- Country cookie remembered in the browser
- Best-effort mapping of form fields onto Ames numeric features; missing features filled with train medians
- Simple buy-date inflation (~0.25%/month, capped)

## Requirements

- Python 3.11
- Local data: `data/train_clean.csv` (or raw `train.csv` + pipeline scripts)
- Optional: `models/rf_house_prices_best.joblib` (trained if missing)

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python server.py
```

Open http://127.0.0.1:5000/ in your browser.

API examples:

```bash
curl http://127.0.0.1:5000/api/meta

curl -X POST http://127.0.0.1:5000/api/estimate ^
  -H "Content-Type: application/json" ^
  -d "{\"country\":\"PL\",\"city\":\"Warsaw\",\"address\":\"Example 1\",\"buy_date\":\"2026-01-15\",\"living_area\":1400,\"bedrooms\":3,\"year_built\":2008,\"mortgage\":true,\"loan_years\":25,\"down_payment_pct\":20}"
```

## Regenerate data / model

CSV and large `.joblib` files are gitignored.

1. Put Kaggle `train.csv` / `test.csv` into `data/`
2. Run cleaning + training:

```bash
python 03_clean.py
python 08_tune_forest.py
```

If `models/rf_house_prices_best.joblib` is absent, `server.py` trains a smaller RF on `data/train_clean.csv` at startup.

## Project layout

```text
server.py              Flask API + static site
app/geo_coeffs.json    country/city multipliers
app/rates.json         default mortgage rates
site/                  calculator UI
data/                  CSV (local)
models/                joblib (local)
01_*.py … 09_*.py      training / EDA pipeline
```

## Disclaimer

Estimates are for demonstration only. They are not an appraisal, loan offer, or investment advice.
