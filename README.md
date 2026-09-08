# House Price Calculator

ML-powered demo that estimates a home price from basic property details, then optionally shows down payment and monthly mortgage payment.

Not a live bank API and not live address comps — geo multipliers and interest rates come from static config files (`app/geo_coeffs.json`, `app/rates.json`). The core price comes from a Random Forest trained on the Ames Housing dataset.

## Features

- Calculator UI (`site/`) — white / blue (`#2B7BBF`)
- `POST /api/estimate` — price + optional mortgage
- `GET /api/meta` — countries, default rates, cookie hint (`hp_country`)
- Country remembered via `hp_country` cookie
- Form living area is **m²** (converted to sq ft for the Ames model)
- Best-effort mapping onto Ames numeric features; other features filled with train medians
- Simple buy-date inflation (~0.25%/month, capped)

## Requirements

- Python 3.11
- Local data: `data/train_clean.csv`
- Optional: `models/rf_house_prices_best.joblib` (a smaller RF is trained at startup if missing)

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python server.py
```

Open http://127.0.0.1:5000/

API examples:

```bash
curl http://127.0.0.1:5000/api/meta

curl -X POST http://127.0.0.1:5000/api/estimate -H "Content-Type: application/json" -d "{\"country\":\"Poland\",\"city\":\"Warsaw\",\"address\":\"Example 1\",\"buy_date\":\"2026-01-15\",\"living_area\":85,\"bedrooms\":3,\"year_built\":2008,\"mortgage\":true,\"loan_years\":25,\"interest_rate\":7.5,\"down_payment_pct\":20}"
```

## Regenerate data / model

CSV and large `.joblib` files are gitignored.

1. Put Kaggle `train.csv` / `test.csv` into `data/`
2. Run:

```bash
python 03_clean.py
python 08_tune_forest.py
```

If `models/rf_house_prices_best.joblib` is absent, `server.py` trains a quick RF on `data/train_clean.csv` at startup.

## Layout

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