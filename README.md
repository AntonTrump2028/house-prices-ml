# House Price Calculator

Locate a property on **OpenStreetMap** (Nominatim + Leaflet) and get an **indicative ML price**.

## Honesty

- **OSM has no sale prices.** The map and Nominatim search only locate the property.
- Price = Ames Housing Random Forest (USD) + inflation (2010→`buy_year`) + geo coeffs + FX (`config.json`, with live Frankfurter/open.er-api when available).
- **Year built does not affect price** (training median `YearBuilt` is kept).
- Mortgage uses demo country rates or a custom rate (default 30 years, 20% down).

## Product files

| File | Role |
|------|------|
| `train.py` | Clean data, tune RF, save `models/rf_house_prices_best.joblib` |
| `server.py` | Flask API + Nominatim/Overpass proxies + static `site/` |
| `site/index.html` | UI shell (Design Spec v4, two-column) |
| `site/styles.css` | White/blue styles |
| `site/app.js` | Leaflet map, search, estimate form |
| `config.json` | Geo coeffs, mortgage rates, static FX |

## API

- `GET /api/meta`
- `GET /api/search?q=` — Nominatim proxy (limit 5, ~1 req/s, proper User-Agent)
- `GET /api/reverse?lat=&lon=`
- `GET|POST /api/place` — Overpass building enrich
- `POST /api/estimate` — body includes `buy_year` (integer), not day date; no year_built effect

## Run

```bash
pip install -r requirements.txt
python server.py
```

Open http://127.0.0.1:5000/

```bash
curl "http://127.0.0.1:5000/api/search?q=Marszalkowska%201%20Warsaw"
curl -X POST http://127.0.0.1:5000/api/estimate -H "Content-Type: application/json" -d "{"country":"Poland","city":"Warsaw","address":"Marszalkowska 1","buy_year":2026,"living_area":85,"bedrooms":3,"mortgage":true,"loan_years":30,"interest_rate":7.5,"down_payment_pct":20}"
```

Retrain:

```bash
python train.py
```

## Disclaimer

Demo only — not an appraisal, loan offer, or investment advice. OSM data © OpenStreetMap contributors.
