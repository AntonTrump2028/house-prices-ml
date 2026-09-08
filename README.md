# house-prices-ml

Estimate a home price and optional mortgage payment from an address and a few basics.

Map search uses OpenStreetMap / Nominatim. The model is a Random Forest trained on Ames Housing (USD), then adjusted for location, FX, and inflation. Not a bank offer or formal appraisal.

## Quick start

```bash
pip install -r requirements.txt
python server.py
```

Open http://127.0.0.1:5000/

Needs Python 3.11 and `data/train_clean.csv` locally (Kaggle Ames train → `03_clean.py`). The model file under `models/` is optional; the server can train a fallback on startup.

## Stack

Python, Flask, scikit-learn, Leaflet, OpenStreetMap

## License

MIT