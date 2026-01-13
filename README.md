# AIS Mediterranean Visualization (Part II)

This project downloads AIS-derived **port visit events** for the Mediterranean and builds an interactive visualization.

## Setup
1. Create a virtual environment (recommended)
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create your `.env`:
   - Copy `.env.example` to `.env`
   - Paste your `GFW_TOKEN`

## Run (Step 1: download data)
```bash
python src/01_download_port_visits.py
```

Outputs:
- `data/raw/port_visits_med_2024_07.parquet`

## Next steps
- `src/02_build_network.py` will build a port-to-port network + centrality
- `app/app.py` will run the Dash app
