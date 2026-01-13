import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API = "https://gateway.api.globalfishingwatch.org/v3/events"
TOKEN = os.getenv("GFW_TOKEN")

if not TOKEN:
    raise RuntimeError(
        "Missing GFW_TOKEN in .env (copy .env.example -> .env and paste your token)"
    )

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

# Bounding box Mediterráneo (aprox) como GeoJSON Geometry (Polygon)
# OJO: coordenadas en orden [lon, lat]
MED_POLYGON = {
    "type": "Polygon",
    "coordinates": [[
        [-6.0, 30.0],
        [36.5, 30.0],
        [36.5, 46.5],
        [-6.0, 46.5],
        [-6.0, 30.0],
    ]]
}

# ✅ Body compatible con POST /v3/events para PORT_VISIT
BODY = {
    "datasets": ["public-global-port-visits-events:latest"],
    "types": ["PORT_VISIT"],
    "startDate": "2024-07-01",
    "endDate": "2024-08-01",
    # Solo aplica a port visits (2/3/4). Usamos calidad media/alta
    "confidences": ["3", "4"],
    "geometry": MED_POLYGON,
}

def fetch_page(limit: int, offset: int) -> dict:
    url = f"{API}?limit={limit}&offset={offset}&sort=-start"
    r = requests.post(url, headers=HEADERS, json=BODY, timeout=60)
    r.raise_for_status()
    return r.json()

def flatten(entries: list[dict]) -> pd.DataFrame:
    rows = []
    for e in entries:
        pv = e.get("port_visit", {}) or {}
        pos = e.get("position", {}) or {}
        vessel = e.get("vessel", {}) or {}

        anch = pv.get("intermediateAnchorage") or pv.get("startAnchorage") or {}

        rows.append({
            "event_id": e.get("id"),
            "type": e.get("type"),
            "start": e.get("start"),
            "end": e.get("end"),

            "lat": pos.get("lat"),
            "lon": pos.get("lon"),

            "vessel_id": vessel.get("id"),
            "ssvid": vessel.get("ssvid"),
            "vessel_name": vessel.get("name"),

            "confidence": pv.get("confidence"),
            "durationHrs": pv.get("durationHrs"),

            "port_id": anch.get("id"),
            "port_name": anch.get("name"),
            "port_flag": anch.get("flag"),
            "port_lat": anch.get("lat"),
            "port_lon": anch.get("lon"),
            "atDock": anch.get("atDock"),
            "distanceFromShoreKm": anch.get("distanceFromShoreKm"),
        })

    df = pd.DataFrame(rows)
    # Limpieza mínima
    df = df.dropna(subset=["start", "vessel_id", "port_id", "port_lat", "port_lon"])
    return df

def main(max_events: int = 50000, limit: int = 2000):
    all_frames = []
    offset = 0
    total = None

    while True:
        data = fetch_page(limit=limit, offset=offset)
        entries = data.get("entries", [])

        if total is None:
            total = data.get("total")
            print(f"Total events (API): {total}")

        if not entries:
            print("No more entries returned.")
            break

        df = flatten(entries)
        all_frames.append(df)

        downloaded = sum(len(x) for x in all_frames)
        offset = data.get("nextOffset")
        print(f"Downloaded: {downloaded} | nextOffset={offset}")

        if offset is None:
            break
        if downloaded >= max_events:
            print("Stopping due to max_events (keeps dataset manageable).")
            break

        time.sleep(0.25)  # suave con rate limits

    out = pd.concat(all_frames, ignore_index=True)
    os.makedirs("data/raw", exist_ok=True)
    out.to_parquet("data/raw/port_visits_med_2024_07.parquet", index=False)

    print("Saved: data/raw/port_visits_med_2024_07.parquet")
    print(out.head())

if __name__ == "__main__":
    main()
