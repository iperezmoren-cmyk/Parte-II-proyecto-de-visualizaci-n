import os
import pandas as pd
import networkx as nx

print("02_build_network v5 ✅ (sin PageRank, sin SciPy, estable)")

RAW_PATH = "data/raw/port_visits_med_2024_07.parquet"
OUT_EDGES = "data/processed/port_network_edges.parquet"
OUT_PORTS = "data/processed/port_metrics.parquet"


def safe_mean_numeric(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce")
    return float(s.mean()) if s.notna().any() else 0.0


def safe_sum_numeric(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce")
    return float(s.sum()) if s.notna().any() else 0.0


def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError("Ejecuta primero 01_download_port_visits.py")

    df = pd.read_parquet(RAW_PATH)

    # Fechas
    df["start_dt"] = pd.to_datetime(df["start"], utc=True, errors="coerce")

    # Filtro mínimo
    df = df.dropna(subset=["start_dt", "vessel_id", "port_id", "port_lat", "port_lon"])

    df["vessel_id"] = df["vessel_id"].astype(str)
    df["port_id"] = df["port_id"].astype(str)

    # ---------- Métricas por puerto ----------
    g = df.groupby("port_id")

    ports_base = g.agg(
        port_name=("port_name", "first"),
        port_flag=("port_flag", "first"),
        port_lat=("port_lat", "first"),
        port_lon=("port_lon", "first"),
        visits=("event_id", "count"),
        vessels_unique=("vessel_id", "nunique"),
    ).reset_index()

    duration_sum = g["durationHrs"].apply(safe_sum_numeric).reset_index(name="total_duration_hrs")
    dist_mean = g["distanceFromShoreKm"].apply(safe_mean_numeric).reset_index(name="avg_distance_shore_km")

    ports = (
        ports_base
        .merge(duration_sum, on="port_id", how="left")
        .merge(dist_mean, on="port_id", how="left")
    )

    # ---------- Aristas puerto → puerto ----------
    df_sorted = df.sort_values(["vessel_id", "start_dt"])
    df_sorted["next_port_id"] = df_sorted.groupby("vessel_id")["port_id"].shift(-1)

    edges_raw = df_sorted.dropna(subset=["next_port_id"])
    edges_raw = edges_raw[edges_raw["port_id"] != edges_raw["next_port_id"]]

    edges = (
        edges_raw.groupby(["port_id", "next_port_id"])
        .agg(
            trips=("event_id", "count"),
            vessels_unique=("vessel_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"port_id": "port_id_from", "next_port_id": "port_id_to"})
    )

    # Coordenadas para mapa
    lookup = ports.set_index("port_id")[["port_name", "port_lat", "port_lon"]]

    edges["from_name"] = edges["port_id_from"].map(lookup["port_name"])
    edges["from_lat"] = edges["port_id_from"].map(lookup["port_lat"])
    edges["from_lon"] = edges["port_id_from"].map(lookup["port_lon"])
    edges["to_name"] = edges["port_id_to"].map(lookup["port_name"])
    edges["to_lat"] = edges["port_id_to"].map(lookup["port_lat"])
    edges["to_lon"] = edges["port_id_to"].map(lookup["port_lon"])

    edges = edges.dropna(subset=["from_lat", "from_lon", "to_lat", "to_lon"])

    # ---------- Centralidad (grado ponderado) ----------
    G = nx.DiGraph()
    for _, r in edges.iterrows():
        G.add_edge(r["port_id_from"], r["port_id_to"], weight=float(r["trips"]))

    ports["in_strength"] = ports["port_id"].map(dict(G.in_degree(weight="weight"))).fillna(0.0)
    ports["out_strength"] = ports["port_id"].map(dict(G.out_degree(weight="weight"))).fillna(0.0)
    ports["total_strength"] = ports["in_strength"] + ports["out_strength"]

    # Guardar
    os.makedirs("data/processed", exist_ok=True)
    edges.to_parquet(OUT_EDGES, index=False)
    ports.to_parquet(OUT_PORTS, index=False)

    print("OK ✅")
    print(f"Eventos usados: {len(df):,}")
    print(f"Puertos únicos: {ports['port_id'].nunique():,}")
    print(f"Aristas (rutas) únicas: {len(edges):,}")

    print("\nTop 10 hubs por total_strength:")
    print(
        ports.sort_values("total_strength", ascending=False)[
            ["port_name", "visits", "vessels_unique", "total_strength"]
        ].head(10)
    )


if __name__ == "__main__":
    main()
