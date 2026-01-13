import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output

PORTS_PATH = "data/processed/port_metrics.parquet"
EDGES_PATH = "data/processed/port_network_edges.parquet"
RAW_PATH = "data/raw/port_visits_med_2024_07.parquet"

if not os.path.exists(PORTS_PATH) or not os.path.exists(EDGES_PATH):
    raise FileNotFoundError(
        "Faltan ficheros procesados. Ejecuta antes:\n"
        "  python src/02_build_network.py"
    )

ports = pd.read_parquet(PORTS_PATH)
edges = pd.read_parquet(EDGES_PATH)

# --- Tipos seguros ---
for col in ["visits", "vessels_unique", "total_strength", "in_strength", "out_strength", "port_lat", "port_lon"]:
    if col in ports.columns:
        ports[col] = pd.to_numeric(ports[col], errors="coerce")

for col in ["trips", "vessels_unique", "from_lat", "from_lon", "to_lat", "to_lon", "median_delta_hours"]:
    if col in edges.columns:
        edges[col] = pd.to_numeric(edges[col], errors="coerce")

ports = ports.dropna(subset=["port_lat", "port_lon"]).copy()
edges = edges.dropna(subset=["from_lat", "from_lon", "to_lat", "to_lon"]).copy()

# --- Temporal (visitas por día) ---
daily_all = None
daily_by_port = None
if os.path.exists(RAW_PATH):
    raw = pd.read_parquet(RAW_PATH)
    raw["start_dt"] = pd.to_datetime(raw["start"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["start_dt", "port_id"])
    raw["day"] = raw["start_dt"].dt.floor("D")

    daily_all = raw.groupby("day").size().reset_index(name="port_visits")
    daily_by_port = raw.groupby(["day", "port_id"]).size().reset_index(name="port_visits")

# Dropdown puertos (top 200 por visitas)
top_ports = ports.sort_values("visits", ascending=False).head(200)[["port_id", "port_name"]]
port_options = [{"label": f"{r.port_name} ({r.port_id})", "value": r.port_id} for r in top_ports.itertuples(index=False)]


# ---------- Figures ----------
def make_hubs_figure(metric: str):
    metric_map = {
        "visits": ("visits", "Visitas a puerto"),
        "vessels_unique": ("vessels_unique", "Barcos únicos"),
        "total_strength": ("total_strength", "Centralidad (grado ponderado)"),
        "in_strength": ("in_strength", "Entradas ponderadas"),
        "out_strength": ("out_strength", "Salidas ponderadas"),
    }
    col, title = metric_map[metric]

    df = ports.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # tamaño estable
    df["size"] = (df[col] ** 0.5).clip(lower=1)

    fig = px.scatter_mapbox(
        df,
        lat="port_lat",
        lon="port_lon",
        size="size",
        hover_name="port_name",
        hover_data={
            "port_lat": False,
            "port_lon": False,
            "visits": True,
            "vessels_unique": True,
            "total_strength": True,
            "in_strength": True,
            "out_strength": True,
        },
        zoom=3.2,
        height=650,
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=10, r=10, t=45, b=10),
        title=f"Hubs portuarios — {title}",
    )
    return fig


def make_routes_figure(rank_by: str, top_n: int):
    df = edges.copy()
    df[rank_by] = pd.to_numeric(df[rank_by], errors="coerce").fillna(0)
    df = df.sort_values(rank_by, ascending=False).head(top_n)

    fig = go.Figure()

    # líneas
    for r in df.itertuples(index=False):
        val = float(getattr(r, rank_by))
        width = max(1, min(6, (val ** 0.5) / 3))

        delta = getattr(r, "median_delta_hours", None)
        delta_txt = f"{delta:.1f}" if isinstance(delta, (int, float)) and pd.notna(delta) else "N/A"

        fig.add_trace(
            go.Scattermapbox(
                lat=[r.from_lat, r.to_lat],
                lon=[r.from_lon, r.to_lon],
                mode="lines",
                line={"width": width},
                hoverinfo="text",
                text=(
                    f"{r.from_name} → {r.to_name}<br>"
                    f"trips: {r.trips} | vessels: {r.vessels_unique}<br>"
                    f"Δ horas (mediana): {delta_txt}"
                ),
                showlegend=False,
            )
        )

    # marcadores de puertos implicados
    port_ids = pd.unique(pd.concat([df["port_id_from"], df["port_id_to"]], ignore_index=True))
    p = ports[ports["port_id"].isin(port_ids)].copy()

    fig.add_trace(
        go.Scattermapbox(
            lat=p["port_lat"],
            lon=p["port_lon"],
            mode="markers",
            marker={"size": 8},
            text=p["port_name"],
            hoverinfo="text",
            showlegend=False,
        )
    )

    title = f"Rutas principales (Top {top_n}) — ranking por {'viajes' if rank_by=='trips' else 'barcos únicos'}"
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=3.2,
        mapbox_center={"lat": 38.5, "lon": 15.0},
        height=650,
        margin=dict(l=10, r=10, t=45, b=10),
        title=title,
    )
    return fig


def make_time_figure(port_id: str | None):
    if daily_all is None:
        fig = go.Figure()
        fig.add_annotation(text="No se encontró data/raw/port_visits_med_2024_07.parquet", showarrow=False)
        fig.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10))
        return fig

    if port_id:
        df = daily_by_port[daily_by_port["port_id"] == port_id].copy()
        name = ports.set_index("port_id")["port_name"].get(port_id, port_id)
        title = f"Visitas por día — {name}"
    else:
        df = daily_all.copy()
        title = "Visitas por día — total Mediterráneo"

    fig = px.line(df, x="day", y="port_visits", markers=True, height=450)
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis_title="Día",
        yaxis_title="Nº visitas (eventos)",
    )
    return fig


# ---------- App ----------
app = Dash(__name__)
app.title = "AIS Mediterranean – Port Visits"

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto"},
    children=[
        html.H2("Tráfico marítimo Mediterráneo (Julio 2024) — Hubs, rutas y temporalidad"),
        html.Div(
            style={"marginBottom": "12px", "color": "#333"},
            children=[
                html.Div("Dataset: eventos de visitas a puerto (submuestra 50k)."),
                html.Div("Idea: el Mediterráneo como red logística dinámica (hubs + conexiones + evolución)."),
            ],
        ),

        dcc.Tabs(
            value="tab-hubs",
            children=[
                dcc.Tab(
                    label="Hubs (Mapa)",
                    value="tab-hubs",
                    children=[
                        html.Div(
                            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "alignItems": "center"},
                            children=[
                                html.Div("Métrica (tamaño del hub):"),
                                dcc.Dropdown(
                                    id="hubs-metric",
                                    value="total_strength",
                                    clearable=False,
                                    options=[
                                        {"label": "Centralidad (grado ponderado)", "value": "total_strength"},
                                        {"label": "Visitas", "value": "visits"},
                                        {"label": "Barcos únicos", "value": "vessels_unique"},
                                        {"label": "Entradas ponderadas", "value": "in_strength"},
                                        {"label": "Salidas ponderadas", "value": "out_strength"},
                                    ],
                                    style={"minWidth": "320px"},
                                ),
                            ],
                        ),
                        dcc.Graph(id="hubs-graph"),
                        html.Div(
                            style={"color": "#333", "marginTop": "6px"},
                            children=[
                                html.B(""),
                                "Los hubs no solo son puertos grandes; también aparecen hubs de ferris/turismo (islas). "
                                "Esto muestra que la red mezcla logística y movilidad regional.",
                            ],
                        ),
                    ],
                ),

                dcc.Tab(
                    label="Rutas (Mapa)",
                    value="tab-routes",
                    children=[
                        html.Div(
                            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "alignItems": "center"},
                            children=[
                                html.Div("Ranking por:"),
                                dcc.Dropdown(
                                    id="routes-rankby",
                                    value="trips",
                                    clearable=False,
                                    options=[
                                        {"label": "Viajes (conteo de transiciones)", "value": "trips"},
                                        {"label": "Barcos únicos", "value": "vessels_unique"},
                                    ],
                                    style={"minWidth": "260px"},
                                ),
                                html.Div("Top N:"),
                                # ✅ En dcc 3.3.0 Slider no acepta style -> usamos wrapper
                                html.Div(
                                    style={"width": "380px"},
                                    children=[
                                        dcc.Slider(
                                            id="routes-topn",
                                            min=50,
                                            max=500,
                                            step=50,
                                            value=200,
                                            marks={50: "50", 200: "200", 500: "500"},
                                            tooltip={"placement": "bottom", "always_visible": False},
                                            updatemode="mouseup",
                                        )
                                    ],
                                ),
                            ],
                        ),
                        dcc.Graph(id="routes-graph"),
                        html.Div(
                            style={"color": "#333", "marginTop": "6px"},
                            children=[
                                html.B(" "),
                                "Este mapa muestra conexiones agregadas entre puertos (red), no trayectorias individuales. "
                                "Sirve para identificar corredores y cuellos de botella.",
                            ],
                        ),
                    ],
                ),

                dcc.Tab(
                    label="Temporal",
                    value="tab-time",
                    children=[
                        html.Div(
                            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "alignItems": "center"},
                            children=[
                                html.Div("Puerto (opcional):"),
                                dcc.Dropdown(
                                    id="time-port",
                                    value=None,
                                    clearable=True,
                                    placeholder="Total Mediterráneo (sin filtro)",
                                    options=port_options,
                                    style={"minWidth": "520px"},
                                ),
                            ],
                        ),
                        dcc.Graph(id="time-graph"),
                        html.Div(
                            style={"color": "#333", "marginTop": "6px"},
                            children=[
                                html.B(" "),
                                "Permite ver variaciones temporales y comparar si un hub es estable o presenta picos puntuales.",
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------- Callbacks ----------
@app.callback(Output("hubs-graph", "figure"), Input("hubs-metric", "value"))
def update_hubs(metric):
    return make_hubs_figure(metric)


@app.callback(
    Output("routes-graph", "figure"),
    Input("routes-rankby", "value"),
    Input("routes-topn", "value"),
)
def update_routes(rank_by, top_n):
    return make_routes_figure(rank_by, int(top_n))


@app.callback(Output("time-graph", "figure"), Input("time-port", "value"))
def update_time(port_id):
    return make_time_figure(port_id)


if __name__ == "__main__":
    app.run(debug=True)
