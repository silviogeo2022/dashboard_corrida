import os
import json
import webbrowser
import html
from datetime import datetime, timedelta
from typing import List, Tuple

import pandas as pd
import numpy as np
import polyline  # pip install polyline
from flask import (
    Flask,
    redirect,
    request,
    url_for,
    session as flask_session,
    jsonify,
    send_file,
    render_template,
    flash,
)
from dotenv import load_dotenv
from requests_oauthlib import OAuth2Session

# Novas imports para geoprocessamento
try:
    import geopandas as gpd
    from shapely.geometry import LineString
except Exception:
    gpd = None
    LineString = None

print(">>> STARTING app.py FROM:", __file__)

load_dotenv()
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"  # dev only

CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")
REDIRECT_URI = os.getenv("STRAVA_REDIRECT_URI", "http://127.0.0.1:5000/callback")

SCOPES = ["profile:read_all", "activity:read_all", "read"]
AUTH_BASE_URL = "https://www.strava.com/oauth/authorize"
TOKEN_URL = "https://www.strava.com/api/v3/oauth/token"

TOKEN_FILE = "strava_token.json"
ACTIVITIES_FILE = "activities.json"
CSV_ENXUTO = "atividades_strava_enxutas.csv"
MUNICIPIOS_GEOJSON = "municipios_final.geojson"

# Ordem/labels da aba Distâncias — ATUALIZADO
DIST_GROUP_ORDER = [
    "Até 5,0 km",
    "De 5 a 10 km",
    "Acima de 10 km",
]

app = Flask(__name__, template_folder="templates")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev_secret_key")


# ----
# Carregar municipios (GeoJSON) — feito ao iniciar a app
# ----
GDF_MUNICIPIOS = None
if gpd is not None and LineString is not None:
    try:
        if os.path.exists(MUNICIPIOS_GEOJSON):
            GDF_MUNICIPIOS = gpd.read_file(MUNICIPIOS_GEOJSON)
            # garantir CRS WGS84 (lon/lat)
            try:
                if GDF_MUNICIPIOS.crs is None:
                    GDF_MUNICIPIOS.set_crs(epsg=4326, inplace=True)
                else:
                    GDF_MUNICIPIOS = GDF_MUNICIPIOS.to_crs(epsg=4326)
            except Exception:
                app.logger.warning("Não foi possível ajustar CRS de municipios_final.geojson")
        else:
            app.logger.warning(f"{MUNICIPIOS_GEOJSON} não encontrado — spatial join desabilitado")
    except Exception as e:
        app.logger.exception("Falha ao carregar municipios_final.geojson: %s", e)
        GDF_MUNICIPIOS = None
else:
    app.logger.warning("geopandas/shapely não disponíveis — spatial join desabilitado")


# ----
# token / oauth helpers
# ----
def save_token(token):
    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        json.dump(token, f, ensure_ascii=False, indent=2)


def load_token():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def is_token_expired(token):
    return token.get("expires_at") and datetime.now().timestamp() > token["expires_at"]


def refresh_token(oauth_session, token):
    try:
        new_token = oauth_session.refresh_token(
            token_url=TOKEN_URL,
            refresh_token=token["refresh_token"],
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
        )
        save_token(new_token)
        return new_token
    except Exception as e:
        app.logger.error("Erro ao atualizar token: %s", e)
        return None


def get_oauth_session():
    token = load_token()
    if not token:
        return None
    if is_token_expired(token):
        oauth = OAuth2Session(client_id=CLIENT_ID, token=token)
        new_token = refresh_token(oauth, token)
        if new_token:
            return OAuth2Session(client_id=CLIENT_ID, token=new_token)
        return None
    return OAuth2Session(client_id=CLIENT_ID, token=token)


# ----
# polyline decode + normalização
# ----
def tentar_decodificar(poly_str: str) -> List[Tuple[float, float]]:
    """Tenta várias limpezas e decodifica uma polyline; retorna [] se falhar."""
    if not isinstance(poly_str, str) or not poly_str.strip():
        return []

    variantes = []
    raw = poly_str
    variantes.append(raw)
    variantes.append(raw.strip())
    variantes.append(raw.strip().lstrip('|` '))
    variantes.append(raw.strip().replace(' ', ''))
    try:
        variantes.append(raw.encode('utf-8', 'ignore').decode('unicode_escape', 'ignore'))
    except Exception:
        pass
    try:
        variantes.append(html.unescape(raw))
    except Exception:
        pass

    seen = set()
    for v in variantes:
        if not v or v in seen:
            continue
        seen.add(v)
        try:
            coords = polyline.decode(v)
            if isinstance(coords, list) and len(coords) >= 2:
                return [[float(lat), float(lon)] for lat, lon in coords]
        except Exception:
            continue
    return []


def normalize_coord_order(coords: List[List[float]]) -> List[List[float]]:
    if not coords or len(coords) == 0:
        return coords

    count_first_out = sum(1 for a, b in coords if (not pd.isna(a) and abs(a) > 90))
    count_second_out = sum(1 for a, b in coords if (not pd.isna(b) and abs(b) > 90))

    if count_first_out > count_second_out:
        try:
            normalized = [[float(b), float(a)] for a, b in coords]
            return normalized
        except Exception:
            pass

    try:
        lat_vals = [float(p[0]) for p in coords]
        if max(lat_vals) > 90 or min(lat_vals) < -90:
            normalized = [[float(b), float(a)] for a, b in coords]
            return normalized
    except Exception:
        pass

    try:
        return [[float(p[0]), float(p[1])] for p in coords]
    except Exception:
        out = []
        for p in coords:
            try:
                out.append([float(p[0]), float(p[1])])
            except Exception:
                out.append([0.0, 0.0])
        return out


# ----
# misc helpers
# ----
def secs_to_hms(s):
    try:
        return str(timedelta(seconds=int(s)))
    except Exception:
        return "0:00:00"


def to_native(obj):
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return to_native(obj.tolist())
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


# ----
# helper: spatial join em um dataframe
# ----
def _apply_spatial_join(df: pd.DataFrame) -> pd.DataFrame:
    """Faz spatial join do df (que já tem map_summary_polyline) com GDF_MUNICIPIOS."""
    if gpd is None or GDF_MUNICIPIOS is None:
        df["NM_UF"] = None
        df["NM_MUN"] = None
        return df

    geom_list = []
    for _, r in df.iterrows():
        raw_poly = r.get("map_summary_polyline") or ""
        coords = tentar_decodificar(raw_poly) if raw_poly else []
        if coords and len(coords) >= 2:
            coords_float = normalize_coord_order(coords)
            try:
                coords_lonlat = [(float(p[1]), float(p[0])) for p in coords_float]
                geom = LineString(coords_lonlat) if LineString is not None else None
            except Exception:
                geom = None
        else:
            geom = None
        geom_list.append(geom)

    try:
        poly_gdf = gpd.GeoDataFrame(df[["id"]].copy(), geometry=geom_list, crs="EPSG:4326")
        joined = gpd.sjoin(
            poly_gdf,
            GDF_MUNICIPIOS[["NM_UF", "NM_MUN", "geometry"]],
            how="left",
            predicate="intersects",
        )
        if not joined.empty:
            joined_simple = joined[["id", "NM_UF", "NM_MUN"]].groupby("id").first()
            df = df.set_index("id")
            df["NM_UF"] = joined_simple["NM_UF"]
            df["NM_MUN"] = joined_simple["NM_MUN"]
            df = df.reset_index()
        else:
            df["NM_UF"] = None
            df["NM_MUN"] = None
    except Exception as e:
        app.logger.exception("Erro durante spatial join: %s", e)
        df["NM_UF"] = None
        df["NM_MUN"] = None

    return df


# ----
# processamento principal
# ----
def process_activities(activities, filters=None, coluna_poly="map.summary_polyline"):
    """Processa atividades e retorna estatísticas + lista de polylines."""
    if filters is None:
        filters = {}

    records = []
    for act in activities:
        try:
            poly_raw = act.get("map", {}).get("summary_polyline", "") or ""
        except Exception:
            poly_raw = ""
        records.append(
            {
                "id": act.get("id"),
                "name": act.get("name"),
                "type": act.get("type"),
                "start_date": act.get("start_date"),
                "distance": act.get("distance"),
                "moving_time": act.get("moving_time"),
                "average_speed": act.get("average_speed"),
                "kilojoules": act.get("kilojoules"),
                "average_heartrate": act.get("average_heartrate"),
                "map_summary_polyline": poly_raw,
            }
        )

    df = pd.DataFrame(records)

    if df.empty:
        return {
            "total_activities": 0,
            "total_distance_km": 0.0,
            "total_moving_hours": 0.0,
            "avg_speed_overall_kmh": 0.0,
            "total_kilojoules": 0.0,
            "avg_pace_minutes": 0,
            "avg_pace_seconds": 0,
            "avg_heartrate": 0.0,
            "avg_daily_distance_km": 0.0,
            "avg_moving_time_min": 0.0,
            "activities_by_type": {},
            "distance_buckets": {},
            "kJ_buckets": {},
            "distance_by_month": {},
            "recent_list": [],
            "points": [],
            "polylines": [],
            "stats": {},
            "years": [],
            "months": [],
            "estados": [],
            "municipios": [],
            "dist_group_sum": {k: 0.0 for k in DIST_GROUP_ORDER},
            "dist_group_count": {k: 0 for k in DIST_GROUP_ORDER},
            "max_dist_km": 0.0,
            "max_dist_date": "",
            "max_dist_name": "",
        }

    # colunas básicas
    df["distance_km"] = pd.to_numeric(df["distance"], errors="coerce") / 1000.0
    df["moving_time"] = pd.to_numeric(df["moving_time"], errors="coerce").fillna(0).astype(int)
    df["moving_hours"] = df["moving_time"] / 3600.0
    df["avg_speed_kmh"] = pd.to_numeric(df["average_speed"], errors="coerce") * 3.6
    df["kilojoules"] = pd.to_numeric(df["kilojoules"], errors="coerce").fillna(0)
    df["average_heartrate"] = pd.to_numeric(df["average_heartrate"], errors="coerce").fillna(0)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["moving_time_hms"] = df["moving_time"].apply(secs_to_hms)
    df["distance_bucket"] = df["distance_km"].apply(
        lambda d: "<5 km" if pd.notna(d) and d < 5
        else ("5-10 km" if pd.notna(d) and d < 10
              else ("10-20 km" if pd.notna(d) and d < 20 else ">20 km"))
    )
    df["year"] = df["start_date"].dt.year
    df["month"] = df["start_date"].dt.month
    df["month_name"] = df["start_date"].dt.strftime("%B")

    # filtros básicos por type/year/month (antes do spatial join)
    if filters.get("type"):
        df = df[df["type"] == filters["type"]]
    if filters.get("year"):
        try:
            df = df[df["year"] == int(filters["year"])]
        except Exception:
            pass
    if filters.get("month"):
        try:
            df = df[df["month"] == int(filters["month"])]
        except Exception:
            pass

    df = df.reset_index(drop=True)

    # --- Spatial join ---
    df = _apply_spatial_join(df)

    # estados/municípios disponíveis (para selects)
    try:
        estados_disponiveis = sorted(df["NM_UF"].dropna().unique().tolist())
    except Exception:
        estados_disponiveis = []

    try:
        if filters.get("estado"):
            municipios_disponiveis = sorted(
                df.loc[df["NM_UF"] == filters["estado"], "NM_MUN"].dropna().unique().tolist()
            )
        else:
            municipios_disponiveis = sorted(df["NM_MUN"].dropna().unique().tolist())
    except Exception:
        municipios_disponiveis = []

    # filtros espaciais
    if filters.get("estado"):
        df = df[df["NM_UF"] == filters["estado"]]
    if filters.get("municipio"):
        df = df[df["NM_MUN"] == filters["municipio"]]

    # métricas
    total_activities = int(len(df))
    total_distance_km = float(df["distance_km"].sum(skipna=True))
    total_moving_hours = float(df["moving_hours"].sum(skipna=True))
    total_kilojoules = float(df["kilojoules"].sum(skipna=True))

    if total_distance_km > 0:
        total_seconds = df["moving_time"].sum()
        avg_pace_per_km = total_seconds / total_distance_km
    else:
        avg_pace_per_km = 0

    avg_pace_minutes = int(avg_pace_per_km // 60) if avg_pace_per_km else 0
    avg_pace_seconds = int(avg_pace_per_km % 60) if avg_pace_per_km else 0

    avg_heartrate = float(df["average_heartrate"].mean()) if len(df) > 0 else 0.0

    if df["moving_time"].sum() > 0:
        total_hours = df["moving_time"].sum() / 3600.0
        avg_speed_overall_kmh = (df["distance_km"].sum() / total_hours) if total_hours > 0 else 0.0
    else:
        avg_speed_overall_kmh = float(df["avg_speed_kmh"].mean(skipna=True) or 0.0)

    activities_by_type = df["type"].fillna("Unknown").value_counts().to_dict()
    distance_buckets = df["distance_bucket"].fillna("Sem dado").value_counts().to_dict()
    kJ_buckets = {}

    distance_by_month = df.groupby("month_name")["distance_km"].sum().to_dict()

    # ----
    # Aba Distâncias — categorias e contagem ATUALIZADAS
    # ----
    def dist_group_label(d):
        if pd.isna(d) or d <= 0:
            return None
        if d <= 5.0:
            return "Até 5,0 km"
        elif d <= 10.0:
            return "De 5 a 10 km"
        else:
            return "Acima de 10 km"

    df["dist_group"] = df["distance_km"].apply(dist_group_label)

    # Soma de km por categoria
    dist_group_sum = (
        df.groupby("dist_group", dropna=False)["distance_km"]
        .sum()
        .reindex(DIST_GROUP_ORDER)
        .fillna(0)
        .to_dict()
    )

    # Contagem de atividades com distância > 0 por categoria
    dist_group_count = (
        df.dropna(subset=["dist_group"])
        .groupby("dist_group")["distance_km"]
        .size()
        .reindex(DIST_GROUP_ORDER, fill_value=0)
        .to_dict()
    )

    # Maior distância
    max_dist_km = 0.0
    max_dist_date = ""
    max_dist_name = ""
    if total_activities > 0 and df["distance_km"].notna().any():
        try:
            max_dist_row = df.loc[df["distance_km"].idxmax()]
            max_dist_km = float(max_dist_row.get("distance_km") or 0.0)
            max_dist_date_raw = max_dist_row.get("start_date")
            try:
                max_dist_date = pd.to_datetime(max_dist_date_raw).strftime("%Y-%m-%d") if pd.notna(max_dist_date_raw) else ""
            except Exception:
                max_dist_date = ""
            max_dist_name = str(max_dist_row.get("name") or "")
        except Exception:
            pass

    # médias
    try:
        daily_sums = df.groupby(df["start_date"].dt.date)["distance_km"].sum()
        avg_daily_distance_km = float(daily_sums.mean()) if len(daily_sums) > 0 else 0.0
    except Exception:
        avg_daily_distance_km = 0.0

    try:
        avg_moving_time_min = float(df["moving_time"].mean() / 60.0) if len(df) > 0 else 0.0
    except Exception:
        avg_moving_time_min = 0.0

    # salvar CSV enxuto
    df_to_save = df[[
        "id", "name", "type", "start_date", "distance_km", "moving_time",
        "moving_time_hms", "avg_speed_kmh", "kilojoules", "map_summary_polyline",
        "NM_UF", "NM_MUN"
    ]].copy()
    df_to_save["start_date"] = pd.to_datetime(df_to_save["start_date"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    try:
        df_to_save.to_csv(CSV_ENXUTO, index=False, encoding="utf-8")
    except Exception:
        pass

    recent_activities = df.sort_values("start_date", ascending=False).head(200)

    recent_list = []
    points = []
    polylines = []
    failed_list = []

    for _, row in recent_activities.iterrows():
        raw_poly = row.get("map_summary_polyline") or ""
        coords = tentar_decodificar(raw_poly) if raw_poly else []

        start_date_txt = ""
        if pd.notna(row.get("start_date")):
            try:
                start_date_txt = pd.to_datetime(row.get("start_date")).strftime("%Y-%m-%d %H:%M")
            except Exception:
                start_date_txt = str(row.get("start_date"))

        recent_item = {
            "id": to_native(row.get("id")),
            "name": to_native(row.get("name") or ""),
            "type": to_native(row.get("type") or "Unknown"),
            "start_date": to_native(start_date_txt),
            "distance_km": to_native(round(float(row.get("distance_km") or 0.0), 2)),
            "moving_time_hms": to_native(row.get("moving_time_hms") or "0:00:00"),
            "avg_speed_kmh": to_native(round(float(row.get("avg_speed_kmh") or 0.0), 2)),
            "kilojoules": to_native(round(float(row.get("kilojoules") or 0.0), 0)),
            "average_heartrate": to_native(round(float(row.get("average_heartrate") or 0.0), 0)),
            "map_summary_polyline": to_native(raw_poly),
            "NM_UF": to_native(row.get("NM_UF")),
            "NM_MUN": to_native(row.get("NM_MUN")),
        }
        recent_list.append(recent_item)

        if coords and len(coords) >= 2:
            coords_float = normalize_coord_order(coords)
            start_coord = coords_float[0] if coords_float else None
            end_coord = coords_float[-1] if coords_float else None

            polylines.append(
                {
                    "id": to_native(row.get("id")),
                    "name": to_native(row.get("name") or ""),
                    "type": to_native(row.get("type") or "Unknown"),
                    "coords": coords_float,
                    "distance_km": to_native(round(float(row.get("distance_km") or 0.0), 2)),
                    "moving_time_hms": to_native(row.get("moving_time_hms") or "0:00:00"),
                    "avg_speed_kmh": to_native(round(float(row.get("avg_speed_kmh") or 0.0), 2)),
                    "kilojoules": to_native(round(float(row.get("kilojoules") or 0.0), 0)),
                    "start_date": to_native(start_date_txt),
                    "start_coord": to_native(start_coord),
                    "end_coord": to_native(end_coord),
                    "NM_UF": to_native(row.get("NM_UF")),
                    "NM_MUN": to_native(row.get("NM_MUN")),
                }
            )
        else:
            if raw_poly:
                preview = (str(raw_poly)[:400] if raw_poly is not None else "")
                failed_list.append({"id": to_native(row.get("id")), "poly_preview": preview})

        if coords:
            mid = coords[len(coords) // 2]
            mid_norm = normalize_coord_order([mid])[0]
            points.append(
                {
                    "id": to_native(row.get("id")),
                    "latitude": float(mid_norm[0]),
                    "longitude": float(mid_norm[1]),
                    "type": to_native(row.get("type") or "Unknown"),
                    "distance_km": to_native(round(float(row.get("distance_km") or 0.0), 2)),
                    "start_date": to_native(start_date_txt),
                    "NM_UF": to_native(row.get("NM_UF")),
                    "NM_MUN": to_native(row.get("NM_MUN")),
                }
            )

    if failed_list:
        try:
            df_failed = pd.DataFrame(failed_list)
            df_failed.to_csv("failed_polylines.csv", index=False, encoding="utf-8")
        except Exception:
            pass

    stats = {
        "total": int(total_activities),
        "avg_distance": float(total_distance_km / total_activities) if total_activities > 0 else 0.0,
        "avg_pace_minutes": avg_pace_minutes,
        "avg_pace_seconds": avg_pace_seconds,
        "avg_heartrate": avg_heartrate,
        "distance_by_month": distance_by_month,
        "avg_daily_distance_km": avg_daily_distance_km,
        "avg_moving_time_min": avg_moving_time_min,
    }

    try:
        with open("polylines_debug_sample.json", "w", encoding="utf-8") as fh:
            json.dump(polylines[:10], fh, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return {
        "total_activities": int(total_activities),
        "total_distance_km": float(total_distance_km),
        "total_moving_hours": float(total_moving_hours),
        "avg_speed_overall_kmh": float(avg_speed_overall_kmh or 0.0),
        "total_kilojoules": float(total_kilojoules),
        "avg_pace_minutes": avg_pace_minutes,
        "avg_pace_seconds": avg_pace_seconds,
        "avg_heartrate": avg_heartrate,
        "avg_daily_distance_km": float(avg_daily_distance_km or 0.0),
        "avg_moving_time_min": float(avg_moving_time_min or 0.0),
        "activities_by_type": to_native(activities_by_type),
        "distance_buckets": to_native(distance_buckets),
        "kJ_buckets": to_native(kJ_buckets),
        "distance_by_month": to_native(distance_by_month),
        "recent_list": to_native(recent_list),
        "points": to_native(points),
        "polylines": to_native(polylines),
        "stats": to_native(stats),
        "years": sorted(df["year"].dropna().unique().tolist()),
        "months": sorted(df["month"].dropna().unique().tolist()),
        "estados": estados_disponiveis,
        "municipios": municipios_disponiveis,
        "dist_group_sum": to_native(dist_group_sum),
        "dist_group_count": to_native(dist_group_count),
        "max_dist_km": float(max_dist_km),
        "max_dist_date": max_dist_date,
        "max_dist_name": max_dist_name,
    }


# ----
# rotas
# ----
@app.route("/")
def index():
    token = load_token()
    ok = token is not None and not is_token_expired(token)
    html_resp = f"""
    <h2>Strava OAuth (Flask)</h2>
    <p>Token presente: {bool(token)}</p>
    <p>Token válido: {ok}</p>
    <ul>
      <li><a href="/authorize">Autorizar Strava</a></li>
      <li><a href="/activities">Baixar atividades</a></li>
      <li><a href="/kpis">Ver KPIs</a></li>
    </ul>
    """
    return html_resp


@app.route("/authorize")
def authorize():
    oauth = OAuth2Session(client_id=CLIENT_ID, redirect_uri=REDIRECT_URI, scope=SCOPES)
    auth_url, state = oauth.authorization_url(AUTH_BASE_URL, approval_prompt="auto")
    flask_session["oauth_state"] = state
    return redirect(auth_url)


@app.route("/callback")
def callback():
    state = flask_session.get("oauth_state")
    oauth = OAuth2Session(client_id=CLIENT_ID, state=state, redirect_uri=REDIRECT_URI)
    try:
        token = oauth.fetch_token(
            token_url=TOKEN_URL,
            client_secret=CLIENT_SECRET,
            authorization_response=request.url,
            include_client_id=True,
        )
    except Exception as e:
        return f"Erro ao trocar code por token: {e}", 400

    save_token(token)
    flask_session["token"] = token
    return redirect(url_for("kpis"))


@app.route("/activities")
def activities():
    oauth = get_oauth_session()
    if not oauth:
        return redirect(url_for("authorize"))

    activities = []
    per_page = 200
    page = 1
    try:
        while True:
            resp = oauth.get(
                "https://www.strava.com/api/v3/athlete/activities",
                params={"per_page": per_page, "page": page},
            )
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            activities.extend(data)
            page += 1
    except Exception as e:
        return f"Erro ao baixar atividades: {e}", 500

    with open(ACTIVITIES_FILE, "w", encoding="utf-8") as f:
        json.dump(activities, f, ensure_ascii=False, indent=2)

    process_activities(activities)
    flash(f"Atividades baixadas: {len(activities)}", "success")
    return redirect(url_for("kpis"))


@app.route("/kpis")
def kpis():
    if os.path.exists(ACTIVITIES_FILE):
        with open(ACTIVITIES_FILE, "r", encoding="utf-8") as f:
            activities = json.load(f)
    else:
        return redirect(url_for("activities"))

    selected_type = request.args.get("type", default="", type=str) or ""
    selected_year = request.args.get("year", default="", type=str) or ""
    selected_month = request.args.get("month", default="", type=str) or ""
    selected_estado = request.args.get("estado", default="", type=str) or ""
    selected_municipio = request.args.get("municipio", default="", type=str) or ""

    filters = {}
    if selected_type:
        filters["type"] = selected_type
    if selected_year:
        filters["year"] = selected_year
    if selected_month:
        filters["month"] = selected_month
    if selected_estado:
        filters["estado"] = selected_estado
    if selected_municipio:
        filters["municipio"] = selected_municipio

    kpi_result = process_activities(activities, filters=filters)

    polylines_raw = kpi_result.get("polylines", []) or []
    try:
        safe_polylines = json.loads(json.dumps(polylines_raw, default=str, ensure_ascii=False))
    except Exception:
        safe_polylines = []

    safe_polylines_sample = safe_polylines[:3] if isinstance(safe_polylines, list) else []

    recent_list = kpi_result.get("recent_list", []) or []
    activities_by_type = kpi_result.get("activities_by_type", {}) or {}
    distance_buckets = kpi_result.get("distance_buckets", {}) or {}
    kJ_buckets = kpi_result.get("kJ_buckets", {}) or {}
    points = kpi_result.get("points", []) or []
    years = kpi_result.get("years", []) or []
    estados = kpi_result.get("estados", []) or []
    municipios = kpi_result.get("municipios", []) or []

    try:
        with open(ACTIVITIES_FILE, "r", encoding="utf-8") as f:
            all_activities_raw = json.load(f)
        all_types = sorted(set(a.get("type") for a in all_activities_raw if a.get("type")))
    except Exception:
        all_types = sorted(list(activities_by_type.keys()))

    return render_template(
        "index.html",
        title="Painel Strava",
        stats=kpi_result.get("stats", {}) or {},
        points=points,
        polylines=safe_polylines,
        polylines_count=len(safe_polylines),
        polylines_sample=safe_polylines_sample,
        tipos_evento=all_types,
        tipos_distance=sorted(list(distance_buckets.keys())),
        tipos_kj=sorted(list(kJ_buckets.keys())),
        anos=sorted(years, reverse=True),
        meses=list(range(1, 13)),
        selected_type=selected_type,
        selected_year=selected_year,
        selected_month=selected_month,
        selected_estado=selected_estado,
        selected_municipio=selected_municipio,
        estados=estados,
        municipios=municipios,
        now=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_activities=int(kpi_result.get("total_activities", 0) or 0),
        total_distance_km=float(kpi_result.get("total_distance_km", 0.0) or 0.0),
        total_moving_hours=float(kpi_result.get("total_moving_hours", 0.0) or 0.0),
        avg_speed_overall_kmh=float(kpi_result.get("avg_speed_overall_kmh", 0.0) or 0.0),
        total_kilojoules=float(kpi_result.get("total_kilojoules", 0.0) or 0.0),
        avg_pace_minutes=int(kpi_result.get("avg_pace_minutes", 0) or 0),
        avg_pace_seconds=int(kpi_result.get("avg_pace_seconds", 0) or 0),
        avg_heartrate=float(kpi_result.get("avg_heartrate", 0.0) or 0.0),
        avg_daily_distance_km=float(kpi_result.get("avg_daily_distance_km", 0.0) or 0.0),
        avg_moving_time_min=float(kpi_result.get("avg_moving_time_min", 0.0) or 0.0),
        activities_by_type=activities_by_type,
        distance_buckets=distance_buckets,
        kJ_buckets=kJ_buckets,
        recent_activities=recent_list,
        dist_group_sum=kpi_result.get("dist_group_sum", {k: 0.0 for k in DIST_GROUP_ORDER}),
        dist_group_count=kpi_result.get("dist_group_count", {k: 0 for k in DIST_GROUP_ORDER}),
        max_dist_km=float(kpi_result.get("max_dist_km", 0.0) or 0.0),
        max_dist_date=kpi_result.get("max_dist_date", ""),
        max_dist_name=kpi_result.get("max_dist_name", ""),
    )


# ----
# ROTA: retorna opções de filtro dependentes via AJAX
# ----
@app.route("/api/filter-options")
def api_filter_options():
    if not os.path.exists(ACTIVITIES_FILE):
        return jsonify({"estados": [], "municipios": []})

    try:
        with open(ACTIVITIES_FILE, "r", encoding="utf-8") as f:
            activities = json.load(f)
    except Exception:
        return jsonify({"estados": [], "municipios": []})

    selected_type = request.args.get("type", "").strip()
    selected_year = request.args.get("year", "").strip()
    selected_month = request.args.get("month", "").strip()
    selected_estado = request.args.get("estado", "").strip()

    records = []
    for act in activities:
        try:
            poly_raw = act.get("map", {}).get("summary_polyline", "") or ""
        except Exception:
            poly_raw = ""
        records.append({
            "id": act.get("id"),
            "type": act.get("type"),
            "start_date": act.get("start_date"),
            "map_summary_polyline": poly_raw,
        })

    df = pd.DataFrame(records)
    if df.empty:
        return jsonify({"estados": [], "municipios": []})

    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["year"] = df["start_date"].dt.year
    df["month"] = df["start_date"].dt.month

    if selected_type:
        df = df[df["type"] == selected_type]
    if selected_year:
        try:
            df = df[df["year"] == int(selected_year)]
        except Exception:
            pass
    if selected_month:
        try:
            df = df[df["month"] == int(selected_month)]
        except Exception:
            pass

    df = df.reset_index(drop=True)
    df = _apply_spatial_join(df)

    try:
        estados = sorted(df["NM_UF"].dropna().unique().tolist())
    except Exception:
        estados = []

    try:
        if selected_estado:
            municipios = sorted(
                df.loc[df["NM_UF"] == selected_estado, "NM_MUN"].dropna().unique().tolist()
            )
        else:
            municipios = sorted(df["NM_MUN"].dropna().unique().tolist())
    except Exception:
        municipios = []

    return jsonify({"estados": estados, "municipios": municipios})


@app.route("/download_csv")
def download_csv():
    if os.path.exists(CSV_ENXUTO):
        return send_file(CSV_ENXUTO, as_attachment=True)
    return redirect(url_for("kpis"))


@app.route("/debug_polylines")
def debug_polylines():
    if os.path.exists("polylines_debug_sample.json"):
        return send_file("polylines_debug_sample.json", as_attachment=True)
    return jsonify({"error": "debug sample not found"}), 404


@app.route("/polylines.geojson")
def polylines_geojson():
    if not os.path.exists(ACTIVITIES_FILE):
        return jsonify({"type": "FeatureCollection", "features": []})

    try:
        with open(ACTIVITIES_FILE, "r", encoding="utf-8") as f:
            activities = json.load(f)
    except Exception as e:
        app.logger.exception("polylines.geojson: erro lendo activities.json: %s", e)
        return jsonify({"type": "FeatureCollection", "features": []})

    filters = {}
    for key in ("type", "year", "month", "estado", "municipio"):
        v = request.args.get(key)
        if v:
            filters[key] = v

    try:
        kpi_result = process_activities(activities, filters=filters)
        polylines = kpi_result.get("polylines", []) or []
    except Exception:
        app.logger.exception("polylines.geojson: process_activities falhou")
        polylines = []

    features = []

    def valid_latlon_pair(p):
        try:
            if p is None or len(p) < 2:
                return False
            lat = float(p[0])
            lon = float(p[1])
            return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0
        except Exception:
            return False

    for p in polylines:
        coords = p.get("coords", []) or []
        if not coords:
            continue
        try:
            coords_clean = [[float(x[0]), float(x[1])] for x in coords if x is not None and len(x) >= 2]
        except Exception:
            continue
        coords_clean = [c for c in coords_clean if valid_latlon_pair(c)]
        if len(coords_clean) < 2:
            continue
        coords_geo = [[float(c[1]), float(c[0])] for c in coords_clean]
        properties = {
            "id": p.get("id"),
            "name": p.get("name"),
            "type": p.get("type"),
            "distance_km": p.get("distance_km"),
            "moving_time_hms": p.get("moving_time_hms"),
            "avg_speed_kmh": p.get("avg_speed_kmh"),
            "kilojoules": p.get("kilojoules"),
            "start_date": p.get("start_date"),
            "NM_UF": p.get("NM_UF"),
            "NM_MUN": p.get("NM_MUN"),
        }
        features.append({
            "type": "Feature",
            "properties": properties,
            "geometry": {"type": "LineString", "coordinates": coords_geo},
        })

    feature_collection = {"type": "FeatureCollection", "features": features}
    try:
        with open("polylines_geojson_debug.json", "w", encoding="utf-8") as fh:
            json.dump(feature_collection, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return jsonify(feature_collection)


if __name__ == "__main__":
    try:
        webbrowser.open("http://127.0.0.1:5000/", new=2)
    except Exception:
        pass
    app.run(host="127.0.0.1", port=5000, debug=True)