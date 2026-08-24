# backend/main.py
import math
import re
from datetime import date, datetime, timedelta
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

app = FastAPI()


class NoCacheStaticFiles(StaticFiles):
    """Serve static files without conditional caching so root always reflects latest frontend."""

    def is_not_modified(self, *args, **kwargs) -> bool:  # noqa: ANN002, ANN003
        return False

    def file_response(self, *args, **kwargs):  # noqa: ANN002, ANN003
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

# Bound how much data a single request can generate/fetch, since this is a
# public, unauthenticated API with no per-user quotas.
MAX_HOURS = 48
HTTP_TIMEOUT = 10.0

# Current speed/direction offsets (in knots) relative to the high/low tide at
# a station, one entry per whole hour starting at hour 0 (the tide event
# itself). Positive values = Flood, negative values = Ebb, 0 = Slack.
CURRENT_STATIONS = {
    "pier66": {
        "name": "Pier 66",
        "after_high": [1.9, 1.7, 0.9, 0.0, -1.3, -2.2],
        "after_low": [-2.8, -2.6, -1.9, -1.0, 0.3, 1.3],
    }
}

# weather.gov requires a descriptive User-Agent and rejects requests without one.
WEATHER_HEADERS = {"User-Agent": "tides-and-currents-app (github.com/tides-and-currents-app)"}
WEATHER_LAT = 40.75281480397881
WEATHER_LON = -74.0151581463272

# The points -> forecast URL mapping never changes for a fixed lat/lon, so cache it.
_weather_forecast_urls_cache: dict[str, str] | None = None


async def fetch_weather_forecast_urls() -> dict[str, str]:
    global _weather_forecast_urls_cache
    if _weather_forecast_urls_cache:
        return _weather_forecast_urls_cache

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=WEATHER_HEADERS, follow_redirects=True) as client:
            response = await client.get(f"https://api.weather.gov/points/{WEATHER_LAT},{WEATHER_LON}")
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Unable to reach weather data provider") from exc

    properties = data.get("properties", {})
    forecast_url = properties.get("forecast")
    forecast_hourly_url = properties.get("forecastHourly")
    forecast_grid_data_url = properties.get("forecastGridData")
    if not forecast_url or not forecast_hourly_url:
        raise HTTPException(status_code=502, detail="Weather data provider returned no forecast URL")

    _weather_forecast_urls_cache = {
        "forecast": forecast_url,
        "forecastHourly": forecast_hourly_url,
        "forecastGridData": forecast_grid_data_url,
    }
    return _weather_forecast_urls_cache


_ISO8601_DURATION_RE = re.compile(
    r"P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?"
)


def _parse_iso8601_duration(duration: str) -> timedelta:
    match = _ISO8601_DURATION_RE.fullmatch(duration)
    if not match:
        return timedelta()
    parts = {key: int(value) for key, value in match.groupdict().items() if value}
    return timedelta(
        days=parts.get("days", 0),
        hours=parts.get("hours", 0),
        minutes=parts.get("minutes", 0),
        seconds=parts.get("seconds", 0),
    )


def _parse_valid_time_interval(valid_time: str) -> tuple[datetime, datetime]:
    start_str, duration_str = valid_time.split("/")
    start = datetime.fromisoformat(start_str)
    return start, start + _parse_iso8601_duration(duration_str)


async def fetch_wind_gusts_mph(grid_data_url: str) -> list[tuple[datetime, datetime, int]]:
    """windGust is only exposed as a km/h time-series on forecastGridData, not on the hourly periods."""
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=WEATHER_HEADERS, follow_redirects=True) as client:
            response = await client.get(grid_data_url)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Unable to reach weather data provider") from exc

    values = data.get("properties", {}).get("windGust", {}).get("values", [])
    gusts = []
    for entry in values:
        if entry.get("value") is None:
            continue
        start, end = _parse_valid_time_interval(entry["validTime"])
        gusts.append((start, end, round(entry["value"] / 1.60934)))
    return gusts


def find_wind_gust_mph(gusts: list[tuple[datetime, datetime, int]], when: datetime) -> int | None:
    for start, end, mph in gusts:
        if start <= when < end:
            return mph
    return None


async def fetch_weather_periods(forecast_url: str) -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=WEATHER_HEADERS, follow_redirects=True) as client:
            response = await client.get(forecast_url)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Unable to reach weather data provider") from exc

    return data.get("properties", {}).get("periods", [])


frontend_path = Path(__file__).parent.parent / "frontend"


async def fetch_tide_events(begin: date, end: date):
    params = {
        "begin_date": begin.strftime("%Y%m%d"),
        "end_date": end.strftime("%Y%m%d"),
        "station": "8518750",
        "product": "predictions",
        "datum": "MLLW",
        "time_zone": "lst_ldt",
        "interval": "hilo",
        "units": "english",
        "format": "json",
        "application": "battery_tide_app",
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.get(
                "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter",
                params=params,
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Unable to reach tide data provider") from exc

    events = [
        {
            "time": datetime.strptime(p["t"], "%Y-%m-%d %H:%M"),
            "type": p["type"],
            "v": p["v"],
        }
        for p in data.get("predictions", [])
    ]
    events.sort(key=lambda e: e["time"])
    return events


def bracket_events(when: datetime, events: list):
    """Return the (prev, next) tide events immediately surrounding `when`."""
    prev_event = next_event = None
    for event in events:
        if event["time"] <= when:
            prev_event = event
        elif next_event is None:
            next_event = event
            break
    return prev_event, next_event


def cosine_ease(v0: float, v1: float, fraction: float) -> float:
    """Ease-in-out interpolation between v0 and v1 (fraction in [0, 1])."""
    return v0 + (v1 - v0) / 2 * (1 - math.cos(math.pi * fraction))


def signed_speed_at(when: datetime, events: list, table: dict) -> float:
    """Interpolate the signed current speed (flood positive, ebb negative) at `when`.

    The configured curve only covers hours 0-5 after a tide event, but the
    real gap between a high and low tide is usually ~6+ hours. Rather than
    flatlining at the hour-5 value for the remainder of the gap, the curve is
    ramped the rest of the way to the next event's hour-0 value (the known
    speed exactly at that next high/low), which is reached at its actual time.
    Each segment is eased (cosine) rather than linear so the transition
    between table entries, and across the boundary into the next table, is
    smooth instead of having abrupt slope changes.
    """
    prev_event, next_event = bracket_events(when, events)
    if prev_event is None:
        prev_event = events[0]

    curve = table["after_high"] if prev_event["type"] == "H" else table["after_low"]
    elapsed_hours = (when - prev_event["time"]).total_seconds() / 3600

    if next_event is None:
        elapsed_hours = max(0.0, min(elapsed_hours, len(curve) - 1))
        lower_idx = int(elapsed_hours)
        upper_idx = min(lower_idx + 1, len(curve) - 1)
        fraction = elapsed_hours - lower_idx
        return cosine_ease(curve[lower_idx], curve[upper_idx], fraction)

    next_curve = table["after_high"] if next_event["type"] == "H" else table["after_low"]
    span_hours = (next_event["time"] - prev_event["time"]).total_seconds() / 3600
    elapsed_hours = max(0.0, min(elapsed_hours, span_hours))

    anchors = [(h, curve[h]) for h in range(len(curve)) if h <= span_hours]
    if not anchors or anchors[-1][0] < span_hours:
        anchors.append((span_hours, next_curve[0]))

    for (h0, v0), (h1, v1) in zip(anchors, anchors[1:]):
        if h0 <= elapsed_hours <= h1:
            fraction = 0 if h1 == h0 else (elapsed_hours - h0) / (h1 - h0)
            return cosine_ease(v0, v1, fraction)

    return anchors[-1][1]


def height_at(when: datetime, events: list) -> float | None:
    """Cosine (ease-in-out) interpolated tide height between the two tide
    events bracketing `when`, since real tides do not change linearly."""
    prev_event, next_event = bracket_events(when, events)

    if prev_event is None or next_event is None:
        return None

    prev_height = float(prev_event["v"])
    next_height = float(next_event["v"])
    span_hours = (next_event["time"] - prev_event["time"]).total_seconds() / 3600
    elapsed_hours = (when - prev_event["time"]).total_seconds() / 3600
    fraction = 0 if span_hours == 0 else elapsed_hours / span_hours

    return cosine_ease(prev_height, next_height, fraction)


def describe_current(signed_speed: float) -> dict:
    epsilon = 0.25
    if signed_speed > epsilon:
        return {"direction": "Flood", "speed": round(signed_speed, 2)}
    if signed_speed < -epsilon:
        return {"direction": "Ebb", "speed": round(abs(signed_speed), 2)}
    return {"direction": "Slack", "speed": round(abs(signed_speed), 2)}


@app.get("/api/weather")
async def get_weather(day: date):
    urls = await fetch_weather_forecast_urls()
    periods = await fetch_weather_periods(urls["forecast"])

    return {
        "periods": [
            {
                "name": p["name"],
                "startTime": p["startTime"],
                "endTime": p["endTime"],
                "isDaytime": p["isDaytime"],
                "temperature": p["temperature"],
                "temperatureUnit": p["temperatureUnit"],
                "shortForecast": p["shortForecast"],
                "windSpeed": p["windSpeed"],
                "windDirection": p["windDirection"],
                "icon": p["icon"],
            }
            for p in periods
            if p["isDaytime"]
            and (
                datetime.fromisoformat(p["startTime"]).date() == day
                or datetime.fromisoformat(p["endTime"]).date() == day
            )
        ]
    }


@app.get("/api/weather/hourly")
async def get_weather_hourly(day: date):
    urls = await fetch_weather_forecast_urls()
    periods = await fetch_weather_periods(urls["forecastHourly"])
    gusts = await fetch_wind_gusts_mph(urls["forecastGridData"]) if urls.get("forecastGridData") else []

    return {
        "periods": [
            {
                "startTime": p["startTime"],
                "temperature": p["temperature"],
                "temperatureUnit": p["temperatureUnit"],
                "shortForecast": p["shortForecast"],
                "windSpeed": p["windSpeed"],
                "windGust": (
                    f"{gust_mph} mph"
                    if (gust_mph := find_wind_gust_mph(gusts, datetime.fromisoformat(p["startTime"]))) is not None
                    else None
                ),
                "windDirection": p["windDirection"],
                "icon": p["icon"],
                "probabilityOfPrecipitation": (p.get("probabilityOfPrecipitation") or {}).get("value"),
            }
            for p in periods
            if datetime.fromisoformat(p["startTime"]).date() == day
        ]
    }


@app.get("/api/tides")
async def get_tides(day: date):
    events = await fetch_tide_events(day, day)
    return {
        "predictions": [
            {"t": e["time"].strftime("%Y-%m-%d %H:%M"), "type": e["type"], "v": e["v"]}
            for e in events
        ]
    }


@app.get("/api/currents")
async def get_currents(
    day: date,
    start_time: str = "00:00",
    station: str = "pier66",
    hours: float = 12,
):
    if station not in CURRENT_STATIONS:
        raise HTTPException(status_code=400, detail=f"Unknown station '{station}'")

    if not (0 < hours <= MAX_HOURS):
        raise HTTPException(status_code=400, detail=f"hours must be between 0 and {MAX_HOURS}")

    try:
        start_hour, start_minute = (int(part) for part in start_time.split(":"))
        start_dt = datetime.combine(day, datetime.min.time()).replace(
            hour=start_hour, minute=start_minute
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail="start_time must be in HH:MM format") from exc

    table = CURRENT_STATIONS[station]

    # Fetch a wider window so interpolation near midnight has enough context.
    events = await fetch_tide_events(day - timedelta(days=1), day + timedelta(days=1))
    if not events:
        raise HTTPException(status_code=502, detail="No tide data available for this date")

    steps = int(hours * 2)  # 30-minute intervals
    results = []
    for i in range(steps + 1):
        when = start_dt + timedelta(minutes=30 * i)
        signed = signed_speed_at(when, events, table)
        entry = describe_current(signed)
        entry["time"] = when.strftime("%Y-%m-%d %H:%M")
        height = height_at(when, events)
        entry["height"] = round(height, 2) if height is not None else None
        results.append(entry)

    return {"station": table["name"], "predictions": results}


@app.get("/")
@app.get("/index.html")
async def frontend_index():
    response = FileResponse(frontend_path / "index.html")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["Clear-Site-Data"] = '"cache"'
    return response


# Mount the frontend static files AFTER defining API routes
app.mount("/", NoCacheStaticFiles(directory=str(frontend_path), html=True), name="static")
