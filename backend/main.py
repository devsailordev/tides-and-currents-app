# backend/main.py
import math
from datetime import date, datetime, timedelta
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()

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
    epsilon = 0.05
    if signed_speed > epsilon:
        return {"direction": "Flood", "speed": round(signed_speed, 2)}
    if signed_speed < -epsilon:
        return {"direction": "Ebb", "speed": round(abs(signed_speed), 2)}
    return {"direction": "Slack", "speed": 0.0}


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


@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


# Mount the frontend static files AFTER defining API routes
frontend_path = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="static")
