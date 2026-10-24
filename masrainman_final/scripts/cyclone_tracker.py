#!/usr/bin/env python3
"""MASRAINMAN Bay of Bengal Cyclone Tracker.

Retrieves ECMWF IFS and AIFS deterministic MSLP forecasts, detects candidate
closed/local pressure minima over the North Indian Ocean, links candidates
between forecast lead times, and writes a map plus machine-readable status.

IMPORTANT: This is experimental deterministic guidance, not an official warning.
Local MSLP minima are candidate lows, not automatically tropical cyclones.
Two deterministic models do not provide calibrated track probabilities.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import xarray as xr
from ecmwf.opendata import Client

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "docs" / "data"
IMAGE_PATH = OUT_DIR / "latest.png"
STATUS_PATH = OUT_DIR / "status.json"
UTC = timezone.utc
IST = ZoneInfo("Asia/Kolkata")

DEFAULT_MODELS = ["ifs", "aifs-single"]
DEFAULT_STEPS = [0, 6, 12, 18, 24, 30, 36, 42, 48, 60, 72]
MODEL_ALIASES = {"ifs": "ifs", "aifs": "aifs-single", "aifs-single": "aifs-single"}
PARAMETERS = ["msl", "10u", "10v"]
MIN_CYCLE_AGE_HOURS = int(os.getenv("ECMWF_MIN_CYCLE_AGE_HOURS", "6"))
CYCLE_FALLBACKS = max(0, int(os.getenv("ECMWF_CYCLE_FALLBACKS", "4")))
RETRIES = max(1, int(os.getenv("ECMWF_RETRIES", "2")))
RETRY_DELAY = max(0, int(os.getenv("ECMWF_RETRY_DELAY_SECONDS", "3")))
# Candidate centres are only searched within this broad regional domain.
REGION = (70.0, 105.0, -2.0, 28.0)  # lon_min, lon_max, lat_min, lat_max
LOW_PRESSURE_LIMIT_HPA = float(os.getenv("LOW_PRESSURE_LIMIT_HPA", "1010"))
TRACK_MAX_STEP_KM = float(os.getenv("TRACK_MAX_STEP_KM", "500"))
MIN_TRACK_POINTS = int(os.getenv("MIN_TRACK_POINTS", "2"))
LOG = logging.getLogger("masrainman")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                    format="%(asctime)s %(levelname)s %(message)s")


def iso(value: datetime, zone=UTC) -> str:
    return value.astimezone(zone).isoformat().replace("+00:00", "Z") if zone is UTC else value.astimezone(zone).isoformat()


def parse_models() -> list[str]:
    raw = os.getenv("ECMWF_MODELS", ",".join(DEFAULT_MODELS))
    models = []
    for part in raw.split(","):
        model = MODEL_ALIASES.get(part.strip().lower())
        if model and model not in models:
            models.append(model)
        elif part.strip():
            LOG.warning("Ignoring unsupported model name: %s", part.strip())
    return models or DEFAULT_MODELS.copy()


def parse_steps() -> list[int]:
    raw = os.getenv("ECMWF_FORECAST_STEPS", ",".join(map(str, DEFAULT_STEPS)))
    values = set()
    for token in raw.split(","):
        try:
            step = int(token.strip())
            if 0 <= step <= 360:
                values.add(step)
        except ValueError:
            LOG.warning("Ignoring invalid forecast step %r", token)
    return sorted(values) or DEFAULT_STEPS.copy()


def floor_cycle(value: datetime) -> datetime:
    value = value.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    return value.replace(hour=(value.hour // 6) * 6)


def candidate_cycles(now: datetime | None = None) -> list[datetime]:
    now = now or datetime.now(UTC)
    newest = floor_cycle(now - timedelta(hours=MIN_CYCLE_AGE_HOURS))
    return [newest - timedelta(hours=6 * i) for i in range(CYCLE_FALLBACKS + 1)]


def cycle_id(cycle: datetime) -> str:
    return cycle.strftime("%Y%m%d%H")


def validate_grib(path: Path) -> None:
    if not path.exists() or path.stat().st_size < 100:
        raise ValueError("Missing, empty, or implausibly small GRIB file")
    ds = None
    try:
        ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
        if not ds.data_vars:
            raise ValueError("No readable variables in GRIB")
    finally:
        if ds is not None:
            ds.close()


def retrieve_step(client, model: str, cycle: datetime, step: int) -> Path:
    path = RAW_DIR / f"{model}_{cycle_id(cycle)}_step{step:03d}.grib2"
    if path.exists():
        try:
            validate_grib(path)
            return path
        except Exception:
            path.unlink(missing_ok=True)
    tmp = path.with_name(path.name + ".part")
    last_error = None
    for attempt in range(1, RETRIES + 1):
        tmp.unlink(missing_ok=True)
        try:
            client.retrieve(date=cycle.strftime("%Y%m%d"), time=int(cycle.strftime("%H")),
                            step=step, type="fc", stream="oper", param=PARAMETERS,
                            target=str(tmp))
            validate_grib(tmp)
            tmp.replace(path)
            return path
        except Exception as exc:
            last_error = str(exc)
            LOG.warning("%s %s T+%03d attempt %d/%d failed: %s",
                        model, cycle_id(cycle), step, attempt, RETRIES, exc)
            tmp.unlink(missing_ok=True)
            if attempt < RETRIES:
                time.sleep(RETRY_DELAY)
    raise RuntimeError(last_error or "unknown retrieval error")


def fetch_model(model: str, steps: list[int], cycles: list[datetime]) -> dict:
    attempts = []
    for cycle in cycles:
        LOG.info("Trying %s cycle %s", model, cycle_id(cycle))
        files, errors = {}, {}
        try:
            client = Client(source="ecmwf", model=model, resol="0p25")
        except Exception as exc:
            return {"model": model, "status": "unavailable", "cycle": None,
                    "files": {}, "steps_successful": [], "steps_failed": steps,
                    "attempts": attempts, "error": f"Client initialization failed: {exc}"}
        for step in steps:
            try:
                files[step] = retrieve_step(client, model, cycle, step)
            except Exception as exc:
                errors[step] = str(exc)
        attempts.append({"cycle": cycle_id(cycle), "steps_successful": sorted(files),
                         "steps_failed": [{"step": k, "error": v} for k, v in errors.items()]})
        # Require T+0 and at least one later lead for a useful track sequence.
        if 0 in files and len(files) >= 2:
            return {"model": model, "status": "success" if not errors else "partial",
                    "cycle": cycle_id(cycle), "cycle_datetime": cycle,
                    "files": files, "steps_successful": sorted(files),
                    "steps_failed": sorted(set(steps) - set(files)),
                    "attempts": attempts, "error": None if not errors else "Some forecast steps failed."}
        LOG.warning("Cycle %s did not provide T+0 plus a later lead; falling back.", cycle_id(cycle))
    return {"model": model, "status": "unavailable", "cycle": None, "files": {},
            "steps_successful": [], "steps_failed": steps, "attempts": attempts,
            "error": "No candidate cycle provided T+0 and at least one later forecast step."}


def read_variable(path: Path, short_name: str):
    ds = xr.open_dataset(path, engine="cfgrib",
                         backend_kwargs={"filter_by_keys": {"shortName": short_name}, "indexpath": ""})
    try:
        if not ds.data_vars:
            raise ValueError(f"No {short_name} variable in {path.name}")
        arr = ds[next(iter(ds.data_vars))]
        for dim in list(arr.dims):
            if dim.lower() not in {"latitude", "longitude", "lat", "lon"} and arr.sizes.get(dim, 0) > 1:
                # A single forecast file should represent one lead time; select first extra coordinate safely.
                arr = arr.isel({dim: 0})
            elif dim.lower() not in {"latitude", "longitude", "lat", "lon"}:
                arr = arr.isel({dim: 0})
        lat_name = next((d for d in arr.dims if d.lower() in {"latitude", "lat"}), None)
        lon_name = next((d for d in arr.dims if d.lower() in {"longitude", "lon"}), None)
        if lat_name is None or lon_name is None:
            raise ValueError(f"Cannot identify lat/lon in {path.name}: {arr.dims}")
        arr = arr.transpose(lat_name, lon_name)
        lat = np.asarray(arr[lat_name].values, dtype=float)
        lon = np.asarray(arr[lon_name].values, dtype=float)
        data = np.asarray(arr.values, dtype=float)
    finally:
        ds.close()
    if np.nanmax(lon) > 180:
        lon = ((lon + 180) % 360) - 180
        order = np.argsort(lon)
        lon, data = lon[order], data[:, order]
    if lat[0] > lat[-1]:
        lat, data = lat[::-1], data[::-1, :]
    if short_name == "msl" and np.nanmedian(data) > 2000:
        data = data / 100.0  # Pa to hPa
    return lat, lon, data


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def detect_lows(lat: np.ndarray, lon: np.ndarray, pressure: np.ndarray, limit_hpa: float) -> list[dict]:
    """Detect local MSLP minima; these are candidate lows, not cyclone declarations."""
    lon_min, lon_max, lat_min, lat_max = REGION
    region = ((lat[:, None] >= lat_min) & (lat[:, None] <= lat_max) &
              (lon[None, :] >= lon_min) & (lon[None, :] <= lon_max))
    valid = np.isfinite(pressure) & region & (pressure <= limit_hpa)
    # Local 8-neighbour minimum test. Boundaries are excluded to avoid wraparound.
    local = valid.copy()
    core = pressure[1:-1, 1:-1]
    core_valid = valid[1:-1, 1:-1].copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbour = pressure[1 + dy: pressure.shape[0] - 1 + dy,
                                 1 + dx: pressure.shape[1] - 1 + dx]
            core_valid &= core <= neighbour
    local[:] = False
    local[1:-1, 1:-1] = core_valid
    ij = np.argwhere(local)
    candidates = [{"lat": float(lat[i]), "lon": float(lon[j]),
                   "mslp_hpa": round(float(pressure[i, j]), 1), "grid_i": int(i), "grid_j": int(j)}
                  for i, j in ij]
    candidates.sort(key=lambda p: p["mslp_hpa"])
    # Keep distinct minima separated by ~2.5 degrees; avoids clusters of same low.
    selected = []
    for cand in candidates:
        if all(haversine_km(cand["lat"], cand["lon"], x["lat"], x["lon"]) >= 275 for x in selected):
            selected.append(cand)
        if len(selected) >= 12:
            break
    return selected


def build_tracks(model_result: dict, limit_hpa: float) -> dict:
    if not model_result.get("files"):
        return {"model": model_result["model"], "cycle": None, "tracks": [], "steps": []}
    steps_out = []
    for step, path in sorted(model_result["files"].items()):
        try:
            lat, lon, p = read_variable(path, "msl")
            lows = detect_lows(lat, lon, p, limit_hpa)
            valid_time = model_result["cycle_datetime"] + timedelta(hours=step)
            steps_out.append({"step": step, "valid_time_utc": iso(valid_time),
                              "valid_time_ist": iso(valid_time, IST), "lows": lows,
                              "grid": {"lat": lat, "lon": lon, "mslp": p}})
            LOG.info("%s T+%s: %d candidate lows", model_result["model"], step, len(lows))
        except Exception as exc:
            LOG.warning("Could not detect lows for %s T+%s: %s", model_result["model"], step, exc)
            steps_out.append({"step": step, "valid_time_utc": iso(model_result["cycle_datetime"] + timedelta(hours=step)),
                              "valid_time_ist": iso(model_result["cycle_datetime"] + timedelta(hours=step), IST),
                              "lows": [], "error": str(exc)})

    tracks = []
    previous_step = None
    for item in steps_out:
        candidates = item["lows"]
        # Greedy one-to-one nearest association only from the immediately
        # preceding successfully processed forecast lead time.
        active = [t for t in tracks if t["points"] and t["last_step"] == previous_step]
        used = set()
        for track in sorted(active, key=lambda t: t["points"][-1]["mslp_hpa"]):
            prev = track["points"][-1]
            choices = []
            for idx, cand in enumerate(candidates):
                if idx in used:
                    continue
                distance = haversine_km(prev["lat"], prev["lon"], cand["lat"], cand["lon"])
                if distance <= TRACK_MAX_STEP_KM:
                    choices.append((distance, idx, cand))
            if choices:
                _, idx, cand = min(choices, key=lambda x: x[0])
                used.add(idx)
                track["points"].append({**cand, "step": item["step"], "valid_time_utc": item["valid_time_utc"]})
                track["last_step"] = item["step"]
        for idx, cand in enumerate(candidates):
            if idx not in used:
                tracks.append({"track_id": f"{model_result['model']}-L{len(tracks)+1:02d}",
                               "model": model_result["model"], "last_step": item["step"],
                               "points": [{**cand, "step": item["step"], "valid_time_utc": item["valid_time_utc"]}]})
        previous_step = item["step"]

    kept = [t for t in tracks if len(t["points"]) >= MIN_TRACK_POINTS]
    for track in kept:
        track.pop("last_step", None)
        track["min_mslp_hpa"] = min(p["mslp_hpa"] for p in track["points"])
        track["first_step"] = track["points"][0]["step"]
        track["last_forecast_step"] = track["points"][-1]["step"]
        track["classification"] = "candidate pressure-low track; not cyclone confirmation"
    return {"model": model_result["model"], "cycle": model_result.get("cycle"),
            "steps": [{k: v for k, v in s.items() if k != "grid"} for s in steps_out],
            "tracks": kept}


def plot_results(model_results: list[dict], track_results: list[dict]) -> dict:
    # Use first available MSLP field at T+0 for background contours.
    base = None
    for result in model_results:
        if 0 in result.get("files", {}):
            try:
                lat, lon, p = read_variable(result["files"][0], "msl")
                base = (lat, lon, p, result["model"], result["cycle"])
                break
            except Exception as exc:
                LOG.warning("No base MSLP field from %s: %s", result["model"], exc)
    if base is None:
        raise RuntimeError("No readable T+0 MSLP field; cannot make dashboard map")
    lat, lon, p, base_model, base_cycle = base
    lomin, lomax, lamin, lamax = REGION
    ix = (lon >= lomin) & (lon <= lomax)
    iy = (lat >= lamin) & (lat <= lamax)
    if not ix.any() or not iy.any():
        raise RuntimeError("Forecast grid does not overlap map region")
    xx, yy, zz = lon[ix], lat[iy], p[np.ix_(iy, ix)]
    fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
    levels = np.arange(math.floor(np.nanmin(zz) / 2) * 2, math.ceil(np.nanmax(zz) / 2) * 2 + 2, 2)
    cf = ax.contourf(xx, yy, zz, levels=levels, extend="both", cmap="viridis")
    cs = ax.contour(xx, yy, zz, levels=levels, colors="black", linewidths=.45, alpha=.75)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.0f")
    fig.colorbar(cf, ax=ax, pad=.02, label="Mean sea-level pressure (hPa)")
    styles = {"ifs": ("#e63946", "o"), "aifs-single": ("#00b4d8", "s")}
    for tr in track_results:
        color, marker = styles.get(tr["model"], (None, "o"))
        for track in tr["tracks"]:
            pts = track["points"]
            ax.plot([q["lon"] for q in pts], [q["lat"] for q in pts], color=color,
                    marker=marker, linewidth=2, markersize=4, label=f"{tr['model']} {track['track_id']}")
            if pts:
                ax.annotate(f"{pts[-1]['mslp_hpa']:.0f} hPa", (pts[-1]["lon"], pts[-1]["lat"]), fontsize=7)
    ax.set(xlim=(lomin, lomax), ylim=(lamin, lamax), xlabel="Longitude", ylabel="Latitude")
    ax.set_xticks(np.arange(70, 106, 5)); ax.set_yticks(np.arange(0, 29, 5))
    ax.grid(True, linestyle=":", linewidth=.5, alpha=.65)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower left", fontsize=7, framealpha=.85)
    ax.set_title(f"MASRAINMAN | Bay of Bengal & South India\nMSLP T+0 background: {base_model.upper()} | Cycle {base_cycle} UTC\nCandidate low tracks: red IFS, blue AIFS", fontsize=12)
    fig.text(.5, .018, "Experimental deterministic guidance • Candidate lows are not cyclone confirmation • Not an official warning",
             ha="center", fontsize=8)
    fig.tight_layout(rect=[0, .035, 1, 1])
    fig.savefig(IMAGE_PATH, bbox_inches="tight")
    plt.close(fig)
    return {"status": "success", "path": "docs/data/latest.png", "background_model": base_model,
            "cycle": base_cycle, "track_count": sum(len(t["tracks"]) for t in track_results)}


def weather_next_status() -> dict:
    url = os.getenv("WEATHERNEXT_EXPORT_URL", "").strip()
    token = os.getenv("WEATHERNEXT_EXPORT_TOKEN", "").strip()
    if not url:
        return {"status": "not_configured", "error": "WEATHERNEXT_EXPORT_URL is not configured; no WeatherNext data used."}
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return {"status": "endpoint_reachable_not_integrated", "http_status": response.status_code,
                "note": "Endpoint checked only; data has not been parsed or used in this forecast."}
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc)}


def write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC)
    models, steps, cycles = parse_models(), parse_steps(), candidate_cycles(started)
    LOG.info("Starting tracker at %s; models=%s; steps=%s; cycles=%s", iso(started), models, steps, [cycle_id(c) for c in cycles])

    results = []
    for model in models:
        result = fetch_model(model, steps, cycles)
        results.append(result)
        LOG.info("%s status=%s cycle=%s successful_steps=%s", model, result["status"], result.get("cycle"), result["steps_successful"])

    track_results = [build_tracks(r, LOW_PRESSURE_LIMIT_HPA) for r in results if r.get("files")]
    plot_status = {"status": "unavailable", "error": "No usable forecast field."}
    if any(r.get("files") for r in results):
        try:
            plot_status = plot_results(results, track_results)
        except Exception as exc:
            LOG.exception("Plot generation failed")
            plot_status = {"status": "failed", "error": str(exc)}

    source_rows = []
    for result in results:
        source_rows.append({"source": f"ECMWF {result['model']}", "status": result["status"],
                            "cycle": result.get("cycle"), "error": result.get("error"),
                            "note": f"Validated forecast lead times: {result['steps_successful']}"})
    wn = weather_next_status()
    source_rows.append({"source": "WeatherNext 3", "status": wn["status"], "error": wn.get("error"), "note": wn.get("note")})
    cycle_used = next((r.get("cycle") for r in results if r.get("cycle")), None)
    cycle_dt = datetime.strptime(cycle_used, "%Y%m%d%H").replace(tzinfo=UTC) if cycle_used else None
    model_ok = any(r.get("files") for r in results)
    if plot_status.get("status") == "success" and all(r["status"] == "success" for r in results):
        overall = "success"
    elif model_ok and plot_status.get("status") == "success":
        overall = "partial"
    else:
        overall = "failed"

    public_models = []
    for r in results:
        public_models.append({"model": r["model"], "status": r["status"], "cycle": r.get("cycle"),
                              "files": [str(p.relative_to(ROOT)) for p in r.get("files", {}).values()],
                              "steps_requested": steps, "steps_successful": r["steps_successful"],
                              "steps_failed": r["steps_failed"], "attempts": r["attempts"], "error": r.get("error")})
    public_tracks = [{"model": tr["model"], "cycle": tr["cycle"], "tracks": tr["tracks"], "steps": tr["steps"]} for tr in track_results]
    status = {
        "project": "MASRAINMAN Bay of Bengal Cyclone Tracker",
        "run_utc": iso(started), "run_ist": iso(started, IST),
        "finished_utc": iso(datetime.now(UTC)), "overall_status": overall,
        "overall_message": {"success": "Forecast data validated and map generated.",
                             "partial": "Some model data/steps unavailable; inspect model diagnostics.",
                             "failed": "No usable forecast map could be generated."}[overall],
        "configured_models": models, "forecast_steps_requested": steps,
        "candidate_cycles": [cycle_id(c) for c in cycles],
        "ecmwf_cycle": {"date": cycle_dt.strftime("%Y-%m-%d") if cycle_dt else None,
                        "hour_utc": cycle_dt.hour if cycle_dt else None, "cycle_id": cycle_used},
        "models": public_models, "sources": source_rows, "weathernext3": wn,
        "plot": plot_status, "tracks": public_tracks,
        "track_method": {"type": "local MSLP-minimum association", "pressure_threshold_hpa": LOW_PRESSURE_LIMIT_HPA,
                         "maximum_step_displacement_km": TRACK_MAX_STEP_KM, "minimum_points_per_track": MIN_TRACK_POINTS,
                         "caveat": "Candidate lows only. No closed-circulation, wind, or official cyclone classification is inferred."},
        "probabilities": None,
        "notes": ["Deterministic model tracks are guidance, not certainty.",
                  "No calibrated track probabilities are computed from two deterministic models.",
                  "Candidate MSLP minima are not automatically tropical cyclones.",
                  "Follow IMD for official warnings."],
    }
    write_json_atomic(STATUS_PATH, status)
    LOG.info("Overall=%s; plot=%s; status=%s", overall, plot_status.get("status"), STATUS_PATH)
    return 0 if plot_status.get("status") == "success" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        LOG.error("Fatal error:\n%s", traceback.format_exc())
        # Best-effort status file so the dashboard shows the failure.
        try:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            write_json_atomic(STATUS_PATH, {"project": "MASRAINMAN Bay of Bengal Cyclone Tracker",
                "run_utc": iso(datetime.now(UTC)), "overall_status": "failed",
                "overall_message": "Fatal tracker error; see GitHub Actions logs.",
                "sources": [], "tracks": [], "probabilities": None,
                "error": traceback.format_exc()})
        except Exception:
            pass
        sys.exit(1)
