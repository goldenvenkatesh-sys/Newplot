#!/usr/bin/env python3
"""MASRAINMAN Bay of Bengal tracker: ECMWF Open Data fetch + transparent status.

This is an experimental forecast-field viewer, not an official cyclone warning
or a validated automatic cyclone-track/probability model.
"""
from __future__ import annotations
import os, json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DOCS, DATA, RAW = ROOT / "docs", ROOT / "docs" / "data", ROOT / "data" / "raw"
for folder in (DOCS, DATA, RAW):
    folder.mkdir(parents=True, exist_ok=True)

def now():
    return datetime.now(timezone.utc)

def iso(dt):
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def ecmwf_fetch_model(model, run_date, run_hour, steps):
    record = {"source": f"ECMWF {model}", "status": "unavailable", "files": [], "error": None}
    target_dir = RAW / f"{model}_{run_date}_{run_hour:02d}"
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        from ecmwf.opendata import Client
        client = Client(source="ecmwf", model=model, resol="0p25")
    except Exception as exc:
        record["error"] = f"Client setup failed: {type(exc).__name__}: {exc}"
        return record

    errors = []
    for step in steps:
        target = target_dir / f"{model}_{run_date}_{run_hour:02d}_step{step:03d}.grib2"
        if target.exists() and target.stat().st_size > 1000:
            record["files"].append(str(target.relative_to(ROOT)))
            continue
        try:
            client.retrieve(date=run_date, time=run_hour, step=step, type="fc",
                            stream="oper", param=["msl", "10u", "10v"], target=str(target))
            if target.exists() and target.stat().st_size > 1000:
                # Validate that the file can actually be decoded before counting it.
                import xarray as xr
                ds = xr.open_dataset(target, engine="cfgrib",
                                     backend_kwargs={"indexpath": ""})
                try:
                    if len(ds.data_vars) == 0:
                        raise ValueError("Decoded GRIB contains no data variables")
                finally:
                    ds.close()
                record["files"].append(str(target.relative_to(ROOT)))
            else:
                errors.append(f"step {step}: empty or very small output")
                target.unlink(missing_ok=True)
        except Exception as exc:
            errors.append(f"step {step}: {type(exc).__name__}: {exc}")
            if target.exists() and target.stat().st_size < 1000:
                target.unlink(missing_ok=True)

    if record["files"]:
        record["status"] = "partial" if errors else "available"
        record["error"] = "; ".join(errors[-3:]) if errors else None
    else:
        record["status"] = "unavailable"
        record["error"] = "; ".join(errors[-3:]) or "No valid GRIB files were retrieved."
    return record

def weathernext_fetch():
    url = os.getenv("WEATHERNEXT_EXPORT_URL", "").strip()
    token = os.getenv("WEATHERNEXT_EXPORT_TOKEN", "").strip()
    out = {"source": "Google DeepMind WeatherNext 3", "status": "unavailable",
           "files": [], "error": None,
           "note": "Requires a permitted export URL; no undocumented API is assumed."}
    if not url:
        out["error"] = "WEATHERNEXT_EXPORT_URL is not configured."
        return out
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        if not response.content:
            raise ValueError("Export endpoint returned an empty response.")
        content_type = response.headers.get("content-type", "")
        suffix = ".json" if "json" in content_type or url.lower().split("?")[0].endswith(".json") else ".csv"
        target = RAW / f"weathernext3_export{suffix}"
        target.write_bytes(response.content)
        out.update(status="downloaded", files=[str(target.relative_to(ROOT))], error=None)
        out["note"] = "Export saved; schema-specific track parsing still requires validation."
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out

def plot_latest(status):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor("#edf4f7")
    ax.set_xlim(70, 105); ax.set_ylim(-2, 25)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.grid(True, alpha=.25)
    ax.set_title("MASRAINMAN | Bay of Bengal Forecast Tracker", loc="left",
                 weight="bold", fontsize=16)
    ax.text(.01, .98, "Forecast fields only • not an official warning",
            transform=ax.transAxes, va="top", fontsize=9, color="#555")
    plotted = False
    colors = ["#008ba3", "#6f2da8", "#d33b48", "#e6a400"]
    for source_i, source in enumerate(status["sources"]):
        for rel in source.get("files", []):
            path = ROOT / rel
            if path.suffix not in (".grib", ".grib2"):
                continue
            try:
                import xarray as xr
                ds = xr.open_dataset(path, engine="cfgrib",
                                     backend_kwargs={"indexpath": ""})
                try:
                    var = next((v for v in ("msl", "prmsl", "mean_sea_level_pressure")
                                if v in ds.data_vars), None)
                    if not var:
                        continue
                    da = ds[var].squeeze()
                    lat = next((x for x in ("latitude", "lat") if x in da.coords), None)
                    lon = next((x for x in ("longitude", "lon") if x in da.coords), None)
                    if not lat or not lon or da.ndim != 2:
                        continue
                    values = da.values
                    if np.nanmedian(values) > 2000:
                        values = values / 100.0
                    contour = ax.contour(da[lon].values, da[lat].values, values,
                                         levels=np.arange(988, 1037, 4),
                                         colors=[colors[source_i % len(colors)]],
                                         linewidths=.8, alpha=.8)
                    ax.clabel(contour, inline=True, fontsize=7, fmt="%d")
                    plotted = True
                finally:
                    ds.close()
            except Exception:
                continue
    if not plotted:
        ax.text(.5, .48, "No compatible forecast fields decoded in this run",
                transform=ax.transAxes, ha="center", va="center", fontsize=13,
                bbox=dict(boxstyle="round,pad=.5", facecolor="white", alpha=.9))
    summary = ", ".join(f'{s["source"]}={s["status"]}' for s in status["sources"])
    ax.text(.01, .02, f'Run: {status["run_utc"]} | {summary}',
            transform=ax.transAxes, fontsize=7, color="#444")
    fig.tight_layout()
    fig.savefig(DATA / "latest.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

def choose_cycle(run):
    # ECMWF Open Data can lag nominal cycle time. Select a cycle at least 12 h old.
    eligible = run - timedelta(hours=12)
    return eligible.date().strftime("%Y%m%d"), (eligible.hour // 6) * 6

def main():
    run = now()
    date_str, hour = choose_cycle(run)
    configured = os.getenv("ECMWF_MODELS", "").strip()
    models = [x.strip().lower() for x in (configured or "ifs,aifs-single").split(",") if x.strip()]
    models = ["aifs-single" if x in ("aifs", "aifs_single") else x for x in models]
    models = list(dict.fromkeys(models))
    steps = [0, 6, 12, 18, 24, 30, 36, 42, 48, 60, 72]
    sources = [ecmwf_fetch_model(model, date_str, hour, steps) for model in models]
    sources.append(weathernext_fetch())
    status = {
        "project": "MASRAINMAN Bay of Bengal Cyclone Tracker",
        "run_utc": iso(run),
        "run_ist": run.astimezone(ZoneInfo("Asia/Kolkata")).isoformat(timespec="seconds"),
        "ecmwf_cycle": {"date": date_str, "hour_utc": hour},
        "sources": sources,
        "probabilities": None,
        "probability_note": "No probability is reported without genuine ensemble/event data and a validated calculation.",
        "warning": "Experimental visualization, not an official warning. Follow IMD bulletins."
    }
    plot_latest(status)
    write_json(DATA / "status.json", status)
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    main()
