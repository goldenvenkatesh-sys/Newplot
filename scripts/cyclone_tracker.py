#!/usr/bin/env python3
"""MASRAINMAN cyclone tracker bootstrap.

Downloads available ECMWF Open Data fields, renders a map from real fields,
and writes a transparent source/status manifest. This is not a validated
cyclone tracker; heuristic minima are labelled as candidate centers only.
"""
from __future__ import annotations
import os, sys, json, time, traceback
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DATA = DOCS / "data"
RAW = ROOT / "data" / "raw"
for p in (DOCS, DATA, RAW):
    p.mkdir(parents=True, exist_ok=True)

def now():
    return datetime.now(timezone.utc)

def iso(dt):
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def ecmwf_fetch_model(model, run_date, run_hour, steps):
    """Attempt ECMWF Open Data retrieval; never silently substitutes models."""
    record = {"source": f"ECMWF {model}", "status": "not_started", "files": [], "error": None}
    try:
        from ecmwf.opendata import Client
        client = Client(source="ecmwf", model=model, resol="0p25")
        target_dir = RAW / f"{model}_{run_date}_{run_hour:02d}"
        target_dir.mkdir(parents=True, exist_ok=True)
        # One GRIB per step allows partial-cycle recovery and clear diagnostics.
        for step in steps:
            target = target_dir / f"{model}_{run_date}_{run_hour:02d}_step{step:03d}.grib2"
            if target.exists() and target.stat().st_size > 1000:
                record["files"].append(str(target.relative_to(ROOT)))
                continue
            try:
                client.retrieve(
                    date=run_date,
                    time=run_hour,
                    step=step,
                    type="fc",
                    stream="oper",
                    param=["msl", "10u", "10v"],
                    target=str(target),
                )
                if target.exists() and target.stat().st_size > 1000:
                    record["files"].append(str(target.relative_to(ROOT)))
            except Exception as e:
                # A product/step may not be available for this model/cycle.
                record["error"] = f"Partial retrieval: {type(e).__name__}: {e}"
                if target.exists() and target.stat().st_size < 1000:
                    target.unlink(missing_ok=True)
        if record["files"]:
            record["status"] = "downloaded"
        else:
            record["status"] = "unavailable"
            if not record["error"]:
                record["error"] = "No forecast files returned for requested model/cycle/steps."
    except Exception as e:
        record["status"] = "unavailable"
        record["error"] = f"{type(e).__name__}: {e}"
    return record

def weathernext_fetch():
    url = os.getenv("WEATHERNEXT_EXPORT_URL", "").strip()
    token = os.getenv("WEATHERNEXT_EXPORT_TOKEN", "").strip()
    out = {"source": "Google DeepMind WeatherNext 3", "status": "unavailable",
           "files": [], "error": None,
           "note": "No undocumented API is assumed. Configure a permitted CSV/JSON export URL."}
    if not url:
        out["error"] = "WEATHERNEXT_EXPORT_URL is not configured."
        return out
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        suffix = ".json" if "json" in content_type or url.lower().split("?")[0].endswith(".json") else ".csv"
        target = RAW / f"weathernext3_export{suffix}"
        target.write_bytes(resp.content)
        if not resp.content:
            raise ValueError("Export endpoint returned an empty response.")
        out.update(status="downloaded", files=[str(target.relative_to(ROOT))], error=None)
        out["note"] = "Saved configured export; schema-specific track parsing must be validated."
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out

def read_grib_candidate(path):
    """Read GRIB fields with cfgrib; return None for unrecognized/partial data."""
    try:
        import xarray as xr
        ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
        return ds
    except Exception:
        return None

def plot_latest(status):
    """Plot actual available MSLP/wind data if a compatible GRIB can be decoded."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor("#edf4f7")
    ax.set_xlim(70, 105); ax.set_ylim(-2, 25)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.grid(True, alpha=.25)
    ax.set_title("MASRAINMAN | Bay of Bengal Forecast Tracker", loc="left", weight="bold", fontsize=16)
    ax.text(0.01, 0.98, "Forecast fields only • candidate tracks require validation",
            transform=ax.transAxes, va="top", fontsize=9, color="#555")
    plotted = False
    colors = ["#008ba3", "#6f2da8", "#d33b48", "#e6a400"]
    for source_i, src in enumerate(status["sources"]):
        for rel in src.get("files", []):
            path = ROOT / rel
            if path.suffix not in (".grib", ".grib2"):
                continue
            ds = read_grib_candidate(path)
            if ds is None:
                continue
            try:
                # Plot a pressure field if the file contains a usable msl variable.
                var = next((v for v in ("msl", "prmsl", "mean_sea_level_pressure") if v in ds.data_vars), None)
                if var:
                    da = ds[var].squeeze()
                    lat_name = next((x for x in ("latitude", "lat") if x in da.coords), None)
                    lon_name = next((x for x in ("longitude", "lon") if x in da.coords), None)
                    if lat_name and lon_name and da.ndim == 2:
                        lats = da[lat_name].values
                        lons = da[lon_name].values
                        vals = da.values
                        # ECMWF msl is generally Pa; convert to hPa for labels.
                        if np.nanmedian(vals) > 2000:
                            vals = vals / 100.0
                        cs = ax.contour(lons, lats, vals, levels=np.arange(990, 1036, 4),
                                        colors=[colors[source_i % len(colors)]], linewidths=.75, alpha=.7)
                        ax.clabel(cs, inline=True, fontsize=7, fmt="%d")
                        plotted = True
            except Exception:
                pass
            finally:
                try: ds.close()
                except Exception: pass
    # Keep map honest when no files could be decoded.
    if not plotted:
        ax.text(.5, .48, "No compatible forecast fields decoded in this run",
                transform=ax.transAxes, ha="center", va="center", fontsize=13,
                bbox=dict(boxstyle="round,pad=.5", facecolor="white", alpha=.9))
    ax.text(.01, .02, f"Run: {status['run_utc']} | Sources: " +
            ", ".join(f"{s['source']}={s['status']}" for s in status["sources"]),
            transform=ax.transAxes, fontsize=7, color="#444")
    fig.tight_layout()
    fig.savefig(DATA / "latest.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

def main():
    run = now()
    # ECMWF cycles are 00/06/12/18 UTC; choose the latest cycle that is at least
    # several hours old to reduce incomplete-product failures.
    rounded_hour = (run.hour // 6) * 6
    if run.hour % 6 == 0 and run.minute < 40:
        rounded_hour = (rounded_hour - 6) % 24
        run_date = (run.date() if rounded_hour != 18 else (run.date() - pd.Timedelta(days=1).to_pytimedelta()).date())
    else:
        run_date = run.date()
    run_date_str = run_date.strftime("%Y%m%d")
    models = [x.strip() for x in os.getenv("ECMWF_MODELS", "ifs,aifs").split(",") if x.strip()]
    steps = [0,6,12,18,24,30,36,42,48,60,72]
    sources = []
    for model in models:
        sources.append(ecmwf_fetch_model(model, run_date_str, rounded_hour, steps))
    sources.append(weathernext_fetch())
    status = {
        "project": "MASRAINMAN Bay of Bengal Cyclone Tracker",
        "run_utc": iso(run),
        "run_ist": run.astimezone(ZoneInfo("Asia/Kolkata")).isoformat(timespec="seconds"),
        "ecmwf_cycle": {"date": run_date_str, "hour_utc": rounded_hour},
        "sources": sources,
        "probabilities": None,
        "probability_note": "No probability is reported unless genuine ensemble/event data and a validated calculation are available.",
        "warning": "Experimental visualization, not an official warning. Follow IMD bulletins.",
    }
    plot_latest(status)
    write_json(DATA / "status.json", status)
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    main()
