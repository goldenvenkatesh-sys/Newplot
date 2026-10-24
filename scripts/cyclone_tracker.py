
#!/usr/bin/env python3
"""MASRAINMAN Bay of Bengal Cyclone Tracker.

Retrieves available ECMWF forecast fields, plots decoded pressure contours,
and publishes source status. Experimental guidance only, not official warnings.
WeatherNext 3 requires a permitted CSV/JSON export URL.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DATA = DOCS / "data"
RAW = ROOT / "data" / "raw"

for directory in (DOCS, DATA, RAW):
    directory.mkdir(parents=True, exist_ok=True)


def now_utc():
    return datetime.now(timezone.utc)


def iso_utc(value):
    return value.astimezone(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def select_ecmwf_cycle(run):
    """Choose a recent 00/06/12/18 UTC cycle with safe date rollover."""
    cycle_hour = (run.hour // 6) * 6
    cycle_date = run.date()

    # Avoid requesting a cycle that has only just started.
    if run.minute < 40 and run.hour % 6 == 0:
        cycle_hour -= 6

        if cycle_hour < 0:
            cycle_hour = 18
            cycle_date = cycle_date - timedelta(days=1)

    return cycle_date.strftime("%Y%m%d"), cycle_hour


def fetch_ecmwf_model(model, run_date, run_hour, steps):
    """Download available ECMWF Open Data fields for one model."""
    result = {
        "source": f"ECMWF {model.upper()}",
        "status": "unavailable",
        "files": [],
        "error": None,
    }

    try:
        from ecmwf.opendata import Client

        client = Client(
            source="ecmwf",
            model=model,
            resol="0p25",
        )

        target_dir = RAW / f"{model}_{run_date}_{run_hour:02d}"
        target_dir.mkdir(parents=True, exist_ok=True)

        for step in steps:
            target = target_dir / (
                f"{model}_{run_date}_{run_hour:02d}_"
                f"step{step:03d}.grib2"
            )

            if target.exists() and target.stat().st_size > 1000:
                result["files"].append(str(target.relative_to(ROOT)))
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
                    result["files"].append(
                        str(target.relative_to(ROOT))
                    )
                elif target.exists():
                    target.unlink(missing_ok=True)

            except Exception as exc:
                if target.exists() and target.stat().st_size <= 1000:
                    target.unlink(missing_ok=True)

                result["error"] = (
                    f"{type(exc).__name__}: {exc}"
                )

        if result["files"]:
            result["status"] = "downloaded"
        elif not result["error"]:
            result["error"] = (
                "No forecast files returned for this model, "
                "cycle and step selection."
            )

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def fetch_weathernext_export():
    """Fetch WeatherNext only from a configured, permitted export endpoint."""
    url = os.getenv("WEATHERNEXT_EXPORT_URL", "").strip()
    token = os.getenv("WEATHERNEXT_EXPORT_TOKEN", "").strip()

    result = {
        "source": "Google DeepMind WeatherNext 3",
        "status": "unavailable",
        "files": [],
        "error": None,
        "note": (
            "Configure a permitted CSV/JSON export URL. "
            "No undocumented public API is assumed."
        ),
    }

    if not url:
        result["error"] = (
            "WEATHERNEXT_EXPORT_URL is not configured."
        )
        return result

    try:
        headers = (
            {"Authorization": f"Bearer {token}"}
            if token else {}
        )

        response = requests.get(
            url,
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()

        content_type = response.headers.get(
            "content-type", ""
        ).lower()
        clean_url = url.lower().split("?", 1)[0]

        suffix = (
            ".json"
            if "json" in content_type or clean_url.endswith(".json")
            else ".csv"
        )

        target = RAW / f"weathernext3_export{suffix}"
        target.write_bytes(response.content)

        if not response.content:
            raise ValueError("Export response was empty.")

        result.update(
            status="downloaded",
            files=[str(target.relative_to(ROOT))],
            error=None,
            note=(
                "Export downloaded. Its schema must be validated "
                "before cyclone tracks can be plotted."
            ),
        )

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def open_grib(path):
    """Decode a GRIB file; return None if decoding fails."""
    try:
        import xarray as xr

        return xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={"indexpath": ""},
        )
    except Exception:
        return None


def plot_latest(status):
    """Plot real decoded MSLP fields; do not fabricate tracks."""
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_facecolor("#edf4f7")
    ax.set_xlim(70, 105)
    ax.set_ylim(-2, 25)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.25)

    ax.set_title(
        "MASRAINMAN | Bay of Bengal Forecast Tracker",
        loc="left",
        weight="bold",
        fontsize=16,
    )

    ax.text(
        0.01,
        0.98,
        "Forecast fields only - no synthetic cyclone tracks",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )

    colors = ["#008ba3", "#6f2da8", "#d33b48", "#e6a400"]
    plotted = False

    for source_index, source in enumerate(status["sources"]):
        for relative_path in source.get("files", []):
            file_path = ROOT / relative_path

            if file_path.suffix not in (".grib", ".grib2"):
                continue

            dataset = open_grib(file_path)

            if dataset is None:
                continue

            try:
                variable = next(
                    (
                        name
                        for name in (
                            "msl",
                            "prmsl",
                            "mean_sea_level_pressure",
                        )
                        if name in dataset.data_vars
                    ),
                    None,
                )

                if variable is None:
                    continue

                field = dataset[variable].squeeze()

                lat_name = next(
                    (
                        name for name in ("latitude", "lat")
                        if name in field.coords
                    ),
                    None,
                )
                lon_name = next(
                    (
                        name for name in ("longitude", "lon")
                        if name in field.coords
                    ),
                    None,
                )

                if not lat_name or not lon_name or field.ndim != 2:
                    continue

                latitudes = field[lat_name].values
                longitudes = field[lon_name].values
                values = field.values

                # Convert Pa to hPa when necessary.
                if np.isfinite(values).any():
                    if np.nanmedian(values) > 2000:
                        values = values / 100.0

                contours = ax.contour(
                    longitudes,
                    latitudes,
                    values,
                    levels=np.arange(990, 1036, 4),
                    colors=[colors[source_index % len(colors)]],
                    linewidths=0.8,
                    alpha=0.75,
                )

                ax.clabel(
                    contours,
                    inline=True,
                    fontsize=7,
                    fmt="%d",
                )

                plotted = True

            except Exception:
                continue

            finally:
                try:
                    dataset.close()
                except Exception:
                    pass

    if not plotted:
        ax.text(
            0.5,
            0.48,
            "No compatible forecast fields decoded in this run",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=13,
            bbox={
                "boxstyle": "round,pad=.5",
                "facecolor": "white",
                "alpha": 0.9,
            },
        )

    summary = ", ".join(
        f"{item['source']}={item['status']}"
        for item in status["sources"]
    )

    ax.text(
        0.01,
        0.02,
        f"Run: {status['run_utc']} | {summary}",
        transform=ax.transAxes,
        fontsize=7,
    )

    fig.tight_layout()
    fig.savefig(DATA / "latest.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    run = now_utc()

    # Corrected cycle/date selection. No pandas .date() bug.
    run_date, run_hour = select_ecmwf_cycle(run)

    models = [
        item.strip()
        for item in os.getenv("ECMWF_MODELS", "ifs,aifs").split(",")
        if item.strip()
    ]

    steps = [0, 6, 12, 18, 24, 30, 36, 42, 48, 60, 72]

    sources = []

    for model in models:
        sources.append(
            fetch_ecmwf_model(
                model,
                run_date,
                run_hour,
                steps,
            )
        )

    sources.append(fetch_weathernext_export())

    status = {
        "project": "MASRAINMAN Bay of Bengal Cyclone Tracker",
        "run_utc": iso_utc(run),
        "run_ist": run.astimezone(
            ZoneInfo("Asia/Kolkata")
        ).isoformat(timespec="seconds"),
        "ecmwf_cycle": {
            "date": run_date,
            "hour_utc": run_hour,
        },
        "sources": sources,
        "probabilities": None,
        "probability_note": (
            "No probability is reported unless genuine ensemble data "
            "and a validated calculation are available."
        ),
        "warning": (
            "Experimental visualization, not an official warning. "
            "Follow IMD bulletins."
        ),
    }

    plot_latest(status)
    write_json(DATA / "status.json", status)

    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
