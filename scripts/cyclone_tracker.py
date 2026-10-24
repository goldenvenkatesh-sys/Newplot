
#!/usr/bin/env python3
"""
MASRAINMAN Bay of Bengal Cyclone Tracker
ECMWF IFS deterministic + ECMWF AIFS deterministic

Outputs:
  docs/data/latest.png
  docs/data/status.json

Optional environment variables:
  ECMWF_MODELS=ifs,aifs-single
  ECMWF_FORECAST_STEPS=0,6,12,18,24,30,36,42,48,60,72
  ECMWF_CYCLE_FALLBACKS=4
  WEATHERNEXT_EXPORT_URL=
  WEATHERNEXT_EXPORT_TOKEN=

This script does not claim that a workflow succeeded unless forecast
data was actually retrieved and validated.
"""

from __future__ import annotations

import json
import logging
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


# ---------------------------------------------------------------------
# 1. PROJECT POSITION / CONFIGURATION
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "docs" / "data"

IMAGE_PATH = OUTPUT_DIR / "latest.png"
STATUS_PATH = OUTPUT_DIR / "status.json"

IST = ZoneInfo("Asia/Kolkata")
UTC = timezone.utc

DEFAULT_MODELS = ["ifs", "aifs-single"]
DEFAULT_STEPS = [0, 6, 12, 18, 24, 30, 36, 42, 48, 60, 72]

MODEL_ALIASES = {
    "aifs": "aifs-single",
    "aifs-single": "aifs-single",
    "ifs": "ifs",
}

PARAMETERS = ["msl", "10u", "10v"]

# A cycle must be at least this old before the first attempt.
# If it is unavailable, older cycles are tried automatically.
MIN_CYCLE_AGE_HOURS = 6

# Number of older six-hourly cycles to try after the initial cycle.
CYCLE_FALLBACKS = int(os.getenv("ECMWF_CYCLE_FALLBACKS", "4"))

REQUEST_TIMEOUT = 45
MAX_RETRIES_PER_FILE = 2
RETRY_DELAY_SECONDS = 3

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOG = logging.getLogger("masrainman-cyclone")


# ---------------------------------------------------------------------
# 2. DIRECTORY AND ENVIRONMENT HELPERS
# ---------------------------------------------------------------------

def ensure_directories():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_models():
    raw = os.getenv("ECMWF_MODELS", ",".join(DEFAULT_MODELS))

    models = []
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue

        model = MODEL_ALIASES.get(item)
        if model is None:
            LOG.warning("Ignoring unsupported ECMWF model: %s", item)
            continue

        if model not in models:
            models.append(model)

    return models or DEFAULT_MODELS.copy()


def parse_steps():
    raw = os.getenv(
        "ECMWF_FORECAST_STEPS",
        ",".join(str(x) for x in DEFAULT_STEPS),
    )

    steps = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue

        try:
            value = int(item)
        except ValueError:
            LOG.warning("Ignoring invalid forecast step: %s", item)
            continue

        if value < 0 or value > 360:
            LOG.warning("Ignoring out-of-range forecast step: %s", value)
            continue

        if value not in steps:
            steps.append(value)

    return sorted(steps) if steps else DEFAULT_STEPS.copy()


def utc_now():
    return datetime.now(UTC)


def iso_utc(value):
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def iso_ist(value):
    return value.astimezone(IST).isoformat()


# ---------------------------------------------------------------------
# 3. FORECAST CYCLE SELECTION AND FALLBACK
# ---------------------------------------------------------------------

def floor_to_synoptic_cycle(value):
    """
    Return the most recent 00, 06, 12 or 18 UTC cycle at or before value.
    """
    value = value.astimezone(UTC).replace(
        minute=0, second=0, microsecond=0
    )
    hour = (value.hour // 6) * 6
    return value.replace(hour=hour)


def candidate_cycles(run_time=None):
    """
    Generate newest-to-oldest candidate cycles.

    A six-hour age buffer reduces the chance of requesting a cycle
    immediately after its nominal initialization time, when files
    may not yet have been published.
    """
    if run_time is None:
        run_time = utc_now()

    safe_reference = run_time - timedelta(
        hours=MIN_CYCLE_AGE_HOURS
    )
    newest = floor_to_synoptic_cycle(safe_reference)

    count = max(0, CYCLE_FALLBACKS) + 1

    return [
        newest - timedelta(hours=6 * offset)
        for offset in range(count)
    ]


def cycle_string(cycle):
    return cycle.strftime("%Y%m%d%H")


# ---------------------------------------------------------------------
# 4. ECMWF OPEN DATA RETRIEVAL
# ---------------------------------------------------------------------

def make_client(model):
    """
    The ECMWF Open Data client uses:
      ifs         - physics-based deterministic forecast
      aifs-single - AI-based deterministic forecast
    """
    return Client(
        source="ecmwf",
        model=model,
        resol="0p25",
    )


def retrieve_one_file(client, model, cycle, step):
    """
    Retrieve one forecast lead time.

    Files are named with both model and cycle to avoid mixing
    forecasts from different initialization times.
    """
    cycle_id = cycle_string(cycle)
    destination = RAW_DIR / (
        f"{model}_{cycle_id}_step{step:03d}.grib2"
    )
    temporary = destination.with_suffix(".grib2.part")

    # Reuse an existing file only if it can be opened as GRIB.
    if destination.exists() and destination.stat().st_size > 0:
        try:
            validate_grib(destination)
            return destination
        except Exception:
            LOG.warning(
                "Existing file failed validation; downloading again: %s",
                destination.name,
            )
            destination.unlink(missing_ok=True)

    last_error = None

    for attempt in range(1, MAX_RETRIES_PER_FILE + 1):
        temporary.unlink(missing_ok=True)

        try:
            LOG.info(
                "Retrieving model=%s cycle=%s step=%s attempt=%s",
                model,
                cycle_id,
                step,
                attempt,
            )

            client.retrieve(
                date=cycle.strftime("%Y%m%d"),
                time=int(cycle.strftime("%H")),
                step=step,
                type="fc",
                stream="oper",
                param=PARAMETERS,
                target=str(temporary),
            )

            if not temporary.exists():
                raise RuntimeError(
                    "ECMWF client returned without creating the file"
                )

            if temporary.stat().st_size < 100:
                raise RuntimeError(
                    f"Downloaded file is unexpectedly small "
                    f"({temporary.stat().st_size} bytes)"
                )

            validate_grib(temporary)

            temporary.replace(destination)

            LOG.info(
                "Validated %s (%s bytes)",
                destination.name,
                destination.stat().st_size,
            )
            return destination

        except Exception as exc:
            last_error = exc
            LOG.warning(
                "Download failed: model=%s cycle=%s step=%s "
                "attempt=%s error=%s",
                model,
                cycle_id,
                step,
                attempt,
                str(exc),
            )

            temporary.unlink(missing_ok=True)

            if attempt < MAX_RETRIES_PER_FILE:
                time.sleep(RETRY_DELAY_SECONDS)

    raise RuntimeError(
        f"Failed model={model}, cycle={cycle_id}, step={step}: "
        f"{last_error}"
    )


def fetch_ecmwf_model(model, steps, cycles):
    """
    Try candidate cycles from newest to oldest.

    A cycle is accepted if at least one requested lead time is
    retrieved and validated. Failed lead times are reported separately.
    """
    cycle_attempts = []
    successful_files = []
    chosen_cycle = None

    for cycle in cycles:
        cycle_id = cycle_string(cycle)
        LOG.info("Trying %s cycle %s", model, cycle_id)

        files_for_cycle = []
        failures_for_cycle = []

        try:
            client = make_client(model)
        except Exception as exc:
            return {
                "model": model,
                "status": "unavailable",
                "cycle": None,
                "files": [],
                "steps_successful": [],
                "steps_failed": steps,
                "cycle_attempts": [],
                "error": f"Unable to initialize ECMWF client: {exc}",
            }

        for step in steps:
            try:
                path = retrieve_one_file(
                    client=client,
                    model=model,
                    cycle=cycle,
                    step=step,
                )
                files_for_cycle.append(path)
            except Exception as exc:
                failures_for_cycle.append({
                    "step": step,
                    "error": str(exc),
                })

        cycle_attempts.append({
            "cycle": cycle_id,
            "successful_steps": [
                int(p.stem.rsplit("step", 1)[1])
                for p in files_for_cycle
            ],
            "failed_steps": failures_for_cycle,
        })

        if files_for_cycle:
            chosen_cycle = cycle
            successful_files = files_for_cycle
            break

        LOG.warning(
            "%s cycle %s returned no usable forecast files; "
            "trying an older cycle",
            model,
            cycle_id,
        )

    if chosen_cycle is None:
        return {
            "model": model,
            "status": "unavailable",
            "cycle": None,
            "files": [],
            "steps_successful": [],
            "steps_failed": steps,
            "cycle_attempts": cycle_attempts,
            "error": "No usable files retrieved from any candidate cycle.",
        }

    successful_steps = sorted([
        int(p.stem.rsplit("step", 1)[1])
        for p in successful_files
    ])

    failed_steps = sorted(set(steps) - set(successful_steps))

    return {
        "model": model,
        "status": (
            "success" if not failed_steps else "partial"
        ),
        "cycle": cycle_string(chosen_cycle),
        "files": [str(p.relative_to(ROOT)) for p in successful_files],
        "file_paths": [str(p) for p in successful_files],
        "steps_successful": successful_steps,
        "steps_failed": failed_steps,
        "cycle_attempts": cycle_attempts,
        "error": None if not failed_steps else (
            "Some requested forecast steps could not be retrieved."
        ),
    }


# ---------------------------------------------------------------------
# 5. GRIB VALIDATION AND READING
# ---------------------------------------------------------------------

def validate_grib(path):
    """
    Verify that the file is readable by cfgrib.

    Requires eccodes and cfgrib, installed through requirements.txt.
    """
    path = Path(path)

    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty GRIB file: {path}")

    dataset = None
    try:
        dataset = xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={
                "indexpath": "",
                "errors": "raise",
            },
        )

        if not dataset.data_vars:
            raise ValueError("GRIB contains no readable data variables")

        return True

    finally:
        if dataset is not None:
            dataset.close()


def open_grib_variable(path, short_name):
    """
    Open one GRIB variable by short name.
    """
    dataset = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"shortName": short_name},
            "indexpath": "",
        },
    )

    if not dataset.data_vars:
        dataset.close()
        raise ValueError(
            f"No variable {short_name} found in {path}"
        )

    variable_name = next(iter(dataset.data_vars))
    return dataset, dataset[variable_name]


def normalize_longitudes(longitudes, values):
    """
    Convert longitudes from 0..360 to -180..180 when necessary,
    sorting the longitude axis and matching data columns.
    """
    lon = np.asarray(longitudes)

    if np.nanmax(lon) > 180:
        lon = ((lon + 180) % 360) - 180
        order = np.argsort(lon)
        lon = lon[order]
        values = values[..., order]

    return lon, values


def select_2d_field(data_array):
    """
    Reduce a forecast variable to a two-dimensional lat/lon field.
    """
    array = data_array

    for dim in list(array.dims):
        if dim.lower() in {
            "latitude", "longitude", "lat", "lon"
        }:
            continue

        if array.sizes.get(dim, 0) == 1:
            array = array.isel({dim: 0})
        elif dim.lower() in {
            "step", "time", "valid_time", "number", "surface"
        }:
            array = array.isel({dim: 0})

    lat_name = next(
        (d for d in array.dims if d.lower() in {"latitude", "lat"}),
        None,
    )
    lon_name = next(
        (d for d in array.dims if d.lower() in {"longitude", "lon"}),
        None,
    )

    if lat_name is None or lon_name is None:
        raise ValueError(
            f"Could not identify latitude/longitude dimensions: "
            f"{array.dims}"
        )

    array = array.transpose(lat_name, lon_name)

    lat = np.asarray(array[lat_name].values)
    lon = np.asarray(array[lon_name].values)
    values = np.asarray(array.values, dtype=float)

    lon, values = normalize_longitudes(lon, values)

    if lat[0] > lat[-1]:
        lat = lat[::-1]
        values = values[::-1, :]

    return lat, lon, values


# ---------------------------------------------------------------------
# 6. FIND A VALID MSLP FIELD
# ---------------------------------------------------------------------

def find_latest_mslp(model_results):
    """
    Find the first valid mean sea-level pressure field in the latest
    usable ECMWF output. Returns lat, lon, hPa field and metadata.
    """
    for result in model_results:
        if result.get("status") not in {"success", "partial"}:
            continue

        for file_path in result.get("file_paths", []):
            dataset = None
            try:
                dataset, field = open_grib_variable(
                    file_path, "msl"
                )

                lat, lon, values = select_2d_field(field)

                # ECMWF MSLP is normally stored in Pa; convert to hPa.
                finite = values[np.isfinite(values)]
                if finite.size == 0:
                    continue

                median = float(np.nanmedian(finite))
                if median > 2000:
                    values = values / 100.0

                return {
                    "lat": lat,
                    "lon": lon,
                    "mslp_hpa": values,
                    "model": result["model"],
                    "cycle": result["cycle"],
                    "step": int(
                        Path(file_path).stem.rsplit("step", 1)[1]
                    ),
                    "source_file": str(
                        Path(file_path).relative_to(ROOT)
                    ),
                }

            except Exception as exc:
                LOG.warning(
                    "Could not read MSLP from %s: %s",
                    file_path,
                    exc,
                )
            finally:
                if dataset is not None:
                    dataset.close()

    return None


# ---------------------------------------------------------------------
# 7. DASHBOARD GRAPHIC
# ---------------------------------------------------------------------

def plot_latest(field_info, status):
    """
    Produce a South India / Bay of Bengal MSLP contour image.
    This is a pressure-field visualization, not a cyclone-track
    probability map.
    """
    lat = field_info["lat"]
    lon = field_info["lon"]
    mslp = field_info["mslp_hpa"]

    # Approximate plotting region: Bay of Bengal and southern India.
    lon_min, lon_max = 70, 105
    lat_min, lat_max = -2, 27

    lon_mask = (lon >= lon_min) & (lon <= lon_max)
    lat_mask = (lat >= lat_min) & (lat <= lat_max)

    if not np.any(lon_mask) or not np.any(lat_mask):
        raise ValueError(
            "Forecast grid does not cover the requested plotting region"
        )

    plot_lon = lon[lon_mask]
    plot_lat = lat[lat_mask]
    plot_data = mslp[np.ix_(lat_mask, lon_mask)]

    finite = plot_data[np.isfinite(plot_data)]
    if finite.size == 0:
        raise ValueError("No valid MSLP data in the plotting region")

    levels = np.arange(
        np.floor(np.nanmin(finite) / 2) * 2,
        np.ceil(np.nanmax(finite) / 2) * 2 + 2,
        2,
    )

    fig, ax = plt.subplots(figsize=(12, 9), dpi=150)

    filled = ax.contourf(
        plot_lon,
        plot_lat,
        plot_data,
        levels=levels,
        extend="both",
    )

    lines = ax.contour(
        plot_lon,
        plot_lat,
        plot_data,
        levels=levels,
        linewidths=0.7,
    )
    ax.clabel(lines, inline=True, fontsize=7, fmt="%.0f")

    fig.colorbar(
        filled,
        ax=ax,
        pad=0.02,
        label="Mean sea-level pressure (hPa)",
    )

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xticks(np.arange(70, 106, 5))
    ax.set_yticks(np.arange(0, 28, 5))
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

    cycle = field_info["cycle"]
    step = field_info["step"]
    valid_time = datetime.strptime(
        cycle, "%Y%m%d%H"
    ).replace(tzinfo=UTC) + timedelta(hours=step)

    ax.set_title(
        "MASRAINMAN | Bay of Bengal & South India\n"
        f"ECMWF {field_info['model'].upper()} deterministic | "
        f"Cycle {cycle} UTC | T+{step:03d} h\n"
        f"Valid: {valid_time.strftime('%Y-%m-%d %H:%M UTC')}",
        fontsize=13,
        pad=14,
    )

    footer = (
        "Forecast pressure field only — not an official warning "
        "or a cyclone-track probability forecast."
    )
    fig.text(0.5, 0.018, footer, ha="center", fontsize=8)

    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(IMAGE_PATH, bbox_inches="tight")
    plt.close(fig)

    LOG.info("Dashboard image saved: %s", IMAGE_PATH)


# ---------------------------------------------------------------------
# 8. OPTIONAL WEATHERNEXT EXPORT CHECK
# ---------------------------------------------------------------------

def fetch_weathernext_export():
    """
    Optional external export endpoint.

    This function does not assume that WeatherNext is publicly
    available. It only checks a URL explicitly configured by the user.
    """
    url = os.getenv("WEATHERNEXT_EXPORT_URL", "").strip()
    token = os.getenv("WEATHERNEXT_EXPORT_TOKEN", "").strip()

    if not url:
        return {
            "status": "unavailable",
            "error": "WEATHERNEXT_EXPORT_URL is not configured.",
        }

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        return {
            "status": "reachable",
            "http_status": response.status_code,
            "content_type": content_type,
            "note": (
                "Endpoint responded. Its contents have not been "
                "integrated into the forecast plot or probabilities."
            ),
        }

    except Exception as exc:
        return {
            "status": "unavailable",
            "error": str(exc),
        }


# ---------------------------------------------------------------------
# 9. STATUS REPORTING
# ---------------------------------------------------------------------

def write_status(status):
    temporary = STATUS_PATH.with_suffix(".json.tmp")

    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2, ensure_ascii=False)

    temporary.replace(STATUS_PATH)
    LOG.info("Status saved: %s", STATUS_PATH)


def summarize_model(result):
    return {
        "model": result.get("model"),
        "status": result.get("status"),
        "cycle": result.get("cycle"),
        "files": result.get("files", []),
        "steps_requested": result.get("steps_requested"),
        "steps_successful": result.get("steps_successful", []),
        "steps_failed": result.get("steps_failed", []),
        "cycle_attempts": result.get("cycle_attempts", []),
        "error": result.get("error"),
    }


# ---------------------------------------------------------------------
# 10. MAIN
# ---------------------------------------------------------------------

def main():
    ensure_directories()

    started = utc_now()
    models = parse_models()
    steps = parse_steps()
    cycles = candidate_cycles(started)

    LOG.info("=" * 72)
    LOG.info("MASRAINMAN BAY OF BENGAL CYCLONE TRACKER")
    LOG.info("Run UTC: %s", iso_utc(started))
    LOG.info("Run IST: %s", iso_ist(started))
    LOG.info("Models: %s", models)
    LOG.info("Forecast steps: %s", steps)
    LOG.info(
        "Candidate cycles: %s",
        [cycle_string(cycle) for cycle in cycles],
    )
    LOG.info("=" * 72)

    model_results = []

    for model in models:
        result = fetch_ecmwf_model(
            model=model,
            steps=steps,
            cycles=cycles,
        )
        result["steps_requested"] = steps
        model_results.append(result)

        LOG.info(
            "Model result: %s",
            json.dumps(summarize_model(result)),
        )

    weathernext_status = fetch_weathernext_export()

    status = {
        "project": "MASRAINMAN Bay of Bengal Cyclone Tracker",
        "run_utc": iso_utc(started),
        "run_ist": iso_ist(started),
        "configured_models": models,
        "forecast_steps_requested": steps,
        "candidate_cycles": [
            cycle_string(cycle) for cycle in cycles
        ],
        "cycle_selection": {
            "minimum_age_hours": MIN_CYCLE_AGE_HOURS,
            "fallback_cycles": CYCLE_FALLBACKS,
        },
        "models": [
            summarize_model(result) for result in model_results
        ],
        "weathernext3": weathernext_status,
        "plot": {
            "path": str(IMAGE_PATH.relative_to(ROOT)),
            "status": "not_generated",
        },
        "probabilities": None,
        "cyclone_tracks": None,
        "notes": [
            "MSLP contours are not cyclone-track probabilities.",
            "No cyclone track is inferred from pressure contours alone.",
            "WeatherNext endpoint reachability does not mean its data "
            "has been integrated.",
        ],
    }

    field_info = find_latest_mslp(model_results)

    if field_info is not None:
        try:
            plot_latest(field_info, status)
            status["plot"] = {
                "path": str(IMAGE_PATH.relative_to(ROOT)),
                "status": "success",
                "model": field_info["model"],
                "cycle": field_info["cycle"],
                "step": field_info["step"],
                "valid_source_file": field_info["source_file"],
            }
        except Exception as exc:
            LOG.exception("Plot generation failed")
            status["plot"] = {
                "path": str(IMAGE_PATH.relative_to(ROOT)),
                "status": "failed",
                "error": str(exc),
            }
    else:
        status["plot"] = {
            "path": str(IMAGE_PATH.relative_to(ROOT)),
            "status": "unavailable",
            "error": "No readable MSLP field was retrieved.",
        }

    finished = utc_now()
    status["finished_utc"] = iso_utc(finished)
    status["finished_ist"] = iso_ist(finished)
    status["duration_seconds"] = round(
        (finished - started).total_seconds(), 2
    )

    successful_models = [
        result for result in model_results
        if result.get("status") in {"success", "partial"}
    ]

    if not successful_models:
        status["overall_status"] = "failed"
        status["overall_message"] = (
            "No usable ECMWF forecast files were retrieved."
        )
    elif field_info is None:
        status["overall_status"] = "partial"
        status["overall_message"] = (
            "Some forecast files were retrieved, but no readable "
            "MSLP field was available for plotting."
        )
    elif any(
        result.get("status") == "partial"
        for result in successful_models
    ):
        status["overall_status"] = "partial"
        status["overall_message"] = (
            "A forecast plot was generated, but some requested "
            "model lead times were unavailable."
        )
    else:
        status["overall_status"] = "success"
        status["overall_message"] = (
            "Forecast data was retrieved and an MSLP plot was generated."
        )

    write_status(status)

    LOG.info("Overall status: %s", status["overall_status"])
    LOG.info("%s", status["overall_message"])
    LOG.info("Dashboard: %s", IMAGE_PATH)
    LOG.info("Status JSON: %s", STATUS_PATH)

    # Fail the workflow if no usable ECMWF data was downloaded.
    # This prevents GitHub Actions from reporting a false success.
    if not successful_models:
        return 1

    if field_info is None:
        return 1

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        LOG.error("Fatal tracker error:\n%s", traceback.format_exc())
        sys.exit(1)
