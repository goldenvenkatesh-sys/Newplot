
#!/usr/bin/env python3
"""
MASRAINMAN BAY OF BENGAL CYCLONE TRACKER
========================================

Sources:
  1. ECMWF IFS deterministic via ECMWF Open Data
  2. ECMWF AIFS via ECMWF Open Data, if supported by the current catalogue
  3. Google DeepMind WeatherNext 3 via a configured, permitted CSV/JSON export

Outputs:
  docs/data/latest.png
  docs/data/status.json

Important:
  - No GCS or Google Cloud billing project is required.
  - No GFS fallback is used.
  - No cyclone tracks or probabilities are invented.
  - A downloaded file is not considered a validated forecast until decoded.
  - Follow official IMD warnings for operational decisions.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import requests


# ============================================================
# 1. PROJECT PATHS AND LOGGING
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DOCS = ROOT / "docs"
DATA = DOCS / "data"
RAW = ROOT / "data" / "raw"

for directory in (DOCS, DATA, RAW):
    directory.mkdir(parents=True, exist_ok=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOG = logging.getLogger("MASRAINMAN")


# ============================================================
# 2. TIME AND JSON HELPERS
# ============================================================

def now_utc() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def iso_utc(value: datetime) -> str:
    """Format a timestamp in ISO-8601 UTC."""
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def write_json(path: Path, payload: dict) -> None:
    """Write JSON safely using UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ============================================================
# 3. ECMWF CYCLE SELECTION
# ============================================================

def select_ecmwf_cycle(run: datetime) -> tuple[str, int]:
    """
    Select a nominal ECMWF 00/06/12/18 UTC cycle.

    When a cycle has just started, use the previous cycle to reduce
    incomplete-product failures.

    Date rollover is handled with datetime.timedelta. This avoids the
    previous AttributeError caused by calling .date() on a date object.
    """

    cycle_hour = (run.hour // 6) * 6
    cycle_date = run.date()

    if run.minute < 40 and run.hour % 6 == 0:
        cycle_hour -= 6

        if cycle_hour < 0:
            cycle_hour = 18
            cycle_date = cycle_date - timedelta(days=1)

    return cycle_date.strftime("%Y%m%d"), cycle_hour


# ============================================================
# 4. MODEL CONFIGURATION
# ============================================================

def configured_ecmwf_models() -> list[str]:
    """
    Read ECMWF_MODELS from the environment.

    Crucial fix:
    If GitHub Actions defines ECMWF_MODELS as an empty string,
    explicitly restore the default instead of returning an empty list.
    """

    configured = os.getenv("ECMWF_MODELS", "").strip()

    if not configured:
        configured = "ifs,aifs"

    models = [
        item.strip().lower()
        for item in configured.split(",")
        if item.strip()
    ]

    # Avoid accidentally running duplicate model identifiers.
    models = list(dict.fromkeys(models))

    if not models:
        models = ["ifs", "aifs"]

    LOG.info("Configured ECMWF models: %s", ", ".join(models))

    return models


def configured_forecast_steps() -> list[int]:
    """Read forecast steps from an optional environment variable."""

    raw_steps = os.getenv(
        "ECMWF_STEPS",
        "0,6,12,18,24,30,36,42,48,60,72",
    )

    try:
        steps = sorted({
            int(value.strip())
            for value in raw_steps.split(",")
            if value.strip()
        })

        if not steps or any(step < 0 for step in steps):
            raise ValueError("Forecast steps must be non-negative.")

        return steps

    except Exception as exc:
        LOG.warning(
            "Invalid ECMWF_STEPS=%r; using default steps. Reason: %s",
            raw_steps,
            exc,
        )

        return [0, 6, 12, 18, 24, 30, 36, 42, 48, 60, 72]


# ============================================================
# 5. ECMWF OPEN DATA RETRIEVAL
# ============================================================

def fetch_ecmwf_model(
    model: str,
    run_date: str,
    run_hour: int,
    steps: list[int],
) -> dict:
    """
    Attempt ECMWF Open Data retrieval for one configured model.

    Each forecast step is requested separately so that partial
    availability is recorded. The catalogue may not support every
    model/parameter/step combination.
    """

    result = {
        "source": f"ECMWF {model.upper()}",
        "model": model,
        "status": "unavailable",
        "files": [],
        "decoded_fields": 0,
        "errors": [],
        "cycle": {
            "date": run_date,
            "hour_utc": run_hour,
        },
    }

    try:
        from ecmwf.opendata import Client

        client = Client(
            source="ecmwf",
            model=model,
            resol="0p25",
        )

    except Exception as exc:
        result["errors"].append(
            f"Client initialization failed: {type(exc).__name__}: {exc}"
        )

        LOG.error(
            "%s client initialization failed: %s",
            model.upper(),
            exc,
        )

        return result

    target_dir = RAW / f"{model}_{run_date}_{run_hour:02d}"
    target_dir.mkdir(parents=True, exist_ok=True)

    for step in steps:
        target = target_dir / (
            f"{model}_{run_date}_{run_hour:02d}_"
            f"step{step:03d}.grib2"
        )

        # Reuse an existing file only if it is non-empty and plausible.
        if target.exists() and target.stat().st_size > 1000:
            result["files"].append(str(target.relative_to(ROOT)))
            continue

        try:
            LOG.info(
                "Requesting %s cycle %s %02d UTC, step %s",
                model.upper(),
                run_date,
                run_hour,
                step,
            )

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
            else:
                if target.exists():
                    target.unlink(missing_ok=True)

                result["errors"].append(
                    f"Step {step}: output file missing or too small."
                )

        except Exception as exc:
            if target.exists() and target.stat().st_size <= 1000:
                target.unlink(missing_ok=True)

            message = (
                f"Step {step}: {type(exc).__name__}: {exc}"
            )

            result["errors"].append(message)

            LOG.warning("%s %s", model.upper(), message)

    if result["files"]:
        result["status"] = "downloaded"

        if result["errors"]:
            result["status"] = "partial"

    else:
        result["status"] = "unavailable"

        if not result["errors"]:
            result["errors"].append(
                "No forecast files returned for this model and cycle."
            )

    LOG.info(
        "%s retrieval status: %s; files: %s",
        model.upper(),
        result["status"],
        len(result["files"]),
    )

    return result


# ============================================================
# 6. WEATHERNEXT 3 EXPORT CONNECTOR
# ============================================================

def fetch_weathernext_export() -> dict:
    """
    Fetch a permitted WeatherNext 3 CSV/JSON export.

    This deliberately does not assume an undocumented public API.
    A successful download alone does not establish that its contents
    are cyclone tracks or that they can be compared with ECMWF fields.
    """

    url = os.getenv("WEATHERNEXT_EXPORT_URL", "").strip()
    token = os.getenv("WEATHERNEXT_EXPORT_TOKEN", "").strip()

    result = {
        "source": "Google DeepMind WeatherNext 3",
        "status": "unavailable",
        "files": [],
        "errors": [],
        "note": (
            "Configure a permitted CSV/JSON export URL. "
            "No undocumented public API is assumed."
        ),
    }

    if not url:
        result["errors"].append(
            "WEATHERNEXT_EXPORT_URL is not configured."
        )

        return result

    try:
        headers = {}

        if token:
            headers["Authorization"] = f"Bearer {token}"

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

        if "json" in content_type or clean_url.endswith(".json"):
            suffix = ".json"
        else:
            suffix = ".csv"

        target = RAW / f"weathernext3_export{suffix}"

        if not response.content:
            raise ValueError("Export response was empty.")

        target.write_bytes(response.content)

        result["status"] = "downloaded"
        result["files"].append(str(target.relative_to(ROOT)))
        result["note"] = (
            "Export downloaded. Its schema must be validated before "
            "it can contribute to cyclone-track plotting."
        )

        LOG.info("WeatherNext export downloaded: %s", target.name)

    except Exception as exc:
        result["errors"].append(
            f"{type(exc).__name__}: {exc}"
        )

        LOG.warning("WeatherNext export unavailable: %s", exc)

    return result


# ============================================================
# 7. GRIB DECODING
# ============================================================

def open_grib(path: Path):
    """Open a GRIB file with cfgrib; return None if decoding fails."""

    try:
        import xarray as xr

        return xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={"indexpath": ""},
        )

    except Exception as exc:
        LOG.debug(
            "Could not decode %s: %s",
            path.name,
            exc,
        )

        return None


def validate_grib_files(sources: list[dict]) -> None:
    """
    Count files with decodable mean-sea-level pressure fields.

    Downloaded files are not automatically considered valid plots.
    """

    for source in sources:
        if not source.get("source", "").startswith("ECMWF"):
            continue

        decoded_count = 0

        for relative_path in source.get("files", []):
            path = ROOT / relative_path

            dataset = open_grib(path)

            if dataset is None:
                continue

            try:
                found_msl = any(
                    name in dataset.data_vars
                    for name in (
                        "msl",
                        "prmsl",
                        "mean_sea_level_pressure",
                    )
                )

                if found_msl:
                    decoded_count += 1

            finally:
                try:
                    dataset.close()
                except Exception:
                    pass

        source["decoded_fields"] = decoded_count

        if decoded_count > 0:
            source["status"] = "validated_fields"

        elif source.get("files"):
            source["status"] = "downloaded_not_decoded"


# ============================================================
# 8. FORECAST MAP
# ============================================================

def plot_latest(status: dict) -> None:
    """
    Plot actual decoded MSLP contours.

    This is not yet an ensemble spaghetti-track plot. No track lines,
    storm centers or probabilities are fabricated.
    """

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
        "Experimental forecast fields - not an official warning",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color="#555555",
    )

    colors = [
        "#008ba3",
        "#6f2da8",
        "#d33b48",
        "#e6a400",
    ]

    plotted = False

    for source_index, source in enumerate(status["sources"]):
        if not source.get("source", "").startswith("ECMWF"):
            continue

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
                        name
                        for name in ("latitude", "lat")
                        if name in field.coords
                    ),
                    None,
                )

                lon_name = next(
                    (
                        name
                        for name in ("longitude", "lon")
                        if name in field.coords
                    ),
                    None,
                )

                if not lat_name or not lon_name or field.ndim != 2:
                    continue

                latitudes = field[lat_name].values
                longitudes = field[lon_name].values
                values = field.values

                if not np.isfinite(values).any():
                    continue

                # ECMWF MSLP is normally in pascals.
                if np.nanmedian(values) > 2000:
                    values = values / 100.0

                contours = ax.contour(
                    longitudes,
                    latitudes,
                    values,
                    levels=np.arange(900, 1041, 4),
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

            except Exception as exc:
                LOG.debug(
                    "Plotting error for %s: %s",
                    file_path.name,
                    exc,
                )

            finally:
                try:
                    dataset.close()
                except Exception:
                    pass

    if not plotted:
        ax.text(
            0.5,
            0.48,
            "No compatible ECMWF pressure fields decoded",
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

    source_summary = " | ".join(
        f"{item['source']}: {item['status']}"
        for item in status["sources"]
    )

    ax.text(
        0.01,
        0.02,
        f"Run: {status['run_utc']} | {source_summary}",
        transform=ax.transAxes,
        fontsize=7,
        color="#444444",
    )

    fig.tight_layout()

    output = DATA / "latest.png"

    fig.savefig(
        output,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    LOG.info("Forecast map written: %s", output)


# ============================================================
# 9. MAIN PIPELINE
# ============================================================

def main() -> int:
    run = now_utc()

    run_date, run_hour = select_ecmwf_cycle(run)

    models = configured_ecmwf_models()
    steps = configured_forecast_steps()

    LOG.info("Project root: %s", ROOT)
    LOG.info("Run time UTC: %s", iso_utc(run))
    LOG.info("ECMWF cycle: %s %02d UTC", run_date, run_hour)
    LOG.info("Forecast steps: %s", steps)

    sources = []

    for model in models:
        sources.append(
            fetch_ecmwf_model(
                model=model,
                run_date=run_date,
                run_hour=run_hour,
                steps=steps,
            )
        )

    sources.append(fetch_weathernext_export())

    validate_grib_files(sources)

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
        "configured_models": models,
        "forecast_steps": steps,
        "sources": sources,
        "probabilities": None,
        "probability_note": (
            "No probability is calculated unless genuine ensemble "
            "members, a defined event and a validated calculation "
            "are available."
        ),
        "warning": (
            "Experimental visualization, not an official warning. "
            "Follow IMD bulletins."
        ),
    }

    plot_latest(status)

    write_json(
        DATA / "status.json",
        status,
    )

    LOG.info(
        "Status JSON written: %s",
        DATA / "status.json",
    )

    print(json.dumps(status, indent=2))

    # The run can complete even when a provider is unavailable.
    # The status manifest records those failures for diagnosis.
    return 0


if __name__ == "__main__":
    sys.exit(main())

