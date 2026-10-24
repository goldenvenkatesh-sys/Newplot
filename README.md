# MASRAINMAN Bay of Bengal Cyclone Tracker

Automated Bay of Bengal forecast-track visualization scaffold for:
- ECMWF IFS deterministic via ECMWF Open Data
- ECMWF AIFS deterministic when the requested product is available
- ECMWF ensemble data when supported by the selected product
- Google DeepMind WeatherNext 3 only through a documented/exported data feed configured by the operator

**No Google Cloud Storage (GCS) or GCS billing project is used.**

## Important status
This initial version is a cautious starter implementation, not an operational cyclone-warning system. ECMWF product names/availability can vary by cycle and Open Data catalogue; the workflow validates that files were retrieved and records source failures. It does not silently substitute GFS, invent tracks, or invent probabilities. The first pass plots actual forecast wind/pressure fields when downloaded. Track extraction is heuristic and should be independently validated before public operational use.

WeatherNext 3: this project does not assume an undocumented public API. Set `WEATHERNEXT_EXPORT_URL` to a URL for a permitted CSV/JSON export, or leave it unset. If it is unset, the run labels WeatherNext as unavailable. If the URL requires authentication, use a GitHub Actions secret named `WEATHERNEXT_EXPORT_TOKEN` and configure the endpoint to accept a Bearer token.

## Run locally
```bash
python -m pip install -r requirements.txt
python scripts/cyclone_tracker.py
```

## GitHub Actions schedule
The workflow runs at 00:00, 03:00, 06:00, 09:00, 12:00, 15:00, 18:00 and 21:00 India Standard Time (Asia/Kolkata), using UTC cron entries converted to India Standard Time (IST). Scheduled workflows can be delayed by GitHub, and may be disabled by GitHub during periods of repository inactivity.

## GitHub Pages
The workflow writes `docs/data/status.json` and `docs/data/latest.png`. Enable Pages in repository Settings → Pages → Build and deployment → Deploy from a branch → `main` / `/docs`. The dashboard is static and can be published publicly even if source privacy is changed later, depending on the account plan and GitHub's current Pages rules.

## Required GitHub settings
- Actions: allow workflows to read/write repository contents (workflow uses `contents: write`).
- Pages: enable deployment from `main` and `/docs`.
- Optional secret: `WEATHERNEXT_EXPORT_TOKEN`.
- Optional variable: `ECMWF_MODELS` (comma-separated model identifiers, default `ifs,aifs`).

## Forecast interpretation
This is not an official warning product. Consult IMD bulletins for operational decisions. Probabilities must not be shown unless supported by genuine ensemble/event data and a documented denominator. A single deterministic forecast cannot provide a meaningful ensemble probability.
