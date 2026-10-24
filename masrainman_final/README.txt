MASRAINMAN BAY OF BENGAL CYCLONE TRACKER — FINAL PACKAGE

Files:
  scripts/cyclone_tracker.py  Complete tracker script
  docs/index.html             Dashboard compatible with status.json schema

Existing repository files to keep:
  .github/workflows/cyclone_update.yml
  requirements.txt

INSTALL:
1. Replace scripts/cyclone_tracker.py with the supplied file.
2. Replace docs/index.html with the supplied file.
3. Keep the existing requirements.txt and GitHub Actions workflow.
4. Commit both files to the main branch.
5. Run Actions > MASRAINMAN Cyclone Tracker > Run workflow.

ENVIRONMENT VARIABLES (optional):
  ECMWF_MODELS=ifs,aifs-single
  ECMWF_FORECAST_STEPS=0,6,12,18,24,30,36,42,48,60,72
  ECMWF_MIN_CYCLE_AGE_HOURS=6
  ECMWF_CYCLE_FALLBACKS=4
  ECMWF_RETRIES=2
  LOW_PRESSURE_LIMIT_HPA=1010
  TRACK_MAX_STEP_KM=500
  MIN_TRACK_POINTS=2

WeatherNext remains optional and is not integrated into track calculations.
The tracker links local MSLP minima as candidate lows. It does not calculate
calibrated probabilities and does not automatically classify lows as cyclones.
