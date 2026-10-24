MASRAINMAN Cyclone Tracker — final corrected script

Replace scripts/cyclone_tracker.py in the existing Newplot repository with the
included file. No repository rebuild is needed. The script now:
- defaults to ECMWF IFS and AIFS Single even if ECMWF_MODELS is unset/empty;
- normalizes the common `aifs` alias to `aifs-single`;
- selects an ECMWF cycle at least 12 hours old to avoid requesting a just-issued cycle;
- validates downloaded GRIB files by decoding them before counting them;
- writes docs/data/status.json and docs/data/latest.png.

After replacing the file, run the existing GitHub Actions workflow manually once.
Only post the generated chart if status.json shows valid ECMWF files and the image
contains decoded forecast contours. This is not an official warning or validated
automatic cyclone track/probability product.
