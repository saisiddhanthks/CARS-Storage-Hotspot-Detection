"""Central path configuration. Override any location with an environment variable.

    CARS_PROJECT_REPO  clone of the HPE CPP team repository (telemetry generator + features)
    CARS_DATA          generated telemetry parquet files
    CARS_OUT           intermediate scores and result tables
    CARS_FIG           output figures
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_REPO = os.environ.get("CARS_PROJECT_REPO", os.path.join(ROOT, "external", "IO-Workload-Pattern-Classification-Hotspot-Detection"))
DATA = os.environ.get("CARS_DATA", os.path.join(ROOT, "work", "data"))
OUT = os.environ.get("CARS_OUT", os.path.join(ROOT, "work", "out"))
FIG = os.environ.get("CARS_FIG", os.path.join(ROOT, "work", "fig"))
for _d in (DATA, OUT, FIG):
    os.makedirs(_d, exist_ok=True)
