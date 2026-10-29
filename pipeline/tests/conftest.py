"""Test setup: run from the pipeline directory and generate outputs on first use."""
import os
import subprocess
import sys

PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PIPELINE_DIR)
os.chdir(PIPELINE_DIR)

if not (os.path.exists("final_hotspot_output.csv") and os.path.exists("noisy_neighbour_events.csv")):
    subprocess.run([sys.executable, "hotspot_pipeline.py"], check=True)
