import os
import pandas as pd
import joblib

# ============================================
# FIND PROJECT ROOT (Desktop/traffic)
# ============================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# move up: model_files → predict → traffic → Desktop/traffic
PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..", "..")
)

DATA_PATH = os.path.join(
    PROJECT_ROOT, "data", "tfdata_sorted.csv"
)

OUT_PATH = os.path.join(
    CURRENT_DIR, "area_stats.pkl"
)

print("Loading:", DATA_PATH)

# ============================================
# LOAD DATA
# ============================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"CSV NOT FOUND: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

# ============================================
# CREATE AREA STATS
# ============================================

stats = {}

for area, g in df.groupby("area"):
    stats[area] = {
        "mean": round(g["traffic_percentage"].mean(), 2),
        "std": round(g["traffic_percentage"].std(), 2),
        "hourly_mean": (
            g.groupby(g["timestamp"].dt.hour)["traffic_percentage"]
            .mean()
            .round(2)
            .to_dict()
        )
    }

# ============================================
# SAVE
# ============================================

joblib.dump(stats, OUT_PATH)

print("✅ area_stats.pkl created successfully")
print("Saved at:", OUT_PATH)
