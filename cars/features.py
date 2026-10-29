import sys, pandas as pd, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import PROJECT_REPO, DATA
sys.path.insert(0, os.path.join(PROJECT_REPO, "src", "data"))
import feature_engineer as fe
name = sys.argv[1]
df = pd.read_parquet(f"{DATA}/{name}.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = fe.engineer_features(df)
df = df.sort_values(["timestamp", "volume_id"]).reset_index(drop=True)
df.to_parquet(f"{DATA}/{name}_feat.parquet", index=False)
print(name, df.shape)
