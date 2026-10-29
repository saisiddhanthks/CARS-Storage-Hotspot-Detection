import sys, os, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, events, evaluate, OUT
name = sys.argv[1]; methods = sys.argv[2:] if len(sys.argv) > 2 else None
df, tr = load(name); test = df[~tr].reset_index(drop=True); evs = events(name)
rows = []
for f in sorted(x for x in glob.glob(f"{OUT}/{name}/*.npz") if not os.path.basename(x).startswith("parts_")):
    m = os.path.basename(f)[:-4]
    if methods and m not in methods: continue
    z = np.load(f); rows.append(evaluate(test, z["score"], z["flag"], evs, m))
r = pd.DataFrame(rows).set_index("method")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(r[["AP","ROC_AUC","precision","recall","F1","best_F1","surge_FPR","aggressor_recall","victim_recall","event_recall","event_recall_strict","mean_delay_min","fa_per_volume_day"]].round(3))
