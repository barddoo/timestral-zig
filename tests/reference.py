"""PyTorch reference for the Zig runner: same CLI and CSV formats, prints series,variate,step,q... to --out."""
import sys, argparse, time, numpy as np, torch
from timestral import TimestralForecaster
ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("series"); ap.add_argument("--horizon", type=int, required=True)
for k in ("past", "known", "cat-past", "cat-known"): ap.add_argument("--" + k)
ap.add_argument("--max-context", type=int); ap.add_argument("--block-patches", type=int, default=32)
ap.add_argument("--no-symmetric", action="store_true"); ap.add_argument("--no-clip", action="store_true")
ap.add_argument("--out", required=True); ap.add_argument("--time", action="store_true")
a = ap.parse_args()

def blocks(path, dtype):
    out, cur = [], []
    for line in open(path).read().splitlines() + [""]:
        s = line.strip()
        if s.startswith("#"): continue
        if not s:
            if cur: out.append(cur); cur = []
            continue
        toks = s.split(",") if "," in s else s.split()
        cur.append([float(t) if t.strip().lower() not in ("", "nan", "na") else np.nan for t in toks])
    return np.array(out, dtype=dtype)

ctx = blocks(a.series, np.float32)
kw = {}
if a.past: kw["past_covariates"] = blocks(a.past, np.float32)
if a.known: kw["known_covariates"] = blocks(a.known, np.float32)
if a.cat_past: kw["categorical_past_covariates"] = blocks(a.cat_past, np.float64)
if a.cat_known: kw["categorical_known_covariates"] = blocks(a.cat_known, np.float64)
fc = TimestralForecaster.from_pretrained(a.model, max_context=a.max_context, per_shot_horizon_patches=a.block_patches)
t0 = time.perf_counter()
f = fc.forecast(ctx, a.horizon, symmetric_averaging=not a.no_symmetric, clip_nonnegative=not a.no_clip, **kw)
if a.time: print(f"torch forecast={time.perf_counter()-t0:.3f}s", file=sys.stderr)
q = f.quantiles.numpy()
with open(a.out, "w") as fh:
    for s in range(q.shape[0]):
        for v in range(q.shape[1]):
            for t in range(q.shape[2]): fh.write(f"{s},{v},{t+1}," + ",".join(repr(float(x)) for x in q[s, v, t]) + "\n")
