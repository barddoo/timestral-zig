"""Diff two forecast CSVs (series,variate,step,q...): `python compare.py zig.csv reference.csv`.

The error is |diff| / max(|reference|, 1), so it stays meaningful for values near zero.
Expect ~1e-6 or better; forecasts that diverge at long horizons amplify float noise in the
extreme quantiles.
"""
import sys

def rows(path):
    return [list(map(float, l.split(",")[3:])) for l in open(path).read().strip().splitlines() if not l.startswith("series")]

a, b = rows(sys.argv[1]), rows(sys.argv[2])
assert len(a) == len(b), f"row count differs: {len(a)} vs {len(b)}"
abs_err = max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))
scaled = max(abs(x - y) / max(abs(y), 1.0) for ra, rb in zip(a, b) for x, y in zip(ra, rb))
print(f"rows={len(a)} max|diff|={abs_err:.2e} max scaled err={scaled:.2e}")
