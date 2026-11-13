"""Accuracy audit of the building layer against the roof-height raster. Read-only.

    python audit.py

Prints how heights were sourced, how much tall raster area still has no building over it,
and what the model holds at a few places that were reported wrong.
"""
import numpy as np
from rasterio import features
from rasterio.transform import from_origin
from scipy import ndimage

import build

SPOTS = {"North Eye (real ~255 m)": (28.57271, 77.38800), "Skytech Matrott": (28.57022, 77.38555),
         "Capetown market": (28.57213, 77.38873), "Supernova Spira": (28.55071, 77.32249)}

if __name__ == "__main__":
    dem, _ = build.dem_grid()
    g, info = build.load_buildings(dem, None)
    h, src, area = info["h"], info["src"], info["area"]
    print("buildings", len(g))
    for s in np.unique(src):
        m = src == s
        print(f"  {s:9s} {m.sum():8d}  median {np.median(h[m]):5.1f} m  p99 {np.percentile(h[m], 99):6.1f} m")
    print("  >= 40 m:", int((h >= 40).sum()), " >= 100 m:", int((h >= 100).sum()))

    res = build.OB_RES
    hr = np.load(build.CACHE / f"{build.OB_NAME}.npy")
    lab = features.rasterize(((p, 1) for p in g), out_shape=hr.shape, transform=from_origin(build.X0, build.Y1, res, res), dtype="uint8")
    covered = ndimage.binary_dilation(lab > 0, iterations=2)
    tall = ndimage.binary_opening((hr >= 20) & ~covered, iterations=1)
    cc, n = ndimage.label(tall)
    sz = np.bincount(cc.ravel())[1:] * res * res
    print(f"tall raster (>= 20 m) with no building over it: {(sz >= 400).sum()} blobs >= 400 m2, {sz[sz >= 400].sum() / 1e6:.2f} km2")
    built = hr > 2.5
    print(f"raster built area {built.sum() * res * res / 1e6:.1f} km2, covered by buildings {(built & covered).sum() * res * res / 1e6:.1f} km2")

    xs, ys = build.to_grid([p[1] for p in SPOTS.values()], [p[0] for p in SPOTS.values()])
    for name, x, y in zip(SPOTS, xs, ys):
        d = np.hypot(info["cx"] - x, info["cy"] - y)
        near = np.nonzero(d < 120)[0]
        top = near[np.argsort(-h[near])][:3]
        print(f"{name:26s} within 120 m: {len(near)} buildings; tallest " + ", ".join(f"{h[i]:.0f} m/{area[i]:.0f} m2/{src[i]}" for i in top))
