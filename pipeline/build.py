"""Noida 3D data pipeline. Open data in, compact streaming files out (noida/data/).

    python build.py dem sat masks pack      # any subset, in this order
    python build.py all

Inputs expected in cache/ (see README section in the plan): Overture GeoParquet
building/segment/water/division downloaded with the `overturemaps` CLI.
Remote rasters (Copernicus DEM, Sentinel-2, WorldCover) are read straight from
their public COGs and cached as .npy.

Grid: metres, x east, y north, azimuthal equidistant centred on ORIGIN.
Chunk file format: see ../decode.py docstring (checked by that script).
"""
import json, math, os, re, struct, sys, zlib
from pathlib import Path

import numpy as np
import rasterio
import shapely
from PIL import Image
from rasterio import features, warp
from rasterio.crs import CRS
from rasterio.transform import from_origin

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).parent
CACHE, OUT = HERE / "cache", HERE.parent / "data"
LAT0, LON0 = 28.50, 77.43
X0, Y0, X1, Y1 = -16000, -24000, 16000, 24000          # detailed extent
RX0, RY0, RX1, RY1 = -32000, -48000, 32000, 48000      # backdrop extent
CHUNK, TILE = 4000, 1000
NCX, NCY = (X1 - X0) // CHUNK, (Y1 - Y0) // CHUNK
GRID = CRS.from_proj4(f"+proj=aeqd +lat_0={LAT0} +lon_0={LON0} +datum=WGS84 +units=m +no_defs")
LL = CRS.from_epsg(4326)
H_OFF, H_SCALE = 50, 20                                  # height.png: u16 = (m + 50) * 20, R = high byte, G = low byte
S2_DATES = ["20251129", "20251126", "20251124", "20251114", "20251109", "20251025", "20251017"]


def to_grid(lon, lat):
    x, y = warp.transform(LL, GRID, np.asarray(lon).ravel().tolist(), np.asarray(lat).ravel().tolist())
    return np.array(x), np.array(y)


def to_lonlat(x, y):
    lon, lat = warp.transform(GRID, LL, np.asarray(x).ravel().tolist(), np.asarray(y).ravel().tolist())
    return np.array(lon), np.array(lat)


def grid_raster(src_path, band, ext, res, resampling=warp.Resampling.bilinear, dtype="float32"):
    """Warp one band of a (remote) raster onto the metric grid. Row 0 = north."""
    x0, y0, x1, y1 = ext
    w, h = int((x1 - x0) / res), int((y1 - y0) / res)
    dst = np.zeros((h, w), dtype)
    with rasterio.open(src_path) as s:
        warp.reproject(rasterio.band(s, band), dst, dst_transform=from_origin(x0, y1, res, res), dst_crs=GRID, resampling=resampling)
    return dst


def cached(name, fn):
    p = CACHE / (name + ".npy")
    if not p.exists():
        np.save(p, fn())
        print("built", name)
    return np.load(p)


# ---------------------------------------------------------------- terrain
DEM_URL = "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N28_00_E077_00_DEM/Copernicus_DSM_COG_10_N28_00_E077_00_DEM.tif"


def dem_grid():
    from scipy.ndimage import gaussian_filter, minimum_filter
    # GLO-30 is a surface model: roofs leak in. Min-filter then blur gives bare-ish ground on this flat plain.
    f = lambda ext, res: gaussian_filter(minimum_filter(grid_raster(DEM_URL, 1, ext, res), 5), 3)
    return cached("dem20", lambda: f((X0, Y0, X1, Y1), 20)), cached("dem80", lambda: f((RX0, RY0, RX1, RY1), 80))


def write_height(a, path):
    v = np.clip((a + H_OFF) * H_SCALE, 0, 65535).astype(np.uint16)
    Image.fromarray(np.dstack([v >> 8, v & 255, np.zeros_like(v)]).astype(np.uint8)).save(path, optimize=True)


def stage_dem():
    d20, d80 = dem_grid()
    write_height(d20, OUT / "height.png")
    write_height(d80, OUT / "region_h.png")
    print("dem: %.1f..%.1f m" % (d20.min(), d20.max()))


# ---------------------------------------------------------------- imagery
def s2_median(ext, res):
    layers = []
    for d in S2_DATES:
        y, m = d[:4], str(int(d[4:6]))
        for sat in "ABC":
            for seq in (0, 1):
                url = f"https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/R/GM/{y}/{m}/S2{sat}_43RGM_{d}_{seq}_L2A/TCI.tif"
                try:
                    rgb = np.dstack([grid_raster(url, b, ext, res, dtype="uint8") for b in (1, 2, 3)])
                except Exception:
                    continue
                layers.append(rgb)
                print("  s2", d, sat, seq)
                break
            else:
                continue
            break
    st = np.stack(layers).astype(np.float32)
    st[st.sum(axis=3) == 0] = np.nan                       # nodata edge of the swath
    med = np.nan_to_num(np.nanmedian(st, axis=0), nan=40)
    # TCI is dark over hazy north India: lift mid-tones, keep hue.
    return np.clip(255 * (med / 255) ** 0.72 * 1.12, 0, 255).astype(np.uint8)


def sat_grid():
    return (cached("sat10", lambda: s2_median((X0, Y0, X1, Y1), 10)),
            cached("sat40", lambda: s2_median((RX0, RY0, RX1, RY1), 40)))


def stage_sat():
    s10, s40 = sat_grid()
    T = 1600
    for ty in range(s10.shape[0] // T):                    # tile row 0 = south, to match the grid's y axis
        for tx in range(s10.shape[1] // T):
            r0 = s10.shape[0] - (ty + 1) * T
            Image.fromarray(s10[r0:r0 + T, tx * T:(tx + 1) * T]).save(OUT / f"sat_{tx}_{ty}.jpg", quality=82)
    Image.fromarray(s10).resize((s10.shape[1] // 4, s10.shape[0] // 4), Image.LANCZOS).save(OUT / "sat_lo.jpg", quality=80)
    Image.fromarray(s40).save(OUT / "region_sat.jpg", quality=80)
    print("sat tiles written")


# ---------------------------------------------------------------- masks
WC_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N27E075_Map.tif"


def load_geoms(name, cols=()):
    import pyarrow.parquet as pq
    t = pq.read_table(CACHE / f"{name}.parquet", columns=["geometry", *cols]).to_pandas()
    g = shapely.from_wkb(t.pop("geometry").values)
    c = shapely.get_coordinates(g)
    x, y = to_grid(c[:, 0], c[:, 1])
    return shapely.set_coordinates(g, np.column_stack([x, y])), t


def stage_masks():
    res = 12.5
    wc = cached("wc", lambda: grid_raster(WC_URL, 1, (X0, Y0, X1, Y1), res, warp.Resampling.nearest, "uint8"))
    water = wc == 80
    g, t = load_geoms("water")
    polys = [p for p in g if p.geom_type in ("Polygon", "MultiPolygon")]
    lines = [l.buffer(6) for l, c in zip(g, t.index) if l.geom_type in ("LineString", "MultiLineString")]
    if polys or lines:
        water |= features.rasterize([(p, 1) for p in polys + lines], out_shape=wc.shape,
                                    transform=from_origin(X0, Y1, res, res), dtype="uint8").astype(bool)
    rgb = np.dstack([np.isin(wc, (10, 95)), water, np.isin(wc, (30, 40))]).astype(np.uint8) * 255   # R trees, G water, B grass/crop
    Image.fromarray(rgb).save(OUT / "masks.png", optimize=True)
    np.save(CACHE / "trees.npy", rgb[..., 0])
    print("masks: trees %.1f%% water %.1f%% crop %.1f%%" % tuple(rgb.reshape(-1, 3).mean(0) / 2.55))


# ---------------------------------------------------------------- building height raster
OB_RES = int(os.environ.get('OB_RES', 4))   # source is 0.5 m; 4 m separates a tower from its podium and from the next tower
OB_NAME = f"ob_h{OB_RES}"


def stage_heights():
    """Google Open Buildings 2.5D Temporal (2023), band 2 = building height in metres. cache/ob25d_tiles.json lists the COGs."""
    def run():
        w, h = (X1 - X0) // OB_RES, (Y1 - Y0) // OB_RES
        import time
        part_npy, part_json = CACHE / f"{OB_NAME}.partial.npy", CACHE / f"{OB_NAME}.partial.json"
        done = json.loads(part_json.read_text()) if part_json.exists() else []      # resume: the network drops tiles
        out = np.load(part_npy) if done else np.zeros((h, w), "float32")
        for name in json.loads((CACHE / "ob25d_tiles.json").read_text()):
            if name in done:
                continue
            url = "https://storage.googleapis.com/open-buildings-temporal-data/" + name
            for attempt in range(8):
                try:
                    part = grid_raster(url, 2, (X0, Y0, X1, Y1), OB_RES, warp.Resampling.average)
                    break
                except Exception as e:
                    print("  retry", name[-20:], e, flush=True)
                    time.sleep(10)
            else:
                raise SystemExit("height tile failed: " + name)
            out = np.maximum(out, np.nan_to_num(part))
            done.append(name)
            np.save(part_npy, out)
            part_json.write_text(json.dumps(done))
            print("  ob", name[-20:], "max %.0f m" % part.max(), flush=True)
        part_npy.unlink(), part_json.unlink()
        return out
    a = cached(OB_NAME, run)
    print("heights: %.1f%% of cells built, max %.0f m" % (100 * (a > 0.5).mean(), a.max()))


# ---------------------------------------------------------------- buildings + roads -> chunks
ROAD_CLS = {"motorway": 0, "trunk": 1, "primary": 2, "secondary": 3, "tertiary": 4, "residential": 5, "unclassified": 5,
            "living_street": 5, "service": 6, "unknown": 6}                # everything else on foot = 7; rail 8; metro 9
ROAD_W = [22, 18, 14, 11, 8, 6, 4, 2, 4, 8]
B_TYPE = {"industrial": 0, "warehouse": 0, "factory": 0, "commercial": 4, "office": 4, "retail": 4, "hotel": 4,
          "education": 5, "school": 5, "university": 5, "college": 5, "medical": 5, "hospital": 5, "civic": 5, "government": 5,
          "religious": 6, "temple": 6, "mosque": 6, "church": 6, "transportation": 7, "train_station": 7, "stadium": 9}


def sample(a, res, ext, x, y):
    c = np.clip(((x - ext[0]) / res).astype(int), 0, a.shape[1] - 1)
    r = np.clip(((ext[3] - y) / res).astype(int), 0, a.shape[0] - 1)
    return a[r, c]


def hash01(i):
    return (zlib.crc32(struct.pack("<I", int(i) & 0xFFFFFFFF)) & 0xFFFF) / 65535


def poly_of(mask, r0, c0):
    """Outline of a boolean raster window (OB_RES cells, window origin at row r0, col c0) as one grid polygon."""
    from shapely.geometry import shape
    tr = from_origin(X0 + c0 * OB_RES, Y1 - r0 * OB_RES, OB_RES, OB_RES)
    parts = [shape(s) for s, v in features.shapes(mask.astype("uint8"), mask=mask, transform=tr)]
    p = shapely.simplify(shapely.union_all(parts), OB_RES * 0.75)
    if p.geom_type == "MultiPolygon":
        p = max(p.geoms, key=lambda q: q.area)
    return p if p.geom_type == "Polygon" and p.is_valid else None


def raster_heights(g, known):
    """Height per footprint from the roof-height raster, plus two kinds of extra buildings it reveals.

    Returns est (90th percentile of the footprint's cells; the mean reads low because edge cells mix in ground),
    towers [(polygon, height, parent index)] where one footprint covers a tower and its podium,
    and blobs [(polygon, height)] for tall structures no source has a footprint for."""
    from scipy import ndimage
    hr = np.load(CACHE / f"{OB_NAME}.npy")
    lab = features.rasterize(((p, i + 1) for i, p in enumerate(g)), out_shape=hr.shape, transform=from_origin(X0, Y1, OB_RES, OB_RES), dtype="int32")
    rr, cc = np.nonzero((lab > 0) & (hr > 0.5))
    L, V = lab[rr, cc] - 1, hr[rr, cc]
    o = np.lexsort((V, L))
    rr, cc, L, V = rr[o], cc[o], L[o], V[o]
    cnt = np.bincount(L, minlength=len(g))
    start = np.cumsum(cnt) - cnt
    q = lambda f: V[np.minimum(start + (cnt * f).astype(int), np.maximum(start + cnt - 1, 0)).clip(0, len(V) - 1)]
    p90, p25 = q(0.9), q(0.25)
    cen = shapely.centroid(g)
    est = np.where(cnt > 0, p90, sample(hr, OB_RES, (X0, Y0, X1, Y1), shapely.get_x(cen), shapely.get_y(cen)))

    towers = []
    for i in np.nonzero((cnt * OB_RES ** 2 >= 1200) & (p90 > 30) & (p90 - p25 > 15) & ~known)[0]:
        s, e = start[i], start[i] + cnt[i]
        r, c, v = rr[s:e], cc[s:e], V[s:e]
        tall = v >= (p90[i] + p25[i]) / 2
        if tall.mean() > 0.6:                                # mostly tower already: one height is right
            continue
        r0, c0 = r.min(), c.min()
        win = np.zeros((r.max() - r0 + 1, c.max() - c0 + 1), bool)
        hw = np.zeros(win.shape, "float32")
        win[r[tall] - r0, c[tall] - c0] = True
        hw[r - r0, c - c0] = v
        comp, n = ndimage.label(win)
        for k in range(1, n + 1):
            m = comp == k
            if m.sum() * OB_RES ** 2 < 200:
                continue
            p = poly_of(m, r0, c0)
            p = p and p.intersection(g[i])
            if p is not None and p.geom_type == "Polygon" and p.area > 150:
                towers.append((p, float(np.percentile(hw[m], 90)), i))
        est[i] = np.median(v[~tall])                         # what is left is the podium

    covered = ndimage.binary_dilation(lab > 0, iterations=2)
    tall = ndimage.binary_opening((hr >= 12) & ~covered, iterations=1)   # opening drops slivers along misaligned footprints
    comp, n = ndimage.label(tall)
    blobs = []
    for k, sl in enumerate(ndimage.find_objects(comp), 1):
        m = comp[sl] == k
        if m.sum() * OB_RES ** 2 < 320:
            continue
        p = poly_of(m, sl[0].start, sl[1].start)
        if p is not None and p.area > 250:
            blobs.append((p, float(np.percentile(hr[sl][m], 90))))
    return est, towers, blobs


def load_buildings(dem, sat):
    g, t = load_geoms("building", ["height", "num_floors", "class", "subtype", "names"])
    idx = np.repeat(np.arange(len(g)), shapely.get_num_geometries(g))
    g = shapely.get_parts(g)                                 # MultiPolygon -> one building per part
    t = t.iloc[idx].reset_index(drop=True)
    g = shapely.simplify(g, 0.3)
    cen = shapely.centroid(g)
    cx, cy = shapely.get_x(cen), shapely.get_y(cen)
    ok = (shapely.area(g) > 6) & (cx >= X0) & (cx < X1) & (cy >= Y0) & (cy < Y1) & (shapely.get_type_id(g) == 3)
    g, t = g[ok], t[ok].reset_index(drop=True)
    hv, fl = np.nan_to_num(t["height"].values.astype(float)), np.nan_to_num(t["num_floors"].values.astype(float))
    cls = t["class"].fillna(t["subtype"]).fillna("").values.astype(object)
    names = np.array([n.get("primary") if isinstance(n, dict) else None for n in t["names"]], object)

    est = np.zeros(len(g))
    forced = np.full(len(g), "", "U9")                       # source for buildings the raster itself outlined
    if (CACHE / f"{OB_NAME}.npy").exists():
        est, towers, blobs = raster_heights(g, (hv > 0) | (fl > 0))
        print("raster: split", len(towers), "towers from podiums; outlined", len(blobs), "buildings with no footprint")
        par = np.array([i for _, _, i in towers], int)
        n_new = len(towers) + len(blobs)
        g = np.concatenate([g, np.array([p for p, *_ in towers] + [p for p, _ in blobs], object)])
        est = np.concatenate([est, [h for _, h, _ in towers], [h for _, h in blobs]])
        hv, fl = np.concatenate([hv, np.zeros(n_new)]), np.concatenate([fl, np.zeros(n_new)])
        cls = np.concatenate([cls, cls[par], np.full(len(blobs), "", object)])
        names = np.concatenate([names, np.full(n_new, None, object)])
        forced = np.concatenate([forced, np.full(len(towers), "satellite"), np.full(len(blobs), "raster")])
    area = shapely.area(g)
    cen = shapely.centroid(g)
    cx, cy = shapely.get_x(cen), shapely.get_y(cen)
    ok = (cx >= X0) & (cx < X1) & (cy >= Y0) & (cy < Y1)
    g, est, hv, fl, cls, names, forced, area, cx, cy = (a[ok] for a in (g, est, hv, fl, cls, names, forced, area, cx, cy))

    guess = np.select([area < 40, area < 150, area < 900], [3.2, 6.4, 9.6], 8) * (0.9 + 0.25 * np.array([hash01(i) for i in range(len(g))]))
    h = np.select([hv > 0, fl > 0, est > 2.5], [hv, fl * 3.1 + 1, est], guess)
    src = np.select([hv > 0, fl > 0, forced != "", est > 2.5], ["measured", "floors", forced, "satellite"], "default")
    h = np.clip(h, 2.5, 6000)

    ov = HERE / "overrides.csv"
    if ov.exists():                                          # towers taller than the raster can report
        import csv
        rows = list(csv.DictReader(l for l in ov.read_text(encoding="utf-8").splitlines() if not l.startswith("#")))
        ox, oy = to_grid([float(r["lon"]) for r in rows], [float(r["lat"]) for r in rows])
        for r, x, y in zip(rows, ox, oy):
            d = np.hypot(cx - x, cy - y)
            cand = np.nonzero((d < float(r["radius_m"])) & (h >= 50))[0]
            cand = cand[h[cand] >= 0.8 * h[cand].max()] if len(cand) else cand     # the towers, not their podium blocks
            hit = cand[np.argsort(d[cand])][:int(r["count"])]                      # nearest first: a taller neighbour must not win
            print(f"  override {r['name']:28s} {len(hit)}/{r['count']} set to {r['height_m']} m (was {', '.join('%.0f m at %.0f m' % (v, w) for v, w in zip(h[hit], d[hit]))})")
            h[hit], src[hit] = float(r["height_m"]), "measured"
            names[hit] = [n or r["name"] for n in names[hit]]

    typ = np.array([B_TYPE.get(c, 2 if hh > 14 else 1) for c, hh in zip(cls, h)])
    typ[(h >= 45) & (typ <= 2)] = 3
    rgb = sample(sat, 10, (X0, Y0, X1, Y1), cx, cy).astype(int) if sat is not None else np.full((len(g), 3), 150)
    r565 = (rgb[:, 0] >> 3 << 11) | (rgb[:, 1] >> 2 << 5) | (rgb[:, 2] >> 3)
    base = sample(dem, 20, (X0, Y0, X1, Y1), cx, cy)
    return g, dict(h=h, src=src, typ=typ, r565=r565, base=base, area=area, cx=cx, cy=cy, names=list(names), cls=cls)


def pack_building(poly, ox, oy, h, base, r565, typ):
    rings = [poly.exterior, *poly.interiors][:255]
    out = bytearray(struct.pack("<HHhHBBB", min(65535, round(h * 10)), 0, int(round(base * 10)), int(r565), int(typ), 0, len(rings)))
    for r in rings:
        c = np.asarray(r.coords)[:-1]
        if len(c) > 255:
            c = c[np.linspace(0, len(c) - 1, 255).astype(int)]
        q = np.clip(np.round((c - (ox, oy)) * 16), -32768, 32767).astype("<i2")
        out += bytes([len(q)]) + q.tobytes()
    return bytes(out)


def load_roads():
    g, t = load_geoms("segment", ["subtype", "class", "road_flags", "level_rules", "rail_flags"])
    roads = {}                                               # (tx, ty) -> [record bytes, cls]
    n = 0
    for line, sub, c, rf, lv in zip(g, t["subtype"], t["class"], t["road_flags"], t["level_rules"]):
        if sub == "road":
            cls = ROAD_CLS.get(c, 7)
        elif sub == "rail":
            cls = 9 if c in ("subway", "light_rail", "monorail") else 8
        else:
            continue
        # Rules apply to a fraction of the segment ("between"); only raise it when most of it is elevated.
        span = lambda r: 1.0 if r.get("between") is None else float(r["between"][1] - r["between"][0])
        cover = {"is_bridge": 0.0, "is_tunnel": 0.0}
        for f in (rf if rf is not None else []):
            for v in (f.get("values") if f.get("values") is not None else []):
                if v in cover:
                    cover[v] += span(f)
        flags = (cover["is_bridge"] > 0.5) | (cover["is_tunnel"] > 0.5) << 1
        level = 1 if lv is not None and sum(span(r) for r in lv if (r.get("value") or 0) > 0) > 0.5 else 0
        deck = min(255, max(0, level) * 7 + (7 if flags & 1 and level <= 0 else 0))
        if cls == 9 and not flags & 2 and deck == 0:
            deck = 9                                         # Noida's Aqua and Blue lines run on viaduct
        for part in (line.geoms if line.geom_type == "MultiLineString" else [line]):
            pts = np.asarray(shapely.simplify(part, 0.5).coords)
            tix = np.floor((pts[:, 0] - X0) / TILE).astype(int) + 100000 * np.floor((pts[:, 1] - Y0) / TILE).astype(int)
            cuts = [0, *(np.nonzero(np.diff(tix))[0] + 1), len(pts)]
            for a, b in zip(cuts[:-1], cuts[1:]):            # split where the line changes tile, sharing the crossing vertex
                seg = pts[max(0, a - 1):b]
                if len(seg) < 2:
                    continue
                tx, ty = int((seg[-1][0] - X0) // TILE), int((seg[-1][1] - Y0) // TILE)
                if not (0 <= tx < NCX * 4 and 0 <= ty < NCY * 4):
                    continue
                q = np.clip(np.round((seg - (X0 + tx * TILE, Y0 + ty * TILE)) * 8), -32768, 32767).astype("<i2")
                rec = struct.pack("<6BH", cls, flags, deck, deck, deck, min(255, ROAD_W[cls] * 10), len(q)) + q.tobytes()
                roads.setdefault((tx, ty), []).append((rec, cls))
                n += 1
    print("road pieces", n)
    return roads


def chunk_blob(tiles_b, tiles_r, sky=False):
    """tiles_b[t] = (records, nSig, nMid); tiles_r[t] = [(rec, cls)]"""
    head, body, off = bytearray(struct.pack("<IHH", 1144209997, 2, 16)), bytearray(), 8 + 16 * 24
    counts = dict(n=[], s=[], m=[], r=[])
    for t in range(16):
        recs, ns, nm = tiles_b[t]
        rr = [r for r, c in tiles_r[t] if not sky or c <= 1 or c >= 8]
        if sky:
            recs, nm = recs[:ns], 0
        b, r = b"".join(recs), b"".join(rr)
        head += struct.pack("<6I", off + len(body), len(recs), ns, nm, off + len(body) + len(b), len(rr))
        body += b + r
        for k, v in zip("nsmr", (len(recs), ns, nm, len(rr))):
            counts[k].append(v)
    return bytes(head + body), counts


def labels(bg, info):
    out = []
    p = CACHE / "division.parquet"
    if p.exists():
        g, t = load_geoms("division", ["subtype", "names", "population"])
        kind = {"locality": "town", "neighborhood": "hood", "microhood": "hood", "macrohood": "hood", "county": "region"}
        for pt, sub, nm in zip(g, t["subtype"], t["names"]):
            n = nm.get("primary") if isinstance(nm, dict) else None
            if n and sub in kind and pt.geom_type == "Point" and X0 < pt.x < X1 and Y0 < pt.y < Y1:
                out.append({"n": n, "x": round(pt.x), "y": round(pt.y), "k": kind[sub]})
    seen = set()
    for i in np.argsort(-info["h"])[:400]:                   # named tall buildings
        n = info["names"][i]
        if n and n not in seen and info["h"][i] >= 60 and len(seen) < 60:
            seen.add(n)
            out.append({"n": n, "x": round(info["cx"][i]), "y": round(info["cy"][i]), "k": "tower",
                        "z": round(float(info["base"][i] + info["h"][i]), 1), "h": round(float(info["h"][i]), 1)})
    return out


GREEK = "Alpha|Beta|Gamma|Delta|Omicron|Zeta|Eta|Chi|Phi|Pi|Mu|Sigma|Omega|Xu|Knowledge Park|Ecotech|Techzone|Tech Zone"
ROMAN = {"I": "1", "II": "2", "III": "3", "IV": "4", "V": "5"}


def stage_search():
    """search.json: one point per sector. Overture has ~15 sector divisions, so the rest come from place addresses.
    ponytail: centre = median of addresses that mention the sector, not a surveyed boundary; swap in authority sector polygons if one is published."""
    g, t = load_geoms("place", ["names", "addresses"])
    xy = shapely.get_coordinates(g)
    pat = re.compile(rf"\b(?:(sec(?:tor)?)[-\s.]*(\d{{1,3}}[a-h]?)|({GREEK})[-\s]*(\d|iii|ii|iv|i|v)?)\b", re.I)
    hits = {}
    for (x, y), nm, ad in zip(xy, t["names"], t["addresses"]):
        if not (X0 < x < X1 and Y0 < y < Y1):
            continue
        a = ad[0] if ad is not None and len(ad) else {}
        text = f"{(nm or {}).get('primary') or ''} {a.get('freeform') or ''} {a.get('locality') or ''}"
        m = pat.search(text)
        if not m or re.search(r"faridabad|ghaziabad|vaishali|vasundhara|indirapuram|gurgaon|gurugram|dwarka|rohini|delhi", text, re.I):
            continue
        if m.group(1):
            name = "Sector " + m.group(2).upper().lstrip("0")
            if re.search(r"greater noida|noida ext|gr\.? noida", text, re.I):
                name += ", Greater Noida"
        else:
            num = (m.group(4) or "").upper()
            name = (m.group(3).title().replace("Tech Zone", "Techzone") + " " + ROMAN.get(num, num)).strip()
        hits.setdefault(name, []).append((x, y))
    out = []
    for name, pts in hits.items():
        p = np.array(pts)
        c = np.median(p, 0)
        near = p[np.hypot(*(p - c).T) < 1500]                 # drop strays that only mention the sector
        if len(near) >= 3 and len(near) >= 0.4 * len(p):
            c = np.median(near, 0)
            out.append({"n": name, "x": round(float(c[0])), "y": round(float(c[1]))})
    base = {s["n"]: s for s in out}
    gn = ", Greater Noida"                                    # same sector tagged both ways: keep one entry
    out = [s for s in out if not (s["n"].endswith(gn) and (b := base.get(s["n"][:-len(gn)]))
                                  and math.hypot(s["x"] - b["x"], s["y"] - b["y"]) < 2500)]
    lg, lt = (load_geoms("land_use", ["names", "subtype", "class"]) if (CACHE / "land_use.parquet").exists() else ([], None))
    lname = [n.get("primary") if isinstance(n, dict) else None for n in lt["names"]] if lt is not None else []
    exact = re.compile(rf"(?:sec(?:tor)?[-\s.]*(\d{{1,3}}[a-h]?)|({GREEK})[-\s]*(\d|iii|ii|iv|i|v)?)$", re.I)
    mapped = {}
    for p, n in zip(lg, lname):                              # OSM maps many sectors as named residential areas: exact centre
        m = n and exact.match(n.strip())
        c = p.centroid
        if m and X0 < c.x < X1 and Y0 < c.y < Y1:
            num = (m.group(3) or "").upper()
            name = "Sector " + m.group(1).upper().lstrip("0") if m.group(1) else (m.group(2).title().replace("Tech Zone", "Techzone") + " " + ROMAN.get(num, num)).strip()
            mapped.setdefault(name, []).append((p.area, c.x, c.y))
    for name, v in mapped.items():
        _, x, y = max(v)
        near = [s for s in out if s["n"].split(",")[0] == name and math.hypot(s["x"] - x, s["y"] - y) < 4000]
        for s in near:
            out.remove(s)
        out.append({"n": name, "x": round(x), "y": round(y)})
    key = lambda s: [int(v) if v.isdigit() else v for v in re.split(r"(\d+)", s["n"])]
    out.sort(key=key)
    assert any(s["n"] == "Sector 18" for s in out) and len(out) > 100, len(out)
    print("sectors", len(out), "of which mapped as areas", len(mapped))

    # ---- zoom-level labels: societies (5 km), named buildings and landmarks (1.8 km), shops (600 m)
    bad = re.compile(r"[|/@]|\d{5,}|\b(best|near me|for sale|pvt|private limited|ltd|llp)\b|\bin (noida|delhi|sector|greater)", re.I)
    def clean(n):
        n = re.sub(r"\s+", " ", n or "").strip(" -,.")
        return n if 2 < len(n) <= 34 and not bad.search(n) and not exact.match(n) else None
    rows = []                                                # (tier, importance, name, x, y, z)
    for p, n, sub in zip(lg, lname, lt["subtype"] if lt is not None else []):
        n = clean(n)
        if n and sub in ("residential", "developed", "education", "medical", "park", "golf", "recreation") and p.area > 2500:
            c = p.representative_point()
            rows.append((0, p.area, n, c.x, c.y, None))
    LM = {"hospital", "college_university", "high_school", "elementary_school", "movie_theater", "stadium_arena", "public_transit_facility_or_service",
          "police_station", "library", "hotel", "park", "government_office", "hindu_place_of_worship", "muslim_place_of_worship", "christian_place_of_worship"}
    pg, pt = load_geoms("place", ["names", "basic_category", "confidence"])
    pxy = shapely.get_coordinates(pg)
    for (x, y), nm, cat, conf in zip(pxy, pt["names"], pt["basic_category"], pt["confidence"]):
        n = clean(nm.get("primary") if isinstance(nm, dict) else None)
        if not n or conf < 0.7:
            continue
        big = cat in ("apartment", "condominium") or (cat == "shopping_mall" and re.search(r"mall|plaza|arcade|cent(re|er)|square|market|complex|city", n, re.I))
        tier = 0 if big and conf >= 0.8 else 2 if cat in LM and conf >= 0.8 else 3
        rows.append((tier, 0 if tier == 0 else conf, n, x, y, None))
    nb = CACHE / "named_buildings.json"                      # written by the pack stage, which knows roof heights
    for n, x, y, z, h in (json.loads(nb.read_text(encoding="utf-8")) if nb.exists() else []):
        if clean(n):
            rows.append((1, h, clean(n), x, y, z))
    rows.sort(key=lambda r: (r[0], -r[1]))
    seen, chunks, n_soc = set(), {}, 0
    for tier, _, n, x, y, z in rows:
        k = (n.lower(), int(x // 150), int(y // 150))        # same name twice within ~150 m: keep the more important
        if k in seen or not (X0 <= x < X1 and Y0 <= y < Y1):
            continue
        seen.add(k)
        rec = [n, round(x), round(y), ("soc", "bld", "lm", "poi")[tier]] + ([round(z, 1)] if z is not None else [])
        chunks.setdefault((int((x - X0) // CHUNK), int((y - Y0) // CHUNK)), []).append(rec)
        if tier == 0:
            out.append({"n": n, "x": round(x), "y": round(y), "k": "soc"})
            n_soc += 1
    for f in OUT.glob("l_*.json"):
        f.unlink()
    for (cx, cy), recs in chunks.items():
        (OUT / f"l_{cx}_{cy}.json").write_text(json.dumps(recs, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    (OUT / "search.json").write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    print("search entries", len(out), "(societies and malls", n_soc, "); labels", sum(map(len, chunks.values())), "in", len(chunks), "chunk files")


def stage_pack():
    dem, _ = dem_grid()
    sat = np.load(CACHE / "sat10.npy") if (CACHE / "sat10.npy").exists() else None
    g, info = load_buildings(dem, sat)
    print("buildings", len(g), {s: int((info["src"] == s).sum()) for s in np.unique(info["src"])})
    roads = load_roads()
    tx = ((info["cx"] - X0) // TILE).astype(int)
    ty = ((info["cy"] - Y0) // TILE).astype(int)
    sig = (info["h"] >= 40) | (info["area"] >= 4000)
    mid = ~sig & ((info["h"] >= 15) | (info["area"] >= 600))
    rank = np.where(sig, 0, np.where(mid, 1, 2))
    order = np.lexsort((-info["h"] * info["area"], rank, tx, ty))    # by tile, then LOD class, then importance
    by_tile = {}
    for i in order:
        by_tile.setdefault((tx[i], ty[i]), []).append(i)

    for f in OUT.glob("c_*.bin"):
        f.unlink()
    chunks, sky_parts, sky_b = [], [], 0
    for cy in range(NCY):
        for cx in range(NCX):
            tb, tr = [], []
            for t in range(16):
                k = (cx * 4 + t % 4, cy * 4 + t // 4)
                ids = by_tile.get(k, [])
                ox, oy = X0 + k[0] * TILE, Y0 + k[1] * TILE
                recs = [pack_building(g[i], ox, oy, info["h"][i], info["base"][i], info["r565"][i], info["typ"][i]) for i in ids]
                tb.append((recs, int(sum(sig[i] for i in ids)), int(sum(mid[i] for i in ids))))
                tr.append(roads.get(k, []))
            blob, cnt = chunk_blob(tb, tr)
            if not sum(cnt["n"]) and not sum(cnt["r"]):
                continue
            (OUT / f"c_{cx}_{cy}.bin").write_bytes(blob)
            chunks.append({"cx": cx, "cy": cy, "bytes": len(blob), **cnt})
            sb, sc = chunk_blob(tb, tr, sky=True)
            if sum(sc["n"]) or sum(sc["r"]):
                sky_parts.append((cx, cy, sb))
                sky_b += sum(sc["n"])
    head = bytearray(struct.pack("<IHH", 1263747661, 1, len(sky_parts)))
    off = 8 + 12 * len(sky_parts)
    for cx, cy, b in sky_parts:
        head += struct.pack("<HHII", cx, cy, off, len(b))
        off += len(b)
    (OUT / "sky.bin").write_bytes(bytes(head) + b"".join(b for _, _, b in sky_parts))

    src = info["src"]
    (CACHE / "named_buildings.json").write_text(json.dumps(
        [[n, round(float(info["cx"][i])), round(float(info["cy"][i])), round(float(info["base"][i] + info["h"][i]), 1), round(float(info["h"][i]), 1)]
         for i, n in enumerate(info["names"]) if n], ensure_ascii=False), encoding="utf-8")
    manifest = {
        "version": 1, "name": "Noida", "extent": [X0, Y0, X1, Y1], "origin": {"lat": LAT0, "lon": LON0},
        "chunk": CHUNK, "tile": TILE, "ncx": NCX, "ncy": NCY, "chunkExt": "bin",
        "height": {"file": "height.png", "res": 20, "w": (X1 - X0) // 20, "h": (Y1 - Y0) // 20, "offset": H_OFF, "scale": H_SCALE},
        "masks": {"file": "masks.png", "res": 12.5},
        "sat": {"res": 10, "tile": 1600, "nx": (X1 - X0) // 16000, "ny": (Y1 - Y0) // 16000},
        "region": {"extent": [RX0, RY0, RX1, RY1], "sat": {"file": "region_sat.jpg", "res": 40},
                   "height": {"file": "region_h.png", "res": 80, "offset": H_OFF, "scale": H_SCALE}},
        "chunks": chunks,
        "sky": {"file": "sky", "chunks": len(sky_parts), "buildings": sky_b},
        "stats": {"total": len(g), **{s: int((src == s).sum()) for s in ("measured", "floors", "satellite", "raster", "default")}},
        "labels": labels(g, info),
        "places": [{"n": n, "x": round(float(x)), "y": round(float(y)), "d": d, "turn": turn}
                   for (n, la, lo, d, turn), x, y in zip(PLACES, *to_grid([p[2] for p in PLACES], [p[1] for p in PLACES]))],
        "imagery_dates": S2_DATES,
        "sources": [
            "Buildings, roads, rail, water, place names: Overture Maps Foundation (includes OpenStreetMap contributors, ODbL; Google Open Buildings; Microsoft ML Buildings)",
            "Terrain: Copernicus DEM GLO-30 (ESA / Airbus)",
            "Ground imagery: Copernicus Sentinel-2 L2A, median of " + ", ".join(S2_DATES),
            "Tree and crop cover: ESA WorldCover 2021 v200, CC BY 4.0",
        ] + (["Building height estimates: Google Open Buildings 2.5D Temporal (2023), CC BY 4.0"] if (CACHE / f"{OB_NAME}.npy").exists() else []) + (["Tower heights: Wikipedia, List of tallest buildings in Delhi-NCR, CC BY-SA 4.0"] if (HERE / "overrides.csv").exists() else []),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    print("packed", len(chunks), "chunks,", sum(c["bytes"] for c in chunks) // 1024, "KB; sky", sky_b, "buildings; labels", len(manifest["labels"]))


# Fly-to chips. Coordinates are approximate centres, good to a few hundred metres.
PLACES = [("Sector 18", 28.5700, 77.3219, 2600, 20), ("Film City", 28.5665, 77.3150, 2200, 60), ("Okhla Bird Sanctuary", 28.5560, 77.3080, 4200, 300),
          ("Sector 62", 28.6270, 77.3650, 3200, 40), ("Expressway", 28.5000, 77.4000, 5200, 315), ("Noida Extension", 28.6080, 77.4320, 3600, 20),
          ("Pari Chowk", 28.4653, 77.5109, 2600, 30), ("Knowledge Park", 28.4720, 77.4900, 3600, 330), ("Buddh Circuit", 28.3487, 77.5331, 4200, 20)]


STAGES = {"dem": stage_dem, "sat": stage_sat, "masks": stage_masks, "heights": stage_heights, "pack": stage_pack, "search": stage_search}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for s in (list(STAGES) if sys.argv[1:] == ["all"] else sys.argv[1:]):
        print("==", s)
        STAGES[s]()
