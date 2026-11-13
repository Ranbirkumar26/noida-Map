"""Realty360 project sites for the Noida 3D map.

    python projects.py fetch     # snapshot the project list from the site (needs http://localhost:3000)
    python projects.py locate    # find each project's real site -> projects.csv
    python projects.py audit     # which sites already have towers in the map data

projects.csv is the reviewable record: one row per project with its coordinates, where they came
from, and the tower facts used when the map has no building there. Rows are only filled in, never
overwritten: a row with source "manual" or "web" is kept as it is, so hand corrections survive.
build.py reads this file (project_buildings) and adds towers for rows marked need=planned.
"""
import csv, json, math, re, sys, time, urllib.parse, urllib.request
from pathlib import Path

import numpy as np
import shapely

from build import CACHE, OUT, X0, Y0, X1, Y1, OB_RES, OB_NAME, load_geoms, to_grid, to_lonlat

HERE = Path(__file__).parent
CSV, API_SNAPSHOT = HERE / "projects.csv", CACHE / "projects_api.json"
FIELDS = ["slug", "name", "city", "sector", "status", "type", "lat", "lon", "source", "matched", "confidence", "need", "towers", "floors", "acres", "facts_url"]
STOP = {"the", "by", "at", "of", "and", "noida", "greater", "phase", "sector", "residences", "residency", "apartments", "project"}

norm = lambda t: re.sub(r"[^a-z0-9]+", " ", re.sub(r"\bsec(rtor|tor)?\b\.?", "sector", (t or "").lower())).strip()
tight = lambda t: norm(t).replace(" ", "")
ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5"}
# The map data spells these differently from the listing. Checked by eye against the candidates near each sector.
ALIAS = {"Samridhi Daksha Avenue": "Samridhi Daksh Avenue", "Experion Saatori": "Experion Noida 151", "Max Estate 105": "Max Estates Sector 105 Noida"}


# Rough town limits (lat0, lat1, lon0, lon1). A name match outside its town is a sales office or a namesake, not the site.
TOWN = {"Noida": (28.40, 28.66, 77.29, 77.50), "Greater Noida": (28.40, 28.56, 77.44, 77.62), "Greater Noida West": (28.54, 28.66, 77.41, 77.50),
        "Yamuna Expressway": (28.20, 28.46, 77.45, 77.70), "Ghaziabad": (28.62, 28.80, 77.30, 77.60)}


def in_town(city, lat, lon):
    b = TOWN.get(city)
    return not b or (b[0] <= lat <= b[1] and b[2] <= lon <= b[3])


def sector_name(t):
    s = norm(re.sub(r"\(.*?\)|/.*", "", t or ""))
    s = re.sub(r"\b(i{1,3}|iv|v)\b", lambda m: ROMAN[m.group(0)], s)
    s = re.sub(r"^sector (?=[a-z])", "", s).replace("tech zone", "techzone")
    return re.sub(r"(\d) ([a-d])$", r"\1\2", s)


def read_rows():
    return {r["slug"]: r for r in csv.DictReader(CSV.open(encoding="utf-8"))} if CSV.exists() else {}


def write_rows(rows):
    with CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(sorted(rows.values(), key=lambda r: r["name"].lower()))


def stage_fetch():
    data = json.load(urllib.request.urlopen("http://localhost:3000/api/map-projects", timeout=30))
    API_SNAPSHOT.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print("projects", len(data))


def anchors(projects):
    """Where the listing says the project is: its sector on the map, else the town-level point. (x, y, search radius)."""
    sectors = {}
    for s in json.loads((OUT / "search.json").read_text(encoding="utf-8")):
        if not s.get("k"):
            sectors.setdefault(norm(s["n"]), s)
    out = {}
    for p in projects:
        s = sector_name(p["sector"])
        numbered = bool(re.match(r"sector \d", s))
        gn = p["city"].startswith("Greater Noida")
        tries = ([s + " greater noida"] + ([] if numbered else [s, re.sub(r"(\d)[a-d]$", r"\1", s)])) if gn else ([s] if p["city"] == "Noida" or not numbered else [])
        hit = next((sectors[t] for t in tries if t in sectors), None)
        if hit:
            out[p["slug"]] = (hit["x"], hit["y"], 3500)
        elif p["lat"] is not None:
            x, y = to_grid([p["lng"]], [p["lat"]])
            out[p["slug"]] = (float(x[0]), float(y[0]), 14000)       # town-level only: search wide
    return out


def name_score(project, candidate):
    """0..1: how well a map name matches a project name. 1 = same name, ignoring spaces and case."""
    if tight(project) == tight(candidate):
        return 1.0
    a = [w for w in norm(project).split() if w not in STOP]
    b = [w for w in norm(candidate).split() if w not in STOP]
    if len(a) < 2 or not b:
        return 0.0
    if all(w in b for w in a):                                # every word of the project name is in the map name
        return 0.9 if len(b) <= len(a) + 2 else 0.75
    if len(b) >= 2 and all(w in a for w in b) and len(" ".join(b)) >= 9:
        return 0.7                                            # the map name is a shorter form: "ATS Pristine" for "ATS Pristine Golf Villas"
    return 0.0


def map_names():
    """Every named thing in the map inputs: (name, x, y, kind, area)."""
    out = []
    g, t = load_geoms("land_use", ["names", "subtype"])
    for p, nm in zip(g, t["names"]):
        n = nm.get("primary") if isinstance(nm, dict) else None
        if n:
            c = p.representative_point()
            out.append((n, c.x, c.y, "land_use", p.area))
    g, t = load_geoms("place", ["names", "confidence"])
    xy = shapely.get_coordinates(g)
    for (x, y), nm, conf in zip(xy, t["names"], t["confidence"]):
        n = nm.get("primary") if isinstance(nm, dict) else None
        if n and conf >= 0.5:
            out.append((n, x, y, "place", 0))
    for n, x, y, z, h in json.loads((CACHE / "named_buildings.json").read_text(encoding="utf-8")):
        out.append((n, x, y, "building", 0))
    return out


def geocode(name, sector, city, cache):
    """OpenStreetMap geocoders, one request per second. Returns (lat, lon, label) or None."""
    q = f"{name}, {sector or ''}, {city}".replace(", ,", ",")
    if q in cache:
        return cache[q]
    hit = None
    for url, pick in (
        ("https://nominatim.openstreetmap.org/search?format=json&limit=3&countrycodes=in&viewbox=77.2,28.85,77.8,28.1&bounded=1&q=" + urllib.parse.quote(q),
         lambda d: [(float(h["lat"]), float(h["lon"]), h["display_name"]) for h in d]),
        ("https://photon.komoot.io/api/?limit=5&bbox=77.2,28.1,77.8,28.85&q=" + urllib.parse.quote(f"{name} {city}"),
         lambda d: [(f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0], f["properties"].get("name") or "") for f in d["features"]]),
    ):
        time.sleep(1.1)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "map-site-locator/1.0"})
            for lat, lon, label in pick(json.load(urllib.request.urlopen(req, timeout=30))):
                if name_score(name, label.split(",")[0]) >= 0.7:
                    hit = (lat, lon, label.split(",")[0])
                    break
        except Exception as e:
            print("  geocoder error", e)
        if hit:
            break
    cache[q] = hit
    return hit


def stage_locate():
    projects = json.loads(API_SNAPSHOT.read_text(encoding="utf-8"))
    rows, anchor, names = read_rows(), anchors(projects), map_names()
    gc_path = CACHE / "projects_geocode.json"
    gc = json.loads(gc_path.read_text(encoding="utf-8")) if gc_path.exists() else {}
    nx, ny = np.array([n[1] for n in names]), np.array([n[2] for n in names])
    for p in projects:
        row = rows.get(p["slug"]) or {f: "" for f in FIELDS}
        row.update(slug=p["slug"], name=p["name"].strip(), city=p["city"], sector=p["sector"] or "", status=p["status"] or "", type=p["type"] or "")
        rows[p["slug"]] = row
        if row["source"] in ("manual", "web"):
            continue
        ax, ay, radius = anchor.get(p["slug"], (0, 0, 40000))
        best, want = None, ALIAS.get(p["name"].strip(), p["name"])
        # Near the listed sector a looser name match is safe. Anywhere else on the map only a full name match counts:
        # the listing's sector is sometimes wrong or only town-level, and the society is still on the map under its own name.
        for reach, floor in ((radius, 0.7), (26000 if p["city"] != "Noida" else 0, 0.9)):   # Noida sectors are on the map, so their anchor is trusted
            for i in np.nonzero(np.hypot(nx - ax, ny - ay) < reach)[0]:
                n, x, y, kind, area = names[i]
                s = name_score(want, n)
                if s >= floor and in_town(p["city"], *reversed([float(v[0]) for v in to_lonlat([x], [y])])):
                    rank = (s, {"land_use": 2, "building": 1, "place": 0}[kind], -math.hypot(x - ax, y - ay))
                    if not best or rank > best[0]:
                        best = (rank, n, x, y, kind)
            if best:
                break
        if best:
            (s, *_), n, x, y, kind = best
            lon, lat = to_lonlat([x], [y])
            row.update(lat=f"{lat[0]:.6f}", lon=f"{lon[0]:.6f}", source="map:" + kind, matched=n, confidence=f"{s:.2f}")
            continue
        hit = geocode(p["name"].strip(), p["sector"], p["city"], gc)
        gc_path.write_text(json.dumps(gc, ensure_ascii=False), encoding="utf-8")
        if hit:
            x, y = to_grid([hit[1]], [hit[0]])
            if math.hypot(x[0] - ax, y[0] - ay) < radius:
                row.update(lat=f"{hit[0]:.6f}", lon=f"{hit[1]:.6f}", source="geocoder", matched=hit[2], confidence="0.70")
                continue
        row.update(lat="", lon="", source="", matched="", confidence="")
    # Not named on any map: stand it in its sector. The sector is where already-located projects of the same
    # listing sector are, else the sector centre from the map. Good to about a kilometre; marked low confidence.
    groups = {}
    for r in rows.values():
        if r["source"].startswith("map:") or r["source"] in ("manual", "web", "geocoder"):
            groups.setdefault((r["city"], sector_name(r["sector"])), []).append((float(r["lat"]), float(r["lon"])))
    for p in projects:
        r = rows[p["slug"]]
        if r["source"] and r["source"] != "sector":
            continue
        # Listings file Yamuna Expressway sectors under other towns, and 22A sits beside 22D.
        near = groups.get((r["city"], sector_name(r["sector"]))) or groups.get(("Yamuna Expressway", sector_name(r["sector"])))             or (groups.get(("Yamuna Expressway", "sector 22d")) if sector_name(r["sector"]).startswith("sector 22") else None)
        if near and r["sector"]:
            lat, lon = float(np.median([v[0] for v in near])), float(np.median([v[1] for v in near]))
            r.update(lat=f"{lat:.6f}", lon=f"{lon:.6f}", source="sector", matched="beside " + str(len(near)) + " located in same sector", confidence="0.30")
        elif p["slug"] in anchor and anchor[p["slug"]][2] < 10000:
            lon, lat = to_lonlat([anchor[p["slug"]][0]], [anchor[p["slug"]][1]])
            r.update(lat=f"{lat[0]:.6f}", lon=f"{lon[0]:.6f}", source="sector", matched="sector centre", confidence="0.30")
    write_rows(rows)
    by = {}
    for r in rows.values():
        by[r["source"] or "unresolved"] = by.get(r["source"] or "unresolved", 0) + 1
    print("located:", by)
    print("unresolved:", "; ".join(f"{r['name']} [{r['sector']}, {r['city']}]" for r in rows.values() if not r["source"]))


def stage_audit():
    """need = built, lowrise, planned or outside (off the detailed grid). Judged from the 2023 satellite roof-height raster."""
    rows = read_rows()
    hr = np.load(CACHE / f"{OB_NAME}.npy", mmap_mode="r")
    for r in rows.values():
        if not r["lat"]:
            r["need"] = ""
            continue
        x, y = to_grid([float(r["lon"])], [float(r["lat"])])
        x, y = float(x[0]), float(y[0])
        if not (X0 + 300 < x < X1 - 300 and Y0 + 300 < y < Y1 - 300):
            r["need"] = "outside"
            continue
        def roofs(radius):                                   # (m2 of roof at 20 m or higher, m2 of any roof) around the site
            c, w, rr = int((x - X0) / OB_RES), int(radius / OB_RES), int((Y1 - y) / OB_RES)
            win = np.asarray(hr[rr - w:rr + w, c - w:c + w])
            return (win >= 20).sum() * OB_RES ** 2, (win > 2.5).sum() * OB_RES ** 2
        tall, low = roofs(150)
        open_now = r["status"] in ("Ready to Move", "Active")
        if tall >= 600:
            r["need"] = "built"                              # towers stand there
        elif low >= 3000 and (open_now or r["type"] == "commercial"):
            r["need"] = "lowrise"                            # a real mall, plaza or villa cluster stands there: label only
        elif open_now and roofs(400)[0] >= 2000:
            r["need"] = "built"                              # the map point is the gate or sales office; the towers are beside it
        else:
            r["need"] = "planned"                            # open ground: towers are added from published project details
    write_rows(rows)
    by = {}
    for r in rows.values():
        by[r["need"] or "unlocated"] = by.get(r["need"] or "unlocated", 0) + 1
    print("audit:", by)
    print("planned:", "; ".join(f"{r['name']} ({r['status']})" for r in rows.values() if r["need"] == "planned"))


if __name__ == "__main__":
    for s in sys.argv[1:]:
        print("==", s)
        {"fetch": stage_fetch, "locate": stage_locate, "audit": stage_audit}[s]()
