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
    apply_web(rows)
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


WEB = HERE / "web_sites.csv"


def apply_web(rows):
    """Coordinates and facts read from property portal pages (web_sites.csv), for projects no map names.
    A point is used only when it is inside the project's town, within 6 km of where the listing's sector puts the
    project, and not a placeholder: portals give one stock point for a whole area, so a point shared by two or more
    different projects is rejected."""
    import csv
    if not WEB.exists():
        return
    web = [r for r in csv.DictReader(l for l in WEB.read_text(encoding="utf-8").splitlines() if not l.startswith("#"))]
    seen = {}
    for w in web:
        seen.setdefault((round(float(w["lat"]), 4), round(float(w["lon"]), 4)), set()).add(w["name"])
    by_name = {r["name"]: r for r in rows.values()}
    used, refused = [], []
    for w in web:
        r = by_name.get(w["name"])
        if not r or r["source"] in ("manual", "web") or (r["source"] or "").startswith("map:"):
            continue
        for k in ("towers", "floors", "acres"):
            if w[k] and not r[k]:
                r[k] = w[k]
        if w[k] and not r["facts_url"]:
            r["facts_url"] = w["url"]
        lat, lon = float(w["lat"]), float(w["lon"])
        why = None
        if len(seen[(round(lat, 4), round(lon, 4))]) > 1:
            why = "placeholder point shared by " + ", ".join(sorted(seen[(round(lat, 4), round(lon, 4))]))
        elif not in_town(r["city"], lat, lon):
            why = "outside " + r["city"]
        elif r["lat"]:
            x0, y0 = to_grid([float(r["lon"])], [float(r["lat"])])
            x1, y1 = to_grid([lon], [lat])
            if math.hypot(x1[0] - x0[0], y1[0] - y0[0]) > 6000:
                why = "more than 6 km from its listed sector"
        if why:
            refused.append(f"{w['name']} ({why})")
            continue
        r.update(lat=f"{lat:.6f}", lon=f"{lon:.6f}", source="web", matched=w["url"][:200], confidence="0.80")
        used.append(w["name"])
    print(f"web: used {len(used)}: {', '.join(used)}")
    print(f"web: refused {len(refused)}: {'; '.join(refused)}")


def kind_of(r):
    """villa, commercial or tower. Decides how tall the building at a project's site must be to count as its building."""
    floors = int(r["floors"]) if (r["floors"] or "").isdigit() else 0
    if "villa" in r["name"].lower() or 0 < floors <= 4:
        return "villa"
    return "commercial" if r["type"] == "commercial" else "tower"


def stage_audit():
    """need = built, lowrise, planned or outside (off the detailed grid).

    Judged from the 2023 satellite roof-height raster at the pin itself, not a wide circle: a project counts as built
    only when its kind of building stands within 40 m of its pin (towers 20 m or more, villas and shops 6 m or more).
    Otherwise the pin moves onto the nearest such building that no other project stands nearer to: within 250 m for a
    ready or active project (its pin may be a gate or sales office), within 160 m for an unfinished project whose
    pin is the centre of its mapped plot (a big society's centre can sit in its courtyard). Any other unfinished
    project gets generated buildings, because the towers beside it may belong to an unrelated society. Everything else gets buildings generated from its published details."""
    rows = read_rows()
    hr = np.load(CACHE / f"{OB_NAME}.npy", mmap_mode="r")
    grid = {}
    for r in rows.values():
        if r["lat"]:
            x, y = to_grid([float(r["lon"])], [float(r["lat"])])
            grid[r["slug"]] = (float(x[0]), float(y[0]))

    def cells(x, y, radius, floor):
        """Cells of the height raster within `radius` of (x, y) with a roof at `floor` metres or higher, as grid points."""
        c, w, rr = int((x - X0) / OB_RES), int(radius / OB_RES), int((Y1 - y) / OB_RES)
        win = np.asarray(hr[max(0, rr - w):rr + w + 1, max(0, c - w):c + w + 1])
        jj, ii = np.nonzero(win >= floor)
        px = X0 + (max(0, c - w) + ii + 0.5) * OB_RES
        py = Y1 - (max(0, rr - w) + jj + 0.5) * OB_RES
        keep = np.hypot(px - x, py - y) <= radius
        return px[keep], py[keep]

    moved = 0
    for r in rows.values():
        if r["slug"] not in grid:
            r["need"] = ""
            continue
        x, y = grid[r["slug"]]
        if not (X0 + 300 < x < X1 - 300 and Y0 + 300 < y < Y1 - 300):
            r["need"] = "outside"
            continue
        kind = kind_of(r)
        floor = 20 if kind == "tower" else 6
        if len(cells(x, y, 40, floor)[0]) >= 4:              # at least 64 m2 of roof at that height
            r["need"] = "built" if kind == "tower" else "lowrise"
            continue
        if r["source"] != "manual":
            # Ready projects: the pin may be a gate or sales office, so look up to 250 m. Others: only their own plot.
            # An unfinished project only takes towers inside its own mapped plot: next to it may stand an unrelated society.
            reach = 250 if r["status"] in ("Ready to Move", "Active") else 160 if r["source"] == "map:land_use" else 0
            px, py = cells(x, y, reach, floor) if reach else (np.array([]), np.array([]))
            others = [v for k, v in grid.items() if k != r["slug"]]
            # A tower belongs to this project only if no other project's pin is nearer to it, or within 90 m of it.
            free = [(math.hypot(a - x, b - y), a, b) for a, b in zip(px, py)
                    if all(math.hypot(a - ox, b - oy) > max(90, math.hypot(a - x, b - y)) for ox, oy in others)]
            if len(free) >= 4:
                _, a, b = min(free)
                lon, lat = to_lonlat([a], [b])
                r.update(lat=f"{lat[0]:.6f}", lon=f"{lon[0]:.6f}", matched=(r["matched"] + " (moved onto its towers)")[:200])
                grid[r["slug"]] = (a, b)
                r["need"] = "built" if kind == "tower" else "lowrise"
                moved += 1
                continue
        r["need"] = "planned"
    write_rows(rows)
    by = {}
    for r in rows.values():
        by[r["need"] or "unlocated"] = by.get(r["need"] or "unlocated", 0) + 1
    print("audit:", by, "| pins moved onto their towers:", moved)
    print("planned:", "; ".join(f"{r['name']} ({r['status']}, facts {'yes' if r['towers'] else 'no'})" for r in rows.values() if r["need"] == "planned"))


if __name__ == "__main__":
    for s in sys.argv[1:]:
        print("==", s)
        {"fetch": stage_fetch, "locate": stage_locate, "audit": stage_audit}[s]()
