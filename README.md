# Noida 3D

Interactive 3D model of Noida and Greater Noida in the browser, built from open map data. About 1.1 million buildings, roads, metro lines, terrain and satellite ground imagery on a 32 × 48 km grid.

## Run

```
python -m http.server 8765
```

Open http://127.0.0.1:8765/viewer/index.html

Any static host works. The viewer reads `../data/` relative to `viewer/`.

## Layout

| Path | What |
|---|---|
| `viewer/` | three.js viewer: `index.html`, `main.js`, `worker.js` (tile mesh builder) |
| `data/` | built output: binary building/road chunks, terrain, imagery, labels, search index |
| `pipeline/build.py` | data pipeline, stages `dem sat masks heights pack search` |
| `pipeline/overrides.csv` | published heights for towers the satellite data cannot measure |
| `pipeline/audit.py` | accuracy report for the building layer |
| `decode.py` | reads the chunk format and checks it against the manifest |

## Rebuild the data

Needs Python with `numpy rasterio shapely scipy pillow pyarrow overturemaps`. Inputs are downloaded into `pipeline/cache/` (not in this repository): Overture GeoParquet for `building`, `segment`, `water`, `division`, `place`, `land_use` over bbox `77.255,28.275,77.605,28.725`, and `ob25d_tiles.json`, the list of Open Buildings 2.5D tiles.

```
cd pipeline
python build.py all
python audit.py
python ../decode.py ../data
```

## What is real and what is generated

Real: building footprints and positions, roads, rail and metro alignments, terrain, water, ground and roof colour, tree cover, sun position.

Generated: windows, rooftop details, tree shapes, traffic, lights, and the height of any building with no mapped or published height. Satellite height estimates stop near 100 m. Some newer societies have no mapped footprint and are outlined from the satellite height data, so their shapes are rough.

## Data sources and licences

- Buildings, roads, rail, water, places, land use: Overture Maps Foundation, which includes OpenStreetMap contributors (ODbL), Google Open Buildings and Microsoft ML Buildings
- Building height estimates: Google Open Buildings 2.5D Temporal (2023), CC BY 4.0
- Tower heights: Wikipedia, "List of tallest buildings in Delhi-NCR", CC BY-SA 4.0
- Terrain: Copernicus DEM GLO-30 (ESA / Airbus)
- Ground imagery: Copernicus Sentinel-2 L2A
- Tree and crop cover: ESA WorldCover 2021 v200, CC BY 4.0

The files in `data/` derive from these sources and carry their licences, including ODbL share-alike for the building and road layers.
