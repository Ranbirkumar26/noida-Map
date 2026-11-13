"""Decoder for mumbai-3d chunk files (c_<cx>_<cy>.bin, magic 'MB3D') and sky.bin ('MBSK').

Format recovered from site/a/*/src/worker.js (fillTile / setSky). All little-endian.

chunk = 4x4 km, holds 16 tiles of 1x1 km:
  u32 magic 'MB3D', u16 version (2), u16 ntiles (16)
  16 x tile entry (24 B), index = (ty%4)*4 + tx%4:
     u32 offB, u32 nB, u32 nSig, u32 nMid, u32 offR, u32 nR
  building @offB (sorted: nSig significant first, then nMid, then rest -> LOD prefix):
     u16 h*10, u16 minh*10, i16 base*10, u16 roofRGB565, u8 type, u8 flags, u8 nrings
     per ring: u8 npts, npts x (i16 x*16, i16 y*16)   metres, tile-local
  road @offR:
     u8 cls, u8 flags, u8 deck, u8 zStart, u8 zEnd, u8 width*10, u16 npts
     npts x (i16 x*8, i16 y*8)                        metres, tile-local
sky.bin = low-detail copy of every chunk in one file:
  u32 magic 'MBSK', u16 version, u16 n, n x (u16 cx, u16 cy, u32 off, u32 len) -> embedded MB3D blobs

Usage: python decode.py site/d/<hash>            # self-check against manifest
       python decode.py site/d/<hash> out.geojson # also export buildings as GeoJSON (lon/lat)
"""
import json, math, struct, sys
from pathlib import Path

MM3D, MMSK = 1144209997, 1263747661  # b'MB3D', b'MBSK'


def tiles(buf):
    """Yield (tile_index, buildings, roads) for one MM3D blob."""
    assert struct.unpack_from("<I", buf, 0)[0] == MM3D, "bad chunk magic"
    for t in range(16):
        offB, nB, nSig, nMid, offR, nR = struct.unpack_from("<6I", buf, 8 + t * 24)
        o, bl = offB, []
        for i in range(nB):
            h, minh, base, r565, typ, flags, nr = struct.unpack_from("<HHhHBBB", buf, o)
            o += 11
            rings = []
            for _ in range(nr):
                n = buf[o]
                o += 1
                c = struct.unpack_from("<%dh" % (n * 2), buf, o)
                o += n * 4
                rings.append([(c[k] / 16, c[k + 1] / 16) for k in range(0, n * 2, 2)])
            bl.append(dict(h=h / 10, minh=minh / 10, base=base / 10, roof565=r565, type=typ, flags=flags,
                           lod="sig" if i < nSig else "mid" if i < nSig + nMid else "rest", rings=rings))
        o, rl = offR, []
        for _ in range(nR):
            cls, flags, deck, zs, ze, w, n = struct.unpack_from("<6BH", buf, o)
            o += 8
            c = struct.unpack_from("<%dh" % (n * 2), buf, o)
            o += n * 4
            rl.append(dict(cls=cls, flags=flags, deck=deck, zs=zs, ze=ze, width=w / 10,
                           pts=[(c[k] / 8, c[k + 1] / 8) for k in range(0, n * 2, 2)]))
        yield t, bl, rl


def sky(buf):
    """Yield ((cx, cy), blob) for each chunk inside sky.bin."""
    assert struct.unpack_from("<I", buf, 0)[0] == MMSK, "bad sky magic"
    for i in range(struct.unpack_from("<H", buf, 6)[0]):
        cx, cy, off, ln = struct.unpack_from("<HHII", buf, 8 + i * 12)
        yield (cx, cy), buf[off:off + ln]


def main(data_dir, out=None):
    d = Path(data_dir)
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    x0, y0 = m["extent"][0], m["extent"][1]
    lat0, lon0 = m["origin"]["lat"], m["origin"]["lon"]
    # ponytail: equirectangular metres->degrees; manifest.lonlat holds the site's exact polynomial if sub-metre matters
    mlat, mlon = 110574.0, 111320.0 * math.cos(math.radians(lat0))
    feats, nb, nr = [], 0, 0
    for c in m["chunks"]:
        buf = (d / f"c_{c['cx']}_{c['cy']}.bin").read_bytes()
        assert len(buf) == c["bytes"], "size mismatch vs manifest"
        for t, bl, rl in tiles(buf):
            assert len(bl) == c["n"][t], f"building count mismatch chunk {c['cx']},{c['cy']} tile {t}"
            assert len(rl) == c["r"][t], "road count mismatch"
            nb += len(bl)
            nr += len(rl)
            if out:
                ox = x0 + (c["cx"] * 4 + t % 4) * m["tile"]
                oy = y0 + (c["cy"] * 4 + t // 4) * m["tile"]
                for b in bl:
                    ring = [[round(lon0 + (ox + x) / mlon, 7), round(lat0 + (oy + y) / mlat, 7)] for x, y in b["rings"][0]]
                    feats.append({"type": "Feature", "properties": {"height": b["h"], "base": b["base"], "type": b["type"]},
                                  "geometry": {"type": "Polygon", "coordinates": [ring + ring[:1]]}})
    # stats.total (600704) counts height-source rows before filtering; shipped count is checked per tile above
    ns = sum(len(bl) for _, blob in sky((d / "sky.bin").read_bytes()) for _, bl, _ in tiles(blob))
    assert ns == m["sky"]["buildings"], (ns, m["sky"]["buildings"])
    print(f"OK: {nb} buildings, {nr} roads in {len(m['chunks'])} chunks; sky.bin {ns} buildings")
    if out:
        Path(out).write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
        print("wrote", out, len(feats), "features")


if __name__ == "__main__":
    main(*sys.argv[1:3])
