"""Smoke test for the viewer. Needs the server on 127.0.0.1:8765 and `pip install playwright`.

    python smoke.py

Fails (exit 1) on console errors, a viewer that never boots, search that goes nowhere,
idle redraws while zoomed out, or labels that do not appear when close.
"""
import sys, time
from playwright.sync_api import sync_playwright

errs, bad = [], []
def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok: bad.append(name)

with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True, args=["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist", "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": 1400, "height": 860})
    pg.on("pageerror", lambda e: errs.append("pageerror " + str(e)[:200]))
    pg.on("console", lambda m: m.type == "error" and "favicon" not in m.text and "404" not in m.text and errs.append(m.text[:200]))
    pg.goto("http://127.0.0.1:8765/viewer/index.html")
    try:
        pg.wait_for_function("document.getElementById('boot').classList.contains('done')", timeout=120000); booted = True
    except Exception: booted = False
    check("boots", booted)
    time.sleep(4)
    n = pg.evaluate("window.__noida.tiles.length")
    check("tiles loaded", n > 1000 and pg.evaluate("window.__noida.tiles.filter(t=>t.far).length") > 500, n)
    for q in ["Sector 62", "Skytech Matrott", "Pari Chowk", "sec 18"]:
        check(f"search '{q}'", bool(pg.evaluate(f"!!window.__noida.find({q!r}, true)")))
    check("search miss is a miss", not pg.evaluate("!!window.__noida.find('zzzzqq', true)"))
    pg.evaluate("window.__noida.fly({d:30000,tilt:0.5,turn:0},10)"); time.sleep(8)
    f0 = pg.evaluate("window.__noida.renderer.info.render.frame"); time.sleep(3)
    check("idle far draws no frames", pg.evaluate("window.__noida.renderer.info.render.frame") - f0 == 0)
    pg.evaluate("window.__noida.find('Sector 62', true)"); time.sleep(9)
    check("fine labels near", pg.evaluate("window.__noida.fine") > 0, pg.evaluate("window.__noida.fine"))
    check("detail tiles near", pg.evaluate("window.__noida.tiles.filter(t=>t.det).length") > 0)
    pg.evaluate("window.__noida.setHour(23)"); time.sleep(2); pg.evaluate("window.__noida.setHour(12)"); time.sleep(2)
    check("time slider no errors", not errs, errs[:3])
    check("labels toggle in DOM", pg.locator("#tags .tag:visible").count() > 0)
    check("search input labelled", pg.evaluate("!!document.getElementById('find').getAttribute('aria-label') || !!document.getElementById('find').placeholder"))
    b.close()
for e in errs: print("ERR", e)
sys.exit(1 if bad or errs else 0)
