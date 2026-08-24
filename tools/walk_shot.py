"""Screenshot the three.js walk viewer, so the lighting can be looked at.

    python tools/walk_shot.py --model build/lodo/lodo.gltf --out build/shots

`shoot.py` drives the dependency-free viewer and answers "did the compiler put
the right things in the right places". This answers a different question --
"does it look like somewhere" -- which needs shadows, image-based lighting and
tone mapping, and therefore the other renderer.

Poses are eye-height by default, because that is the only view the product is
actually judged from. An overview looks fine long after a street does not.
"""
from __future__ import annotations

import argparse
import http.server
import shutil
import socketserver
import threading
from pathlib import Path

#: (eye offset from centre, look-at offset from centre) in metres, z up in the
#: source frame -- converted to the viewer's y-up below. Eye z is above the
#: model's own floor, not above its centre.
VIEWS = {
    "street":   ((-55, -55, 1.7), (30, 30, 6)),
    "approach": ((0, -90, 1.7), (0, 40, 14)),
    "corner":   ((-28, 26, 1.7), (45, -25, 10)),
    "roofline": ((-70, 0, 38), (55, 0, 10)),
}

#: Elevation, bearing. Low sun is the honest test: it is what makes relief read
#: and what makes a flat facade obvious.
LIGHTS = {"morning": (18, 95), "noon": (68, 180), "evening": (12, 262)}


def serve(root: Path, port: int):
    handler = type("Q", (http.server.SimpleHTTPRequestHandler,), {
        "__init__": lambda self, *a, **k: http.server.SimpleHTTPRequestHandler.__init__(
            self, *a, directory=str(root), **k),
        "log_message": lambda *a: None,
    })
    server = socketserver.TCPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="compiled .gltf or .glb")
    ap.add_argument("--out", default="build/shots")
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--lights", default="evening")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--chrome", default="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # The page and the model have to be under one document root or the fetch is
    # cross-origin and the loader gets an opaque failure that looks like a
    # missing file.
    root = Path("viewer").resolve()
    staged = root / "three" / "_model"
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)
    model = Path(args.model)
    shutil.copy2(model, staged / model.name)
    for extra in (".bin",):                      # a .gltf keeps its buffer beside it
        beside = model.with_suffix(extra)
        if beside.exists():
            shutil.copy2(beside, staged / beside.name)
    if (model.parent / "tex").is_dir():
        shutil.copytree(model.parent / "tex", staged / "tex")

    server = serve(root, args.port)
    problems: list[str] = []
    try:
        with sync_playwright() as pw:
            launch = {"args": ["--use-gl=angle", "--use-angle=swiftshader",
                               "--enable-unsafe-swiftshader"]}
            if Path(args.chrome).exists():
                launch["executable_path"] = args.chrome
            browser = pw.chromium.launch(**launch)
            page = browser.new_page(viewport={"width": args.width, "height": args.height})
            page.on("console", lambda m: problems.append(f"console.{m.type}: {m.text}")
                    if m.type in ("error", "warning") else None)
            page.on("requestfailed", lambda r: problems.append(f"failed: {r.url}"))

            page.goto(f"http://127.0.0.1:{args.port}/three/index.html"
                      f"?model=./_model/{model.name}", wait_until="load")
            page.wait_for_function("window.walk && window.walk.ready", timeout=180_000)

            bounds = page.evaluate("() => window.walk.bounds")
            centre = page.evaluate("() => window.walk.centre")
            size = [bounds["max"][i] - bounds["min"][i] for i in range(3)]
            print(f"model {model.name}  extent "
                  f"{size[0]:.0f} x {size[1]:.0f} x {size[2]:.0f} m (y is up)")

            for light in args.lights.split(","):
                if light not in LIGHTS:
                    continue
                page.evaluate("([e, b]) => window.walk.setSun(e, b)", list(LIGHTS[light]))
                page.wait_for_timeout(400)
                for name in args.views.split(","):
                    if name not in VIEWS:
                        continue
                    eye, target = VIEWS[name]
                    # Source frame is z-up; the exporter already rotated the
                    # model, so a camera aimed in source coordinates ends up
                    # under the pavement looking at its underside.
                    page.evaluate("""([eye, target, centre, floor]) => {
                        window.walk.look(
                            [centre[0] + eye[0], floor + eye[2], centre[2] + eye[1]],
                            [centre[0] + target[0], floor + target[2], centre[2] + target[1]]);
                    }""", [list(eye), list(target), centre, bounds["min"][1]])
                    page.wait_for_timeout(900)
                    path = out / f"{light}-{name}.png"
                    page.screenshot(path=str(path))
                    print(f"  wrote {path}")
            browser.close()
    finally:
        server.shutdown()
        shutil.rmtree(staged, ignore_errors=True)

    if problems:
        print("\nproblems the render reported:")
        for line in dict.fromkeys(problems):
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
