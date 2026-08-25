"""Screenshot the viewer, so the world can be looked at instead of guessed at.

Every metric this project reports -- tiles, patches, residuals -- can look fine
while the world looks like nothing. Rendering it is the only check that catches
a wall in the wrong place, a camera in a gap, or a theme that resolves to grey.

    python tools/shoot.py --out build/shots --theme victorian

Runs headless Chromium against a local server, drives the viewer through its own
debug handle (`window.lidarworld`), and writes one PNG per camera pose. Console
errors and failed requests are reported: a black screenshot with a shader error
in the log is a different problem from a black screenshot without one.
"""
from __future__ import annotations

import argparse
import http.server
import socketserver
import threading
from pathlib import Path

# Eye height is measured from the ground under the camera, not from the world's
# minimum z. A 320 m block spans 78 m of relief, so anchoring to the bounds puts
# a "street level" camera tens of metres underground and half the frame is void.
VIEWS = {
    # name:      (eye offset from centre, look-at offset), metres. z is above
    #            local terrain for the camera, above local terrain for the target.
    "street":    ((-60, -60, 1.7), (40, 40, 8)),
    "corner":    ((-30, 30, 1.7), (50, -30, 12)),
    "overview":  ((-140, -140, 95), (0, 0, 0)),
    "roofline":  ((-70, 0, 40), (60, 0, 12)),
}


def serve(root: Path, port: int):
    handler = type("Q", (http.server.SimpleHTTPRequestHandler,), {
        "directory_": str(root),
        "__init__": lambda self, *a, **k: http.server.SimpleHTTPRequestHandler.__init__(
            self, *a, directory=str(root), **k),
        "log_message": lambda *a: None,
    })
    server = socketserver.TCPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", default="viewer")
    ap.add_argument("--out", default="build/shots")
    ap.add_argument("--theme", action="append", default=None)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--pose", action="append", default=None,
                    help="an explicit pose in BUNDLE coordinates: "
                         "name=ex,ey,ez,tx,ty,tz. The fixed VIEWS are offsets "
                         "from the block centre, which aims a 'street' camera "
                         "at whatever happens to be there -- in a canal block "
                         "that is usually a facade 2 m away. Use this to stand "
                         "somewhere the world actually is. Note the bundle "
                         "frame is centred on the block, not the seed's local "
                         "0,0 corner -- read window.lidarworld.world.bounds "
                         "before writing a pose, or you will stand outside the "
                         "city looking at the underside of it.")
    ap.add_argument("--chrome", default="/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
                    help="pre-installed browser; the pip package's pinned build "
                         "may not match what this container ships")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    server = serve(Path(args.viewer), args.port)
    problems: list[str] = []

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
        page.goto(f"http://127.0.0.1:{args.port}/", wait_until="networkidle")

        page.wait_for_function("window.lidarworld !== undefined", timeout=90_000)
        info = page.evaluate("""() => {
            const w = window.lidarworld.world;
            return {themes: w.header.themes, nodes: w.header.summary.nodes,
                    triangles: w.indices.length / 3, bounds: w.bounds,
                    points: w.points ? w.points.count : 0};
        }""")
        print(f"themes {info['themes']}  nodes {info['nodes']}  "
              f"triangles {info['triangles']:,.0f}  points {info['points']:,}")
        lo, hi = info["bounds"]
        centre = [(lo[i] + hi[i]) / 2 for i in range(3)]
        print(f"extent {hi[0]-lo[0]:.0f} x {hi[1]-lo[1]:.0f} x {hi[2]-lo[2]:.0f} m")

        themes = args.theme or info["themes"]
        for theme in themes:
            page.evaluate("(t) => window.lidarworld.switchTheme(t)", theme)
            page.wait_for_timeout(1200)
            poses = [(name, VIEWS[name], centre) for name in args.views.split(",")
                     if name in VIEWS]
            for spec in args.pose or []:
                name, _, numbers = spec.partition("=")
                values = [float(v) for v in numbers.split(",")]
                if len(values) != 6:
                    raise SystemExit(f"--pose {spec!r}: need name=ex,ey,ez,tx,ty,tz")
                # Absolute poses are offsets from a zero centre.
                poses.append((name, (tuple(values[:3]), tuple(values[3:])), [0, 0, 0]))

            for name, (eye, target), origin in poses:
                # The camera exposes yaw/pitch, not a look-at, so aim it the
                # same way it aims itself: forward = (cos yaw, sin yaw).
                page.evaluate("""([eye, target, centre]) => {
                    const w = window.lidarworld.world;
                    const c = window.lidarworld.camera;
                    const ground = (x, y) => w.heightfield.sample(x, y);
                    c.fly = true;
                    const ex = centre[0] + eye[0], ey = centre[1] + eye[1];
                    const tx = centre[0] + target[0], ty = centre[1] + target[1];
                    c.position[0] = ex;
                    c.position[1] = ey;
                    c.position[2] = ground(ex, ey) + eye[2];
                    const dx = tx - c.position[0];
                    const dy = ty - c.position[1];
                    const dz = (ground(tx, ty) + target[2]) - c.position[2];
                    c.yaw = Math.atan2(dy, dx);
                    c.pitch = Math.atan2(dz, Math.hypot(dx, dy));
                }""", [list(eye), list(target), origin])
                page.wait_for_timeout(700)
                path = out / f"{theme}-{name}.png"
                page.screenshot(path=str(path))
                print(f"  {path}")
        browser.close()
    server.shutdown()

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in dict.fromkeys(problems):
            print(f"  {p[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
