"""Render a time-driven page (window.renderAt(t), window.DURATION) to MP4, 1920x1080, 30 fps.

Drives your installed Chrome headlessly (no browser download), renders each frame, screenshots it,
and pipes the frames into ffmpeg (bundled by imageio-ffmpeg).

Run:  uv run --no-project --with playwright --with imageio-ffmpeg video/export.py            # results video
      uv run --no-project --with playwright video/export.py --page demo-animation.html --stills 5 12
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
FPS = 30


def _open(p, uri: str):
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
    page.goto(uri)
    page.wait_for_function("typeof window.renderAt === 'function'")
    return browser, page


def save_stills(uri: str, times: list[float], prefix: str = "still") -> None:
    with sync_playwright() as p:
        browser, page = _open(p, uri)
        for t in times:
            page.evaluate(f"window.renderAt({t})")
            out = HERE / f"{prefix}-{t:05.1f}.png"
            page.screenshot(path=str(out))
            print(out)
        browser.close()


def render_mp4(uri: str, out: Path | str) -> None:
    import imageio_ffmpeg

    with sync_playwright() as p:
        browser, page = _open(p, uri)
        duration = page.evaluate("window.DURATION")
        proc = subprocess.Popen(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "image2pipe",
             "-framerate", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "18", "-preset", "medium", str(out)],
            stdin=subprocess.PIPE,
        )
        n = int(duration * FPS)
        for i in range(n):
            page.evaluate(f"window.renderAt({i / FPS})")
            proc.stdin.write(page.screenshot(type="png"))
            if i % (FPS * 5) == 0:
                print(f"  {i / FPS:4.0f}s / {duration}s", flush=True)
        proc.stdin.close()
        proc.wait()
        browser.close()
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", default="results-animation.html")
    ap.add_argument("--stills", nargs="*", type=float, help="only save PNGs at these seconds")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    uri = (HERE / a.page).resolve().as_uri()
    if a.stills is not None:
        save_stills(uri, a.stills, prefix=Path(a.page).stem)
    else:
        render_mp4(uri, a.out or HERE / (Path(a.page).stem + ".mp4"))


if __name__ == "__main__":
    main()
