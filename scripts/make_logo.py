"""Draw the Parallaxa logo: two offset discs (left eye red, right eye cyan) whose overlap is bright.

Writes parallaxa/assets/logo.svg, logo-512.png and parallaxa.ico (16-256 px).
Run:  python scripts/make_logo.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "parallaxa" / "assets"

SIZE = 512
RADIUS_TILE = 112
BG_TOP, BG_BOTTOM = (22, 28, 48), (38, 50, 86)
LEFT, RIGHT, BOTH = (255, 84, 92), (38, 200, 222), (246, 243, 236)
DISC_R = 128
OFFSET = 46  # half the horizontal separation between the two discs
CY = 256


def draw(scale: int = 4) -> Image.Image:
    s = SIZE * scale
    # Background tile: vertical gradient inside a rounded square.
    grad = Image.new("RGB", (1, s))
    for y in range(s):
        t = y / (s - 1)
        grad.putpixel((0, y), tuple(int(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOTTOM)))
    grad = grad.resize((s, s))
    tile = Image.new("L", (s, s), 0)
    ImageDraw.Draw(tile).rounded_rectangle((0, 0, s - 1, s - 1), RADIUS_TILE * scale, fill=255)
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    img.paste(grad, mask=tile)

    def disc(cx: float) -> Image.Image:
        m = Image.new("L", (s, s), 0)
        r = DISC_R * scale
        ImageDraw.Draw(m).ellipse((cx * scale - r, CY * scale - r, cx * scale + r, CY * scale + r), fill=255)
        return m

    left, right = disc(256 - OFFSET), disc(256 + OFFSET)
    both = ImageChops.darker(left, right)  # pixels inside both discs
    img.paste(LEFT, mask=left)
    img.paste(RIGHT, mask=right)
    img.paste(BOTH, mask=both)
    return img.resize((SIZE, SIZE), Image.LANCZOS)


def svg() -> str:
    lx, rx = 256 - OFFSET, 256 + OFFSET
    hexc = lambda c: "#%02x%02x%02x" % c  # noqa: E731
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}" width="{SIZE}" height="{SIZE}">
  <title>Parallaxa</title>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="{hexc(BG_TOP)}"/>
      <stop offset="1" stop-color="{hexc(BG_BOTTOM)}"/>
    </linearGradient>
    <clipPath id="left"><circle cx="{lx}" cy="{CY}" r="{DISC_R}"/></clipPath>
  </defs>
  <rect width="{SIZE}" height="{SIZE}" rx="{RADIUS_TILE}" fill="url(#bg)"/>
  <circle cx="{lx}" cy="{CY}" r="{DISC_R}" fill="{hexc(LEFT)}"/>
  <circle cx="{rx}" cy="{CY}" r="{DISC_R}" fill="{hexc(RIGHT)}"/>
  <circle cx="{rx}" cy="{CY}" r="{DISC_R}" fill="{hexc(BOTH)}" clip-path="url(#left)"/>
</svg>
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    img = draw()
    img.save(OUT / "logo-512.png")
    img.save(OUT / "parallaxa.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    (OUT / "logo.svg").write_text(svg(), encoding="utf-8")
    print("wrote", ", ".join(p.name for p in sorted(OUT.iterdir())))


if __name__ == "__main__":
    main()
