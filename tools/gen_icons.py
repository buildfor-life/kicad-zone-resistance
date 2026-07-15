"""Generate the plugin icons: a bold copper Ω filling the canvas.

    python tools/gen_icons.py

Writes icons/fill_res_24_light.png (dark copper, for light toolbars),
icons/fill_res_24_dark.png (bright copper, for dark toolbars) and
resources/icon.png (64 px, for the PCM package listing). Rendered
supersampled with matplotlib's bundled DejaVu Sans Bold, then
downsampled for crisp antialiasing.
"""
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

SS = 20                                  # supersampling factor
MARGIN = 1                               # px kept clear per 24 px

COLORS = {
    "light": (158, 88, 28),              # dark copper on light toolbar
    "dark": (232, 162, 94),              # bright copper on dark toolbar
}


def render(color: tuple[int, int, int], size: int) -> Image.Image:
    canvas = size * SS
    img = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font_path = (Path(matplotlib.get_data_path())
                 / "fonts" / "ttf" / "DejaVuSans-Bold.ttf")
    # size the glyph so its ink box fills the canvas minus the margin
    target = canvas - 2 * MARGIN * SS * size // 24
    font = ImageFont.truetype(str(font_path), target)
    x0, y0, x1, y1 = font.getbbox("Ω")
    scale = min(target / (x1 - x0), target / (y1 - y0))
    font = ImageFont.truetype(str(font_path), int(target * scale))
    x0, y0, x1, y1 = font.getbbox("Ω")
    pos = ((canvas - (x1 - x0)) // 2 - x0, (canvas - (y1 - y0)) // 2 - y0)
    draw.text(pos, "Ω", font=font, fill=color + (255,))
    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    for theme, color in COLORS.items():
        p = root / "icons" / f"fill_res_24_{theme}.png"
        render(color, 24).save(p)
        print(f"wrote {p}")
    p = root / "resources" / "icon.png"
    p.parent.mkdir(exist_ok=True)
    render(COLORS["light"], 64).save(p)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
