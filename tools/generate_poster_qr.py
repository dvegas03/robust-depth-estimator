"""
Generate a high-resolution QR code with an embedded logo for posters.

Usage (from engine_grounder/):
    pip install 'qrcode[pil]'
    python generate_poster_qr.py

Logos: Prefer PNG with **alpha** (transparent background). If the image is
opaque, the dominant color at the **four corners** is treated as background
and made transparent so no solid square is pasted onto the QR (works well for
black logos on white; for black-on-black assets, use a PNG with transparency).

If ``github_logo.png`` is missing, a minimal transparent placeholder is created
(no filled background square).
"""

from __future__ import annotations

import os

import qrcode
from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://github.com/dvegas03/robust-depth-estimator"
LOGO_PATH = os.path.join(_REPO_ROOT, "assets", "github_logo.png")
OUTPUT_PATH = os.path.join(_REPO_ROOT, "assets", "poster_qr_code.png")

# Pixels within this distance from averaged corner color become transparent
_CORNER_BG_TOLERANCE = 38


def _make_fallback_logo(path: str, size: int = 256) -> None:
    """Transparent background, black monogram only — no filled square."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", size=size // 3
        )
    except OSError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=size // 3)
        except OSError:
            font = ImageFont.load_default()
    text = "GH"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        ((size - tw) // 2, (size - th) // 2 - size // 32),
        text,
        fill=(0, 0, 0, 255),
        font=font,
    )
    img.save(path)
    print(f"  Created fallback logo → {path}")


def _prepare_logo(img: Image.Image) -> Image.Image:
    """
    Ensure RGBA. Use existing alpha if present; otherwise remove uniform
    background matching the average of the four corner colors.
    """
    rgba = img.convert("RGBA")
    w, h = rgba.size
    if w < 2 or h < 2:
        return rgba

    alpha = rgba.split()[3]
    a_min, a_max = alpha.getextrema()
    if a_max < 255 or a_min < 250:
        return rgba

    corners = [
        rgba.getpixel((0, 0))[:3],
        rgba.getpixel((w - 1, 0))[:3],
        rgba.getpixel((0, h - 1))[:3],
        rgba.getpixel((w - 1, h - 1))[:3],
    ]
    bg_r = sum(c[0] for c in corners) // 4
    bg_g = sum(c[1] for c in corners) // 4
    bg_b = sum(c[2] for c in corners) // 4

    out = Image.new("RGBA", rgba.size)
    px = rgba.load()
    po = out.load()
    t = _CORNER_BG_TOLERANCE
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if (
                abs(r - bg_r) <= t
                and abs(g - bg_g) <= t
                and abs(b - bg_b) <= t
            ):
                po[x, y] = (0, 0, 0, 0)
            else:
                po[x, y] = (r, g, b, a)
    return out


def main() -> None:
    if not os.path.isfile(LOGO_PATH):
        print(f"No '{os.path.basename(LOGO_PATH)}' found — generating placeholder.")
        _make_fallback_logo(LOGO_PATH)

    qr = qrcode.QRCode(
        version=5,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=15,
        border=4,
    )
    qr.add_data(URL)
    qr.make(fit=True)

    qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")

    logo = _prepare_logo(Image.open(LOGO_PATH))
    qr_width, qr_height = qr_img.size
    logo_max_size = (int(qr_width * 0.25), int(qr_height * 0.25))
    logo.thumbnail(logo_max_size, Image.Resampling.LANCZOS)

    logo_width, logo_height = logo.size
    center_pos = (
        (qr_width - logo_width) // 2,
        (qr_height - logo_height) // 2,
    )

    qr_img = qr_img.convert("RGBA")
    qr_img.paste(logo, center_pos, logo)
    qr_img.convert("RGB").save(OUTPUT_PATH)

    print(f"Success! High-res QR code saved as '{OUTPUT_PATH}'")
    print(f"  URL: {URL}")


if __name__ == "__main__":
    main()
