"""OSD text rendering via Pillow (proper TTF fonts, not OpenCV Hershey).

Fonts are packaged under assets/fonts/ (all SIL Open Font License). Text is
drawn to an RGBA layer and alpha-composited onto the BGR frame, so it can be
positioned, glowed, and — because it's baked into the frame before the analogue
stage — degraded by the VHS effects.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

_FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"


def _pretty(stem: str) -> str:
    name = stem.replace("-Regular", "").replace("_", " ").replace("-", " ")
    name = re.sub(r"\s+v?\d[\d.]*\s*$", "", name)   # drop trailing version
    return " ".join(name.split()).strip()


def available_fonts() -> dict[str, str]:
    """Map display name -> font file path for every packaged .ttf."""
    fonts: dict[str, str] = {}
    if _FONT_DIR.is_dir():
        for p in sorted(_FONT_DIR.glob("*.ttf")):
            fonts[_pretty(p.stem)] = str(p)
    return fonts


def default_font() -> str:
    fonts = available_fonts()
    # prefer the authentic VCR OSD face if the user fetched it
    for pref in ("VCR", "VT323", "Share Tech Mono", "PressStart2P"):
        for name, path in fonts.items():
            if pref.lower() in name.lower():
                return path
    return next(iter(fonts.values())) if fonts else ""


def resolve_font(path: str) -> str:
    """A usable font path for `path`, which may come from another computer's take.

    Takes store absolute font paths, so a take moved between machines points at
    a file that isn't there. Match the bundled font by file name, else use the
    default bundled font, so the OSD keeps its face and size.
    """
    if path and Path(path).is_file():
        return path
    name = Path(path.replace("\\", "/")).name.lower() if path else ""
    for candidate in _FONT_DIR.glob("*.ttf"):
        if candidate.name.lower() == name:
            return str(candidate)
    return default_font()


@lru_cache(maxsize=64)
def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(resolve_font(path), size)
    except Exception:
        # Last resort only: Pillow's bitmap font ignores the requested size.
        return ImageFont.load_default()


# Transport glyphs the retro fonts lack -> drawn as shapes instead. Keys are the
# unicode symbols (plus a few ASCII aliases) that may appear in OSD text.
_ICONS = {
    "\u25b6": "play", "\u23f5": "play",           # ▶
    "\u25c0": "rew1", "\u23f4": "rew1",            # ◀
    "\u25a0": "stop", "\u23f9": "stop",            # ■
    "\u25cf": "rec", "\u23fa": "rec",              # ●
    "\u23f8": "pause",                              # ⏸
    "\u23e9": "ffwd", "\u23ea": "rewd",             # ⏩ ⏪
    "\u23ed": "next", "\u23ee": "prev",             # ⏭ ⏮
    "\u23cf": "eject",                              # ⏏
    "\u25b2": "up", "\u25bc": "down",               # ▲ ▼
}


def _segments(text: str):
    segs, buf = [], ""
    for ch in text:
        if ch in _ICONS:
            if buf:
                segs.append(("t", buf)); buf = ""
            segs.append(("i", _ICONS[ch]))
        else:
            buf += ch
    if buf:
        segs.append(("t", buf))
    return segs


def _draw_icon(draw, kind, box, fill):
    x0, y0, x1, y1 = box
    my = (y0 + y1) / 2
    if kind == "play":
        draw.polygon([(x0, y0), (x0, y1), (x1, my)], fill=fill)
    elif kind == "rew1":
        draw.polygon([(x1, y0), (x1, y1), (x0, my)], fill=fill)
    elif kind == "stop":
        draw.rectangle([x0, y0, x1, y1], fill=fill)
    elif kind == "rec":
        draw.ellipse([x0, y0, x1, y1], fill=fill)
    elif kind == "pause":
        w = (x1 - x0)
        draw.rectangle([x0, y0, x0 + w * 0.32, y1], fill=fill)
        draw.rectangle([x1 - w * 0.32, y0, x1, y1], fill=fill)
    elif kind == "ffwd":
        mx = (x0 + x1) / 2
        draw.polygon([(x0, y0), (x0, y1), (mx, my)], fill=fill)
        draw.polygon([(mx, y0), (mx, y1), (x1, my)], fill=fill)
    elif kind == "rewd":
        mx = (x0 + x1) / 2
        draw.polygon([(mx, y0), (mx, y1), (x0, my)], fill=fill)
        draw.polygon([(x1, y0), (x1, y1), (mx, my)], fill=fill)
    elif kind == "up":
        draw.polygon([(x0, y1), (x1, y1), ((x0 + x1) / 2, y0)], fill=fill)
    elif kind == "down":
        draw.polygon([(x0, y0), (x1, y0), ((x0 + x1) / 2, y1)], fill=fill)
    elif kind == "eject":
        w = x1 - x0
        bar = (y1 - y0) * 0.16
        draw.polygon([(x0, y1 - bar * 1.8), (x1, y1 - bar * 1.8),
                      ((x0 + x1) / 2, y0)], fill=fill)
        draw.rectangle([x0, y1 - bar, x1, y1], fill=fill)
    elif kind == "next":
        w = x1 - x0
        xe = x0 + w * 0.78
        m2 = (x0 + xe) / 2
        draw.polygon([(x0, y0), (x0, y1), (m2, my)], fill=fill)
        draw.polygon([(m2, y0), (m2, y1), (xe, my)], fill=fill)
        draw.rectangle([x1 - w * 0.14, y0, x1, y1], fill=fill)
    elif kind == "prev":
        w = x1 - x0
        xs = x0 + w * 0.22
        m2 = (xs + x1) / 2
        draw.rectangle([x0, y0, x0 + w * 0.14, y1], fill=fill)
        draw.polygon([(m2, y0), (m2, y1), (xs, my)], fill=fill)
        draw.polygon([(x1, y0), (x1, y1), (m2, my)], fill=fill)


def _measure(segs, font, icon_w, gap):
    w = 0.0
    for kind, val in segs:
        if kind == "t":
            try:
                w += font.getlength(val)
            except Exception:
                w += font.getsize(val)[0]
        else:
            w += icon_w + gap
    return w


def _render_line(draw, segs, x0, y, font, fill, icon_w, icon_h, ascent, gap):
    x = x0
    iy0 = y + (ascent - icon_h) / 2
    for kind, val in segs:
        if kind == "t":
            draw.text((x, y), val, font=font, fill=fill, anchor="la")
            try:
                x += font.getlength(val)
            except Exception:
                x += font.getsize(val)[0]
        else:
            _draw_icon(draw, val, (x, iy0, x + icon_w, iy0 + icon_h), fill)
            x += icon_w + gap


@lru_cache(maxsize=128)
def _render_tile(text: str, size: int, glow_q: float, color: tuple,
                 font_path: str):
    """Render one text string to an RGBA tile (cached). Returns
    (rgba_uint8_array, total_width, margin). Position is applied by the caller,
    so a static label/tag is rendered once and reused every frame."""
    font = _load_font(font_path, size)
    ascent, desc = font.getmetrics()
    icon_h = ascent * 0.72
    icon_w = icon_h
    gap = size * 0.12
    segs = _segments(text)
    total = _measure(segs, font, icon_w, gap)
    if total <= 0:
        return None, 0, 0
    margin = int(4 + 6 * glow_q)
    tw = int(total) + 2 * margin
    th = int(ascent + desc) + 2 * margin
    tile = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    if glow_q > 1e-3:
        gl = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        gd = ImageDraw.Draw(gl)
        _render_line(gd, segs, margin, margin, font, (0, 0, 0, 255),
                     icon_w, icon_h, ascent, gap)
        gl = gl.filter(ImageFilter.GaussianBlur(radius=2 + 4 * glow_q))
        tile.alpha_composite(gl)
    _render_line(d, segs, margin, margin, font, (*color, 255),
                 icon_w, icon_h, ascent, gap)
    return np.asarray(tile), int(total), margin


def draw_texts(frame_bgr: np.ndarray, items, font_path: str) -> np.ndarray:
    """Composite text items onto a BGR frame. Each item's rendered tile is
    cached, so per-frame cost is just small-region alpha compositing (cost
    scales with the text size, not the frame size)."""
    if not items or not font_path:
        return frame_bgr
    H, W = frame_bgr.shape[:2]
    out = frame_bgr
    made_copy = False

    for it in items:
        text = it.get("text", "")
        if not text:
            continue
        size = max(6, int(it.get("size_px", 24)))
        color = tuple(it.get("color", (235, 235, 235)))
        glow_q = round(float(it.get("glow", 0.0)), 2)
        arr, total, margin = _render_tile(text, size, glow_q, color, font_path)
        if arr is None or total <= 0:
            continue
        th, tw = arr.shape[:2]
        x = int(it["x"]); y = int(it["y"])
        x0 = x - total - margin if it.get("anchor") == "ra" else x - margin
        y0 = y - margin

        fx0, fy0 = max(0, x0), max(0, y0)
        fx1, fy1 = min(W, x0 + tw), min(H, y0 + th)
        if fx1 <= fx0 or fy1 <= fy0:
            continue
        sub = arr[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
        a = sub[..., 3:4].astype(np.float32) / 255.0
        if float(a.max()) <= 0.0:
            continue
        if not made_copy:
            out = frame_bgr.copy()
            made_copy = True
        rgb = np.ascontiguousarray(sub[..., :3][..., ::-1]).astype(np.float32)
        region = out[fy0:fy1, fx0:fx1].astype(np.float32)
        out[fy0:fy1, fx0:fx1] = (region * (1 - a) + rgb * a).astype(np.uint8)

    return out
