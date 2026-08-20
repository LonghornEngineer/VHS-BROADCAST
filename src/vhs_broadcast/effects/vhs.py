"""Concrete VHS effects.

Convention: frames are BGR uint8, shape (H, W, 3), as produced by OpenCV.

Pipeline order: TextOverlay first (so OSD is degraded), then Signal, Slew,
Wobble, Tracking, Chroma, Static, Fuzz, Scanlines, Tearing. Motor (wow/flutter)
is a passthrough here -- it is consumed by the engine/renderer to warp time.
"""
from __future__ import annotations

import math
import time as _time

import cv2
import numpy as np

from ..core.parameters import ParameterSet
from .base import Effect
from .text_render import available_fonts, default_font, draw_texts


# --------------------------------------------------------------------------- #
#  Signal: brightness / contrast / saturation ("intensity")
# --------------------------------------------------------------------------- #
class SignalEffect(Effect):
    id = "signal"
    label = "SIGNAL"

    def build(self, params):
        self.param("brightness", "Brightness", 0.0, -1.0, 1.0)
        self.param("contrast", "Contrast", 1.0, 0.0, 2.0)
        self.param("saturation", "Saturation", 0.85, 0.0, 2.0)

    def process(self, frame, t, v):
        f = frame.astype(np.float32)
        f = (f - 128.0) * v["contrast"] + 128.0 + v["brightness"] * 128.0
        if abs(v["saturation"] - 1.0) > 1e-3:
            gray = (f @ np.array([0.114, 0.587, 0.299], np.float32))[..., None]
            f = gray + (f - gray) * v["saturation"]
        return np.clip(f, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
#  Slew / misalignment: horizontal luma smear + odd/even interlace offset
# --------------------------------------------------------------------------- #
class SlewEffect(Effect):
    id = "slew"
    label = "SLEW / ALIGN"

    def build(self, params):
        self.param("smear", "Smear", 0.25, 0.0, 1.0)
        self.param("misalign", "Misalign", 0.15, 0.0, 1.0)

    def process(self, frame, t, v):
        out = frame
        # horizontal rightward smear (amp not keeping up on transitions)
        L = int(v["smear"] * 30)
        if L >= 2:
            decay = 0.7
            k = decay ** np.arange(L, dtype=np.float32)
            k /= k.sum()
            kernel = k.reshape(1, L)
            anchor = (0, 0)  # causal -> trails to the right
            out = cv2.filter2D(out, -1, kernel, anchor=anchor,
                               borderType=cv2.BORDER_REPLICATE)
        # odd/even row misalignment (interlace comb)
        if v["misalign"] > 1e-3:
            w = out.shape[1]
            d = max(1, int(v["misalign"] * w * 0.04))
            out = out.copy()
            out[0::2] = np.roll(out[0::2], d, axis=1)
            out[1::2] = np.roll(out[1::2], -d, axis=1)
        return out


# --------------------------------------------------------------------------- #
#  Wobble: per-row horizontal sine displacement ("movement")
# --------------------------------------------------------------------------- #
class WobbleEffect(Effect):
    id = "wobble"
    label = "WOBBLE"

    def build(self, params):
        self.param("amount", "Amount", 0.25, 0.0, 1.0)
        self.param("freq", "Frequency", 3.0, 0.5, 20.0, step=0.1)
        self.param("speed", "Speed", 1.5, 0.0, 10.0, step=0.1)

    def process(self, frame, t, v):
        h, w = frame.shape[:2]
        amp = v["amount"] * w * 0.03
        if amp < 0.5:
            return frame
        rows = np.arange(h, dtype=np.float32)
        shift = amp * np.sin(rows / h * v["freq"] * 2 * math.pi + t * v["speed"])
        map_x = np.arange(w, dtype=np.float32)[None, :] + shift[:, None]
        map_y = np.repeat(rows[:, None], w, axis=1)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)


# --------------------------------------------------------------------------- #
#  Tracking: scrolling distortion band + vertical roll (signed + continuous)
# --------------------------------------------------------------------------- #
class TrackingEffect(Effect):
    id = "tracking"
    label = "TRACKING"

    def build(self, params):
        self.param("band", "Band Size", 0.15, 0.0, 0.5)
        self.param("jitter", "Jitter", 0.4, 0.0, 1.0)
        # V-Roll can go negative; modulate it to make the picture drift up/down
        self.param("roll", "V-Roll", 0.0, -1.0, 1.0)
        # continuous roll rate: quadratic response so small values stay gentle
        self.param("rollspeed", "Roll Rate", 0.0, -1.0, 1.0, step=0.01)
        self.param("speed", "Scroll", 0.3, 0.0, 3.0, step=0.05)

    def process(self, frame, t, v):
        h, w = frame.shape[:2]
        out = frame
        # signed static roll + gentle continuous wrapping roll
        rate = v["rollspeed"] * abs(v["rollspeed"]) * 0.6   # max ~0.6 screens/s
        offset = int(round((v["roll"] + rate * t) * h))
        if offset % h != 0:
            out = np.roll(out, offset, axis=0)
        # distortion band that scrolls up the frame
        band_h = int(v["band"] * h)
        if band_h > 1 and v["jitter"] > 1e-3:
            centre = int((1.0 - (t * v["speed"]) % 1.0) * h)
            y0 = max(0, centre - band_h // 2)
            y1 = min(h, centre + band_h // 2)
            if y1 > y0:
                n = y1 - y0
                prof = np.sin(np.linspace(0, math.pi, n))
                shift = (prof * v["jitter"] * w * 0.08).astype(np.int32)
                out = out.copy()
                band = out[y0:y1]
                for i in range(n):
                    out[y0 + i] = np.roll(band[i], shift[i], axis=0)
                edge = min(y1, y0 + max(1, n // 6))
                out[y0:edge] = np.clip(out[y0:edge].astype(np.float32)
                                       + 60 * v["jitter"], 0, 255).astype(np.uint8)
        return out


# --------------------------------------------------------------------------- #
#  Chroma shift: horizontal RGB channel separation / colour bleed
# --------------------------------------------------------------------------- #
class ChromaEffect(Effect):
    id = "chroma"
    label = "CHROMA"

    def build(self, params):
        # signed per-channel horizontal shift -> move just one colour if wanted
        self.param("red", "Red Shift", 0.4, -1.0, 1.0)
        self.param("green", "Grn Shift", 0.0, -1.0, 1.0)
        self.param("blue", "Blu Shift", -0.4, -1.0, 1.0)
        self.param("bleed", "Bleed", 0.3, 0.0, 1.0)

    def process(self, frame, t, v):
        w = frame.shape[1]
        b, g, r = cv2.split(frame)

        def sh(ch, val):
            px = int(val * w * 0.02)
            return np.roll(ch, px, axis=1) if px else ch

        r = sh(r, v["red"])
        g = sh(g, v["green"])
        b = sh(b, v["blue"])
        out = cv2.merge([b, g, r])
        if v["bleed"] > 1e-3:
            k = max(1, int(v["bleed"] * 15)) | 1
            blurred = cv2.GaussianBlur(out, (k, 1), 0)
            out = cv2.addWeighted(out, 1 - 0.5 * v["bleed"],
                                  blurred, 0.5 * v["bleed"], 0)
        return out


# --------------------------------------------------------------------------- #
#  Static: full-screen TV snow blended with alpha
# --------------------------------------------------------------------------- #
class StaticEffect(Effect):
    id = "static"
    label = "TV STATIC"

    def build(self, params):
        self.param("alpha", "Alpha", 0.15, 0.0, 1.0)
        self.param("coarse", "Coarse", 0.2, 0.0, 1.0)
        self.param("tint", "Colour", 0.0, 0.0, 1.0)

    def process(self, frame, t, v):
        if v["alpha"] < 1e-3:
            return frame
        h, w = frame.shape[:2]
        rng = np.random.default_rng(int(t * 1000) & 0xFFFFFFFF)
        k = 1 + int(v["coarse"] * 6)
        sh, sw = max(1, h // k), max(1, w // k)
        if v["tint"] > 1e-3:
            snow = rng.integers(0, 256, (sh, sw, 3), dtype=np.uint8)
            gray = snow.mean(axis=2, keepdims=True).astype(np.uint8)
            snow = (gray * (1 - v["tint"]) + snow * v["tint"]).astype(np.uint8)
        else:
            g = rng.integers(0, 256, (sh, sw, 1), dtype=np.uint8)
            snow = np.repeat(g, 3, axis=2)
        if (sh, sw) != (h, w):
            snow = cv2.resize(snow, (w, h), interpolation=cv2.INTER_NEAREST)
        a = float(v["alpha"])
        return cv2.addWeighted(frame, 1 - a, snow, a, 0)


# --------------------------------------------------------------------------- #
#  Ghost: RF multipath -- faint horizontally-offset copies of the picture
# --------------------------------------------------------------------------- #
class GhostEffect(Effect):
    id = "ghost"
    label = "GHOST"

    def build(self, params):
        self.param("amount", "Amount", 0.35, 0.0, 1.0)
        self.param("delay", "Delay", 0.3, 0.0, 1.0)

    def process(self, frame, t, v):
        if v["amount"] < 1e-3:
            return frame
        w = frame.shape[1]
        dx = int(v["delay"] * w * 0.12) + 2
        out = cv2.addWeighted(frame, 1.0, np.roll(frame, dx, axis=1),
                              v["amount"] * 0.6, 0)
        out = cv2.addWeighted(out, 1.0, np.roll(frame, dx * 2, axis=1),
                              v["amount"] * 0.25, 0)
        return out


# --------------------------------------------------------------------------- #
#  Head switch: torn, noisy band at the very bottom (VHS head-switching point)
# --------------------------------------------------------------------------- #
class HeadSwitchEffect(Effect):
    id = "headswitch"
    label = "HEAD SWITCH"

    def build(self, params):
        self.param("height", "Height", 0.05, 0.0, 0.2)
        self.param("noise", "Noise", 0.6, 0.0, 1.0)
        self.param("shift", "Shift", 0.4, 0.0, 1.0)

    def process(self, frame, t, v):
        h, w = frame.shape[:2]
        bh = int(v["height"] * h)
        if bh < 1:
            return frame
        out = frame.copy()
        y0 = h - bh
        band = np.roll(out[y0:h], int(v["shift"] * w * 0.3), axis=1)
        if v["noise"] > 1e-3:
            rng = np.random.default_rng(int(t * 1000) & 0xFFFFFFFF)
            snow = np.repeat(rng.integers(0, 255, (bh, w, 1), np.uint8), 3, axis=2)
            band = cv2.addWeighted(band, 1 - v["noise"], snow, v["noise"], 0)
        out[y0:h] = band
        return out


# --------------------------------------------------------------------------- #
#  Laser rot: LaserDisc bit-rot -- coloured sparkle, clustered in drifting patches
# --------------------------------------------------------------------------- #
class LaserRotEffect(Effect):
    id = "laserrot"
    label = "LASER ROT"

    def build(self, params):
        self.param("density", "Density", 0.35, 0.0, 1.0)
        self.param("coarse", "Size", 0.2, 0.0, 1.0)
        self.param("locality", "Locality", 0.5, 0.0, 1.0)   # patchy <-> full frame
        self.param("color", "Colour", 0.9, 0.0, 1.0)

    def process(self, frame, t, v):
        if v["density"] < 1e-3:
            return frame
        h, w = frame.shape[:2]
        rng = np.random.default_rng((int(t * 30) ^ 0x9E3779B9) & 0xFFFFFFFF)

        # coloured noise stretched horizontally so it follows the raster lines
        streak = 2 + int(v["coarse"] * 10)          # horizontal smear length
        nw = max(1, w // streak)
        cn = rng.integers(0, 256, (h, nw, 3), np.uint8)
        cn = cv2.resize(cn, (w, h), interpolation=cv2.INTER_LINEAR)
        cn = cv2.blur(cn, (streak, 1))              # smooth along the line
        if v["color"] < 0.999:                       # colour <-> mono sparkle
            gray = cn.mean(2, keepdims=True)
            cn = (cn * v["color"] + gray * (1 - v["color"])).astype(np.uint8)

        # drifting patch field (locality) -- smooth interp between two seeds
        ph, pw = max(2, h // 24), max(2, w // 24)
        i0 = int(t * 2)
        frac = float(t * 2 - i0)
        f0 = np.random.default_rng(i0 & 0xFFFFFFFF).random((ph, pw)).astype(np.float32)
        f1 = np.random.default_rng((i0 + 1) & 0xFFFFFFFF).random((ph, pw)).astype(np.float32)
        field = cv2.resize(f0 * (1 - frac) + f1 * frac, (w, h),
                           interpolation=cv2.INTER_LINEAR)
        dens_field = np.power(field, v["locality"] * 8.0)

        # sparkle alpha, also horizontally streaked, soft-thresholded
        amp = rng.random((h, nw)).astype(np.float32)
        amp = cv2.resize(amp, (w, h), interpolation=cv2.INTER_LINEAR)
        thresh = 1.0 - v["density"]
        spark = np.clip((amp - thresh) / max(1e-3, 1.0 - thresh), 0.0, 1.0)
        alpha = np.clip(spark * dens_field, 0.0, 1.0) * 0.9
        alpha = cv2.blur(alpha, (streak, 1))         # smear the blend along x
        a = alpha[..., None].astype(np.float32)

        f = frame.astype(np.float32)
        return (f * (1 - a) + cn.astype(np.float32) * a).astype(np.uint8)


# --------------------------------------------------------------------------- #
#  Fuzz: luminance grain + sparse dropout specks
# --------------------------------------------------------------------------- #
class NoiseEffect(Effect):
    id = "noise"
    label = "FUZZ"

    def build(self, params):
        self.param("grain", "Grain", 0.3, 0.0, 1.0)
        self.param("dropout", "Dropout", 0.1, 0.0, 1.0)

    def process(self, frame, t, v):
        h, w = frame.shape[:2]
        out = frame
        if v["grain"] > 1e-3:
            rng = np.random.default_rng(int(t * 1000) & 0xFFFFFFFF)
            noise = rng.normal(0, v["grain"] * 40, (h, w, 1)).astype(np.float32)
            out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if v["dropout"] > 1e-3:
            rng = np.random.default_rng((int(t * 1000) ^ 0x5bd1e995) & 0xFFFFFFFF)
            n = int(v["dropout"] * w * 0.5)
            if n > 0:
                ys = rng.integers(0, h, n)
                xs = rng.integers(0, w, n)
                ln = rng.integers(2, max(3, w // 40), n)
                out = out.copy()
                for y, x, L in zip(ys, xs, ln):
                    out[y, x:min(w, x + L)] = 235
        return out


# --------------------------------------------------------------------------- #
#  Scanlines + vignette (CRT feel)
# --------------------------------------------------------------------------- #
class ScanlinesEffect(Effect):
    id = "scanlines"
    label = "SCANLINES"

    def build(self, params):
        self.param("strength", "Strength", 0.3, 0.0, 1.0)
        self.param("vignette", "Vignette", 0.25, 0.0, 1.0)
        self._cache_key = None
        self._mask = None

    def _build_mask(self, h, w, strength, vig):
        lines = np.ones((h, 1, 1), np.float32)
        lines[::2] = 1.0 - 0.6 * strength
        mask = np.repeat(lines, w, axis=1)
        if vig > 1e-3:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            cx, cy = w / 2, h / 2
            d = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
            mask = mask * np.clip(1.0 - (d - 0.6) * vig, 0.0, 1.0)[..., None]
        return mask

    def process(self, frame, t, v):
        h, w = frame.shape[:2]
        key = (h, w, round(v["strength"], 3), round(v["vignette"], 3))
        if key != self._cache_key:
            self._mask = self._build_mask(h, w, v["strength"], v["vignette"])
            self._cache_key = key
        return np.clip(frame.astype(np.float32) * self._mask, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
#  Tearing: random horizontal slices shifted sideways ("glitch")
# --------------------------------------------------------------------------- #
class TearEffect(Effect):
    id = "tear"
    label = "TEARING"

    def build(self, params):
        self.param("amount", "Amount", 0.2, 0.0, 1.0)
        self.param("count", "Slices", 0.3, 0.0, 1.0)

    def process(self, frame, t, v):
        h, w = frame.shape[:2]
        n = int(v["count"] * 12)
        if n == 0 or v["amount"] < 1e-3:
            return frame
        rng = np.random.default_rng(int(t * 1000) & 0xFFFFFFFF)
        out = frame.copy()
        for _ in range(n):
            y0 = int(rng.integers(0, h))
            th = int(rng.integers(2, max(3, h // 12)))
            y1 = min(h, y0 + th)
            sh = int(rng.integers(-1, 2) * v["amount"] * w * 0.15)
            if sh:
                out[y0:y1] = np.roll(out[y0:y1], sh, axis=1)
        return out


# --------------------------------------------------------------------------- #
#  OSD / text overlay + clock/timer (PIL fonts, composited early)
# --------------------------------------------------------------------------- #
class TextOverlayEffect(Effect):
    id = "text"
    label = "OSD / TEXT"

    CLOCK_NONE = 0
    CLOCK_BLINK = 1        # unset-VCR blinking 12:00
    CLOCK_TIME = 2
    CLOCK_DATE = 3
    CLOCK_DATETIME = 4
    CLOCK_TIMER = 5

    def __init__(self, params):
        self.custom_text = "PLAY \u25b6"
        self.corner_text = "SP"
        self.clock_mode = self.CLOCK_BLINK
        self.font_path = default_font()
        super().__init__(params)
        self.enabled_param().set_base(1.0, notify=False)  # OSD on by default

    def build(self, params):
        self.param("scale", "Size", 1.0, 0.3, 3.0, step=0.05)
        self.param("x", "Pos X", 0.05, 0.0, 1.0)
        self.param("y", "Pos Y", 0.85, 0.0, 1.0)
        self.param("glow", "Glow", 0.4, 0.0, 1.0)

    def fonts(self):
        return available_fonts()

    def build_items(self, t, w, h):
        g = lambda k: self.params.get(f"text.{k}").evaluate(t)  # noqa: E731
        scale, x, y, glow = g("scale"), g("x"), g("y"), g("glow")
        base = int(48 * scale * (w / 720.0))
        items = []
        if self.custom_text:
            items.append(dict(text=self.custom_text, x=int(x * w) + 8,
                              y=int(y * h), size_px=base, glow=glow,
                              anchor="la"))
        clk = None
        show = True
        m = self.clock_mode
        if m == self.CLOCK_BLINK:
            clk = "12:00"
            show = (int(t * 1.5) % 2 == 0)      # blink ~0.75 Hz (deterministic)
        elif m == self.CLOCK_TIME:
            clk = _time.strftime("%H:%M:%S")
        elif m == self.CLOCK_DATE:
            clk = _time.strftime("%b %d %Y").upper()
        elif m == self.CLOCK_DATETIME:
            clk = _time.strftime("%m/%d %H:%M")
        elif m == self.CLOCK_TIMER:
            clk = f"{int(t // 60):02d}:{int(t % 60):02d}:{int((t * 100) % 100):02d}"
        if clk and show:
            items.append(dict(text=clk, x=int(w * 0.97), y=int(h * 0.06),
                              size_px=base, glow=glow, anchor="ra"))
        if self.corner_text:
            items.append(dict(text=self.corner_text, x=int(w * 0.03),
                              y=int(h * 0.06), size_px=int(base * 0.8),
                              glow=glow, anchor="la"))
        return items

    def process(self, frame, t, v):
        h, w = frame.shape[:2]
        return draw_texts(frame, self.build_items(t, w, h), self.font_path)


# --------------------------------------------------------------------------- #
#  Downsample: resolution crush (pixelate)
# --------------------------------------------------------------------------- #
class DownsampleEffect(Effect):
    id = "downsample"
    label = "DOWNSAMPLE"

    def build(self, params):
        self.param("amount", "Amount", 0.4, 0.0, 1.0)
        self.param("fps", "FPS %", 100.0, 5.0, 100.0, step=1.0)  # temporal hold

    def process(self, frame, t, v):
        # spatial pixelate only; the temporal FPS hold is applied by the engine
        # (so it works identically on the CPU and GPU paths)
        if v["amount"] < 1e-3:
            return frame
        h, w = frame.shape[:2]
        blk = 1 + int(v["amount"] * 15)
        if blk <= 1:
            return frame
        small = cv2.resize(frame, (max(1, w // blk), max(1, h // blk)),
                           interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


# --------------------------------------------------------------------------- #
#  Bit crush: colour-depth reduction (posterize)
# --------------------------------------------------------------------------- #
class BitCrushEffect(Effect):
    id = "bitcrush"
    label = "BIT CRUSH"

    def build(self, params):
        self.param("bits", "Bits", 5.0, 1.0, 8.0, step=1.0)

    def process(self, frame, t, v):
        bits = int(round(v["bits"]))
        if bits >= 8:
            return frame
        levels = max(2, 2 ** bits)
        q = 255.0 / (levels - 1)
        f = frame.astype(np.float32)
        return (np.round(f / q) * q).clip(0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
#  Interlace: field lines + comb offset (temporal fields)
# --------------------------------------------------------------------------- #
class InterlaceEffect(Effect):
    id = "interlace"
    label = "INTERLACE"

    def build(self, params):
        self.param("strength", "Line Dim", 0.4, 0.0, 1.0)
        self.param("offset", "Comb", 0.3, 0.0, 1.0)

    def process(self, frame, t, v):
        w = frame.shape[1]
        out = frame.copy()
        field = int(t * 50.0) % 2          # ~50 Hz field flicker
        if v["strength"] > 1e-3:
            out[1 - field::2] = (out[1 - field::2].astype(np.float32)
                                 * (1.0 - 0.5 * v["strength"])).astype(np.uint8)
        if v["offset"] > 1e-3:
            sh = int(v["offset"] * w * 0.03)
            if sh:
                out[field::2] = np.roll(out[field::2], sh, axis=1)
        return out


# --------------------------------------------------------------------------- #
#  Scramble: analog "premium channel" sync-suppression look
# --------------------------------------------------------------------------- #
class ScrambleEffect(Effect):
    id = "scramble"
    label = "SCRAMBLE"

    def build(self, params):
        self.param("amount", "Desync", 0.6, 0.0, 1.0)      # horizontal line skew
        self.param("wave", "Wave", 0.4, 0.0, 1.0)          # sine frequency
        self.param("roll", "V-Hold", 0.3, 0.0, 1.0)        # vertical roll/tear
        self.param("invert", "Invert", 0.35, 0.0, 1.0)     # inversion bands
        self.param("desat", "Colour Loss", 0.5, 0.0, 1.0)  # chroma corruption

    def process(self, frame, t, v):
        h, w = frame.shape[:2]
        rows = np.arange(h, dtype=np.float32)
        out = frame

        # 1) per-line horizontal desync: a sinusoid (sync-suppression) + jitter,
        #    lines wrap around so the picture tears diagonally
        if v["amount"] > 1e-3:
            freq = 1.0 + v["wave"] * 12.0
            shift = (v["amount"] * w * 0.28) * np.sin(
                rows / h * freq * 2 * math.pi + t * 2.0)
            rng = np.random.default_rng(int(t * 30) & 0xFFFFFFFF)
            shift += rng.normal(0.0, v["amount"] * w * 0.02, h).astype(np.float32)
            map_x = (np.arange(w, dtype=np.float32)[None, :] + shift[:, None])
            map_y = np.repeat(rows[:, None], w, axis=1)
            out = cv2.remap(out, map_x, map_y, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_WRAP)

        # 2) vertical roll / loss of V-hold (continuous, wraps)
        if v["roll"] > 1e-3:
            voff = int((t * v["roll"] * 1.7 * h) % h)
            if voff:
                out = np.roll(out, voff, axis=0)

        # 3) inversion bands scrolling down the frame (video inversion scramble)
        if v["invert"] > 1e-3:
            band = np.sin(rows / h * 3.0 * 2 * math.pi - t * 3.0) > (1.0 - 2.0 * v["invert"])
            if band.any():
                out = out.copy()
                out[band] = 255 - out[band]

        # 4) colour loss / corruption (scrambled chroma -> mostly B/W)
        if v["desat"] > 1e-3:
            gray = out.mean(axis=2, keepdims=True)
            out = (out * (1.0 - v["desat"]) + gray * v["desat"]).astype(np.uint8)

        return out.astype(np.uint8)


# --------------------------------------------------------------------------- #
#  Motor: wow/flutter time-warp. Passthrough in the pixel pipeline; the engine
#  and renderer read its parameters to warp playback speed (and audio pitch).
# --------------------------------------------------------------------------- #
class MotorEffect(Effect):
    id = "motor"
    label = "MOTOR  WOW/FLUTTER"

    def build(self, params):
        self.param("depth", "Depth", 0.0, 0.0, 0.5)
        self.param("rate", "Rate", 0.6, 0.05, 5.0, step=0.01)
        # boolean param -> renders as a checkbox in the UI
        self.param("audio", "Affect Audio", 1.0, 0.0, 1.0, step=1.0)

    def process(self, frame, t, v):
        return frame  # time-warp is applied outside the pixel pipeline


def default_effects(params: ParameterSet) -> list[Effect]:
    return [
        TextOverlayEffect(params),
        SignalEffect(params),
        SlewEffect(params),
        WobbleEffect(params),
        TrackingEffect(params),
        ChromaEffect(params),
        GhostEffect(params),
        StaticEffect(params),
        LaserRotEffect(params),
        HeadSwitchEffect(params),
        NoiseEffect(params),
        ScanlinesEffect(params),
        DownsampleEffect(params),
        BitCrushEffect(params),
        InterlaceEffect(params),
        TearEffect(params),
        ScrambleEffect(params),
        MotorEffect(params),
    ]
