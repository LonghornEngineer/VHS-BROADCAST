"""Concrete VHS effects.

Convention: frames are BGR uint8, shape (H, W, 3), as produced by OpenCV.

Pipeline order: TextOverlay first (so OSD is degraded), then Signal, Slew,
Wobble, Tracking, Chroma, Static, Fuzz, Scanlines, Tearing. Motor (wow/flutter)
is a passthrough here -- it is consumed by the engine/renderer to warp time.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..core.parameters import ParameterSet
from .base import Effect
from .text_render import available_fonts, default_font, draw_texts
from .advanced import ADVANCED_EFFECTS


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
        phase = self.params.motion_phase("wobble.speed", t)
        amp = v["amount"] * w * 0.03
        if amp < 0.5:
            return frame
        rows = np.arange(h, dtype=np.float32)
        shift = amp * np.sin(rows / h * v["freq"] * 2 * math.pi + phase)
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
        # A direct offset is separate from the accumulated roll speed.
        self.param("roll", "V-Offset", 0.0, -1.0, 1.0)
        # continuous roll rate: quadratic response so small values stay gentle
        self.param("rollspeed", "Roll Rate", 0.0, -1.0, 1.0, step=0.01)
        self.param("speed", "Scroll", 0.3, 0.0, 3.0, step=0.05)

    def process(self, frame, t, v):
        h, w = frame.shape[:2]
        out = frame
        # signed static roll + gentle continuous wrapping roll
        travel = self.params.motion_phase("tracking.rollspeed", t, quadratic=True) * 0.6
        scroll = self.params.motion_phase("tracking.speed", t)
        offset = int(round((v["roll"] + travel) * h))
        if offset % h != 0:
            out = np.roll(out, offset, axis=0)
        # distortion band that scrolls up the frame
        band_h = int(v["band"] * h)
        if band_h > 1 and v["jitter"] > 1e-3:
            centre = int((1.0 - scroll % 1.0) * h)
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
        self.param("thickness", "Thickness", 0.2, 0.0, 1.0)  # vertical height of rot
        self.param("locality", "Locality", 0.5, 0.0, 1.0)   # patchy <-> full frame
        self.param("color", "Colour", 0.9, 0.0, 1.0)

    def process(self, frame, t, v):
        if v["density"] < 1e-3:
            return frame
        h, w = frame.shape[:2]
        rng = np.random.default_rng((int(t * 30) ^ 0x9E3779B9) & 0xFFFFFFFF)

        # coloured noise stretched horizontally so it follows the raster lines,
        # and blocked vertically so the rot lines can gain height ("chonk")
        streak = 2 + int(v["coarse"] * 10)          # horizontal smear length
        vthick = 1 + int(v["thickness"] * 10)       # vertical line height
        nw = max(1, w // streak)
        nh = max(1, h // vthick)
        cn = rng.integers(0, 256, (nh, nw, 3), np.uint8)
        cn = cv2.resize(cn, (w, h), interpolation=cv2.INTER_LINEAR)
        cn = cv2.blur(cn, (streak, vthick))         # smooth along the line + height
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

        # sparkle alpha, streaked horizontally and blocked vertically, soft-thresholded
        amp = rng.random((nh, nw)).astype(np.float32)
        amp = cv2.resize(amp, (w, h), interpolation=cv2.INTER_LINEAR)
        thresh = 1.0 - v["density"]
        spark = np.clip((amp - thresh) / max(1e-3, 1.0 - thresh), 0.0, 1.0)
        alpha = np.clip(spark * dens_field, 0.0, 1.0) * 0.9
        alpha = cv2.blur(alpha, (streak, vthick))    # smear the blend along x + height
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

    # Settings a text key may change. A key holds until the next key (like HOLD
    # interpolation); fields it omits keep their earlier value.
    KEYED_FIELDS = ("custom_text", "corner_text", "clock_mode", "manual_datetime")

    def __init__(self, params):
        self.custom_text = "PLAY \u25b6"
        self.corner_text = "SP"
        self.clock_mode = self.CLOCK_BLINK
        self.font_path = default_font()
        self.manual_clock = False
        self.manual_datetime = "1994-01-01T12:00:00"
        self.clock_run = True
        self.text_keys = []           # [{"t": seconds, <KEYED_FIELDS subset>}], sorted by t
        super().__init__(params)
        self.enabled_param().set_base(1.0, notify=False)  # OSD on by default
        self.enabled_param().default = 1.0

    def build(self, params):
        self.param("scale", "Size", 1.0, 0.3, 3.0, step=0.05)
        self.param("x", "Pos X", 0.05, 0.0, 1.0)
        self.param("y", "Pos Y", 0.85, 0.0, 1.0)
        self.param("glow", "Glow", 0.4, 0.0, 1.0)
        # Clock is right-anchored, tag left-anchored; defaults match the original fixed layout.
        self.param("clock_x", "Clock X", 0.97, 0.0, 1.0)
        self.param("clock_y", "Clock Y", 0.06, 0.0, 1.0)
        self.param("tag_x", "Tag X", 0.03, 0.0, 1.0)
        self.param("tag_y", "Tag Y", 0.06, 0.0, 1.0)

    def fonts(self):
        return available_fonts()

    def settings(self):
        data = {k: getattr(self, k) for k in (
            "custom_text", "corner_text", "clock_mode", "font_path",
            "manual_clock", "manual_datetime", "clock_run")}
        data["text_keys"] = [dict(key) for key in self.text_keys]
        return data

    @classmethod
    def normalize_text_keys(cls, keys):
        """Validate text keys and return them sorted; a later key at the same time wins."""
        from datetime import datetime
        if not isinstance(keys, list):
            raise ValueError("text.text_keys must be a list")
        by_time = {}
        for key in keys:
            if not isinstance(key, dict) or "t" not in key or set(key) - {"t", *cls.KEYED_FIELDS}:
                raise ValueError(f"text.text_keys: each key needs 't' plus any of {list(cls.KEYED_FIELDS)}")
            t = key["t"]
            if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
                raise ValueError("text.text_keys: 't' must be a nonnegative number of seconds")
            for field in ("custom_text", "corner_text", "manual_datetime"):
                if field in key and not isinstance(key[field], str):
                    raise ValueError(f"text.text_keys: {field} must be text")
            if "manual_datetime" in key:
                datetime.fromisoformat(key["manual_datetime"])
            if "clock_mode" in key and (isinstance(key["clock_mode"], bool) or
                                        key["clock_mode"] not in range(cls.CLOCK_NONE, cls.CLOCK_TIMER + 1)):
                raise ValueError("text.text_keys: clock_mode must be an integer in [0, 5]")
            by_time[float(t)] = {**key, "t": float(t)}
        return [by_time[t] for t in sorted(by_time)]

    def restore_settings(self, data):
        from datetime import datetime
        for k in self.settings():
            if k in data:
                if k == "manual_datetime":
                    try:
                        datetime.fromisoformat(data[k])
                    except (ValueError, TypeError):
                        continue
                if k == "text_keys":
                    self.text_keys = self.normalize_text_keys(data[k])
                    continue
                setattr(self, k, data[k])

    def set_text_key(self, t, **fields):
        """Add or replace the key at time t (seconds)."""
        self.text_keys = self.normalize_text_keys(
            [k for k in self.text_keys if abs(k["t"] - t) > 1e-6] + [{"t": t, **fields}])

    def state_at(self, t):
        """Text settings in effect at time t, plus the time the shown date was keyed."""
        state = {f: getattr(self, f) for f in self.KEYED_FIELDS}
        date_anchor = 0.0
        for key in self.text_keys:
            if key["t"] > t:
                break
            state.update({f: key[f] for f in self.KEYED_FIELDS if f in key})
            if "manual_datetime" in key:
                date_anchor = key["t"]
        return state, date_anchor

    def build_items(self, t, w, h):
        g = lambda k: self.params.get(f"text.{k}").evaluate(t)  # noqa: E731
        scale, x, y, glow = g("scale"), g("x"), g("y"), g("glow")
        base = int(48 * scale * (w / 720.0))
        state, date_anchor = self.state_at(t)
        items = []
        if state["custom_text"]:
            items.append(dict(text=state["custom_text"], x=int(x * w) + 8,
                              y=int(y * h), size_px=base, glow=glow,
                              anchor="la"))
        clk = None
        show = True
        m = state["clock_mode"]
        from datetime import datetime, timedelta
        now = datetime.now()
        if self.manual_clock:
            now = datetime.fromisoformat(state["manual_datetime"])
            if self.clock_run:
                # a keyed date starts counting from its own key time
                now += timedelta(seconds=max(0.0, t - date_anchor))
        if m == self.CLOCK_BLINK:
            clk = "12:00"
            show = (int(t * 1.5) % 2 == 0)      # blink ~0.75 Hz (deterministic)
        elif m == self.CLOCK_TIME:
            clk = now.strftime("%H:%M:%S")
        elif m == self.CLOCK_DATE:
            clk = now.strftime("%b %d %Y").upper()
        elif m == self.CLOCK_DATETIME:
            clk = now.strftime("%m/%d/%Y %H:%M:%S")
        elif m == self.CLOCK_TIMER:
            clk = f"{int(t // 60):02d}:{int(t % 60):02d}:{int((t * 100) % 100):02d}"
        if clk and show:
            items.append(dict(text=clk, x=round(w * g("clock_x")), y=round(h * g("clock_y")),
                              size_px=base, glow=glow, anchor="ra"))
        if state["corner_text"]:
            items.append(dict(text=state["corner_text"], x=round(w * g("tag_x")),
                              y=round(h * g("tag_y")), size_px=int(base * 0.8),
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
        # before displaying the processed output
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
        travel = self.params.motion_phase("scramble.roll", t) * 1.7
        voff = int((travel * h) % h)
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


class EdgeRingEffect(Effect):
    """Bright/dark analogue enhancer halos, with a shifted ringing echo."""
    id = "edge_ring"
    label = "EDGE RING"

    def build(self, params):
        self.param("amount", "Amount", 0.4)
        self.param("width", "Width", 0.25)
        self.param("echo", "Echo", 0.3)

    def process(self, frame, t, v):
        if v["amount"] <= 0:
            return frame
        f = frame.astype(np.float32)
        sigma = max(0.3, frame.shape[1] / 720 * (0.5 + 7 * v["width"]))
        edge = f - cv2.GaussianBlur(f, (0, 0), sigma)
        offset = max(1, int(sigma * 2))
        ringing = edge - np.roll(edge, offset, axis=1) * v["echo"]
        return np.clip(f + ringing * v["amount"] * 4, 0, 255).astype(np.uint8)


class SolarizeEffect(Effect):
    """Fold highlights back into shadows for a bent colour-inverter look."""
    id = "solarize"
    label = "SOLARIZE"

    def build(self, params):
        self.param("amount", "Amount", 0.5)
        self.param("threshold", "Cutoff", 0.55)

    def process(self, frame, t, v):
        if v["amount"] <= 0:
            return frame
        f = frame.astype(np.float32)
        bent = np.where(f >= v["threshold"] * 255, 255 - f, f)
        return np.clip(f + (bent - f) * v["amount"], 0, 255).astype(np.uint8)


class SignalFoldEffect(Effect):
    """Video voltage overdrive folded into repeated colour contours."""
    id = "signal_fold"
    label = "SIGNAL FOLD"

    def build(self, params):
        self.param("amount", "Amount", 0.35)
        self.param("drive", "Drive", 2.5, 1.0, 8.0)
        self.param("bias", "Bias", 0.0, -1.0, 1.0)

    def process(self, frame, t, v):
        if v["amount"] <= 0:
            return frame
        f = frame.astype(np.float32) / 255
        folded = 1 - np.abs(np.mod(f * v["drive"] + v["bias"], 2) - 1)
        return np.clip((f + (folded - f) * v["amount"]) * 255, 0, 255).astype(np.uint8)


class FrameMoveEffect(Effect):
    """Center-based camera transform with toroidal (tiled) image boundaries."""
    id = "frame_move"
    label = "FRAME MOVE / WRAP"

    def build(self, params):
        self.param("zoom", "Zoom", 1, .25, 4, step=.01)
        self.param("x", "Pan X", 0, -1, 1)
        self.param("y", "Pan Y", 0, -1, 1)
        self.param("angle", "Rotate", 0, -180, 180, step=1)
        self.param("xspeed", "X Rate", 0, -1, 1)
        self.param("yspeed", "Y Rate", 0, -1, 1)

    def process(self, frame, t, v):
        h, w = frame.shape[:2]
        travel_x = self.params.motion_phase("frame_move.xspeed", t)
        travel_y = self.params.motion_phase("frame_move.yspeed", t)
        pan_x, pan_y = v["x"] + travel_x, v["y"] + travel_y
        zoom = max(.25, v["zoom"])
        if zoom == 1 and pan_x == 0 and pan_y == 0 and v["angle"] == 0:
            return frame
        cx, cy = (w - 1) / 2, (h - 1) / 2
        angle = math.radians(v["angle"])
        cosine, sine = math.cos(angle), math.sin(angle)
        # Translate the picture in output coordinates, then inverse-rotate and
        # inverse-scale around its centre to find the source pixel. Reduce the
        # offset in source space so long continuous rolls retain precision.
        offset_x = ((cosine * pan_x * w + sine * pan_y * h) / zoom) % w
        offset_y = ((-sine * pan_x * w + cosine * pan_y * h) / zoom) % h
        x = np.arange(w, dtype=np.float32)[None, :] - cx
        y = np.arange(h, dtype=np.float32)[:, None] - cy
        map_x = ((cosine * x + sine * y) / zoom + cx - offset_x).astype(np.float32)
        map_y = ((-sine * x + cosine * y) / zoom + cy - offset_y).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_WRAP)


class BloomEffect(Effect):
    """Soft phosphor-like glow around bright highlights."""
    id = "bloom"
    label = "CRT BLOOM"

    def build(self, params):
        self.param("amount", "Amount", .35)
        self.param("threshold", "Cutoff", .65)
        self.param("radius", "Radius", .3)

    def process(self, frame, t, v):
        if v["amount"] <= 0:
            return frame
        f = frame.astype(np.float32) / 255
        highlights = np.maximum(f - v["threshold"], 0)
        sigma = max(.3, frame.shape[1] / 720 * (1 + 18 * v["radius"]))
        glow = cv2.GaussianBlur(highlights, (0, 0), sigma)
        return np.clip((f + glow * v["amount"] * 3) * 255, 0, 255).astype(np.uint8)


class HueShiftEffect(Effect):
    """Chroma rotation with an optional continuous colour cycle."""
    id = "hue_shift"
    label = "HUE SHIFT"

    def build(self, params):
        self.param("amount", "Mix", .75)
        self.param("angle", "Hue", 0, -180, 180, step=1)
        self.param("speed", "Cycle", 0, -1, 1)

    def process(self, frame, t, v):
        phase = self.params.motion_phase("hue_shift.speed", t)
        if v["amount"] <= 0:
            return frame
        hsv = cv2.cvtColor(frame.astype(np.float32) / 255, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = (hsv[..., 0] + v["angle"] + phase * 360) % 360
        shifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) * 255
        return np.clip(frame * (1 - v["amount"]) + shifted * v["amount"], 0, 255).astype(np.uint8)


class PixelSortEffect(Effect):
    """Sort contiguous bright runs along scanlines, keeping dark boundaries."""
    id = "pixel_sort"
    label = "PIXEL SORT"

    def build(self, params):
        self.param("amount", "Mix", .6)
        self.param("threshold", "Cutoff", .45)
        self.param("length", "Length", .35)
        self.param("reverse", "Reverse", 0, 0, 1, step=1)
        self.param("upper", "High Cut", 1)
        self.choice("direction", "Direction", ("Horizontal", "Vertical"))
        self.choice("metric", "Sort By", ("Brightness", "Hue", "Saturation", "Red", "Green", "Blue"))
        self.param("outside", "Outside Band", 0, 0, 1, step=1)

    def process(self, frame, t, v):
        if v["amount"] <= 0:
            return frame
        source = frame.transpose(1, 0, 2) if v["direction"] >= .5 else frame
        metric = int(v["metric"])
        if metric == 0:
            luminance = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        elif metric in (1, 2):
            hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
            luminance = hsv[..., metric - 1].astype(np.float32)
            if metric == 1:
                luminance = np.minimum(255, luminance * (255 / 179))
        else:
            luminance = source[..., 5 - metric]
        low, high = sorted((v["threshold"] * 255, v["upper"] * 255))
        mask = (luminance >= low) & (luminance <= high)
        if v["outside"] >= .5:
            mask = ~mask
        out = source.copy()
        run_limit = max(2, int(source.shape[1] * max(.01, v["length"])))
        for y in range(source.shape[0]):
            changes = np.flatnonzero(np.diff(np.r_[False, mask[y], False]))
            for start, end in zip(changes[::2], changes[1::2]):
                for left in range(start, end, run_limit):
                    right = min(left + run_limit, end)
                    if right - left < 2:
                        continue
                    order = np.argsort(luminance[y, left:right], kind="stable")
                    if v["reverse"] >= .5:
                        order = order[::-1]
                    out[y, left:right] = source[y, left:right][order]
        if v["direction"] >= .5:
            out = np.ascontiguousarray(out.transpose(1, 0, 2))
        return cv2.addWeighted(frame, 1 - v["amount"], out, v["amount"], 0)


def default_effects(params: ParameterSet) -> list[Effect]:
    from .glitch import CRTRasterEffect, RasterHoldEffect, RollingShutterEffect, SyncLossEffect
    from .retro import GameBoyEffect, LEDMatrixEffect, NESCorruptionEffect, RGBTortureEffect
    return [
        TextOverlayEffect(params),
        FrameMoveEffect(params),
        SignalEffect(params),
        EdgeRingEffect(params),
        SolarizeEffect(params),
        SignalFoldEffect(params),
        BloomEffect(params),
        HueShiftEffect(params),
        PixelSortEffect(params),
        GameBoyEffect(params),
        LEDMatrixEffect(params),
        NESCorruptionEffect(params),
        RGBTortureEffect(params),
        RasterHoldEffect(params),
        RollingShutterEffect(params),
        CRTRasterEffect(params),
        SyncLossEffect(params),
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
        *[effect(params) for effect in ADVANCED_EFFECTS],
    ]
