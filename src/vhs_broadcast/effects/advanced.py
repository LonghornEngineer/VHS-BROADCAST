"""CPU video synthesis, temporal effects and signal/display treatments.

Temporal buffers hold preceding processed frames, never paused GUI refreshes.
These are aesthetic approximations, not codec corruption or signal emulation.
"""
from __future__ import annotations

import math
from pathlib import Path
import cv2
import numpy as np

from .base import Effect


def _u8(frame):
    return np.clip(frame, 0, 255).astype(np.uint8)


class TemporalEffect(Effect):
    """Keep prior-time state immutable so repeated timestamps can be re-edited."""
    def __init__(self, params):
        super().__init__(params)
        self.param("reset", "Reset pulse", 0, 0, 1, step=1)
        self.reset_state()

    def reset_state(self):
        self._previous = {}
        self._pending = {}
        self._time = None
        self._shape = None
        self._reset_requested = False

    def request_reset(self):
        self._reset_requested = True

    def apply(self, frame, t):
        if self.enabled_param().evaluate(t) < .5:
            self.reset_state()
            return frame
        if (self._reset_requested or self._shape != frame.shape or
                (self._time is not None and (t < self._time or t - self._time > 1))):
            self.reset_state()
        if self._time is None or t > self._time:
            self._previous = self._pending
            self._time = t
        self._shape = frame.shape
        v = {p.id.split(".")[-1]: p.evaluate(t) for p in self.own_params()}
        previous = self._previous
        reset = v["reset"] >= .5
        if reset and not previous.get("_reset", False):
            previous = {}
        dt = max(1 / 240, t - previous.get("_time", t - 1 / 30))
        out, state = self.process_history(frame, t, v, previous, dt)
        self._pending = dict(state, _time=t, _reset=reset)
        return out


class VideoFeedbackEffect(TemporalEffect):
    id = "feedback"
    label = "VIDEO FEEDBACK"

    def build(self, params):
        self.param("gain", "Gain", .65, 0, 1.25)
        self.param("decay", "Decay", .96)
        self.param("zoom", "Zoom", 1.02, .75, 1.25)
        self.param("angle", "Rotate", 1, -30, 30)
        self.param("x", "Offset X", 0, -.25, .25)
        self.param("y", "Offset Y", 0, -.25, .25)
        self.choice("blend", "Blend", ("Mix", "Add", "Screen"), 2)

    def process_history(self, frame, t, v, previous, dt):
        old = previous.get("output")
        out = frame
        if old is not None and v["gain"] > 0:
            h, w = frame.shape[:2]
            matrix = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), v["angle"], v["zoom"])
            matrix[:, 2] += (v["x"] * w, v["y"] * h)
            trail = cv2.warpAffine(old, matrix, (w, h), borderMode=cv2.BORDER_WRAP).astype(np.float32)
            trail *= v["gain"] * v["decay"] ** (dt * 30)
            f = frame.astype(np.float32)
            mode = int(v["blend"])
            if mode == 0:
                out = _u8(f * (1 - min(1, v["gain"])) + trail)
            elif mode == 1:
                out = _u8(f + trail)
            else:
                out = _u8(255 - (255 - f) * (255 - np.minimum(trail, 255)) / 255)
        return out, {"output": out.copy()}


class MotionSmearEffect(TemporalEffect):
    id = "motion_smear"
    label = "DATAMOSH / MOTION SMEAR"

    def build(self, params):
        self.param("block", "Block px", 16, 4, 96, step=1)
        self.param("motion", "Motion", 1, 0, 4)
        self.param("persistence", "Persist", .9, 0, .995)
        self.choice("mode", "Motion", ("Optical Flow", "Block Matching"))
        self.param("search", "Search px", 8, 1, 24, step=1)
        self.param("hold", "Vector s", 0, 0, 2)
        self.param("damage", "Bad Vectors", 0)
        self.param("seed", "Seed", 0, 0, 9999, step=1)

    def process_history(self, frame, t, v, previous, dt):
        h, w = frame.shape[:2]
        scale = min(1, 192 / max(w, h))
        sw, sh = max(8, round(w * scale)), max(8, round(h * scale))
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (sw, sh))
        old = previous.get("output")
        out = frame
        saved_flow = None
        flow_time = t
        if old is not None and v["persistence"] > 0:
            # Backward flow: for each current pixel, find its location in the
            # previous frame, then drag the accumulated colour history along.
            signature = (sw, sh, int(v["mode"]), round(v["block"]), round(v["search"]))
            cached = previous.get("flow")
            if cached is not None and previous.get("signature") == signature and t - previous.get("flow_time", t) < v["hold"]:
                flow = cached.copy()
                flow_time = previous["flow_time"]
            elif int(v["mode"]) == 1:
                flow = np.zeros((sh, sw, 2), np.float32)
                block = max(2, round(v["block"] * scale))
                radius = round(v["search"])
                for top in range(0, sh, block):
                    for left in range(0, sw, block):
                        patch = gray[top:top + block, left:left + block]
                        ph, pw = patch.shape
                        x0, y0 = max(0, left - radius), max(0, top - radius)
                        x1, y1 = min(sw, left + pw + radius), min(sh, top + ph + radius)
                        region = previous["gray"][y0:y1, x0:x1]
                        scores = cv2.matchTemplate(region, patch, cv2.TM_SQDIFF)
                        # Prefer zero displacement for uniform/tied blocks.
                        if scores[top - y0, left - x0] <= scores.min() + .01:
                            dx, dy = 0, 0
                        else:
                            _, _, location, _ = cv2.minMaxLoc(scores)
                            dx, dy = location[0] + x0 - left, location[1] + y0 - top
                        flow[top:top + ph, left:left + pw] = (dx, dy)
            else:
                flow = cv2.calcOpticalFlowFarneback(gray, previous["gray"], None, .5, 2, 15, 2, 5, 1.1, 0)
            saved_flow = flow.copy()
            flow[..., 0] *= w / sw
            flow[..., 1] *= h / sh
            grid = (max(1, math.ceil(w / v["block"])), max(1, math.ceil(h / v["block"])))
            small_flow = cv2.resize(flow, grid, interpolation=cv2.INTER_AREA)
            if v["damage"] > 0:
                rng = np.random.default_rng((round(v["seed"]) + round(flow_time * 30)) & 0xFFFFFFFF)
                damaged = rng.random(small_flow.shape[:2]) < v["damage"]
                random_vectors = rng.uniform(-v["block"] * 2, v["block"] * 2, small_flow.shape)
                small_flow[damaged] = random_vectors[damaged]
            flow = cv2.resize(small_flow,
                              (w, h), interpolation=cv2.INTER_NEAREST)
            x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            trail = cv2.remap(old, x + flow[..., 0] * v["motion"],
                              y + flow[..., 1] * v["motion"], cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_WRAP)
            keep = v["persistence"] ** (dt * 30)
            out = cv2.addWeighted(frame, 1 - keep, trail, keep, 0)
        return out, {"output": out.copy(), "gray": gray, "flow": saved_flow,
                     "flow_time": flow_time,
                     "signature": (sw, sh, int(v["mode"]), round(v["block"]), round(v["search"]))}


class CompositeEffect(Effect):
    id = "composite"
    label = "COMPOSITE ARTIFACTS"

    def build(self, params):
        self.param("dots", "Dot Crawl", .35)
        self.param("color", "Cross Color", .3)
        self.param("bandwidth", "Chroma BW", .6, .05, 1)
        self.param("speed", "Drift", .5, -3, 3)

    def process(self, frame, t, v):
        h, w = frame.shape[:2]
        phase = self.params.motion_phase("composite.speed", t) * 2 * math.pi
        signal = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        luma = signal[..., 0]
        # Measure detail over a few pixels: soft analog footage barely differs
        # from a 3x3 blur, which left the artifacts invisible on real video.
        detail = luma - cv2.GaussianBlur(luma, (0, 0), 1.5)
        x = np.arange(w, dtype=np.float32)[None, :]
        y = np.arange(h, dtype=np.float32)[:, None]
        carrier = np.cos(math.pi * (x + y) + phase)
        signal[..., 0] += np.clip(np.abs(detail) * 4, 0, 64) * carrier * v["dots"]
        signal[..., 1] += detail * np.sin(math.pi * x / 2 + phase) * v["color"] * 6
        signal[..., 2] += detail * np.cos(math.pi * x / 2 + phase) * v["color"] * 6
        smear = 1 - v["bandwidth"]
        kernel = int(smear * 40) | 1
        if kernel > 1:
            signal[..., 1:] = cv2.GaussianBlur(signal[..., 1:], (kernel, 1), 0)
        # Narrow chroma also arrives late, so colour slips right of its edges.
        lag = int(round(smear * w / 160))
        if lag:
            signal[:, lag:, 1:] = signal[:, :-lag, 1:].copy()
        return cv2.cvtColor(_u8(signal), cv2.COLOR_YCrCb2BGR)


class PaletteDitherEffect(Effect):
    id = "palette"
    label = "PALETTE / DITHER"
    BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16 - .46875
    EGA_RGB = np.array([[0, 0, 0], [0, 0, 170], [0, 170, 0], [0, 170, 170],
                       [170, 0, 0], [170, 0, 170], [170, 85, 0], [170, 170, 170],
                       [85, 85, 85], [85, 85, 255], [85, 255, 85], [85, 255, 255],
                       [255, 85, 85], [255, 85, 255], [255, 255, 85], [255, 255, 255]], np.float32)

    def build(self, params):
        self.choice("palette", "Palette", ("Grayscale", "Green CRT", "Amber CRT", "CGA", "EGA",
                                               "Handheld", "Sepia", "Ice", "Purple",
                                               "Riso Pop", "Field", "Dusk"), 4)
        self.param("colors", "Colors", 8, 2, 32, step=1)
        # Append modes: existing take/keyframe indices must keep their meanings.
        self.choice("mode", "Dither", ("None", "Bayer 4x4", "Noise", "Bayer 2x2",
                                          "Bayer 8x8", "Clustered Dots", "Horizontal Lines",
                                          "Diagonal Lines"), 1)
        self.param("amount", "Amount", .65)
        self.param("scale", "Cell px", 1, 1, 16, step=1)
        self.param("gamma", "Gamma", 1, .25, 3)
        self.param("bias", "Bias", 0, -.5, .5)
        self.param("animate", "Animate Noise", 1, 0, 1, step=1)
        self.param("seed", "Seed", 0, 0, 9999, step=1)
        # Appended controls default off so earlier takes render unchanged.
        self.param("shimmer", "Shimmer", 0)
        self.param("shimmer_speed", "Shimmer Hz", 2, 0, 12)
        self.param("gap", "LED Gap", 0)
        self.choice("space", "Match", ("RGB", "Luma"))
        self.param("blocks", "Block average", 0, 0, 1, step=1)
        self.choice("grade", "Grade", ("None", "Instant", "Slide", "Faded", "Night"))
        self.param("curve", "S-curve", 0)

    # Multi-hue palettes (RGB), designed for photographic footage.
    MULTI_RGB = (
        # Riso Pop: inks, their overprints and tints on cream paper.
        [[24, 22, 38], [250, 244, 228], [255, 92, 150], [40, 110, 200], [255, 214, 70],
         [0, 160, 150], [120, 70, 160], [240, 120, 70], [70, 150, 90], [255, 170, 200],
         [150, 190, 235], [255, 236, 160], [140, 210, 200], [60, 60, 110], [150, 50, 90],
         [200, 200, 190]],
        # Field: neutrals, earth, skin, foliage and sky for everyday scenes.
        [[18, 20, 24], [52, 48, 56], [96, 92, 96], [160, 156, 150], [236, 232, 220],
         [70, 40, 32], [140, 74, 48], [196, 120, 80], [232, 180, 140], [170, 120, 90],
         [40, 64, 36], [78, 112, 52], [140, 166, 80], [220, 200, 120], [30, 52, 90],
         [64, 110, 170], [150, 190, 220], [90, 150, 150], [160, 60, 70], [230, 150, 60]],
        # Dusk: violet night through sunset oranges to pale sky.
        [[16, 12, 30], [44, 24, 64], [84, 36, 96], [140, 48, 110], [200, 70, 110],
         [240, 110, 100], [255, 170, 110], [255, 226, 160], [30, 40, 90], [40, 80, 140],
         [60, 140, 180], [120, 200, 210], [210, 240, 230], [90, 30, 60], [170, 100, 170],
         [250, 250, 240]],
    )

    def colors(self, v):
        count = int(round(v["colors"]))
        mode = int(v["palette"])
        if mode >= 9:
            rgb = np.array(self.MULTI_RGB[mode - 9], np.float32)
            # Fewer colours keep an even spread from dark to light.
            rgb = rgb[np.argsort(rgb @ [.299, .587, .114], kind="stable")]
            return rgb[np.linspace(0, len(rgb) - 1, min(count, len(rgb))).round().astype(int), ::-1]
        count = min(count, 16)
        if mode < 3:
            tint = ([255, 255, 255], [60, 255, 95], [20, 185, 255])[mode]
            return np.linspace(0, 1, count)[:, None] * np.array(tint)[None, :]
        if mode == 5:
            rgb = np.array([[15, 56, 15], [48, 98, 48], [139, 172, 15], [155, 188, 15]])
            return rgb[np.linspace(0, 3, min(count, 4)).round().astype(int), ::-1]
        if mode == 8:
            # BGR stops: violet black, magenta, pale pink.
            stops = np.array([[30, 4, 18], [220, 60, 200], [255, 235, 255]], np.float32)
            at = np.linspace(0, 2, count)
            return np.array([np.interp(at, [0, 1, 2], stops[:, c]) for c in range(3)]).T
        if mode >= 6:
            low, high = (([12, 20, 32], [180, 225, 255]) if mode == 6
                         else ([35, 12, 5], [255, 245, 210]))
            return np.linspace(low, high, count)
        rgb = self.EGA_RGB if mode == 4 else self.EGA_RGB[[0, 11, 13, 15]]
        return rgb[np.linspace(0, len(rgb) - 1, min(count, len(rgb))).round().astype(int), ::-1]

    def process(self, frame, t, v):
        h, w = frame.shape[:2]
        scale = max(1, int(round(v["scale"])))
        x, y = np.arange(w) // scale, np.arange(h) // scale
        mode = int(v["mode"])
        if mode == 1:
            noise = self.BAYER[y[:, None] % 4, x[None, :] % 4]
        elif mode == 2:
            tick = int(t * 30) if v["animate"] >= .5 else 0
            rng = np.random.default_rng((tick + round(v["seed"]) * 104729) & 0xFFFFFFFF)
            grid = rng.random((h // scale + 1, w // scale + 1), dtype=np.float32) - .5
            noise = grid[y[:, None], x[None, :]]
        elif mode in (3, 4, 5):
            if mode == 3:
                matrix = np.array([[0, 2], [3, 1]], np.float32) / 4 - .375
            elif mode == 4:
                ranks = (self.BAYER + .46875) * 16
                matrix = np.block([[4 * ranks, 4 * ranks + 2],
                                   [4 * ranks + 3, 4 * ranks + 1]]) / 64 - 31.5 / 64
            else:
                matrix = np.array([[12, 5, 6, 13], [4, 0, 1, 7],
                                   [11, 3, 2, 8], [15, 10, 9, 14]], np.float32) / 16 - .46875
            size = matrix.shape[0]
            noise = matrix[y[:, None] % size, x[None, :] % size]
        elif mode in (6, 7):
            noise = ((y[:, None] + (x[None, :] if mode == 7 else np.zeros((1, w), np.int32))) % 8) / 8 - .4375
        else:
            noise = np.zeros((h, w), np.float32)
        palette = _u8(self.colors(v)).astype(np.float32)
        paint = self._grade(palette, int(v["grade"]), v["curve"])
        luma = int(v["space"]) == 1
        # Luma matching dithers brightness only, so a fixed swing suits any palette size.
        strength = v["amount"] * (96 if luma else 255 / max(1, math.sqrt(len(palette))))
        if v["shimmer"] > 0:
            noise = noise + self._shimmer(t, v, x, y, scale) * v["shimmer"] / max(.05, v["amount"])
        if v["blocks"] >= .5 and scale > 1:
            small = cv2.resize(frame, (-(-w // scale), -(-h // scale)), interpolation=cv2.INTER_AREA)
            frame = np.repeat(np.repeat(small, scale, 0), scale, 1)[:h, :w]
        if luma:
            target = cv2.cvtColor(palette[None] / 255, cv2.COLOR_BGR2YCrCb)[0] * 255
        out = np.empty_like(frame)
        for start in range(0, h, 32):
            pixels = frame[start:start + 32].astype(np.float32)
            if v["gamma"] != 1:
                pixels = 255 * np.power(pixels / 255, 1 / v["gamma"])
            dither = noise[start:start + 32, :, None] * strength
            if luma:
                pixels = cv2.cvtColor(pixels / 255, cv2.COLOR_BGR2YCrCb) * 255
                pixels[..., :1] += v["bias"] * 255 + dither
                flat = pixels.reshape(-1, 1, 3)
                # Brightness counts double: tone carries the picture, hue the mood.
                distance = (np.abs(flat - target[None]) * np.array([2, 1, 1], np.float32)).sum(2)
            else:
                pixels += v["bias"] * 255 + dither
                flat = pixels.reshape(-1, 3)
                distance = (flat * flat).sum(1)[:, None] - 2 * flat @ palette.T + (palette * palette).sum(1)
            shape = pixels.shape
            out[start:start + 32] = paint[distance.argmin(1)].reshape(shape).astype(np.uint8)
        if v["gap"] > 0 and scale >= 2:
            edge = ((np.arange(h) % scale == scale - 1)[:, None]
                    | (np.arange(w) % scale == scale - 1)[None, :])
            out[edge] = (out[edge] * (1 - v["gap"])).astype(np.uint8)
        return out

    @staticmethod
    def _grade(palette, grade, curve):
        """Recolour matched palette entries, index for index, like a film stock would."""
        if grade == 0 and curve <= 0:
            return palette
        c = palette / 255
        if curve > 0:
            c = c + curve * (c * c * (3 - 2 * c) - c)
        gray = c.mean(1, keepdims=True)
        if grade == 1:    # Instant: warm, lifted blacks, a little muted
            c = gray + (c - gray) * .85
            c = c * [.88, .98, 1.04] + [.05, .04, .03]
        elif grade == 2:  # Slide: saturated and punchy, cool shadows
            c = gray + (c - gray) * 1.3
            c = (c - .5) * 1.12 + .5 + [.03, 0, 0] * (1 - gray)
        elif grade == 3:  # Faded: soft contrast, raised blacks, washed colour
            c = gray + (c - gray) * .75
            c = c * .8 + .12
        elif grade == 4:  # Night: cool blue cast, deeper shadows
            c = c * [1.1, .95, .78] + [.04, .01, 0]
            c = (c - .5) * 1.05 + .48
        return np.clip(c, 0, 1) * 255

    def _shimmer(self, t, v, x, y, scale):
        """Per-cell brightness wave, each cell on its own phase and rate."""
        rng = np.random.default_rng(round(v["seed"]) * 7919 + 17)
        rows, cols = y[-1] + 1, x[-1] + 1
        phase = rng.random((rows, cols), dtype=np.float32)
        rate = .5 + rng.random((rows, cols), dtype=np.float32)
        clock = self.params.motion_phase("palette.shimmer_speed", t)
        wave = np.sin(2 * math.pi * (clock * rate + phase)) * .5
        return wave[y[:, None], x[None, :]]


class TemporalDisplacementEffect(TemporalEffect):
    id = "time_displace"
    label = "TEMPORAL DISPLACEMENT"
    MEMORY_BYTES = 96 * 1024 * 1024

    def build(self, params):
        self.param("history", "History s", 1, .1, 4)
        self.choice("map", "Map", ("Horizontal", "Vertical", "Radial"))
        self.param("depth", "Depth", .8)
        self.param("speed", "Animate", .1, -2, 2)

    def process_history(self, frame, t, v, previous, dt):
        h, w = frame.shape[:2]
        limit = max(1, min(120, self.MEMORY_BYTES // max(1, frame.nbytes)))
        history = tuple((stamp, image) for stamp, image in previous.get("history", ())
                        if stamp >= t - v["history"])
        history = (history + ((t, frame.copy()),))[-limit:]
        phase = self.params.motion_phase("time_displace.speed", t)
        x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
        y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        mode = int(v["map"])
        spatial = np.broadcast_to(x if mode == 0 else y, (h, w)) if mode < 2 else np.minimum(1, np.hypot(x - .5, y - .5) * math.sqrt(2))
        delay = np.mod(spatial + phase, 1) * v["history"] * v["depth"]
        stamps = np.array([item[0] for item in history])
        indices = np.clip(np.searchsorted(stamps, t - delay, side="right") - 1, 0, len(history) - 1)
        out = frame.copy()
        for index in np.unique(indices):
            mask = indices == index
            out[mask] = history[index][1][mask]
        return out, {"history": history}


class BlockCorruptionEffect(TemporalEffect):
    id = "block_corrupt"
    label = "BLOCK CORRUPTION"

    def build(self, params):
        self.param("block", "Block px", 32, 8, 128, step=1)
        self.param("density", "Density", .25)
        self.param("hold", "Hold s", .4, .05, 3)
        self.param("shift", "Displace", .2)
        self.choice("mode", "Mode", ("Freeze", "Repeat", "Shift", "Tile Replace"))
        self.param("chroma", "Chroma Only", 0, 0, 1, step=1)
        self.param("pool", "Patterns", 1, 1, 64, step=1)
        self.param("clusters", "Clusters", 0)
        self.param("lock", "Lock Layout", 0, 0, 1, step=1)
        self.param("seed", "Seed", 0, 0, 9999, step=1)
        self.param("amount", "Mix", 1)

    def process_history(self, frame, t, v, previous, dt):
        h, w = frame.shape[:2]
        block = max(1, int(round(v["block"])))
        bucket = int(t / max(.05, v["hold"]))
        anchor = previous.get("anchor") if previous.get("bucket") == bucket else None
        if anchor is None:
            anchor = frame.copy()
        layout_bucket = 0 if v["lock"] >= .5 else bucket
        rng = np.random.default_rng(((layout_bucket ^ 0xB10C) + round(v["seed"]) * 104729) & 0xFFFFFFFF)
        gh, gw = math.ceil(h / block), math.ceil(w / block)
        active = rng.random((gh, gw)) < v["density"]
        dx = rng.integers(-max(1, int(w * v["shift"])), max(1, int(w * v["shift"])) + 1, (gh, gw))
        dy = rng.integers(-max(1, int(h * v["shift"])), max(1, int(h * v["shift"])) + 1, (gh, gw))
        if v["shift"] == 0:
            dx.fill(0); dy.fill(0)
        if v["clusters"] > 0:
            grouped = rng.random((math.ceil(gh / 2), math.ceil(gw / 2))).repeat(2, 0).repeat(2, 1)[:gh, :gw]
            active = np.where(rng.random((gh, gw)) < v["clusters"], grouped < v["density"], active)
        xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        by, bx = np.arange(h) // block, np.arange(w) // block
        mask = active[by[:, None], bx[None, :]]
        mode = int(v["mode"])
        source = frame if mode in (2, 3) else anchor
        if mode == 1 and round(v["pool"]) == 1:
            sx, sy = int(rng.integers(w)), int(rng.integers(h))
            xx, yy = xx % block + sx, yy % block + sy
        elif mode in (1, 3):
            count = round(v["pool"])
            # Reuse a small bank of image fragments, instead of generating
            # synthetic characters or changing the source's color palette.
            sx = rng.integers(gw, size=count) * block
            sy = rng.integers(gh, size=count) * block
            addresses = rng.integers(count, size=(gh, gw))
            selected = addresses[by[:, None], bx[None, :]]
            xx = xx % block + sx[selected]
            yy = yy % block + sy[selected]
            xx, yy = xx.astype(np.float32), yy.astype(np.float32)
        else:
            xx += dx[by[:, None], bx[None, :]].astype(np.float32)
            yy += dy[by[:, None], bx[None, :]].astype(np.float32)
        corrupt = cv2.remap(source, xx, yy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
        if v["chroma"] >= .5:
            signal = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            chroma = cv2.cvtColor(corrupt, cv2.COLOR_BGR2YCrCb)
            signal[mask, 1:] = chroma[mask, 1:]
            converted = cv2.cvtColor(signal, cv2.COLOR_YCrCb2BGR)
            out = frame.copy()
            out[mask] = converted[mask]
        else:
            out = frame.copy()
            out[mask] = corrupt[mask]
        return cv2.addWeighted(frame, 1 - v["amount"], out, v["amount"], 0), {"anchor": anchor, "bucket": bucket}


class PhosphorPersistenceEffect(TemporalEffect):
    id = "phosphor"
    label = "PHOSPHOR PERSISTENCE"

    def build(self, params):
        self.param("red", "R Decay s", .3, .01, 3)
        self.param("green", "G Decay s", .6, .01, 3)
        self.param("blue", "B Decay s", .15, .01, 3)
        self.param("threshold", "Cutoff", .6)
        self.param("strength", "Trail", .85)

    def process_history(self, frame, t, v, previous, dt):
        luma = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        charge = np.where((luma >= v["threshold"] * 255)[..., None], frame, 0).astype(np.float32)
        old = previous.get("charge")
        if old is not None:
            decay = np.exp2(-dt / np.array([v["blue"], v["green"], v["red"]], np.float32))
            charge = np.maximum(charge, old * decay)
        return _u8(np.maximum(frame, charge * v["strength"])), {"charge": charge}


class CRTGeometryEffect(Effect):
    id = "crt_geometry"
    label = "CRT GEOMETRY / MASK"

    def build(self, params):
        self.param("curve", "Curvature", .12, 0, .5)
        self.param("corner", "Corners", .08, 0, .5)
        self.choice("mask", "Mask", ("RGB Triads", "Aperture Grille"))
        self.param("pitch", "Pitch px", 1, 1, 8, step=1)
        self.param("strength", "Mask Amt", .3)

    def process(self, frame, t, v):
        h, w = frame.shape[:2]
        xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        x, y = (xx - (w - 1) / 2) / max(1, w / 2), (yy - (h - 1) / 2) / max(1, h / 2)
        warp = 1 + v["curve"] * (x * x + y * y)
        out = cv2.remap(frame, x * warp * (w / 2) + (w - 1) / 2,
                        y * warp * (h / 2) + (h - 1) / 2, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
        pitch = max(1, int(round(v["pitch"])))
        stagger = ((yy.astype(int) // (pitch * 3)) % 2) * pitch if int(v["mask"]) == 0 else 0
        stripe = ((xx.astype(int) + stagger) // pitch) % 3
        for channel in range(3):
            out[..., channel] *= np.where(stripe == 2 - channel, 1, 1 - v["strength"])
        if int(v["mask"]) == 0:
            out *= np.where((yy.astype(int) % (pitch * 3)) == pitch * 3 - 1,
                            1 - v["strength"] * .7, 1)[..., None]
        radius = min(w, h) * v["corner"]
        if radius > 0:
            qx = np.abs(xx - (w - 1) / 2) - ((w - 1) / 2 - radius)
            qy = np.abs(yy - (h - 1) / 2) - ((h - 1) / 2 - radius)
            distance = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - radius
            out *= np.clip(1 - distance, 0, 1)[..., None]
        return _u8(out)


class RFInterferenceEffect(Effect):
    id = "rf_interference"
    label = "RF / HUM BARS"

    def build(self, params):
        self.param("frequency", "Frequency", 8, 1, 60)
        self.param("angle", "Angle", 15, -180, 180, step=1)
        self.param("speed", "Drift", .3, -3, 3)
        self.param("width", "Band Width", .25, .02, 1)
        self.param("amount", "Intensity", .35)
        self.choice("mode", "Pattern", ("Hum Bars", "Herringbone", "Both"), 2)

    def process(self, frame, t, v):
        if v["amount"] <= 0:
            return frame
        h, w = frame.shape[:2]
        x, y = np.arange(w, dtype=np.float32)[None, :] / w, np.arange(h, dtype=np.float32)[:, None] / h
        phase = self.params.motion_phase("rf_interference.speed", t)
        angle = math.radians(v["angle"])
        axis = x * math.sin(angle) + y * math.cos(angle)
        wave = np.mod(axis * v["frequency"] + phase, 1)
        bands = np.exp(-((wave - .5) / max(.01, v["width"] * .35)) ** 2)
        zigzag = np.abs(np.mod(y * 12, 2) - 1)
        weave = (1 + np.sin(2 * math.pi * (x * v["frequency"] * 3 + zigzag + phase))) / 2
        mode = int(v["mode"])
        pattern = bands if mode == 0 else weave if mode == 1 else bands * .7 + weave * .3
        return _u8(frame * (1 - pattern[..., None] * v["amount"]))


class LumaKeyEffect(Effect):
    id = "luma_key"
    label = "LUMA KEY / THRESHOLD"
    requires_source = True

    def __init__(self, params):
        self.background_path = ""
        self.background_image = None
        self.source_frame = None
        super().__init__(params)

    def build(self, params):
        self.choice("mode", "Mode", ("Silhouette", "Stepped Bands", "Luma Key"), 2)
        self.param("threshold", "Threshold", .5)
        self.param("softness", "Softness", .1, 0, .5)
        self.param("levels", "Levels", 4, 2, 12, step=1)
        self.param("invert", "Invert", 0, 0, 1, step=1)
        self.choice("fill", "Reveal", ("Black", "White", "Tint", "Original Feed", "Background Image"))
        self.param("red", "Fill R", .1)
        self.param("green", "Fill G", .5)
        self.param("blue", "Fill B", .2)

    def load_background(self, path):
        data = np.fromfile(path, dtype=np.uint8)
        if not data.size:
            raise ValueError("The selected image file is empty.")
        try:
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except cv2.error as error:
            raise ValueError("The selected file is not a readable image.") from error
        if image is None:
            raise ValueError("The selected file is not a readable image.")
        self.background_image = image
        self.background_path = str(Path(path).resolve())

    def settings(self):
        return {"background_path": self.background_path}

    def restore_settings(self, data):
        self.background_image = None
        self.background_path = data.get("background_path", "")
        if self.background_path:
            try:
                self.load_background(self.background_path)
            except (OSError, ValueError):
                pass  # Missing image uses black; its path stays visible in UI.

    def process(self, frame, t, v):
        h, w = frame.shape[:2]
        luma = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        if v["softness"] > 0:
            mask = np.clip((luma - v["threshold"]) / v["softness"] + .5, 0, 1)
            mask = mask * mask * (3 - 2 * mask)
        else:
            mask = (luma >= v["threshold"]).astype(np.float32)
        if v["invert"] >= .5:
            mask = 1 - mask
        mode = int(v["mode"])
        if mode == 0:
            return np.repeat(_u8(mask * 255)[..., None], 3, axis=2)
        if mode == 1:
            levels = max(2, int(round(v["levels"])))
            bands = np.round(mask * (levels - 1)) / (levels - 1)
            return np.repeat(_u8(bands * 255)[..., None], 3, axis=2)
        fill = int(v["fill"])
        if fill == 3 and self.source_frame is not None:
            background = self.source_frame
        elif fill == 4 and self.background_image is not None:
            background = cv2.resize(self.background_image, (w, h), interpolation=cv2.INTER_LINEAR)
        elif fill == 2:
            background = np.array([v["blue"], v["green"], v["red"]], np.float32) * 255
        else:
            background = 255 if fill == 1 else 0
        return _u8(frame * mask[..., None] + background * (1 - mask[..., None]))


class LineDitherEffect(Effect):
    """Thin parallel lines bent sideways by brightness, drawn only where the picture is lit."""
    id = "line_dither"
    label = "LINE DITHER"
    # BGR line colours; index 0 keeps the source colour.
    COLORS = np.array([[0, 0, 0], [255, 255, 255], [255, 150, 110],
                       [120, 255, 140], [60, 180, 255]], np.float32)

    def build(self, params):
        self.param("amount", "Mix", 1)
        self.param("spacing", "Spacing px", 6, 2, 32, step=1)
        self.param("thickness", "Line px", 1.5, .5, 8)
        self.param("bend", "Bend px", 10, 0, 64)
        self.param("threshold", "Threshold", .25)
        self.param("weight", "Weight by light", .3)
        self.param("smooth", "Smooth", 1.5, 0, 8)
        self.choice("direction", "Lines", ("Vertical", "Horizontal"))
        self.choice("color", "Color", ("Source", "White", "Periwinkle", "Green", "Amber"), 2)
        self.param("background", "Background", 0)
        self.param("drift", "Drift px/s", 0, -60, 60)
        self.choice("follow", "Follow", ("Off", "Leading edge", "Trailing edge", "Outline"), 1)
        self.param("count", "Lines from edge", 0, 0, 200, step=1)

    def process(self, frame, t, v):
        if v["amount"] <= 0:
            return frame
        horizontal = int(v["direction"]) == 1
        source = frame.transpose(1, 0, 2) if horizontal else frame
        h, w = source.shape[:2]
        luma = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        if v["smooth"] > 0:
            luma = cv2.GaussianBlur(luma, (0, 0), v["smooth"])
        spacing = max(2.0, float(round(v["spacing"])))
        drift = self.params.motion_phase("line_dither.drift", t)
        inside = luma >= v["threshold"]
        offset = self._edge_offset(inside, int(v["follow"]))
        # Lines sit at fixed steps in from the subject's edge, so each one echoes
        # the outline; brightness then pushes them sideways on top of that.
        phase = offset - luma * v["bend"] - drift
        distance = np.abs((phase + spacing / 2) % spacing - spacing / 2)
        width = v["thickness"] * (1 - v["weight"] + v["weight"] * luma)
        line = np.clip(width / 2 + .5 - distance, 0, 1)
        lit = np.clip((luma - v["threshold"]) / .05 + .5, 0, 1)
        if v["count"] >= 1 and int(v["follow"]):
            line *= offset < (round(v["count"]) - .5) * spacing
        mask = (line * lit)[..., None]
        index = int(v["color"])
        ink = source.astype(np.float32) if index == 0 else self.COLORS[index]
        base = source.astype(np.float32) * v["background"]
        out = _u8(base * (1 - mask) + ink * mask)
        if horizontal:
            out = np.ascontiguousarray(out.transpose(1, 0, 2))
        return cv2.addWeighted(frame, 1 - v["amount"], out, v["amount"], 0)

    @staticmethod
    def _edge_offset(inside, follow):
        """Distance in px from the subject's edge that each line is measured from."""
        h, w = inside.shape
        x = np.broadcast_to(np.arange(w, dtype=np.float32), (h, w))
        if follow == 0:
            return x
        if follow == 3:
            return np.maximum(0, cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 5) - 1)
        if follow == 2:
            return LineDitherEffect._edge_offset(inside[:, ::-1], 1)[:, ::-1]
        # Leading edge: distance since the run of lit pixels began on this row,
        # zero on the edge pixel so the first line traces the outline itself.
        last_off = np.maximum.accumulate(np.where(inside, -1, np.arange(w)), axis=1)
        return (x - last_off - 1).astype(np.float32)


ADVANCED_EFFECTS = (VideoFeedbackEffect, MotionSmearEffect, CompositeEffect,
                    PaletteDitherEffect, TemporalDisplacementEffect,
                    BlockCorruptionEffect, PhosphorPersistenceEffect,
                    CRTGeometryEffect, RFInterferenceEffect, LumaKeyEffect,
                    LineDitherEffect)
