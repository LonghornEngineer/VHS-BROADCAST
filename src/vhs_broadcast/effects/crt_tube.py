"""A CRT drawn the way the tube draws it: electron beams on a phosphor mask.

The overlay filters multiply the picture by a fixed pattern at its own resolution.
A real tube does not know about pixels. Each scanline of the signal is swept by
three electron beams (one per colour gun). A beam is a soft spot whose width grows
with its current, so dark lines are thin with black gaps between them and bright
lines swell until the gaps close. The beams land on a mask of phosphor dots or
stripes with its own pitch, and each phosphor only glows from its own gun. The
glass then scatters a little of that light (halation), and phosphors keep glowing
for a moment after the beam has passed (decay, which matters for interlace).

So this stage renders the tube at a display scale above the signal: the signal's
lines stay where they are, and the mask is drawn at output resolution where its
dots can actually be resolved. All light maths happens in linear light.
"""
from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

MASKS = ("RGB Triads", "Aperture Grille", "Slot Mask")
TRIADS, GRILLE, SLOT = range(3)
# BGR channel order, as everywhere in the app. Phosphors run R, G, B left to right.
_RGB_TO_BGR = (2, 1, 0)


# --------------------------------------------------------------------------- #
#  Beam: each source line as a spot whose width follows its own brightness
# --------------------------------------------------------------------------- #
def _erf(x):
    """Vectorised error function (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7)."""
    sign = np.sign(x)
    a = np.abs(x)
    k = 1.0 / (1.0 + 0.3275911 * a)
    poly = k * (0.254829592 + k * (-0.284496736 + k * (1.421413741 + k * (-1.453152027 + k * 1.061405429))))
    return sign * (1.0 - poly * np.exp(-a * a))


def beam_widths(gap: float, fill: float) -> tuple[float, float]:
    """Spot sigma (in line heights) for a black and a full-white line.

    Line gap 0 keeps even dark lines touching; 1 makes them hairlines. Bright fill
    is how far a full-current beam swells towards closing the gap."""
    dark = 0.42 - 0.30 * float(np.clip(gap, 0, 1))
    bright = dark + float(np.clip(fill, 0, 1)) * max(0.0, 0.6 - dark)
    return dark, bright


def beam_weights(h: int, rows: int, sigma_dark: float, sigma_bright: float,
                 field: int | None = None, shift: float = 0.0):
    """Cached; see _beam_weights."""
    return _beam_weights(int(h), int(rows), round(float(sigma_dark), 5), round(float(sigma_bright), 5),
                         None if field is None else int(field), round(float(shift), 5))


@lru_cache(maxsize=12)
def _beam_weights(h, rows, sigma_dark, sigma_bright, field, shift):
    """Spot profiles as (rows, h) weight matrices, one for a dark and one for a bright
    beam. Row y of a matrix says how much of each source line lands on output row y.

    Each output pixel holds the spot's light averaged over the pixel's own height, so a
    spot narrower than one output row is still counted in full rather than missed, and
    every source line's weights sum to one line's worth of light: the beam spreads its
    light, it never adds or loses any. `field` keeps only lines of that parity
    (interlace); `shift` moves the beam vertically, in lines.
    """
    scale = rows / h
    pos = (np.arange(rows, dtype=np.float64) + 0.5) / scale - 0.5 - shift   # in source lines
    half = 0.5 / scale
    lines = np.arange(h, dtype=np.float64)
    d = pos[:, None] - lines[None, :]
    near = np.abs(d) < 3.0
    if field is not None:
        near &= (lines[None, :] % 2) == field
    out = []
    for sigma in (sigma_dark, sigma_bright):
        s = sigma * math.sqrt(2.0)
        w = 0.5 * (_erf((d + half) / s) - _erf((d - half) / s)) / (2 * half)
        out.append(np.where(near, w, 0).astype(np.float32))
    for m in out:
        m.setflags(write=False)                 # shared through the cache
    return tuple(out)


def split_by_width(lin: np.ndarray):
    """Share each pixel's light between the dark-spot and bright-spot profiles.

    The spot widens with the square root of the beam current, so a pixel at level I
    draws sqrt(I) of its light with the bright profile and the rest with the dark one."""
    mix = np.sqrt(lin)
    bright = lin * mix
    return lin - bright, bright


def draw_beams(lin: np.ndarray, rows: int, sigma_dark: float, sigma_bright: float,
               field: int | None = None, shift: tuple[float, float, float] = (0, 0, 0)) -> np.ndarray:
    """Sweep every source line onto `rows` output rows; returns (rows, w, 3) light.
    `shift` moves each channel's beam vertically, in lines (convergence)."""
    h, w = lin.shape[:2]
    dark, bright = split_by_width(lin)
    out = np.empty((rows, w, 3), np.float32)
    cache = {}
    for c in range(3):
        key = round(float(shift[c]), 6)
        if key not in cache:
            cache[key] = beam_weights(h, rows, sigma_dark, sigma_bright, field, shift[c])
        wd, wb = cache[key]
        out[..., c] = wd @ dark[..., c] + wb @ bright[..., c]
    return out


# --------------------------------------------------------------------------- #
#  Phosphor mask, drawn at output resolution with anti-aliased edges
# --------------------------------------------------------------------------- #
def _cover(distance):
    """Coverage of a pixel whose centre is `distance` px inside an edge (box AA)."""
    return np.clip(distance + 0.5, 0.0, 1.0)


def build_mask(rows: int, cols: int, kind: int, pitch: float) -> np.ndarray:
    """(rows, cols, 3) BGR phosphor coverage, 0..1. `pitch` = output px between
    phosphors of the same colour along a row."""
    pitch = max(1.5, float(pitch))
    third = pitch / 3
    x = np.arange(cols, dtype=np.float32) + 0.5
    y = np.arange(rows, dtype=np.float32) + 0.5
    mask = np.zeros((rows, cols, 3), np.float32)
    if kind in (GRILLE, SLOT):
        stripe = third * 0.82                              # black matrix between stripes
        local = np.mod(x, pitch)
        columns = np.floor(x / pitch).astype(np.int64)
        if kind == SLOT:
            slot = pitch * 1.05                            # slot length along the stripe
            bar = slot * 0.18                              # mask bar between slots
        for i, channel in enumerate(_RGB_TO_BGR):
            centre = (i + 0.5) * third
            # Distance to the nearest same-colour stripe edge, with wrap-around.
            offset = np.abs(np.mod(local - centre + pitch / 2, pitch) - pitch / 2)
            across = _cover(stripe / 2 - offset)            # (cols,)
            if kind == GRILLE:
                mask[..., channel] = across[None, :]
            else:
                # Alternate triad columns are staggered by half a slot.
                phase = y[:, None] + (columns[None, :] % 2) * (slot / 2)
                along = np.abs(np.mod(phase, slot) - slot / 2)
                mask[..., channel] = across[None, :] * _cover((slot - bar) / 2 - along)
        return mask
    # Delta triads: round dots on a hexagonal lattice. Dots in a row step one third
    # of the pitch and cycle R, G, B; odd rows shift by half a dot and start two
    # colours on, so every little triangle of neighbours holds one of each.
    step = third
    row_h = step * math.sqrt(3) / 2
    radius = step * 0.45
    best = np.zeros((rows, cols), np.float32)
    colour = np.zeros((rows, cols), np.int64)
    j0 = np.floor(y / row_h).astype(np.int64)
    for dj in (0, 1):
        j = (j0 + dj)[:, None]                                         # (rows, 1)
        offset = (j % 2) * (step / 2)
        i = np.round((x[None, :] - offset) / step).astype(np.int64)    # (rows, cols)
        dx = x[None, :] - (i * step + offset)
        dy = y[:, None] - j * row_h
        cov = _cover(radius - np.hypot(dx, dy)).astype(np.float32)
        take = cov > best
        best = np.where(take, cov, best)
        colour = np.where(take, (i + 2 * (j % 2)) % 3, colour)
    for i, channel in enumerate(_RGB_TO_BGR):
        mask[..., channel] = np.where(colour == i, best, 0)
    return mask


# --------------------------------------------------------------------------- #
#  Tone: phosphors saturate softly instead of clipping to a flat white
# --------------------------------------------------------------------------- #
def soft_clip(x, knee=0.8):
    over = np.maximum(x - knee, 0)
    return np.where(x > knee, knee + (1 - knee) * (1 - np.exp(-over / (1 - knee))), x)


_OUT_STEPS = 4096   # output table, indexed by sqrt(light) over 0..4 so darks stay smooth


def _output_table(gamma, brightness):
    root = np.linspace(0, 2, _OUT_STEPS, dtype=np.float64)
    light = soft_clip(root * root * brightness)
    return np.clip(np.power(light, 1 / gamma) * 255 + 0.5, 0, 255).astype(np.uint8)


_GPU = None


def gpu_available() -> bool:
    """True when OpenCV has an OpenCL device to run on (the R9700 through ROCm's
    OpenCL). OpenCV picks a GPU by default; OPENCV_OPENCL_DEVICE overrides it."""
    global _GPU
    if _GPU is None:
        try:
            cv2.ocl.setUseOpenCL(True)
            _GPU = bool(cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL()
                        and cv2.ocl.Device.getDefault().available())
        except cv2.error:
            _GPU = False
    return _GPU


def _blur_small(img, sigma, factor, rows, cols):
    """Wide Gaussian blur computed at reduced size (it is smooth anyway)."""
    small = cv2.resize(img, (max(1, cols // factor), max(1, rows // factor)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(0.5, sigma / factor))
    return cv2.resize(small, (cols, rows), interpolation=cv2.INTER_LINEAR)


class Tube:
    """Caches the per-size work (mask, curvature maps) between frames."""

    def __init__(self):
        self._mask_key = None
        self._mask = None
        self._gain = None
        self._warp_key = None
        self._maps = None
        self._mask_gpu = None
        self._maps_gpu = None
        self._tables = {}

    def tables(self, gamma, brightness):
        """Input (code value -> linear light) and output (light -> code value) tables."""
        key = (round(gamma, 4), round(brightness, 4))
        if key not in self._tables:
            self._tables.clear()
            decode = np.power(np.arange(256, dtype=np.float64) / 255, gamma).astype(np.float32)
            self._tables[key] = (decode, _output_table(gamma, brightness))
        return self._tables[key]

    def mask(self, rows, cols, kind, pitch, depth, gpu=False):
        key = (rows, cols, kind, round(pitch, 3), round(depth, 3))
        if key != self._mask_key:
            raw = build_mask(rows, cols, kind, pitch)
            mask = 1 - depth * (1 - raw)
            # The mask blocks most of the beam; real tubes are driven harder to
            # make up for it. Gain per colour keeps a flat field at its level.
            gain = 1.0 / np.maximum(mask.reshape(-1, 3).mean(axis=0), 1e-3)
            self._mask, self._mask_key = (mask * gain).astype(np.float32), key
            self._mask_gpu = None
        if gpu:
            if self._mask_gpu is None:
                self._mask_gpu = cv2.UMat(self._mask)
            return self._mask_gpu
        return self._mask

    def warp(self, rows, cols, curve, corner, gpu=False):
        key = (rows, cols, round(curve, 4), round(corner, 4))
        if key != self._warp_key:
            xx, yy = np.meshgrid(np.arange(cols, dtype=np.float32), np.arange(rows, dtype=np.float32))
            x, y = (xx - (cols - 1) / 2) / max(1, cols / 2), (yy - (rows - 1) / 2) / max(1, rows / 2)
            bend = 1 + curve * (x * x + y * y)
            mx = x * bend * (cols / 2) + (cols - 1) / 2
            my = y * bend * (rows / 2) + (rows - 1) / 2
            edge = None
            radius = min(cols, rows) * corner
            if radius > 0:
                qx = np.abs(xx - (cols - 1) / 2) - ((cols - 1) / 2 - radius)
                qy = np.abs(yy - (rows - 1) / 2) - ((rows - 1) / 2 - radius)
                dist = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - radius
                edge = np.repeat(np.clip(1 - dist, 0, 1)[..., None], 3, axis=2).astype(np.float32)
            self._maps = (mx.astype(np.float32), my.astype(np.float32), edge)
            self._maps_gpu = None
            self._warp_key = key
        if gpu:
            if self._maps_gpu is None:
                self._maps_gpu = tuple(None if m is None else cv2.UMat(m) for m in self._maps)
            return self._maps_gpu
        return self._maps

    def _bands_gpu(self, key, wd, wb, cols):
        """The weight matrices as six bands for the card: for each band, a row map that
        picks one source line per output row, and that line's dark/bright weights as
        images. The beam then is six row-picks (remap) and multiply-adds, which every
        OpenCL device runs fast, instead of a large matrix product."""
        cache = self.__dict__.setdefault("_bands_cache", {})
        if key not in cache:
            if len(cache) > 6:
                cache.clear()
            rows, h = wd.shape
            first = np.argmax((wd + wb) > 0, axis=1)               # first line each row uses
            xs = np.tile(np.arange(cols, dtype=np.float32), (rows, 1))
            bands = []
            for j in range(6):
                k = first + j
                inside = k < h
                k = np.minimum(k, h - 1)
                pick = np.arange(rows)
                dark = np.where(inside, wd[pick, k], 0).astype(np.float32)
                bright = np.where(inside, wb[pick, k], 0).astype(np.float32)
                if not (dark.any() or bright.any()):
                    continue
                ys = np.repeat(k.astype(np.float32)[:, None], cols, axis=1)
                bands.append((cv2.UMat(xs), cv2.UMat(ys),
                              cv2.UMat(np.repeat(dark[:, None], cols, axis=1)),
                              cv2.UMat(np.repeat(bright[:, None], cols, axis=1))))
            cache[key] = bands
        return cache[key]

    def render(self, frame, v, *, scale, field=None, history=None, bloom=None, gpu=False):
        """Draw `frame` (BGR uint8) on the tube. Returns (BGR uint8 at display size,
        beam light to carry to the next field).

        With `gpu`, everything after the small source-size steps runs as OpenCV
        OpenCL (UMat) work on the graphics card; the maths is the same as on the CPU."""
        h, w = frame.shape[:2]
        rows = max(2, int(round(h * scale)))
        cols = max(2, int(round(w * scale)))
        rows -= rows % 2
        cols -= cols % 2
        decode, encode = self.tables(float(v["gamma"]), float(v["brightness"]))
        lin = decode[frame]

        # Convergence: the red and blue guns land a little right/down and left/up of green.
        miss = float(v["converge"])
        if miss > 1e-4:
            for channel, sign in ((2, 1.0), (0, -1.0)):
                shift = np.float32([[1, 0, sign * miss], [0, 1, 0]])
                lin[..., channel] = cv2.warpAffine(lin[..., channel], shift, (w, h),
                                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        vertical = (-0.5 * miss, 0.0, 0.5 * miss)          # B, G, R in lines

        dark, bright = beam_widths(v["gap"], v["fill"])
        if gpu:
            low, high = split_by_width(lin)
            planes = []
            for c in range(3):
                wd, wb = beam_weights(h, rows, dark, bright, field, vertical[c])
                bands = self._bands_gpu((h, rows, w, round(dark, 5), round(bright, 5), field,
                                         round(vertical[c], 5)), wd, wb, w)
                lo = cv2.UMat(np.ascontiguousarray(low[..., c]))
                hi = cv2.UMat(np.ascontiguousarray(high[..., c]))
                plane = None
                for xs, ys, dark_w, bright_w in bands:
                    term = cv2.add(cv2.multiply(cv2.remap(lo, xs, ys, cv2.INTER_NEAREST), dark_w),
                                   cv2.multiply(cv2.remap(hi, xs, ys, cv2.INTER_NEAREST), bright_w))
                    plane = term if plane is None else cv2.add(plane, term)
                planes.append(plane)
            light = cv2.merge(planes)
        else:
            light = draw_beams(lin, rows, dark, bright, field=field, shift=vertical)
        # Phosphors drawn by the previous field are still glowing (history is that
        # field's light, kept only if it was drawn at this size on this device).
        key = (rows, w, gpu)
        if history is not None and history[0] == key:
            light = cv2.addWeighted(light, 1.0, history[1], float(v["field_decay"]), 0)
        carry = (key, light)

        # The beams sweep continuously along the line; resample to display width.
        light = cv2.resize(light, (cols, rows), interpolation=cv2.INTER_CUBIC)

        curve, corner = float(v["curve"]), float(v["corner"])
        edge = None
        if curve > 1e-4 or corner > 1e-4:
            mx, my, edge = self.warp(rows, cols, curve, corner, gpu)
            light = cv2.remap(light, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        glow = cv2.multiply(light, self.mask(rows, cols, int(v["mask"]), float(v["triad"]) * scale,
                                             float(v["depth"]), gpu))

        soft = float(v["soft"])
        if soft > 1e-3:
            sigma = soft * 0.22 * float(v["triad"]) * scale
            glow = cv2.GaussianBlur(glow, (0, 0), max(0.3, sigma))

        # Light scattered back out of the faceplate glass: wide and faint.
        halation = float(v["halation"])
        spill = None
        if halation > 1e-3:
            spill = cv2.multiply(_blur_small(glow, rows * 0.03, 8, rows, cols), (0.08 * halation,) * 3 + (0,))
        if bloom is not None:
            # CRT BLOOM's glow, from the lit phosphors. It is a wide blur, so it is
            # worked out at a quarter of the display size.
            amount, threshold, radius = bloom
            small = cv2.resize(glow, (max(1, cols // 4), max(1, rows // 4)), interpolation=cv2.INTER_AREA)
            small = small.get() if gpu else small
            luma = small.max(axis=2)
            knee = threshold * max(0.25, float(np.percentile(luma, 99.5)))
            hot = small * np.clip((luma - knee) / max(0.05, 1 - knee), 0, 1)[..., None]
            # Same reach as CRT BLOOM on its own: (2 + 40 x radius) px per 720 px of width.
            spread = (2 + 40 * radius) * cols / 720 / 4
            hot = (cv2.GaussianBlur(hot, (0, 0), max(0.5, spread * .25)) * 0.6
                   + cv2.GaussianBlur(hot, (0, 0), max(0.5, spread)) * 1.0
                   + cv2.GaussianBlur(hot, (0, 0), max(0.5, spread * 3)) * 0.8) * (0.5 * amount)
            add = cv2.resize(cv2.UMat(hot) if gpu else hot, (cols, rows), interpolation=cv2.INTER_LINEAR)
            spill = add if spill is None else cv2.add(spill, add)
        if spill is not None:
            glow = cv2.add(glow, spill)
        if edge is not None:
            glow = cv2.multiply(glow, edge)
        if gpu:
            return self._encode_gpu(glow, float(v["gamma"]), float(v["brightness"])), carry
        # Encode through the output table, indexed by sqrt(light) so darks stay smooth.
        # (numpy, not cv2, for the scalar steps: cv2 applies a bare scalar to channel 0 only)
        np.maximum(glow, 0, out=glow)
        index = cv2.sqrt(glow)
        index *= (_OUT_STEPS - 1) / 2
        np.minimum(index, _OUT_STEPS - 1, out=index)
        return encode[index.astype(np.uint16)], carry

    @staticmethod
    def _encode_gpu(glow, gamma, brightness, knee=0.8):
        """soft_clip(light x brightness) ** (1 / gamma) -> uint8, all on the card.
        (Per-channel scalars: cv2 applies a bare number to channel 0 only.)"""
        def per(value):
            return (value, value, value, 0)
        x = cv2.max(cv2.multiply(glow, per(brightness)), per(0.0))
        low = cv2.min(x, per(knee))
        over = cv2.max(cv2.subtract(x, per(knee)), per(0.0))
        fade = cv2.exp(cv2.multiply(over, per(-1.0 / (1 - knee))))
        y = cv2.add(low, cv2.addWeighted(fade, -(1 - knee), fade, 0, 1 - knee))
        y = cv2.pow(y, 1.0 / gamma)
        return cv2.convertScaleAbs(y, alpha=255).get()
