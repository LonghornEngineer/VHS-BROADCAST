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


_BINS = 64          # beam widths are tabulated over this many brightness steps


def draw_beams(lin: np.ndarray, rows: int, sigma_dark: float, sigma_bright: float,
               field: int | None = None, shift: tuple[float, float, float] = (0, 0, 0)) -> np.ndarray:
    """Sweep every source line onto `rows` output rows; returns (rows, w, 3) light.

    Each output pixel holds the beam's light averaged over the pixel's own height, so
    a spot narrower than one output row is still counted in full rather than missed.
    The average over a line period equals the line's intensity: the beam spreads its
    light, it does not add or lose any. `field` lights only lines of that parity
    (interlace); `shift` moves each channel's beam vertically, in lines (convergence).

    The spot profile for every (output row, brightness step) pair is tabulated once
    per tap, then looked up per pixel, so the error function is never evaluated per pixel.
    """
    h, w = lin.shape[:2]
    scale = rows / h
    centre = (np.arange(rows, dtype=np.float32) + 0.5) / scale - 0.5     # in source lines
    half = 0.5 / scale                                                     # half an output row
    # Spot width grows with the square root of the beam current (brightness).
    steps = np.linspace(0, 1, _BINS, dtype=np.float32)
    sigmas = (sigma_dark + (sigma_bright - sigma_dark) * steps)[None, :] * math.sqrt(2.0)
    bins = np.minimum(np.sqrt(lin) * (_BINS - 1) + 0.5, _BINS - 1).astype(np.intp)
    out = np.zeros((rows, w, 3), np.float32)
    for c in range(3):
        pos = centre - np.float32(shift[c])
        base = np.floor(pos).astype(np.int32)
        for tap in (-1, 0, 1, 2):
            k = base + tap
            valid = (k >= 0) & (k < h)
            if field is not None:
                valid &= (k % 2) == field
            if not valid.any():
                continue
            idx = np.clip(k, 0, h - 1)
            d = (pos - k)[:, None]
            table = 0.5 * (_erf((d + half) / sigmas) - _erf((d - half) / sigmas)) / (2 * half)
            table *= valid[:, None]                                        # (rows, bins)
            share = np.take_along_axis(table.astype(np.float32), bins[idx, :, c], axis=1)
            out[..., c] += lin[idx, :, c] * share
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


def _blur_small(img, sigma, factor):
    """Wide Gaussian blur computed at reduced size (it is smooth anyway)."""
    rows, cols = img.shape[:2]
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
        self._tables = {}

    def tables(self, gamma, brightness):
        """Input (code value -> linear light) and output (light -> code value) tables."""
        key = (round(gamma, 4), round(brightness, 4))
        if key not in self._tables:
            self._tables.clear()
            decode = np.power(np.arange(256, dtype=np.float64) / 255, gamma).astype(np.float32)
            self._tables[key] = (decode, _output_table(gamma, brightness))
        return self._tables[key]

    def mask(self, rows, cols, kind, pitch, depth):
        key = (rows, cols, kind, round(pitch, 3), round(depth, 3))
        if key != self._mask_key:
            raw = build_mask(rows, cols, kind, pitch)
            mask = 1 - depth * (1 - raw)
            # The mask blocks most of the beam; real tubes are driven harder to
            # make up for it. Gain per colour keeps a flat field at its level.
            gain = 1.0 / np.maximum(mask.reshape(-1, 3).mean(axis=0), 1e-3)
            self._mask, self._mask_key = (mask * gain).astype(np.float32), key
        return self._mask

    def warp(self, rows, cols, curve, corner):
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
            self._warp_key = key
        return self._maps

    def render(self, frame, v, *, scale, field=None, history=None, bloom=None):
        """Draw `frame` (BGR uint8) on the tube. Returns (BGR uint8 at display size,
        beam light to carry to the next field)."""
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
        light = draw_beams(lin, rows, dark, bright, field=field, shift=vertical)
        if history is not None and history.shape == light.shape:
            light = light + history * float(v["field_decay"])
        carry = light

        # The beams sweep continuously along the line; resample to display width.
        light = cv2.resize(light, (cols, rows), interpolation=cv2.INTER_CUBIC)

        curve, corner = float(v["curve"]), float(v["corner"])
        edge = None
        if curve > 1e-4 or corner > 1e-4:
            mx, my, edge = self.warp(rows, cols, curve, corner)
            light = cv2.remap(light, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        glow = cv2.multiply(light, self.mask(rows, cols, int(v["mask"]), float(v["triad"]) * scale,
                                             float(v["depth"])))

        soft = float(v["soft"])
        if soft > 1e-3:
            sigma = soft * 0.22 * float(v["triad"]) * scale
            glow = cv2.GaussianBlur(glow, (0, 0), max(0.3, sigma))

        # Light scattered back out of the faceplate glass: wide and faint.
        halation = float(v["halation"])
        spill = None
        if halation > 1e-3:
            spill = _blur_small(glow, rows * 0.03, 8) * (0.08 * halation)
        if bloom is not None:
            # CRT BLOOM's glow, from the lit phosphors. It is a wide blur, so it is
            # worked out at a quarter of the display size.
            amount, threshold, radius = bloom
            small = cv2.resize(glow, (max(1, cols // 4), max(1, rows // 4)), interpolation=cv2.INTER_AREA)
            luma = small.max(axis=2)
            knee = threshold * max(0.25, float(np.percentile(luma, 99.5)))
            hot = small * np.clip((luma - knee) / max(0.05, 1 - knee), 0, 1)[..., None]
            # Same reach as CRT BLOOM on its own: (2 + 40 x radius) px per 720 px of width.
            spread = (2 + 40 * radius) * cols / 720 / 4
            hot = (cv2.GaussianBlur(hot, (0, 0), max(0.5, spread * .25)) * 0.6
                   + cv2.GaussianBlur(hot, (0, 0), max(0.5, spread)) * 1.0
                   + cv2.GaussianBlur(hot, (0, 0), max(0.5, spread * 3)) * 0.8) * (0.5 * amount)
            add = cv2.resize(hot, (cols, rows), interpolation=cv2.INTER_LINEAR)
            spill = add if spill is None else cv2.add(spill, add)
        if spill is not None:
            glow = cv2.add(glow, spill)
        if edge is not None:
            glow = cv2.multiply(glow, edge)
        # Encode through the output table, indexed by sqrt(light) so darks stay smooth.
        # (numpy, not cv2, for the scalar steps: cv2 applies a bare scalar to channel 0 only)
        np.maximum(glow, 0, out=glow)
        index = cv2.sqrt(glow)
        index *= (_OUT_STEPS - 1) / 2
        np.minimum(index, _OUT_STEPS - 1, out=index)
        return encode[index.astype(np.uint16)], carry
