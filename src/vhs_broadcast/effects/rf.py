"""Analog signal models behind the tape and TV static.

Static is drawn by modelling where it comes from rather than painting noise over the
picture:

* ``track_envelope`` - how much RF a spinning VHS head picks up from where it sits
  over the helical tracks. Search bars, pause bars and mistracking all come from the
  head crossing onto tracks recorded at the other azimuth.
* ``fm_playback`` - what the VCR's luma FM demodulator makes of that RF level: clean
  picture while the carrier is strong, then FM "clicks" (white and black dashes with
  a de-emphasis tail) as it nears threshold, then nothing but noise. The dropout
  compensator patches weak spots with the line above, the colour killer greys out
  the chroma, and noisy sync makes lines wander.
* ``receiver_snow`` - a TV on a weak or empty channel: Gaussian RF noise through the
  envelope detector of a negative-modulation NTSC receiver, with AGC and colour
  killer.

All functions work on BGR uint8 frames and scale their pixel sizes with the frame
width (measured at 640).
"""
from __future__ import annotations

import cv2
import numpy as np


def _erfc(x):
    """Complementary error function (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7)."""
    x = np.asarray(x, np.float32)
    t = 1 / (1 + .3275911 * np.abs(x))
    poly = t * (.254829592 + t * (-.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    out = poly * np.exp(-x * x)
    return np.where(x >= 0, out, 2 - out)


def _unit(w):
    return max(1.0, w / 640)


def _tail(field, length, tau):
    """Causal exponential smear along each line (the de-emphasis/low-pass after the FM
    demodulator): an impulse becomes a dash that fades to the right."""
    n = max(2, int(length))
    k = np.exp(-np.arange(n, dtype=np.float32) / max(.3, tau))
    k = (k / k.sum())[::-1][None, :].copy()
    return cv2.filter2D(field, -1, k, anchor=(n - 1, 0), borderType=cv2.BORDER_REPLICATE)


def smooth_rows(rng, h, scale):
    """Slowly varying per-line noise (~unit std) correlated over about `scale` lines."""
    n = max(2, int(h / max(1.0, scale)) + 2)
    coarse = rng.normal(0, 1, (n, 1)).astype(np.float32)
    return cv2.resize(coarse, (1, h), interpolation=cv2.INTER_CUBIC)[:, 0]


# --------------------------------------------------------------------------- #
#  Head over the tape
# --------------------------------------------------------------------------- #
def track_envelope(offset, head=1.0):
    """RF level 0..1 read by a head whose centre sits `offset` track pitches from the
    centre of a track recorded at its own azimuth.

    VHS lays tracks side by side with no guard band, alternating azimuth. Azimuth loss
    keeps a head from reading the other head's tracks at luma frequencies, so the level
    is the share of the head's width over tracks of its own azimuth, which repeat every
    two pitches."""
    d = np.abs((np.asarray(offset, np.float32) + 1) % 2 - 1)           # 0..1 off-centre
    lo, hi = d - head / 2, d + head / 2
    own = np.clip(np.minimum(hi, .5) - np.maximum(lo, -.5), 0, None)
    next_own = np.clip(np.minimum(hi, 2.5) - np.maximum(lo, 1.5), 0, None)
    return np.clip((own + next_own) / min(head, 1.0), 0, 1)


def head_contact(rng, h, amount=.12):
    """Per-line wobble in RF level as the head rides over the tape surface."""
    return np.clip(1 + amount * smooth_rows(rng, h, 3), 0, 1.2)


# --------------------------------------------------------------------------- #
#  VCR playback: luma FM demodulation
# --------------------------------------------------------------------------- #
def fm_rows(env, cnr):
    """Linear CNR and FM click probability for an RF level `env` ((h, 1) or (h, w)), and
    the lines that need processing (the rest are effectively clean)."""
    rho = 10 ** (cnr / 10) * np.square(np.clip(env, 0, 1)) + 1e-6       # linear CNR
    # Rice: below FM threshold the phase slips 2*pi now and then; each slip is a click.
    clicks = .5 * _erfc(np.sqrt(rho))
    rows = np.flatnonzero((clicks.max(axis=1) > 2e-4) | (rho.min(axis=1) < 60))
    return rho, clicks, rows


def fm_demod(luma, rho, clicks, rng, cnr, unit=1.0, level=118.0, kill=.12, grain=1.0):
    """Demodulate luma (float, 0-255 scale, shape (n, w)) whose carrier has linear CNR
    `rho` and click probability `clicks` (both broadcastable to it).

    Returns the demodulated luma, how much of the picture survives (`keep`, for adding
    noise to the chroma) and the colour-killer gain for the chroma. `kill` is the
    relative RF level where the colour starts to go; `grain` scales the FM grain."""
    n, w = luma.shape
    rho = np.broadcast_to(rho, (n, w))
    p = np.broadcast_to(clicks, (n, w))
    # Above threshold: grain that rises with frequency (FM's triangular noise spectrum),
    # rolled off again by the de-emphasis, so it sits in the mid detail band.
    sigma = np.minimum(34 * grain / np.sqrt(rho), 90)
    fine = rng.normal(0, 1, (n, w)).astype(np.float32)
    fine = fine - cv2.GaussianBlur(fine, (0, 0), sigmaX=4 * unit, sigmaY=.01)
    fine = cv2.GaussianBlur(fine, (0, 0), sigmaX=.7 * unit, sigmaY=.01) * 1.6
    # Clicks: impulses mostly toward white, each far bigger than the luma swing.
    hit = rng.random((n, w)) < p
    sign = np.where(rng.random((n, w)) < .62, 1.0, -1.0).astype(np.float32)
    impulse = hit * sign * rng.uniform(4, 9, (n, w)).astype(np.float32)
    burst = cv2.GaussianBlur(impulse, (0, 0), sigmaX=.6 * unit, sigmaY=.01)
    noise = _tail(burst * 60, 7 * unit, 2.2 * unit) + fine * sigma
    # As clicks take over, the demodulator follows the noise instead of the picture.
    keep = np.clip(1 - 2.1 * p, 0, 1)
    out = np.clip(luma * keep + level * (1 - keep) + noise, 4, 250)
    # Colour killer: no burst, no colour. Chroma fades out well before the luma does.
    e = np.sqrt(rho / 10 ** (cnr / 10))
    chroma = np.clip((e - kill) / (2.5 * kill), 0, 1)
    return out, keep, chroma


def fm_playback(frame, env, rng, cnr=18.0, doc=True, sync=1.0, level=118.0, doc_lines=4):
    """Play `frame` back through a VHS luma FM channel whose RF level is `env`.

    `env` is 0..1 per line (shape (h,) or (h, 1)) or per pixel (h, w); 1 is a head
    squarely on its own track. `cnr` is the carrier-to-noise ratio in dB at full level.
    `level` is the grey the demodulator settles to on pure noise. Returns uint8 BGR."""
    h, w = frame.shape[:2]
    env = np.asarray(env, np.float32).reshape(h, -1)                  # (h, 1) or (h, w)
    unit = _unit(w)
    rho, clicks, rows = fm_rows(env, cnr)
    if not len(rows):
        return frame
    ycc = cv2.cvtColor(frame[rows], cv2.COLOR_BGR2YCrCb).astype(np.float32)
    n = len(rows)
    ycc[..., 0], keep, chroma = fm_demod(ycc[..., 0], rho[rows], clicks[rows], rng, cnr, unit, level)
    tint = rng.normal(0, 1, (max(2, n // 4), max(2, w // 32), 2)).astype(np.float32)
    tint = cv2.resize(tint, (w, n), interpolation=cv2.INTER_LINEAR) * 5
    for c in (1, 2):
        ycc[..., c] = 128 + (ycc[..., c] - 128) * chroma + tint[..., c - 1] * (1 - chroma) * (1 - keep)
    out = frame.copy()
    out[rows] = cv2.cvtColor(np.clip(ycc, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    if doc:
        # doc_lines counts real scan lines; frames are often scaled from 480 active lines.
        dropout_compensate([out], env, rng, rows, unit, max(1, round(doc_lines * h / 480)))
    if sync > 0:
        dx = afc_drift(env, rng, sync, unit)
        if dx is not None:
            map_x = np.arange(w, dtype=np.float32)[None, :] - dx[:, None]
            map_y = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1)
            out = cv2.remap(out, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out


def dropout_compensate(planes, env, rng, rows, unit, max_lines):
    """Where the RF falls below the DOC threshold the deck substitutes the line above
    from its 1H delay line - up to `max_lines` in a row before it gives up. Works in
    place on each array in `planes` (all the same height and width)."""
    w = planes[0].shape[1]
    # The DOC watches a noisy envelope detector, so it switches in ragged segments.
    wobble = rng.normal(0, 1, (len(rows), w)).astype(np.float32)
    wobble = cv2.GaussianBlur(wobble, (0, 0), sigmaX=10 * unit, sigmaY=.01)
    wobble /= wobble.std() + 1e-6
    e = np.broadcast_to(env[rows], (len(rows), w))
    drop = e * (1 + .35 * wobble) < .2
    if not drop.any():
        return
    run = np.zeros(w, np.int32)
    last = -2
    for i, y in enumerate(rows):
        if y == 0:
            continue
        if y != last + 1:
            run[:] = 0
        last = y
        m = drop[i] & (run < max_lines)
        if m.any():
            for plane in planes:
                plane[y, m] = plane[y - 1, m]
        run = np.where(drop[i], run + 1, 0)


def afc_drift(env, rng, amount, unit):
    """Weak RF makes the H-sync noisy; the TV's horizontal AFC follows it sluggishly, so
    the lines drift sideways in soft wiggles. Returns a shift per line in pixels, or None
    when it would not be visible."""
    line = env.mean(axis=1)
    weak = np.clip(1 - line / .3, 0, 1) ** 2               # sync outlasts the picture
    if weak.max() < 1e-3:
        return None
    kick = rng.normal(0, 1, len(line)).astype(np.float32) * weak
    # AFC: a one-pole loop filter down the frame.
    dx = np.zeros(len(line), np.float32)
    acc = 0.0
    for y in range(len(line)):
        acc = acc * .75 + kick[y]
        dx[y] = acc
    dx *= amount * 1.6 * unit
    return dx if np.abs(dx).max() >= .3 else None


# --------------------------------------------------------------------------- #
#  TV receiver: weak or missing station
# --------------------------------------------------------------------------- #
def envelope_detect(video, level, rng, unit=1.0, bandwidth=1.0):
    """Send `video` (float, 0 = blanking, 1 = peak white; any shape (h, w)) over the air
    with RF noise `level` 0..1 and envelope-detect it again.

    NTSC is negative AM: sync tips are full carrier, blanking 75 % and peak white
    12.5 %. Tuner noise is added to the carrier, the envelope detector takes
    |carrier + noise| (Rician, so pure noise comes out Rayleigh), and AGC scales the
    result so the sync tips sit at full scale. `bandwidth` < 1 coarsens the grains."""
    h, w = video.shape
    level = float(np.clip(level, 0, 1))
    carrier = (1 - level) * (.75 - .625 * video)
    sigma = .38 * level
    grain = 1.0 / max(.15, bandwidth)
    i_n, q_n = (cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0),
                                 sigmaX=.8 * unit * grain, sigmaY=.01) for _ in range(2))
    norm = np.sqrt((i_n.std() ** 2 + q_n.std() ** 2) / 2) + 1e-6
    det = np.sqrt(np.square(carrier + sigma * i_n / norm) + np.square(sigma * q_n / norm))
    agc = 1 / ((1 - level) + 3.0 * sigma)                       # sync tips to full scale
    return (.75 - det * agc) / .625


COLOUR_KILL = .55          # RF noise level at which the colour killer drops the chroma


def receiver_snow(frame, level, rng, bandwidth=1.0, colour=0.0):
    """Receive `frame` over the air with RF noise `level` 0..1 (1 = no station at all).

    Snow comes out grey with white and black specks, as wide as the video bandwidth
    allows (see ``envelope_detect``). `colour` > 0 defeats the colour killer so the
    noise decodes as confetti."""
    if level < 1e-3:
        return frame
    h, w = frame.shape[:2]
    unit = _unit(w)
    level = float(np.clip(level, 0, 1))
    ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y = np.clip((ycc[..., 0] - 16) / 219, 0, 1)
    ycc[..., 0] = np.clip(16 + 219 * envelope_detect(y, level, rng, unit, bandwidth), 0, 255)
    # Colour killer drops out once the burst is buried; defeated, noise turns to confetti.
    keep = 1.0 if level < COLOUR_KILL else 0.0
    conf = rng.normal(0, 1, (max(2, h // 2), max(2, w // 6), 2)).astype(np.float32)
    conf = cv2.resize(conf, (w, h), interpolation=cv2.INTER_LINEAR) * 40 * level * colour
    for c in (1, 2):
        ycc[..., c] = 128 + (ycc[..., c] - 128) * keep + conf[..., c - 1]
    out = cv2.cvtColor(np.clip(ycc, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    if level > .6:
        # Sync is lost in the noise: the H oscillator wanders on its own.
        env = np.full((h, 1), max(0.0, 1 - level) * .8, np.float32)
        dx = afc_drift(env, rng, (level - .6) * 2.5, unit)
        if dx is not None:
            map_x = np.arange(w, dtype=np.float32)[None, :] - dx[:, None]
            map_y = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1)
            out = cv2.remap(out, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out
