"""Physically modelled NTSC composite + VHS tape signal ("accurate tape").

The picture is turned into the signal a VCR actually records, damaged where real
gear damages it, and decoded back:

    RGB -> YIQ -> band-limit chroma -> modulate onto the colour subcarrier
    -> over-the-air RF noise through an envelope detector (snow), head switching
    -> separate luma/chroma again (notch or box decoder)
    -> chroma noise, phase error, chroma delay
    -> tape: luma FM noise and clicks (tracking, head switch, worn tape), edge wave,
       luma/chroma bandwidth for SP/LP/EP, chroma lag, dropouts
    -> RGB

Because every artifact comes out of the same signal, they interact the way real
ones do: rainbow fringes on fine detail, colour lagging behind edges, smeared
chroma at slower tape speeds. Inspired by the stage list of ntsc-rs (credited
in THIRD_PARTY_NOTICES.md); the implementation here is our own.

Signal model: one image pixel is one sample at four times the colour subcarrier
(about 14.3 MHz) for 480-line video, so the subcarrier repeats every 4 pixels.
Taller video scales the bandwidths so 720p and 480p look alike.
"""
from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from .base import Effect
from .rf import (COLOUR_KILL, dropout_compensate, envelope_detect, fm_demod, fm_rows, afc_drift,
                 track_envelope)

SUBCARRIER = 315e6 / 88               # NTSC colour subcarrier, Hz
SAMPLE_RATE = SUBCARRIER * 4          # one pixel per sample at 480 lines

# BT.601 RGB <-> YIQ (rows act on R, G, B).
RGB_TO_YIQ = np.array([[0.299, 0.587, 0.114],
                       [0.596, -0.274, -0.322],
                       [0.211, -0.523, 0.312]], np.float32)
YIQ_TO_RGB = np.linalg.inv(RGB_TO_YIQ).astype(np.float32)

# Tape speed: luma bandwidth, chroma bandwidth, chroma lag in samples.
TAPE = {1: (2.4e6, 320e3, 4), 2: (1.9e6, 300e3, 5), 3: (1.4e6, 280e3, 6)}

# Subcarrier phase pattern over 4 samples: I on even samples, Q on odd ones.
_I_WAVE = np.array([1, 0, -1, 0], np.float32)
_Q_WAVE = np.array([0, 1, 0, -1], np.float32)


# --------------------------------------------------------------------------- #
#  Filters: classic analog responses, run along each scanline
# --------------------------------------------------------------------------- #
def _one_pole(cutoff, rate):
    """RC low-pass as (b, a)."""
    dt = 1 / rate
    rc = 1 / (2 * math.pi * cutoff)
    k = dt / (rc + dt)
    return np.array([k]), np.array([1.0, -(1 - k)])


def _butterworth(cutoff, rate):
    """Second-order Butterworth low-pass (bilinear transform), as (b, a)."""
    w = math.pi * min(2 * cutoff, rate) / rate
    alpha = math.sin(w) / math.sqrt(2)
    c = math.cos(w)
    norm = 1 + alpha
    b = np.array([(1 - c) / 2, 1 - c, (1 - c) / 2]) / norm
    a = np.array([1.0, -2 * c / norm, (1 - alpha) / norm])
    return b, a


def _notch(freq, q):
    """Band-stop at `freq` (fraction of Nyquist) with quality q, as (b, a)."""
    w = math.pi * freq
    beta = math.tan(w / q / 2)
    g = 1 / (1 + beta)
    b = np.array([g, -2 * math.cos(w) * g, g])
    a = np.array([1.0, -2 * math.cos(w) * g, 2 * g - 1])
    return b, a


@lru_cache(maxsize=64)
def _impulse(b, a, limit=1e-5, longest=4096):
    """Impulse response of a causal recursive filter, cut where it has decayed."""
    b, a = np.asarray(b), np.asarray(a)
    x = np.zeros(longest); x[0] = 1
    y = np.zeros(longest)
    for n in range(longest):
        acc = sum(b[k] * x[n - k] for k in range(len(b)) if n >= k)
        acc -= sum(a[k] * y[n - k] for k in range(1, len(a)) if n >= k)
        y[n] = acc
        if n > 8 and abs(y[n]) < limit and abs(y[n - 1]) < limit:
            return tuple(y[:n + 1].astype(np.float32))
    return tuple(y.astype(np.float32))


def run_rows(plane, ba, *, start="zero", delay=0, mix=1.0):
    """Filter every scanline left to right; `mix` blends filtered and dry signal.

    start: what the filter assumes was left of the picture ("zero" or "edge").
    delay: shift the output left by this many samples (to undo filter lag).
    mix:   1 = filtered; negative values push away from the filtered signal (sharpen).
    """
    b, a = ba
    h = np.asarray(_impulse(tuple(np.round(b, 12)), tuple(np.round(a, 12))), np.float32)
    n = len(h)
    if start == "zero":
        left = np.zeros((plane.shape[0], n - 1), np.float32)
    else:
        # Repeat the first whole subcarrier cycle so a modulated signal keeps its phase.
        cycle = plane[:, :4] if plane.shape[1] >= 4 else plane[:, :1]
        reps = -(-(n - 1) // cycle.shape[1])
        left = np.tile(cycle, (1, reps))[:, cycle.shape[1] * reps - (n - 1):]
    right = np.repeat(plane[:, -1:], delay, axis=1)
    padded = np.ascontiguousarray(np.hstack([left, plane, right]))
    # filter2D correlates; reverse the response and anchor at its end for a causal filter.
    out = cv2.filter2D(padded, -1, h[::-1][None, :], anchor=(n - 1, 0),
                       borderType=cv2.BORDER_CONSTANT)
    out = out[:, n - 1 + delay:n - 1 + delay + plane.shape[1]]
    return plane + mix * (out - plane) if mix != 1.0 else out


def shift_rows(plane, shifts, edge=False):
    """Slide each row right by its own (fractional) number of samples."""
    h, w = plane.shape
    map_x = np.arange(w, dtype=np.float32)[None, :] - np.asarray(shifts, np.float32)[:, None]
    map_y = np.broadcast_to(np.arange(h, dtype=np.float32)[:, None], (h, w))
    border = cv2.BORDER_REPLICATE if edge else cv2.BORDER_CONSTANT
    return cv2.remap(plane, np.ascontiguousarray(map_x), np.ascontiguousarray(map_y),
                     cv2.INTER_LINEAR, borderMode=border, borderValue=0)


def smooth_noise(rng, rows, width, frequency, octaves=1):
    """Band-limited noise along each row, roughly in [-1, 1], with its own phase per row."""
    total = np.zeros((rows, width), np.float32)
    amp, freq, norm = 1.0, frequency, 0.0
    for _ in range(max(1, octaves)):
        knots = max(2, int(width * freq) + 3)
        coarse = rng.uniform(-1, 1, (rows, knots)).astype(np.float32)
        fine = cv2.resize(coarse, (int(knots / max(freq, 1e-4)), rows), interpolation=cv2.INTER_CUBIC)
        start = rng.integers(0, max(1, fine.shape[1] - width), rows)
        idx = np.clip(start[:, None] + np.arange(width)[None, :], 0, fine.shape[1] - 1)
        total += amp * np.take_along_axis(fine, idx, axis=1)
        norm += amp; amp *= .5; freq *= 2
    return total / norm


# --------------------------------------------------------------------------- #
#  The effect
# --------------------------------------------------------------------------- #
class NTSCSignalEffect(Effect):
    """NTSC composite and VHS tape artifacts from a simulated signal path."""
    id = "ntsc"
    label = "NTSC / VHS SIGNAL (ACCURATE)"
    stage = "tape"

    def build(self, params):
        self.param("amount", "Mix", 1)
        self.choice("tape", "Tape speed", ("Off (broadcast)", "SP", "LP", "EP"), 2)
        self.choice("fields", "Fields", ("Interlaced", "One field (line-doubled)", "Progressive"))
        self.choice("decoder", "Chroma decoder", ("Notch", "Box", "Line comb"))
        self.param("noise", "Signal noise", .05, 0, 1)
        self.param("snow", "Snow", .02, 0, 1)
        self.param("sharpen", "Pre-emphasis", 1, 0, 3)
        self.param("smear", "Luma smear", .5, 0, 1)
        self.param("ringing", "Ringing", 0, 0, 1)
        self.param("head_switch", "Head switch lines", 8, 0, 48, step=1)
        self.param("head_shift", "Head switch shift", 72, -150, 150)
        self.param("tracking", "Tracking lines", 0, 0, 240, step=1)
        self.param("tracking_wave", "Tracking wave", 15, 0, 60)
        self.param("tracking_snow", "Tracking snow", .25, 0, 1)
        self.param("chroma_noise", "Chroma noise", .1, 0, 1)
        self.param("luma_noise", "Luma noise", .01, 0, 1)
        self.param("phase_error", "Hue error", 0, -.5, .5)
        self.param("phase_noise", "Hue noise", .001, 0, .5)
        self.param("chroma_lag", "Chroma lag px", 0, -40, 40)
        self.param("chroma_drop", "Chroma lag lines", 0, -10, 10, step=1)
        self.param("edge_wave", "Edge wave", .5, 0, 4)
        self.param("edge_speed", "Edge wave speed", 4, 0, 20)
        self.param("chroma_loss", "Colour dropouts", .0005, 0, .2)
        self.param("tape_sharpen", "Tape sharpen", .25, 0, 1)
        self.param("vertical_blend", "Chroma line blend", 1, 0, 1, step=1)
        self.param("seed", "Seed", 0, 0, 9999, step=1)
        # NTSC carries 480 picture lines; running the signal there is authentic and fast.
        self.choice("resolution", "Signal lines", ("480 (NTSC, fast)", "Full frame"))

    # ------------------------------------------------------------------ #
    def process(self, frame, t, v):
        if v["amount"] <= 0:
            return frame
        source = frame
        h0, w0 = frame.shape[:2]
        if int(v["resolution"]) == 0 and h0 > 480:
            frame = cv2.resize(frame, (max(4, round(w0 * 480 / h0)), 480), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
        frame_no = int(math.floor(t * 30000 / 1001 + 1e-6))   # NTSC frame rate
        scale = h / 480                                         # bandwidths act like 480i at any size
        fields = int(v["fields"])
        yiq = cv2.transform(frame[..., ::-1].astype(np.float32) / 255, RGB_TO_YIQ)
        if fields == 2:
            out = self._signal(yiq, v, frame_no, scale, 0)
        else:
            out = np.empty_like(yiq)
            for parity in (0, 1):
                if fields == 1 and parity != frame_no % 2:
                    continue
                part = self._signal(yiq[parity::2], v, frame_no * 2 + parity, scale, parity)
                if fields == 1:      # one field shown on both lines of each pair
                    out[0::2] = part[:len(out[0::2])]
                    out[1::2] = part[:len(out[1::2])]
                else:
                    out[parity::2] = part
        rgb = cv2.transform(out, YIQ_TO_RGB)
        bgr = np.clip(rgb[..., ::-1] * 255 + .5, 0, 255).astype(np.uint8)
        if bgr.shape[:2] != (h0, w0):
            bgr = cv2.resize(bgr, (w0, h0), interpolation=cv2.INTER_LINEAR)
        return cv2.addWeighted(source, 1 - v["amount"], bgr, v["amount"], 0) if v["amount"] < 1 else bgr

    def _rng(self, v, stage, field_no):
        return np.random.default_rng([int(v["seed"]), stage, field_no])

    def _signal(self, yiq, v, field_no, scale, parity):
        y = np.ascontiguousarray(yiq[..., 0]); i = np.ascontiguousarray(yiq[..., 1])
        q = np.ascontiguousarray(yiq[..., 2])
        h, w = y.shape
        rate = SAMPLE_RATE * scale
        lowpass = _butterworth

        # ---- broadcast encoder: limit chroma bandwidth, then modulate
        y = run_rows(y, _notch(.5, 2), start="edge")
        i = run_rows(i, lowpass(1.3e6, rate), start="edge", delay=2)
        q = run_rows(q, lowpass(0.6e6, rate), start="edge", delay=4)
        if v["sharpen"] > 0:
            # Pre-emphasis lifts luma detail only; boosting the whole composite signal
            # would also boost the colour subcarrier and oversaturate the picture.
            y = run_rows(y, _one_pole(SUBCARRIER / 2, rate), start="edge", mix=-v["sharpen"])
        phase = self._phase(h, w, field_no)
        signal = y + i * _I_WAVE[phase] + q * _Q_WAVE[phase]

        # ---- damage on the composite signal
        tape = int(v["tape"])
        unit = max(1.0, w / 640)
        # Over the air: the composite rides a negative-AM carrier with tuner noise (rf.py).
        # Without a tape in the path, Signal noise is weak reception too.
        air = min(1.0, v["snow"] + (0 if tape else v["noise"] * .5))
        if air > 1e-3:
            signal = envelope_detect(signal, air, self._rng(v, 2, field_no), unit)
        if v["head_switch"] >= 1:
            signal = self._head_switch(signal, v, field_no, scale)
        if v["tracking"] >= 1:
            signal = self._tracking(signal, v, field_no, scale)

        # ---- receiver decoder: pull luma and chroma apart again
        y, i, q = self._decode(signal, int(v["decoder"]), phase, h, w)
        if air >= COLOUR_KILL:
            i = np.zeros_like(i); q = np.zeros_like(q)          # colour killer: burst is lost
        if v["smear"] > 0:
            y = run_rows(y, _one_pole(2 ** (-4 * v["smear"]) * .25, scale), start="edge")
        if v["ringing"] > 0:
            y = run_rows(y, _notch(min(1, .45 / scale), 4), start="edge", delay=1, mix=v["ringing"] * 4)
        if v["luma_noise"] > 0:
            y = y + smooth_noise(self._rng(v, 3, field_no), h, w, .5 / scale) * v["luma_noise"] * .25
        if v["chroma_noise"] > 0:
            n = v["chroma_noise"] * .25
            i = i + smooth_noise(self._rng(v, 4, field_no), h, w, .05 / scale, 2) * n
            q = q + smooth_noise(self._rng(v, 5, field_no), h, w, .05 / scale, 2) * n
        turn = np.full(h, v["phase_error"], np.float32)
        if v["phase_noise"] > 0:
            turn += (self._rng(v, 6, field_no).random(h).astype(np.float32) - .5) * 2 * v["phase_noise"]
        if np.any(turn):
            cos, sin = np.cos(turn * 2 * math.pi)[:, None], np.sin(turn * 2 * math.pi)[:, None]
            i, q = i * cos - q * sin, i * sin + q * cos
        if v["chroma_lag"] or v["chroma_drop"]:
            i, q = (self._lag(p, v["chroma_lag"] * scale, int(round(v["chroma_drop"] * scale))) for p in (i, q))

        # ---- the tape itself
        # Tape noise, mistracking and the head switch come from the luma FM channel.
        y, i, q = self._fm_playback(y, i, q, v, field_no, scale, unit, tape)
        if tape:
            if v["edge_wave"] > 0:
                wave = smooth_noise(self._rng(v, 7, 0), 1, h + 64, .05 / scale, 2)[0]
                drift = int(field_no * v["edge_speed"]) % 64
                shifts = wave[drift:drift + h] * v["edge_wave"] * 4 * scale
                y, i, q = (shift_rows(p, shifts, edge=True) for p in (y, i, q))
            luma_cut, chroma_cut, lag = TAPE[tape]
            y = run_rows(y, lowpass(luma_cut, rate), start="edge")
            i = run_rows(i, lowpass(chroma_cut, rate), start="edge", delay=int(round(lag * scale)))
            q = run_rows(q, lowpass(chroma_cut, rate), start="edge", delay=int(round(lag * scale)))
            y = run_rows(y, _one_pole(luma_cut, rate), start="edge", mix=-1.6)
            if v["chroma_loss"] > 0:
                gone = self._rng(v, 8, field_no).random(h) < v["chroma_loss"]
                i[gone] = 0; q[gone] = 0
            if v["tape_sharpen"] > 0:
                y = run_rows(y, lowpass(luma_cut, rate), start="edge", mix=-v["tape_sharpen"] * 2)
        if v["vertical_blend"] >= .5:
            i = (i + np.vstack([np.zeros((1, w), np.float32), i[:-1]])) * .5
            q = (q + np.vstack([np.zeros((1, w), np.float32), q[:-1]])) * .5
        i = run_rows(i, lowpass(1.3e6, rate), start="edge", delay=2)
        q = run_rows(q, lowpass(0.6e6, rate), start="edge", delay=4)
        return np.dstack([y, i, q])

    # ------------------------------------------------------------------ #
    @staticmethod
    def _phase(h, w, field_no):
        """Subcarrier phase per sample: it advances half a cycle every line and frame."""
        line_offset = ((field_no + 2 * np.arange(h)) & 2)[:, None]
        return (np.arange(w)[None, :] + line_offset) & 3

    @staticmethod
    def _decode(signal, mode, phase, h, w):
        if mode == 2 and h > 2:
            # Line comb: the subcarrier flips between lines, so averaging cancels it.
            above = np.vstack([signal[1:2], signal[:-1]])
            y = (signal + above) * .5
        elif mode == 1:
            k = np.array([[.25, .25, .25, .25]], np.float32)
            y = cv2.filter2D(signal, -1, k, anchor=(1, 0), borderType=cv2.BORDER_REPLICATE)
        else:
            y = run_rows(signal, _notch(.5, 2), start="edge")
        chroma = y - signal
        # Demodulate with the inverse carrier, weighting each sample's neighbours by half.
        i_inv, q_inv = -_I_WAVE[phase], -_Q_WAVE[phase]
        ci, cq = chroma * i_inv, chroma * q_inv
        k = np.array([[.5, 1, .5]], np.float32)
        i = cv2.filter2D(ci, -1, k, borderType=cv2.BORDER_CONSTANT)
        q = cv2.filter2D(cq, -1, k, borderType=cv2.BORDER_CONSTANT)
        return y, i, q

    def _head_switch(self, signal, v, field_no, scale):
        h, w = signal.shape
        lines = min(h, int(round(v["head_switch"] * scale)))
        rng = self._rng(v, 9, field_no)
        depth = np.arange(lines, 0, -1, dtype=np.float32)           # top of band = deepest
        shifts = (v["head_shift"] * (depth / max(1, lines)) ** 1.5 + rng.random(lines) - .5) * scale
        out = signal.copy()
        band = shift_rows(signal[h - lines:], shifts)
        # The switch lands part-way across the first damaged line, with a bright transient.
        start = int(w * np.clip(.95 + (rng.random() - .5) * .03, 0, 1))
        band[0, :start] = signal[h - lines, :start]
        span = int(math.ceil(16 * scale))
        x = np.arange(min(span, w - start), dtype=np.float32)
        band[0, start:start + len(x)] += (1 - x / span) ** 3 * (rng.random() + .5) * .5
        out[h - lines:] = band
        return out

    def _tracking(self, signal, v, field_no, scale):
        """Mistracking's sync wobble over the bottom lines; its noise comes from _fm_playback."""
        h, w = signal.shape
        lines = min(h, int(round(v["tracking"] * scale)))
        rng = self._rng(v, 10, field_no)
        ramp = np.arange(lines, dtype=np.float32) / max(1, lines)  # worse toward the bottom
        wobble = smooth_noise(rng, 1, lines, .5)[0]
        out = signal.copy()
        out[h - lines:] = shift_rows(signal[h - lines:], wobble * ramp * v["tracking_wave"] * .25 * scale)
        return out

    def _fm_playback(self, y, i, q, v, field_no, scale, unit, tape):
        """Luma FM playback with the RF level the head actually reads.

        Mistracking runs the head toward the neighbouring track over the bottom Tracking
        lines (Tracking snow sets how far), the head switch dips the RF where the heads
        hand over, and Signal noise lowers the tape's carrier-to-noise ratio. Clicks and
        grain are added before the tape's luma filters, so they take its bandwidth."""
        h, w = y.shape
        env = np.ones((h, 1), np.float32)
        lines = min(h, int(round(v["tracking"] * scale)))
        if lines >= 1:
            rng = self._rng(v, 11, field_no)
            ramp = np.arange(lines, dtype=np.float32) / max(1, lines)
            wander = .06 * smooth_noise(rng, 1, lines, .5)[0]
            reach = .3 + .75 * v["tracking_snow"]
            env[h - lines:, 0] = track_envelope(.35 + reach * ramp ** 1.5 + wander)
        switch = min(h, int(round(v["head_switch"] * scale)))
        if switch >= 1:
            at = h - switch
            rows = np.arange(h, dtype=np.float32)
            dip = np.exp(-np.square((rows - at) / max(1.0, 1.2 * scale)))
            env[:, 0] *= 1 - .85 * dip
            env[at:, 0] *= .8
        if tape and v["noise"] > 0:
            # The colour-under channel is noisy on worn tape too: slow, streaky colour noise.
            n = v["noise"] * .12
            i = i + smooth_noise(self._rng(v, 13, field_no), h, w, .03 / scale, 2) * n
            q = q + smooth_noise(self._rng(v, 14, field_no), h, w, .03 / scale, 2) * n
        if not tape and env.min() > .97:
            return y, i, q
        cnr = 26 - 22 * v["noise"] if tape else 40              # Signal noise: worn, cheap tape
        rho, clicks, todo = fm_rows(env, cnr)
        if not len(todo):
            return y, i, q
        rng = self._rng(v, 12, field_no)
        y = y.copy(); i = i.copy(); q = q.copy()
        # The colour-under carrier is low in frequency, so azimuth loss barely touches it and
        # the colour holds on off-track until the RF is nearly gone.
        luma, keep, chroma = fm_demod(y[todo] * 255, rho[todo], clicks[todo], rng, cnr, unit, kill=.04,
                                      grain=1.35)          # EP/LP filtering softens it; worn tape is grainy
        y[todo] = luma / 255
        i[todo] *= chroma; q[todo] *= chroma
        dropout_compensate([y, i, q], env, rng, todo, unit, max(1, round(4 * scale)))
        dx = afc_drift(env, rng, 1.0, unit)
        if dx is not None:
            y, i, q = (shift_rows(p, dx, edge=True) for p in (y, i, q))
        return y, i, q

    @staticmethod
    def _lag(plane, px, lines):
        if px:
            plane = shift_rows(plane, np.full(plane.shape[0], px, np.float32))
        if lines:
            plane = np.roll(plane, lines, axis=0)
            if lines > 0:
                plane[:lines] = 0
            else:
                plane[lines:] = 0
        return plane
