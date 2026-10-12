"""Audio reactivity: turn the loaded audio into per-frame band envelopes that
any Parameter can react to, exactly like the LFO modulator.

Determinism: the whole envelope is precomputed once from the decoded audio and
sampled as a pure function of output time t. So an audio-reactive take renders
bit-identically to what you previewed -- same guarantee as the sine/random LFO.

Bands:
  low   ~20-250 Hz   (kick / bass)
  mid   ~250-2000 Hz (vocals / snare)
  high  ~2k-16k Hz   (hats / cymbals)
  level full-band RMS (overall loudness)

Spectral ranges share one loudness reference. Causal FFT power supports custom
cutoffs without decoding again. Threshold gates and linear attack/release ramps
are cached by settings and sampled by time, so seeking never changes the result.
The two smoothed envelopes remain available for older gain-based takes.
"""
from __future__ import annotations

from typing import Optional
from functools import lru_cache

import numpy as np

from . import parameters as _P

BANDS = ("low", "mid", "high", "level")
_BAND_HZ = {"low": (20, 250), "mid": (250, 2000), "high": (2000, 16000)}


def _one_pole(x: np.ndarray, a: float) -> np.ndarray:
    """Causal one-pole smoother y[i] = a*x[i] + (1-a)*y[i-1] (vectorised-ish)."""
    y = np.empty_like(x)
    acc = 0.0
    for i in range(x.shape[0]):
        acc = a * x[i] + (1 - a) * acc
        y[i] = acc
    return y


class AudioAnalysis:
    """Precomputed band envelopes aligned to video frames."""

    def __init__(self, samples: np.ndarray, sr: int, fps: float, n_frames: int):
        if samples.ndim == 2:
            mono = samples.mean(axis=1)
        else:
            mono = samples
        self.fps = float(fps)
        self.n = max(1, int(n_frames))
        win = 2048
        window = np.hanning(win).astype(np.float32)
        freqs = np.fft.rfftfreq(win, 1.0 / sr)
        self.freqs = freqs
        self.power = np.zeros((self.n, len(freqs)), np.float32)
        band_bins = {b: np.where((freqs >= lo) & (freqs < hi))[0]
                     for b, (lo, hi) in _BAND_HZ.items()}

        raw = {b: np.zeros(self.n, np.float32) for b in BANDS}
        for f in range(self.n):
            centre = int(round(f * sr / self.fps))
            start = centre - win
            seg = mono[max(0, start):max(0, centre)]
            if start < 0:
                seg = np.pad(seg, (-start, 0))
            if seg.shape[0] < win:
                seg = np.pad(seg, (0, win - seg.shape[0]))
            raw["level"][f] = float(np.sqrt(np.mean(seg * seg)))
            spec = np.abs(np.fft.rfft(seg * window))
            self.power[f] = spec ** 2
            for b in ("low", "mid", "high"):
                bins = band_bins[b]
                raw[b][f] = float(np.sqrt(np.mean(spec[bins] ** 2))) \
                    if bins.size else 0.0

        # normalise: the 3 spectral bands share one reference so a quiet band
        # reads low relative to a loud one (real spectral balance); level is
        # self-referential. Then smooth two ways.
        spec_vals = np.concatenate([raw["low"], raw["mid"], raw["high"]])
        spec_ref = np.percentile(spec_vals, 97) or (spec_vals.max() or 1.0)
        lvl_ref = np.percentile(raw["level"], 97) or (raw["level"].max() or 1.0)
        self.spec_ref = float(spec_ref)
        self.level = np.clip(raw["level"] / max(float(lvl_ref), 1e-9), 0, 1)
        # Instance-local bounded caches: changing settings never retains old clips.
        self.filtered = lru_cache(maxsize=16)(self._filtered)
        self.trigger = lru_cache(maxsize=32)(self._trigger)
        refs = {"low": spec_ref, "mid": spec_ref, "high": spec_ref,
                "level": lvl_ref}
        self.fast = {}
        self.slow = {}
        for b in BANDS:
            v = np.clip(raw[b] / (refs[b] + 1e-9), 0.0, 1.3)
            self.fast[b] = _one_pole(v, 0.5)
            self.slow[b] = _one_pole(v, 0.08)

    def _filtered(self, band, low_hz, high_hz):
        if band == "level":
            return self.level
        bins = (self.freqs >= low_hz) & (self.freqs < high_hz)
        if not bins.any() or low_hz >= high_hz:
            return np.zeros(self.n, np.float32)
        energy = np.sqrt(self.power[:, bins].mean(axis=1))
        return np.clip(energy / max(self.spec_ref, 1e-9), 0, 1)

    def _trigger(self, band, low_hz, high_hz, threshold, ramp):
        level = self.filtered(band, low_hz, high_hz)
        active = (level > 0) & (level >= threshold)
        if ramp <= 0:
            return active.astype(np.float32)
        result = np.empty(self.n, np.float32)
        value = 0.0
        step = 1.0 / max(1.0, ramp * self.fps)
        for i, on in enumerate(active):
            value = min(1.0, value + step) if on else max(0.0, value - step)
            result[i] = value
        return result

    def selected_level(self, band, t, low_hz, high_hz):
        i = min(self.n - 1, max(0, int(t * self.fps)))
        return float(self.filtered(band, low_hz, high_hz)[i])

    def activation(self, band, t, low_hz, high_hz, threshold, ramp):
        i = min(self.n - 1, max(0, int(t * self.fps)))
        return float(self.trigger(band, low_hz, high_hz, threshold, ramp)[i])

    def sample(self, band: str, t: float, smooth: float) -> float:
        arrf = self.fast.get(band)
        arrs = self.slow.get(band)
        if arrf is None:
            return 0.0
        pos = t * self.fps
        i0 = int(pos)
        if i0 < 0:
            return float(arrf[0] * (1 - smooth) + arrs[0] * smooth)
        if i0 >= self.n - 1:
            f, s = arrf[-1], arrs[-1]
            return float(f * (1 - smooth) + s * smooth)
        frac = pos - i0
        f = arrf[i0] * (1 - frac) + arrf[i0 + 1] * frac
        s = arrs[i0] * (1 - frac) + arrs[i0 + 1] * frac
        return float(f * (1 - smooth) + s * smooth)


# --- global provider the Parameter layer reads through -------------------- #
_ANALYSIS: Optional[AudioAnalysis] = None


def set_analysis(analysis: Optional[AudioAnalysis]) -> Optional[AudioAnalysis]:
    """Install (or clear) the active analysis; returns the previous one."""
    global _ANALYSIS
    prev = _ANALYSIS
    _ANALYSIS = analysis
    _P.set_audio_provider(_provider if analysis is not None else None)
    return prev


def current() -> Optional[AudioAnalysis]:
    return _ANALYSIS


def _provider(band: str, t: float, smooth: float, low_hz=None, high_hz=None,
              threshold=0.35, ramp=0.15) -> float:
    if _ANALYSIS and low_hz is not None:
        return _ANALYSIS.activation(band, t, low_hz, high_hz, threshold, ramp)
    return _ANALYSIS.sample(band, t, smooth) if _ANALYSIS else 0.0


def levels_at(t: float, smooth: float = 0.25) -> dict:
    """For the on-screen meter."""
    if _ANALYSIS is None:
        return {b: 0.0 for b in BANDS}
    return {b: _ANALYSIS.selected_level(b, t, *_BAND_HZ.get(b, (0, 20000)))
            for b in BANDS}


def build_analysis(audio_path: str, fps: float, n_frames: int
                   ) -> Optional[AudioAnalysis]:
    """Decode `audio_path` and build an analysis, or None if it has no audio."""
    from .audio import load_audio
    loaded = load_audio(audio_path)
    if not loaded:
        return None
    samples, sr = loaded
    return AudioAnalysis(samples, sr, fps, n_frames)
