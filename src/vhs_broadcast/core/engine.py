"""PlaybackEngine: a QThread that decodes, warps, processes and emits frames.

Modes:
  * LIVE     -- user drives controls; the Recorder captures changes.
  * PLAYBACK -- the Timeline drives parameter base values from the recording.

Fixes/behaviours:
  * While PAUSED the current source frame is re-processed every tick, so OSD
    text and any control change update the preview immediately (no need to play).
  * Motor wow/flutter warps source-frame selection (dup on slow, skip on fast).
  * Optional audio playback runs alongside; the same speed curve wobbles it.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QMutex, QMutexLocker, QThread, Signal
from PySide6.QtGui import QImage

from ..config import PREVIEW_MAX_WIDTH
from .automation import Recorder, Timeline
from .motor import motor_speed
from .parameters import ParameterSet
from .pipeline import Pipeline, VideoSource, scale_to_width


class Mode(Enum):
    LIVE = "live"
    PLAYBACK = "playback"


def to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    rgb = np.ascontiguousarray(rgb)
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class PlaybackEngine(QThread):
    frameReady = Signal(QImage)
    positionChanged = Signal(float, int)
    finished_playing = Signal()

    def __init__(self, params, pipeline, recorder, timeline):
        super().__init__()
        self.params = params
        self.pipeline = pipeline
        self.recorder = recorder
        self.timeline = timeline

        self._source: Optional[VideoSource] = None
        self._mutex = QMutex()
        self._running = False
        self._playing = False
        self._mode = Mode.LIVE
        self._seek_to: Optional[int] = None
        self._out_index = 0        # output frame counter (defines time t)
        self._src_pos = 0.0        # fractional source-frame read head
        self._raw: Optional[np.ndarray] = None
        self._last_frame: Optional[np.ndarray] = None
        self._audio_clock = None   # callable -> played source seconds, or None
        self._av_offset = 0.0      # seconds; manual A/V sync trim
        self._shown = -1           # last rendered output index (audio-master)
        self._frame_sink = None    # optional callable(bgr) for virtual cam / rec
        self._use_gpu = False
        self._gl = None            # lazily created in the engine thread
        self._text_eff = next(
            (e for e in getattr(pipeline, "effects", [])
             if getattr(e, "id", None) == "text"), None)
        self._downsample_eff = next(
            (e for e in getattr(pipeline, "effects", [])
             if getattr(e, "id", None) == "downsample"), None)
        self._fps_bucket = -1
        self._held_frame = None
        # --- live performance stats (polled by the UI) ---
        self._fps = 0.0
        self._last_emit = 0.0
        self._desync = 0.0
        self._proc_size = (0, 0)
        self._preview_w = PREVIEW_MAX_WIDTH   # 0 = full source resolution
        self._preview_scale = 1.0             # fraction of the (capped) width
        self._drop_frames = True    # stay time-locked (drop) vs show every frame
        self._preview = True        # emit frames to the on-screen preview
        self._wall_start = None     # wall-clock anchor for free-run drop mode
        self._frames_shown = 0      # rendered frames since play
        self._frames_dropped = 0    # skipped frames since play
        self._acct_prev = 0         # last accounted output index

    def set_drop_frames(self, on: bool):
        """On: stay locked to the master clock, dropping frames that can't be
        rendered in time. Off: show every frame in sequence (may drift)."""
        self._drop_frames = bool(on)

    def set_preview(self, on: bool):
        """Turn the on-screen preview on/off. Off skips the QImage conversion and
        GUI paint, leaving more headroom for the virtual camera / recorder."""
        self._preview = bool(on)

    def set_preview_width(self, w: int):
        """Cap the preview/processing width (0 = full source resolution).
        Lower = faster on CPU; higher is where the GPU backend pays off."""
        self._preview_w = int(w) or 0

    def set_preview_scale(self, frac: float):
        """Scale factor applied on top of the width cap (fraction of source /
        of the cap). 1.0 = no scaling."""
        self._preview_scale = max(0.05, min(1.0, float(frac)))

    def _effective_width(self, src_w: int) -> int:
        """Target processing width for a source of width src_w, from the width
        cap and scale. Never upscales past the source."""
        base = self._preview_w if self._preview_w else src_w
        w = int(round(base * self._preview_scale))
        return max(16, min(w, src_w))

    def stats(self) -> dict:
        total = self._frames_shown + self._frames_dropped
        return {
            "fps": self._fps,
            "backend": "GPU" if (self._use_gpu and self._gl is not None) else "CPU",
            "desync_ms": self._desync * 1000.0,
            "width": self._proc_size[0],
            "height": self._proc_size[1],
            "playing": self._playing,
            "dropped": self._frames_dropped,
            "drop_pct": (self._frames_dropped / total * 100.0) if total else 0.0,
            "decode": getattr(self._source, "decoder", "?") if self._source else "?",
        }

    def set_gpu(self, on: bool):
        """Enable the GPU live-preview backend (CPU stays the offline renderer)."""
        self._use_gpu = bool(on)

    def _gpu_process(self, work, t):
        try:
            if self._gl is None:
                from ..gpu.glrender import GLRenderer
                self._gl = GLRenderer()
            order = [e.id for e in self.pipeline.effects]
            osd_items = None
            font = None
            if (self._text_eff is not None
                    and self._text_eff.enabled_param().value >= 0.5):
                h, w = work.shape[:2]
                osd_items = self._text_eff.build_items(t, w, h)
                font = self._text_eff.font_path
            return self._gl.render_chain(work, self.params, t, order,
                                         osd_items, font)
        except Exception:  # noqa: BLE001 -> never let GPU break playback
            self._use_gpu = False
            return self.pipeline.process(work.copy(), t)

    def set_frame_sink(self, fn):
        """Install a callable that receives each new processed BGR frame (for
        the virtual camera / recorder), or None to detach."""
        self._frame_sink = fn

    def set_av_offset(self, seconds: float):
        """Manual A/V trim: >0 nudges the video ahead (use if video lags the
        audio), <0 nudges it back (if video leads)."""
        self._av_offset = float(seconds)

    def set_audio_clock(self, fn):
        """Install a callable returning the audio's played source-time (seconds)
        while it's playing, else None. When present, video slaves to it."""
        self._audio_clock = fn

    def _advance_to(self, src, target: int):
        """Return the frame for output index `target`, reading forward, or
        seeking on a big jump. None at end of stream."""
        cur = src.index - 1
        if target == cur:
            return self._raw
        if target < cur or target - cur > 8:
            src.seek_frame(max(0, target))
            return src.read()
        frame = self._raw
        while (src.index - 1) < target:
            f = src.read()
            if f is None:
                return None
            frame = f
        return frame

    # ---- source -----------------------------------------------------------
    def set_source(self, source: VideoSource, audio_path: Optional[str] = None):
        with QMutexLocker(self._mutex):
            if self._source:
                self._source.release()
            self._source = source
            self._out_index = 0
            self._src_pos = 0.0
            self._seek_to = 0
            self._raw = None
            self._wall_start = None
            self._fps_bucket = -1
            self._held_frame = None

    @property
    def fps(self) -> float:
        return self._source.fps if self._source else 30.0

    @property
    def duration(self) -> float:
        return self._source.duration if self._source else 0.0

    # ---- transport --------------------------------------------------------
    def set_mode(self, mode: Mode):
        self._mode = mode

    def _fps_downsample(self, frame, t):
        """Hold the last output frame within an FPS bucket, so the content
        updates at a reduced rate (temporal 'downsample the FPS'). Driven by the
        downsample effect's FPS% param; works on both CPU and GPU paths."""
        ds = self._downsample_eff
        if ds is None or ds.enabled_param().value < 0.5:
            return frame
        frac = self.params.get("downsample.fps").evaluate(t) / 100.0
        if frac >= 0.999:
            return frame
        ofps = max(1.0, frac * self.fps)
        b = int(t * ofps)
        if (b == self._fps_bucket and self._held_frame is not None
                and self._held_frame.shape == frame.shape):
            return self._held_frame
        self._fps_bucket = b
        self._held_frame = frame
        return frame

    def current_time(self) -> float:
        """Source-time (seconds) of the frame currently shown."""
        return self._out_index / max(1.0, self.fps)

    def play(self):
        self._playing = True
        self._frames_shown = 0          # fresh drop accounting per play
        self._frames_dropped = 0
        self._acct_prev = self._out_index

    def pause(self):
        self._playing = False
        self._wall_start = None

    def seek(self, index: int):
        with QMutexLocker(self._mutex):
            self._seek_to = index
            self._wall_start = None
            self._fps_bucket = -1
            self._held_frame = None

    def seek_seconds(self, t: float):
        self.seek(int(t * self.fps))

    def stop(self):
        self._running = False
        self._playing = False
        self.wait(2000)

    # ---- main loop --------------------------------------------------------
    def run(self):
        self._running = True
        period = 1.0 / max(1.0, self.fps)
        _winmm = None
        try:                                   # 1ms sleep resolution on Windows
            import ctypes
            import sys
            if sys.platform.startswith("win"):
                _winmm = ctypes.windll.winmm
                _winmm.timeBeginPeriod(1)
        except Exception:  # noqa: BLE001
            _winmm = None
        while self._running:
            start = time.perf_counter()

            with QMutexLocker(self._mutex):
                poll_only = False
                audio_master = False
                src = self._source
                did_seek = False
                if src is not None and self._seek_to is not None:
                    idx = max(0, min(self._seek_to, max(0, src.frame_count - 1)))
                    src.seek_frame(idx)
                    self._out_index = idx
                    self._src_pos = float(idx)
                    self._seek_to = None
                    self._raw = src.read()
                    did_seek = True
                    self._acct_prev = idx

                if src is None:
                    self.msleep(30)
                    continue

                if self._playing:
                    t = self._out_index / max(1.0, self.fps)
                    if getattr(src, "is_live", False):
                        # live capture: grab the latest frame, keep last on miss
                        f = src.read()
                        if f is not None:
                            self._raw = f
                        self._out_index += 1
                    else:
                        atime = self._audio_clock() if self._audio_clock else None
                        if atime is not None:
                            audio_master = True
                            # AUDIO IS MASTER: show the frame that matches what
                            # the speakers are playing (motor wobble already
                            # baked into the audio read head, so video follows).
                            fc = src.frame_count or 0
                            if self._drop_frames:
                                target = int(round((atime + self._av_offset)
                                                   * self.fps))
                            else:
                                target = self._out_index + 1   # every frame
                            if fc:
                                target = max(0, min(target, fc - 1))
                            frame = self._advance_to(src, target)
                            if frame is None:
                                self._playing = False
                                self.finished_playing.emit()
                                continue
                            self._raw = frame
                            self._out_index = target
                            if target == self._shown:
                                poll_only = True   # nothing new to render
                        else:
                            self._desync = 0.0
                            fc = src.frame_count or 0
                            if self._drop_frames:
                                # no audio: hold real time via wall clock,
                                # dropping frames the machine can't keep up with
                                audio_master = True
                                nowt = time.perf_counter()
                                if self._wall_start is None:
                                    self._wall_start = nowt
                                    self._wall_base = self._out_index
                                target = self._wall_base + int(round(
                                    (nowt - self._wall_start) * self.fps))
                                if fc:
                                    target = max(0, min(target, fc - 1))
                                frame = self._advance_to(src, target)
                                if frame is None:
                                    self._playing = False
                                    self.finished_playing.emit()
                                    continue
                                self._raw = frame
                                self._out_index = target
                                if target == self._shown:
                                    poll_only = True
                            else:
                                # show every frame in sequence (motor time-warp)
                                speed = motor_speed(self.params, t)
                                self._src_pos += speed
                                target = int(self._src_pos)
                                frame = self._raw
                                while src.index <= target:
                                    f = src.read()
                                    if f is None:
                                        frame = None
                                        break
                                    frame = f
                                if frame is None:
                                    self._playing = False
                                    self.finished_playing.emit()
                                    continue
                                self._raw = frame
                                self._out_index += 1
                elif self._raw is None:
                    # nothing to show yet
                    f = src.read()
                    if f is not None:
                        self._raw = f

                raw = self._raw
                if raw is None:
                    self.msleep(20)
                    continue

                if not poll_only:
                    t = self._out_index / max(1.0, self.fps)
                    src_w = raw.shape[1]
                    tw = self._effective_width(src_w)
                    work = scale_to_width(raw, tw) if tw < src_w else raw
                    self.recorder.set_time(t)
                    if self._mode == Mode.PLAYBACK:
                        self.timeline.apply(t, self.params)
                    if self._use_gpu:
                        processed = self._gpu_process(work, t)
                    else:
                        processed = self.pipeline.process(work.copy(), t)
                    processed = self._fps_downsample(processed, t)
                    self._last_frame = processed
                    self._shown = self._out_index
                    gap = self._out_index - self._acct_prev
                    if gap >= 1:
                        self._frames_shown += 1
                        if gap > 1:
                            self._frames_dropped += gap - 1   # skipped in between
                    self._acct_prev = self._out_index
                    self._proc_size = (processed.shape[1], processed.shape[0])
                    if self._frame_sink is not None:
                        try:
                            self._frame_sink(processed)
                        except Exception:  # noqa: BLE001
                            pass

            if not poll_only:
                now = time.perf_counter()
                if self._playing and self._last_emit:
                    dt = now - self._last_emit
                    if dt > 1e-6:
                        inst = 1.0 / dt
                        self._fps = inst if self._fps == 0 else \
                            self._fps * 0.85 + inst * 0.15
                elif not self._playing:
                    self._fps = 0.0
                self._last_emit = now
                # REAL A/V lag: how far the frame we're showing now is from what
                # the speakers are actually playing at this instant (negative =
                # video behind audio).
                if audio_master and self._audio_clock:
                    a_now = self._audio_clock()
                    self._desync = (t - a_now) if a_now is not None else 0.0
                elif not audio_master:
                    self._desync = 0.0
                if self._preview:
                    self._emit(processed)
                self.positionChanged.emit(t, self._out_index)

            if audio_master:
                self.msleep(4)       # poll audio often; render on frame boundary
            elif poll_only:
                self.msleep(5)
            elif self._playing:
                sleep = period - (time.perf_counter() - start)
                if sleep > 0:
                    self.msleep(int(sleep * 1000))
            else:
                self.msleep(33)      # ~30fps idle refresh so edits show live

        with QMutexLocker(self._mutex):
            if self._source:
                self._source.release()
            if self._gl is not None:
                self._gl.release()
                self._gl = None
        if _winmm is not None:
            try:
                _winmm.timeEndPeriod(1)
            except Exception:  # noqa: BLE001
                pass

    def _emit(self, bgr: np.ndarray):
        self.frameReady.emit(to_qimage(bgr))
