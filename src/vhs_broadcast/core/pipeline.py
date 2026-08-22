"""Frame pipeline, video sources, and the deterministic offline renderer."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..effects.base import Effect
from .automation import Timeline
from .parameters import ParameterSet


class Pipeline:
    """Runs the ordered effect stack over a single frame at time t."""

    def __init__(self, effects: list[Effect]) -> None:
        self.effects = effects

    def process(self, frame: np.ndarray, t: float) -> np.ndarray:
        for eff in self.effects:
            frame = eff.apply(frame, t)
        return frame


class VideoSource:
    """Abstract source. Subclasses: FileSource; ScreenSource/WindowSource/
    DeviceSource (live). Live sources are infinite, non-seekable and real-time."""

    fps: float = 30.0
    frame_count: int = 0
    width: int = 0
    height: int = 0
    is_live: bool = False

    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def seek_frame(self, index: int) -> None:
        pass

    def release(self) -> None:
        pass

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


class _CvFileSource(VideoSource):
    """Fallback decoder using OpenCV (single-threaded ffmpeg)."""
    def __init__(self, path: str) -> None:
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"cannot open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.index = 0
        self.decoder = "CPU"

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.index += 1
        return frame

    def seek_frame(self, index: int) -> None:
        index = max(0, min(index, max(0, self.frame_count - 1)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        self.index = index

    def release(self) -> None:
        self.cap.release()


class _AvFileSource(VideoSource):
    """Decoder using PyAV. Tries the platform's GPU video decoder (D3D11VA/
    DXVA2/QSV/CUDA on Windows, VideoToolbox on macOS, VAAPI on Linux) and falls
    back to multithreaded (FRAME) software decode across all cores."""

    @staticmethod
    def _detect_hw(path):
        try:
            import av
            from av.codec.hwaccel import HWAccel
        except Exception:  # noqa: BLE001
            return None
        import sys
        if sys.platform.startswith("win"):
            pref = ["d3d11va", "dxva2", "cuda", "qsv", "amf"]
        elif sys.platform == "darwin":
            pref = ["videotoolbox"]
        else:
            pref = ["vaapi", "cuda", "qsv"]
        for dt in pref:
            try:
                hw = HWAccel(device_type=dt, allow_software_fallback=False)
                c = av.open(path, hwaccel=hw)
                next(c.decode(c.streams.video[0]))   # prove it really decodes
                c.close()
                return dt
            except Exception:  # noqa: BLE001
                continue
        return None

    def __init__(self, path: str) -> None:
        import av
        self.path = path
        self._hw = self._detect_hw(path)
        if self._hw:
            from av.codec.hwaccel import HWAccel
            hw = HWAccel(device_type=self._hw, allow_software_fallback=True)
            self._c = av.open(path, hwaccel=hw)
            self.decoder = f"GPU:{self._hw}"
        else:
            self._c = av.open(path)
            self.decoder = "CPU"
        self._s = self._c.streams.video[0]
        if not self._hw:
            try:
                self._s.thread_type = "FRAME"
                self._s.codec_context.thread_count = 0   # 0 = all cores
            except Exception:  # noqa: BLE001
                pass
        self.fps = float(self._s.average_rate) if self._s.average_rate else 30.0
        self.width = self._s.codec_context.width
        self.height = self._s.codec_context.height
        self._tb = self._s.time_base
        self._start = self._s.start_time or 0
        if self._s.frames:
            self.frame_count = self._s.frames
        elif self._c.duration:
            self.frame_count = int(round((self._c.duration / 1e6) * self.fps))
        else:
            self.frame_count = 0
        self.index = 0
        self._gen = self._c.decode(self._s)
        self._skip_to = None
        self._counter = 0

    def _index_of(self, frame):
        if frame.pts is None:
            self._counter += 1
            return self._counter
        return int(round(float(frame.pts - self._start) * float(self._tb)
                         * self.fps))

    def read(self):
        try:
            frame = next(self._gen)
        except StopIteration:
            return None
        if self._skip_to is not None:
            while frame.pts is not None and frame.pts < self._skip_to:
                try:
                    frame = next(self._gen)
                except StopIteration:
                    return None
            self._skip_to = None
        self.index = self._index_of(frame) + 1
        return np.ascontiguousarray(frame.to_ndarray(format="bgr24"))

    def seek_frame(self, index: int) -> None:
        if self.frame_count:
            index = max(0, min(index, self.frame_count - 1))
        else:
            index = max(0, index)
        ts = int(index / self.fps / float(self._tb)) + self._start
        try:
            self._c.seek(ts, stream=self._s, backward=True, any_frame=False)
        except Exception:  # noqa: BLE001
            self._c.seek(ts, backward=True)
        self._gen = self._c.decode(self._s)
        self._skip_to = ts
        self.index = index

    def release(self) -> None:
        try:
            self._c.close()
        except Exception:  # noqa: BLE001
            pass


class ReadAheadSource(VideoSource):
    """Wraps a decoder with a background thread that decodes ahead into a small
    buffer, so the render loop never blocks on decode (decode overlaps scaling,
    effects and display, and decode jitter is absorbed). Same interface as the
    inner source. All decoding happens on the worker thread only."""
    is_live = False

    def __init__(self, inner, ahead: int = 8):
        import queue
        import threading
        self._inner = inner
        self.decoder = getattr(inner, "decoder", "CPU")
        self.fps = inner.fps
        self.frame_count = inner.frame_count
        self.width = inner.width
        self.height = inner.height
        self.index = 0
        self._q = queue.Queue(maxsize=ahead)
        self._Empty = queue.Empty
        self._Full = queue.Full
        self._lock = threading.Lock()
        self._seek_to = None
        self._gen = 0
        self._eos_gen = -1
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _drain(self):
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except self._Empty:
                break

    def _run(self):
        import time as _t
        gen = 0
        while not self._stop:
            with self._lock:
                if self._seek_to is not None:
                    target = self._seek_to
                    self._seek_to = None
                    gen = self._gen
                    self._drain()
                    try:
                        self._inner.seek_frame(target)
                    except Exception:  # noqa: BLE001
                        pass
            try:
                f = self._inner.read()
            except Exception:  # noqa: BLE001
                f = None
            if f is None:                # end of stream: mark and idle
                with self._lock:
                    if self._gen == gen:
                        self._eos_gen = gen
                while not self._stop:
                    with self._lock:
                        if self._seek_to is not None or self._gen != gen:
                            break
                    _t.sleep(0.02)
                continue
            idx = self._inner.index - 1
            item = (gen, f, idx)
            while not self._stop:
                with self._lock:
                    if self._gen != gen:
                        break            # a seek happened -> drop this frame
                try:
                    self._q.put(item, timeout=0.05)
                    break
                except self._Full:
                    continue

    def read(self):
        import time as _t
        while not self._stop:
            with self._lock:
                gen = self._gen
                pending = self._seek_to is not None
                eos = (self._eos_gen == gen)
            if pending:
                _t.sleep(0.002)
                continue
            try:
                g, f, idx = self._q.get(timeout=0.1)
            except self._Empty:
                if eos:
                    return None          # drained and at end of stream
                continue
            if g != gen:
                continue                 # stale frame from before a seek
            if f is None:
                return None
            self.index = idx + 1
            return f
        return None

    def seek_frame(self, n: int) -> None:
        with self._lock:
            self._seek_to = max(0, n)
            self._gen += 1
            self._drain()
            self.index = max(0, n)

    def release(self) -> None:
        self._stop = True
        self._drain()
        self._thread.join(timeout=1.0)
        self._inner.release()


class FileSource(VideoSource):
    """Video file source. Prefers multithreaded PyAV decode wrapped in a
    read-ahead thread (falls back to OpenCV). Pass read_ahead=False for the
    deterministic offline render, which doesn't need the smoothing."""
    def __new__(cls, path: str, read_ahead: bool = True):
        try:
            inner = _AvFileSource(path)
        except Exception:  # noqa: BLE001
            inner = _CvFileSource(path)
        return ReadAheadSource(inner) if read_ahead else inner


def scale_to_width(frame: np.ndarray, max_w: int,
                   interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame
    scale = max_w / w
    return cv2.resize(frame, (max_w, int(h * scale)), interpolation=interp)


def render_to_file(
    source_path: str,
    output_path: str,
    params: ParameterSet,
    timeline: Timeline,
    effects: list[Effect],
    *,
    max_width: Optional[int] = None,
    with_audio: bool = True,
    progress=None,
) -> str:
    """Deterministically re-render the take to a new video file.

    Output frames are selected through the motor wow/flutter time-warp (source
    frames are duplicated when slow, skipped when fast). If the source has audio
    and PyAV is available, the audio is muxed in -- wobbled to match when the
    motor is set to affect audio.
    """
    import os
    import tempfile

    from . import audio as audio_mod
    from . import audio_react
    from .motor import motor_affects_audio, motor_enabled, motor_speed

    src = FileSource(source_path, read_ahead=False)
    pipeline = Pipeline(effects)
    fps = src.fps or 30.0

    # install audio-reactive analysis for this clip (restored afterwards) so the
    # render matches the preview bit-for-bit
    prev_analysis = audio_react.set_analysis(
        audio_react.build_analysis(source_path, fps, src.frame_count))

    w, h = src.width, src.height
    if max_width and w > max_width:
        scale = max_width / w
        w, h = max_width, int(h * scale)

    # decide whether we can/should attach audio
    audio = None
    if with_audio and audio_mod.HAVE_AV:
        loaded = audio_mod.load_audio(source_path)
        if loaded:
            samples, sr = loaded
            if motor_affects_audio(params):
                samples = audio_mod.resample_wobble(
                    samples, sr, lambda t: motor_speed(params, t))
            audio = (samples, sr)

    # write video (silent) to a temp file if we will mux audio, else direct
    video_target = output_path
    tmp_video = None
    if audio is not None:
        fd, tmp_video = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        video_target = tmp_video

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_target, fourcc, fps, (w, h))
    if not writer.isOpened():
        src.release()
        raise IOError(f"cannot open writer: {video_target}")

    src.seek_frame(0)
    frame = src.read()
    out_index = 0
    src_pos = 0.0
    total = src.frame_count or 0
    try:
        while frame is not None:
            t = out_index / fps
            work = (scale_to_width(frame, max_width, cv2.INTER_AREA)
                    if max_width else frame)
            timeline.apply(t, params)
            writer.write(pipeline.process(work.copy(), t))
            out_index += 1

            # advance the (warped) source read head
            src_pos += motor_speed(params, t) if motor_enabled(params) else 1.0
            target = int(src_pos)
            nxt = frame
            while src.index <= target:
                f = src.read()
                if f is None:
                    nxt = None
                    break
                nxt = f
            frame = nxt
            if progress and total:
                progress(min(1.0, src.index / total))
    finally:
        writer.release()
        src.release()

    # mux audio if we have it
    if audio is not None:
        samples, sr = audio
        ok = audio_mod.mux_audio_into_video(video_target, samples, sr,
                                            output_path, fps)
        try:
            if tmp_video and os.path.exists(tmp_video):
                if not ok:
                    # muxing failed -> fall back to the silent video
                    import sys as _sys
                    print("warning: could not attach audio; wrote silent video",
                          file=_sys.stderr)
                    import shutil
                    shutil.move(tmp_video, output_path)
                else:
                    os.remove(tmp_video)
        except OSError:
            pass
    audio_react.set_analysis(prev_analysis)
    return output_path
