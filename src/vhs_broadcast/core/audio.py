"""Audio: decode (PyAV), wow/flutter resample, optional live playback.

Everything here is optional and degrades gracefully:
  * PyAV missing        -> no audio decode/mux (video still works).
  * sounddevice/PortAudio missing -> no live playback (render can still mux).

The motor wow/flutter warps audio by variable-rate resampling with the SAME
speed curve the video uses, so picture and sound wobble together.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

try:
    import av  # PyAV
    HAVE_AV = True
except Exception:  # noqa: BLE001
    HAVE_AV = False


def have_audio_output() -> bool:
    try:
        import sounddevice  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def load_audio(path: str) -> Optional[tuple[np.ndarray, int]]:
    """Decode the first audio stream to float32 [N, ch] and its samplerate."""
    if not HAVE_AV:
        return None
    try:
        container = av.open(path)
        astreams = [s for s in container.streams if s.type == "audio"]
        if not astreams:
            container.close()
            return None
        stream = astreams[0]
        sr = stream.rate or 44100
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=sr)
        chunks = []
        for frame in container.decode(stream):
            for rf in resampler.resample(frame):
                arr = rf.to_ndarray()          # (channels, samples) planar float
                chunks.append(arr.T)           # -> (samples, channels)
        container.close()
        if not chunks:
            return None
        samples = np.concatenate(chunks, axis=0).astype(np.float32)
        return samples, int(sr)
    except Exception:  # noqa: BLE001
        return None


def resample_wobble(samples: np.ndarray, sr: int,
                    speed_at: Callable[[float], float]) -> np.ndarray:
    """Variable-rate resample: read position advances by speed(t) each output
    sample, so a >1 speed raises pitch/tempo and <1 lowers it (tape wow)."""
    n_in = samples.shape[0]
    if n_in == 0:
        return samples
    out = []
    pos = 0.0
    i = 0
    ch = samples.shape[1]
    idx = np.arange(ch)
    while pos < n_in - 1:
        t = i / sr
        i0 = int(pos)
        frac = pos - i0
        s = samples[i0] * (1 - frac) + samples[min(i0 + 1, n_in - 1)] * frac
        out.append(s)
        pos += speed_at(t)
        i += 1
    if not out:
        return samples
    return np.asarray(out, dtype=np.float32).reshape(-1, ch)[:, idx]


def mux_audio_into_video(video_path: str, samples: np.ndarray, sr: int,
                         out_path: str, fps: float) -> bool:
    """Copy an existing (silent) video file and add a PCM/AAC audio track."""
    if not HAVE_AV or samples is None or samples.size == 0:
        return False
    try:
        src = av.open(video_path)
        vin = src.streams.video[0]
        out = av.open(out_path, "w")
        vout = out.add_stream("mpeg4", rate=int(round(fps)))
        vout.width = vin.codec_context.width
        vout.height = vin.codec_context.height
        vout.pix_fmt = "yuv420p"
        aout = out.add_stream("aac", rate=sr)
        aout.layout = "stereo"

        for frame in src.decode(vin):
            frame = frame.reformat(format="yuv420p")
            for pkt in vout.encode(frame):
                out.mux(pkt)
        for pkt in vout.encode():
            out.mux(pkt)

        # audio: feed planar float in 1024-sample frames with running pts
        from fractions import Fraction
        planar = np.ascontiguousarray(samples.T)   # (channels, N)
        n = planar.shape[1]
        block = 1024
        pts = 0
        for start in range(0, n, block):
            chunk = np.ascontiguousarray(planar[:, start:start + block])
            aframe = av.AudioFrame.from_ndarray(chunk, format="fltp",
                                                layout="stereo")
            aframe.rate = sr
            aframe.pts = pts
            aframe.time_base = Fraction(1, sr)
            pts += chunk.shape[1]
            for pkt in aout.encode(aframe):
                out.mux(pkt)
        for pkt in aout.encode():
            out.mux(pkt)

        out.close()
        src.close()
        return True
    except Exception:  # noqa: BLE001
        return False


class AudioPlayer:
    """Optional live playback via sounddevice, with motor wobble applied.

    Uses a callback that advances a fractional read head by speed(t) per output
    sample. Silent no-op if sounddevice/PortAudio is unavailable.
    """

    def __init__(self, samples: np.ndarray, sr: int,
                 speed_at: Callable[[float], float]) -> None:
        self.samples = samples
        self.sr = sr
        self.speed_at = speed_at
        self._pos = 0.0
        self._stream = None
        self._ok = False
        try:
            import sounddevice as sd
            ch = samples.shape[1] if samples.ndim == 2 else 1
            self._stream = sd.OutputStream(
                samplerate=sr, channels=ch, dtype="float32",
                callback=self._callback, blocksize=1024)
            self._ok = True
        except Exception:  # noqa: BLE001
            self._ok = False

    def _callback(self, outdata, frames, time_info, status):  # noqa: D401
        n = self.samples.shape[0]
        for k in range(frames):
            if self._pos >= n - 1:
                outdata[k:] = 0
                return
            i0 = int(self._pos)
            frac = self._pos - i0
            outdata[k] = (self.samples[i0] * (1 - frac)
                          + self.samples[i0 + 1] * frac)
            t = self._pos / self.sr
            self._pos += max(0.05, self.speed_at(t))

    def play(self):
        if self._ok and self._stream:
            try:
                self._stream.start()
            except Exception:  # noqa: BLE001
                pass

    def pause(self):
        if self._ok and self._stream:
            try:
                self._stream.stop()
            except Exception:  # noqa: BLE001
                pass

    def seek_seconds(self, t: float):
        self._pos = max(0.0, t * self.sr)

    def stop(self):
        if self._ok and self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
#  Qt-based live playback (no PortAudio needed -- ships with PySide6)
# --------------------------------------------------------------------------- #
try:
    from PySide6.QtCore import QIODevice
    from PySide6.QtMultimedia import QAudio, QAudioFormat, QAudioSink
    HAVE_QT_AUDIO = True
except Exception:  # noqa: BLE001
    HAVE_QT_AUDIO = False


if HAVE_QT_AUDIO:

    class _WobbleDevice(QIODevice):
        """Pull-mode PCM source: emits int16 stereo, advancing a fractional read
        head by the motor speed curve (block-constant for speed)."""

        def __init__(self, samples: np.ndarray, sr: int, speed_at):
            super().__init__()
            if samples.ndim == 1:
                samples = np.stack([samples, samples], axis=1)
            self.samples = np.ascontiguousarray(samples, dtype=np.float32)
            self.sr = sr
            self.speed_at = speed_at
            self.n = self.samples.shape[0]
            self.pos = 0.0

        def bytesAvailable(self):
            return 4 * self.sr + super().bytesAvailable()

        def isSequential(self):
            return True

        def writeData(self, _data):
            return -1

        def readData(self, maxlen):
            frames = int(maxlen) // 4
            if frames <= 0:
                return bytes()
            sp = 1.0
            if self.speed_at is not None:
                sp = max(0.05, float(self.speed_at(self.pos / self.sr)))
            idx = self.pos + sp * np.arange(frames, dtype=np.float64)
            i0 = idx.astype(np.int64)
            valid = i0 < (self.n - 1)
            i0c = np.clip(i0, 0, max(0, self.n - 2))
            frac = (idx - i0c)[:, None]
            smp = self.samples[i0c] * (1 - frac) + self.samples[i0c + 1] * frac
            smp[~valid] = 0.0
            pcm = np.clip(smp * 32767.0, -32768, 32767).astype("<i2")
            self.pos = float(idx[-1] + sp)
            return pcm.tobytes()

    class QtAudioPlayer:
        def __init__(self, samples: np.ndarray, sr: int, speed_at=None):
            self.ok = False
            try:
                fmt = QAudioFormat()
                fmt.setSampleRate(int(sr))
                fmt.setChannelCount(2)
                fmt.setSampleFormat(QAudioFormat.Int16)
                self.sink = QAudioSink(fmt)
                self.dev = _WobbleDevice(samples, sr, speed_at)
                self.ok = True
            except Exception:  # noqa: BLE001
                self.ok = False

        def is_active(self):
            if not self.ok:
                return False
            try:
                return self.sink.state() == QAudio.State.ActiveState
            except Exception:  # noqa: BLE001
                return False

        def source_time(self):
            """Current *played* source position in seconds, or None if not
            actively playing. Video slaves to this so A/V stay locked. Accounts
            for the sink's un-played buffer so it reflects what you hear."""
            if not self.is_active():
                return None
            try:
                bpf = 4  # stereo int16
                buffered = max(0, (self.sink.bufferSize()
                                   - self.sink.bytesFree())) // bpf
            except Exception:  # noqa: BLE001
                buffered = 0
            played = max(0.0, self.dev.pos - buffered)
            return played / float(self.dev.sr)

        def play(self):
            if not self.ok:
                return
            try:
                # already playing? leave it. Otherwise (re)start from the
                # device's current position. We use stop()/start() rather than
                # suspend()/resume() because resume() is unreliable on some
                # backends -- it leaves the sink alive but silent.
                if self.sink.state() == QAudio.State.ActiveState:
                    return
                if not self.dev.isOpen():
                    self.dev.open(QIODevice.ReadOnly)
                self.sink.start(self.dev)
            except Exception:  # noqa: BLE001
                pass

        def pause(self):
            if not self.ok:
                return
            try:
                # stop (not suspend); the device stays open so its read head
                # persists and playback continues from here on the next play().
                self.sink.stop()
            except Exception:  # noqa: BLE001
                pass

        def seek_seconds(self, t: float):
            if self.ok:
                self.dev.pos = max(0.0, t * self.dev.sr)

        def stop(self):
            if self.ok:
                try:
                    self.sink.stop()
                    self.dev.close()
                except Exception:  # noqa: BLE001
                    pass

else:
    class QtAudioPlayer:   # graceful stub
        def __init__(self, *a, **k):
            self.ok = False
        def play(self): ...
        def pause(self): ...
        def seek_seconds(self, t): ...
        def stop(self): ...
