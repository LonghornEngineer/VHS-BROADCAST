"""Real period codecs: the frame is encoded, the packets are damaged, then decoded.

Blocking, mosquito noise, smeared motion and datamosh all come from actual
compression, not from drawing look-alikes. Encoder and decoder keep their state
between frames, so motion compensation, keyframe intervals and packet loss behave
as they did: a dropped keyframe lets later frames paint motion onto an old picture.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import av
import cv2
import numpy as np

from .base import Effect


@dataclass(frozen=True)
class Format:
    label: str
    codec: str
    width: int              # coded width at size 1 (height follows the frame's aspect)
    fps: float
    bitrate: int            # bits/s at quality 0.5; 0 = codec has no bitrate control
    gop: int                # keyframe interval in frames
    pix_fmt: str = "yuv420p"
    fixed: tuple = ()       # (w, h) when the codec only allows one picture size


FORMATS = (
    Format("Webcam 2006 (MPEG-4)", "mpeg4", 320, 15, 160_000, 90),
    Format("Video call (H.263)", "h263", 352, 12, 96_000, 120, fixed=(352, 288)),
    Format("MiniDV (DV)", "dvvideo", 720, 30000 / 1001, 0, 1, "yuv411p", fixed=(720, 480)),
    Format("Early web video (Sorenson Spark)", "flv", 320, 15, 250_000, 150),
    Format("VCD (MPEG-1)", "mpeg1video", 352, 30000 / 1001, 1_150_000, 18),
    Format("CD-ROM (Cinepak)", "cinepak", 320, 15, 0, 15, "rgb24"),
    Format("Windows Media (WMV)", "wmv2", 320, 15, 250_000, 150),
    Format("Videophone 1990 (H.261)", "h261", 176, 10, 64_000, 132, fixed=(176, 144)),
    Format("QuickTime (Sorenson Video)", "svq1", 320, 15, 250_000, 60, "yuv410p"),
)
# MPEG-1 only accepts broadcast frame rates.
_STANDARD_RATES = (Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001), Fraction(30))

av.logging.set_level(av.logging.FATAL)    # damaged packets make decoders complain on every frame


class CodecEffect(Effect):
    id = "codec"
    label = "DIGITAL CODEC (REAL)"
    stage = "digital"

    def __init__(self, params):
        self._state = None
        super().__init__(params)

    def build(self, params):
        self.param("amount", "Mix", 1)
        self.choice("format", "Format", tuple(f.label for f in FORMATS))
        self.param("quality", "Quality", .5)
        self.param("size", "Size", 1, .25, 2)
        self.param("fps", "Frame rate", 0, 0, 30, step=1)
        self.param("keyframe", "Keyframe every", 0, 0, 600, step=1)
        self.param("loss", "Packet loss", 0, 0, .5)
        self.param("errors", "Bit errors", 0)
        self.param("datamosh", "Datamosh", 0, 0, 1, step=1)
        self.choice("scaling", "Scaling", ("Smooth", "Chunky"))
        self.param("seed", "Seed", 0, 0, 9999, step=1)

    def reset_state(self):
        self._state = None

    # ------------------------------------------------------------------ #
    def _geometry(self, fmt, v, h, w):
        if fmt.fixed:
            return fmt.fixed
        cw = max(64, int(round(fmt.width * v["size"] / 16)) * 16)
        ch = max(48, int(round(cw * h / w / 16)) * 16)
        return cw, ch

    def _rate(self, fmt, v):
        fps = v["fps"] or fmt.fps
        rate = Fraction(fps).limit_denominator(1001)
        if fmt.codec in ("mpeg1video", "mpeg2video"):
            rate = min(_STANDARD_RATES, key=lambda r: abs(float(r) - fps))
        return rate

    def _open(self, fmt, v, size, rate):
        enc = av.CodecContext.create(fmt.codec, "w")
        enc.width, enc.height = size
        enc.pix_fmt = fmt.pix_fmt
        enc.time_base = 1 / rate
        enc.framerate = rate
        enc.thread_count = 1                       # single-threaded: identical output every render
        enc.gop_size = int(v["keyframe"]) or fmt.gop
        if fmt.bitrate:
            enc.bit_rate = int(fmt.bitrate * 10 ** (2 * (v["quality"] - .5)))   # 0 = a tenth, 1 = ten times
            # These encoders follow their quantizer more than the bitrate: pin it.
            quant = str(int(round(31 - v["quality"] * 29)))
            enc.options = {"qmin": quant, "qmax": quant}
        try:
            enc.max_b_frames = 0                   # no reordering: each frame comes straight back
        except (AttributeError, ValueError):
            pass
        return {"enc": enc, "dec": None, "fmt": fmt, "size": size, "rate": rate,
                "tick": None, "time": None, "out": None, "keyed": False}

    def _decoder(self, st):
        dec = av.CodecContext.create(st["fmt"].codec, "r")
        dec.width, dec.height = st["size"]
        if st["enc"].extradata:
            dec.extradata = st["enc"].extradata
        dec.thread_count = 1
        dec.flags |= av.codec.context.Flags.low_delay
        return dec

    # ------------------------------------------------------------------ #
    def process(self, frame, t, v):
        if v["amount"] <= 0:
            return frame
        h, w = frame.shape[:2]
        fmt = FORMATS[int(v["format"])]
        size = self._geometry(fmt, v, h, w)
        rate = self._rate(fmt, v)
        key = (int(v["format"]), size, rate, round(v["quality"], 3), int(v["keyframe"]), (h, w))
        st = self._state
        # Codecs carry history; restart on a new setup, a backward seek or a jump.
        if (st is None or st["key"] != key or (st["time"] is not None and (t < st["time"] or t - st["time"] > 1))):
            st = self._state = self._open(fmt, v, size, rate)
            st["key"] = key
        st["time"] = t
        tick = int(math.floor(t * float(rate) + 1e-6))
        if tick != st["tick"]:
            st["tick"] = tick
            decoded = self._step(st, frame, tick, v)
            if decoded is not None:
                interp = cv2.INTER_NEAREST if int(v["scaling"]) == 1 else cv2.INTER_LINEAR
                st["out"] = cv2.resize(decoded, (w, h), interpolation=interp)
        out = st["out"] if st["out"] is not None else frame
        return out if v["amount"] >= 1 else cv2.addWeighted(frame, 1 - v["amount"], out, v["amount"], 0)

    def _step(self, st, frame, tick, v):
        small = cv2.resize(frame, st["size"], interpolation=cv2.INTER_AREA)
        picture = av.VideoFrame.from_ndarray(small, format="bgr24").reformat(format=st["fmt"].pix_fmt)
        picture.pts = tick
        # Datamosh needs a keyframe at every cut to throw away. Encoders often code a cut as a
        # P-frame full of intra blocks instead, so force one when the picture changes hard.
        thumb = cv2.resize(frame, (32, 18), interpolation=cv2.INTER_AREA).astype(np.int16)
        last = st.get("thumb")
        st["thumb"] = thumb
        if v["datamosh"] >= .5 and last is not None and np.abs(thumb - last).mean() > 12:
            picture.pict_type = av.video.frame.PictureType.I
        rng = np.random.default_rng([int(v["seed"]), tick])
        decoded = None
        for packet in st["enc"].encode(picture):
            data = bytearray(bytes(packet))
            keyframe = bool(packet.is_keyframe)
            if keyframe and st["keyed"] and v["datamosh"] >= .5:
                continue                           # keep predicting from the old picture
            if not keyframe and rng.random() < v["loss"]:
                continue
            if v["errors"] > 0 and len(data) > 32 and rng.random() < v["errors"]:
                hits = 1 + int(v["errors"] * 8)
                for pos in rng.integers(16, len(data), hits):
                    data[pos] = int(rng.integers(0, 256))
            if st["dec"] is None:
                if not keyframe:
                    continue                       # a decoder can only start on a keyframe
                st["dec"] = self._decoder(st)
            try:
                for image in st["dec"].decode(av.Packet(bytes(data))):
                    decoded = image.to_ndarray(format="bgr24")
            except av.error.FFmpegError:
                pass                               # too damaged: the screen keeps the last picture
            st["keyed"] = st["keyed"] or keyframe
        return decoded
