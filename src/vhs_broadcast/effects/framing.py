"""Aspect: frame the picture as 4:3, 16:9, scope or vertical, with bars, crop or stretch.

The bars are part of the picture from the start (camera stage), so tape and TV effects
damage them like a real letterboxed or pillarboxed broadcast. Amount slides between the
full frame and the target framing, so keying it animates the bars in or the squeeze on.
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import Effect

ASPECTS = (4 / 3, 16 / 9, 2.39, 1.0, 9 / 16)


class AspectEffect(Effect):
    id = "aspect"
    label = "ASPECT"
    stage = "camera"

    def build(self, params):
        self.choice("target", "Aspect", ("4:3", "16:9", "2.39:1 (scope)", "1:1", "9:16 (vertical)"))
        self.choice("fit", "Fit", ("Crop to fill", "Whole picture (bars)", "Stretch"))
        self.param("amount", "Amount", 1)                          # 0 = full frame, 1 = target
        self.param("pan", "Pan", 0, -1, 1)                          # pan and scan the crop
        self.choice("bars", "Bars", ("Black", "Blurred picture", "Grey"))

    def process(self, frame, t, v):
        h, w = frame.shape[:2]
        amount = float(np.clip(v["amount"], 0, 1))
        if amount < 1e-3:
            return frame
        # The active picture: the largest rectangle of the target aspect, eased in by Amount.
        target = ASPECTS[int(v["target"])]
        rw, rh = (w, w / target) if w / h < target else (h * target, h)
        rw, rh = w + (rw - w) * amount, h + (rh - h) * amount
        rw, rh = max(2, int(round(rw))), max(2, int(round(rh)))
        x0, y0 = (w - rw) // 2, (h - rh) // 2
        fit = int(v["fit"])
        src = frame
        if fit == 0:
            # Crop the source to the box's shape; Pan slides the crop window.
            box = rw / rh
            cw, ch = (w, w / box) if w / h < box else (h * box, h)
            cw, ch = max(2, int(round(cw))), max(2, int(round(ch)))
            pan = float(v["pan"])
            cx = int(round((w - cw) / 2 * (1 + pan)))
            cy = int(round((h - ch) / 2 * (1 + pan))) if ch < h else 0
            src = frame[cy:cy + ch, cx:cx + cw]
        picture = cv2.resize(src, (rw, rh), interpolation=cv2.INTER_AREA if rw * rh < w * h else cv2.INTER_LINEAR)
        if fit == 1:
            # The whole picture inside the box, with its own bars.
            scale = min(rw / w, rh / h)
            pw, ph = max(2, int(round(w * scale))), max(2, int(round(h * scale)))
            inner = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
            picture = self._backdrop(frame, (rh, rw), int(v["bars"]))
            py, px = (rh - ph) // 2, (rw - pw) // 2
            picture[py:py + ph, px:px + pw] = inner
        out = self._backdrop(frame, (h, w), int(v["bars"]))
        out[y0:y0 + rh, x0:x0 + rw] = picture
        return out

    @staticmethod
    def _backdrop(frame, shape, style):
        h, w = shape
        if style == 1:
            small = cv2.resize(frame, (max(2, w // 16), max(2, h // 16)), interpolation=cv2.INTER_AREA)
            blur = cv2.resize(cv2.GaussianBlur(small, (0, 0), 1.5), (w, h), interpolation=cv2.INTER_LINEAR)
            return (blur.astype(np.float32) * .55).astype(np.uint8)
        if style == 2:
            return np.full((h, w, 3), 64, np.uint8)
        return np.zeros((h, w, 3), np.uint8)
