"""CPU retro-console looks and independent RGB channel processing."""
from __future__ import annotations

import cv2
import numpy as np

from .base import Effect
from .advanced import PaletteDitherEffect, TemporalEffect, _u8


class GameBoyEffect(TemporalEffect):
    id = "gameboy"
    label = "GAME BOY / POCKET VIDEO"
    # Dark-to-light RGB ramps. Inspired looks, not hardware emulation.
    PALETTES = np.array([
        [[15, 56, 15], [48, 98, 48], [139, 172, 15], [155, 188, 15]],
        [[28, 26, 32], [92, 91, 102], [166, 164, 175], [234, 232, 240]],
        [[48, 28, 76], [132, 70, 119], [205, 136, 135], [247, 234, 184]],
        [[17, 33, 51], [52, 92, 102], [132, 174, 165], [223, 239, 211]],
    ], np.uint8)

    def build(self, params):
        self.param("amount", "Mix", 1)
        self.param("width", "Grid W", 160, 32, 320, step=1)
        self.choice("aspect", "Pixels", ("Square / Fit", "160:144 Stretch"))
        self.choice("palette", "Palette", ("Classic Green", "Pocket Gray", "Cream / Plum", "Blue / Mint"))
        self.choice("dither", "Dither", ("None", "Bayer 4x4", "Bayer 2x2", "Noise"), 1)
        self.param("strength", "Dither", .8)
        self.param("contrast", "Contrast", 1.25, .25, 3)
        self.param("brightness", "Bright", 0, -.5, .5)
        self.param("ghost", "LCD Trail", 0, 0, .95)
        self.param("grid", "LCD Grid", 0, 0, .8)

    def process_history(self, frame, t, v, previous, dt):
        h, w = frame.shape[:2]
        gw = min(w, max(1, round(v["width"])))
        gh = max(1, min(h, round(gw * (144 / 160 if v["aspect"] >= .5 else h / w))))
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (gw, gh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        gray = np.clip((gray - .5) * v["contrast"] + .5 + v["brightness"], 0, 1)
        old = previous.get("gray")
        if old is not None and old.shape == gray.shape and v["ghost"] > 0:
            keep = v["ghost"] ** (dt * 30)
            gray = gray * (1 - keep) + old * keep
        yy, xx = np.indices((gh, gw))
        mode = int(v["dither"])
        if mode == 1:
            noise = PaletteDitherEffect.BAYER[yy % 4, xx % 4]
        elif mode == 2:
            noise = np.array([[0, 2], [3, 1]], np.float32)[yy % 2, xx % 2] / 4 - .375
        elif mode == 3:
            # Static noise locks to the pixel grid so still pictures do not fizz.
            noise = np.random.default_rng(1989).random(gray.shape) - .5
        else:
            noise = 0
        indices = np.clip(np.floor(gray * 3 + .5 + noise * v["strength"]), 0, 3).astype(np.uint8)
        colors = self.PALETTES[int(v["palette"]), :, ::-1]
        picture = cv2.resize(colors[indices], (w, h), interpolation=cv2.INTER_NEAREST)
        if v["grid"] > 0:
            # Darken the trailing edge of enlarged cells; no forced display border.
            x = np.arange(w) * gw // w
            y = np.arange(h) * gh // h
            lines = np.zeros((h, w), bool)
            if w >= 2 * gw:
                lines[:, :-1] |= x[:-1] != x[1:]
            if h >= 2 * gh:
                lines[:-1] |= (y[:-1] != y[1:])[:, None]
            picture = _u8(picture * (1 - lines[..., None] * v["grid"]))
        return cv2.addWeighted(frame, 1 - v["amount"], picture, v["amount"], 0), {"gray": gray.copy()}


class RGBTortureEffect(Effect):
    id = "rgb_torture"
    label = "RGB CHANNEL TORTURE"

    def build(self, params):
        self.param("amount", "Mix", 1)
        self.choice("order", "Routing", ("RGB", "RBG", "GRB", "GBR", "BRG", "BGR"))
        for name in ("r", "g", "b"):
            self.param(name + "x", name.upper() + " X", 0, -.5, .5)
            self.param(name + "y", name.upper() + " Y", 0, -.5, .5)
            self.param(name + "gain", name.upper() + " Gain", 1, 0, 3)
            self.param(name + "bits", name.upper() + " Bits", 8, 1, 8, step=1)
            self.param(name + "invert", name.upper() + " Invert", 0, 0, 1, step=1)

    def process(self, frame, t, v):
        if v["amount"] <= 0:
            return frame
        h, w = frame.shape[:2]
        route = ("rgb", "rbg", "grb", "gbr", "brg", "bgr")[int(v["order"])]
        out = np.empty_like(frame)
        for target, source in zip("rgb", route):
            channel = np.roll(frame[..., "bgr".index(source)],
                              (round(v[target + "y"] * h), round(v[target + "x"] * w)), axis=(0, 1)).astype(np.float32)
            channel = np.clip(channel * v[target + "gain"], 0, 255)
            if v[target + "invert"] >= .5:
                channel = 255 - channel
            levels = 2 ** round(v[target + "bits"]) - 1
            out[..., "bgr".index(target)] = _u8(np.round(channel * levels / 255) * 255 / levels)
        return cv2.addWeighted(frame, 1 - v["amount"], out, v["amount"], 0)


class NESCorruptionEffect(Effect):
    """Stylized indexed-color tile scramble; legacy id preserves saved takes."""
    id = "nes_corrupt"
    label = "TILE SCRAMBLE"
    COLORS = np.array([[15, 16, 28], [41, 44, 106], [80, 52, 160], [154, 43, 122],
                       [192, 52, 49], [156, 82, 28], [103, 120, 28], [41, 130, 66],
                       [25, 112, 136], [83, 157, 229], [149, 113, 233], [233, 119, 186],
                       [242, 162, 120], [239, 213, 137], [153, 219, 179], [241, 238, 224]], np.uint8)

    def build(self, params):
        self.param("amount", "Mix", 1)
        self.param("width", "Grid W", 256, 64, 512, step=1)
        self.param("tile", "Tile px", 8, 4, 32, step=1)
        self.param("density", "Scramble", .25)
        self.param("planes", "Bit Damage", .15)
        self.param("attributes", "Color Swap", .3)
        self.param("tear", "Row Shift", .1)
        self.param("rate", "Change Hz", 4, 0, 30)
        self.param("seed", "Seed", 8, 0, 9999, step=1)

    def process(self, frame, t, v):
        if v["amount"] <= 0:
            return frame
        h, w = frame.shape[:2]
        gw = min(w, round(v["width"]))
        gh = max(1, round(h * gw / w))
        image = cv2.resize(frame, (gw, gh), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        indices = np.minimum(3, gray.astype(np.uint16) * 4 // 256).astype(np.uint8)
        tile = round(v["tile"])
        rows, cols = (gh + tile - 1) // tile, (gw + tile - 1) // tile
        tick = int(self.params.motion_phase("nes_corrupt.rate", t))
        rng = np.random.default_rng((round(v["seed"]) + tick * 104729) & 0xFFFFFFFF)
        yy, xx = np.indices((gh, gw))
        by, bx = yy // tile, xx // tile
        damaged = rng.random((rows, cols)) < v["density"]
        sy = rng.integers(0, rows, (rows, cols)) * tile
        sx = rng.integers(0, cols, (rows, cols)) * tile
        mapped_y = np.where(damaged[by, bx], sy[by, bx] + yy % tile, yy) % gh
        mapped_x = np.where(damaged[by, bx], sx[by, bx] + xx % tile, xx) % gw
        picture = indices[mapped_y, mapped_x]
        plane_damage = rng.random((rows, cols)) < v["planes"]
        # XOR one of the two indexed-colour bitplanes in repeatable tile rows.
        plane = rng.integers(1, 4, (rows, cols), dtype=np.uint8)
        picture = picture ^ np.where(plane_damage[by, bx] & ((xx + yy) % 3 != 0), plane[by, bx], 0).astype(np.uint8)
        ar, ac = (rows + 1) // 2, (cols + 1) // 2
        attributes = rng.integers(0, 4, (ar, ac))
        swap = rng.random((ar, ac)) < v["attributes"]
        bank = np.where(swap, attributes, 0)[by // 2, bx // 2]
        # Four color banks, each spanning dark to light.
        palette_index = picture.astype(np.int32) * 4 + bank
        color = self.COLORS[palette_index, ::-1]
        shifted = rng.random(rows) < v["tear"]
        offsets = rng.integers(-gw, gw + 1, rows)
        color = color[yy, (xx + np.where(shifted[by], offsets[by], 0)) % gw]
        color = cv2.resize(color, (w, h), interpolation=cv2.INTER_NEAREST)
        return cv2.addWeighted(frame, 1 - v["amount"], color, v["amount"], 0)
