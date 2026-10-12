"""A group of controls for one effect. The enable checkbox carries the effect
name (checked = on), so there's no separate title crowding the checkbox."""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (QCheckBox, QGroupBox, QHBoxLayout, QVBoxLayout,
                              QPushButton, QFileDialog, QMessageBox, QLabel)

from ...effects.base import Effect
from .param_row import ParamRow


class EffectPanel(QGroupBox):
    def __init__(self, effect: Effect, parent=None, header_text: Optional[str] = None):
        super().__init__("", parent)          # no title -> checkbox names it
        self.effect = effect
        lay = QVBoxLayout(self)
        lay.setSpacing(2)
        lay.setContentsMargins(6, 4, 6, 6)

        header = QHBoxLayout()
        self.enable = QCheckBox(header_text or effect.label)
        self.enable.setObjectName("EffectTitle")
        self.enable.setChecked(effect.enabled_param().value >= 0.5)
        self.enable.toggled.connect(self._on_enable)
        header.addWidget(self.enable)
        header.addStretch(1)
        if hasattr(effect, "request_reset"):
            reset = QPushButton("RESET HISTORY")
            reset.setToolTip("Clear this effect's frame memory. To save a reset on the timeline, key the Reset pulse control off then on.")
            reset.clicked.connect(effect.request_reset)
            header.addWidget(reset)
            self.setToolTip("Uses earlier frames. History clears on seek, source/size/chain changes, and disabling the effect. Pausing does not advance it.")
        lay.addLayout(header)

        self.rows: list[ParamRow] = []
        for p in effect.own_params():
            row = ParamRow(p)
            palette_help = {
                "palette.mode": "Ordered patterns stay fixed to the picture. Noise can be animated or frozen. None still applies palette reduction.",
                "palette.colors": "Number of palette entries. CGA and Handheld have at most four; EGA has at most sixteen.",
                "palette.amount": "Dither strength. Zero keeps palette reduction without the pattern.",
                "palette.scale": "Size of each pattern cell in processing pixels; preview/export resolution affects its appearance.",
                "palette.gamma": "1 is neutral. Higher values brighten midtones before palette reduction. Double-click to reset.",
                "palette.bias": "Shift the image brighter or darker before palette reduction; 0 is neutral.",
                "palette.animate": "Noise only: on updates at 30 Hz; off locks the pattern to the seed. Pausing never changes the noise.",
                "palette.seed": "Noise only: choose a repeatable random pattern.",
                "gameboy.width": "Logical pixel width before nearest-neighbor enlargement. 160 gives a classic handheld-sized grid.",
                "gameboy.aspect": "Square / Fit preserves the source aspect ratio. 160:144 Stretch uses a handheld-shaped grid stretched into the source frame.",
                "gameboy.ghost": "LCD response trails. Zero gives crisp four-color output. History clears on seek and does not advance while paused.",
                "gameboy.palette": "Four-tone display ramps, including Classic Green and the cream/purple reference look.",
                "gameboy.grid": "Darkens enlarged pixel boundaries. Leave at zero for strict four-color output.",
                "nes_corrupt.density": "Borrow other tiles from the image, like corrupted tile addresses. The console color reduction remains when damage is zero.",
                "nes_corrupt.planes": "Damage the two-bit pixel indices within tiles, producing repeated stripe fragments.",
                "nes_corrupt.attributes": "Swap four-color groups across 2x2 tile regions.",
                "nes_corrupt.rate": "Pattern changes per second. Zero holds the corruption pattern in place.",
                "block_corrupt.mode": "Freeze holds damaged source regions; Repeat borrows held image fragments; Shift displaces the current image; Tile Replace repeats aligned tiles from the current frame.",
                "block_corrupt.pool": "Repeat / Tile Replace: number of shared source patterns. One repeats a single fragment; higher values add variety.",
                "block_corrupt.clusters": "Group damaged blocks into 2x2 neighborhoods. Zero keeps independent scattered blocks.",
                "block_corrupt.lock": "Keep the damage locations and source addresses fixed while video continues. Hold s still controls refresh of frozen picture content.",
                "block_corrupt.hold": "How long Freeze / Repeat retain captured content. Also controls layout refresh when Lock Layout is off.",
                "block_corrupt.shift": "Displacement distance for Freeze and Shift. Repeat / Tile Replace use shared source fragments instead.",
                "motion_smear.mode": "Block Matching searches previous-frame neighborhoods for each block. Optical Flow keeps the original motion-smear behavior. This simulates datamoshing without changing the video codec.",
                "motion_smear.hold": "Hold detected motion vectors for this many seconds; picture history continues moving. Zero re-estimates every frame.",
                "motion_smear.search": "Block Matching only: search radius in analysis pixels. A larger radius can track larger motion but costs more CPU time.",
                "motion_smear.damage": "Replace a fraction of block vectors with repeatable incorrect motion.",
                "pixel_sort.metric": "Sort actual source pixels by this channel or color measurement. Cutoff / High Cut select the sortable range in this measurement.",
                "pixel_sort.amount": "At 1, pixels are rearranged without inventing colors. Lower mix blends that result with the source.",
                "rgb_torture.order": "Route source channels into the R, G and B outputs. Each output then has independent wrapped offsets, gain, bit depth and inversion.",
            }
            if p.id in palette_help:
                row.setToolTip(palette_help[p.id])
            self.rows.append(row)
            lay.addWidget(row)
        if hasattr(effect, "load_background"):
            button = QPushButton("LOAD BACKGROUND IMAGE…")
            button.clicked.connect(self._load_background)
            lay.addWidget(button)
            self.background_status = QLabel()
            self.background_status.setWordWrap(True)
            lay.addWidget(self.background_status)
            self._background_status()

    def _load_background(self):
        path, _ = QFileDialog.getOpenFileName(self, "Luma-key background", "",
                                              "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)")
        if path:
            try:
                self.effect.load_background(path)
                self.effect.params.get(f"{self.effect.id}.fill").set_base(4)
                self.refresh()
            except (OSError, ValueError) as error:
                QMessageBox.warning(self, "Background image", str(error))

    def _background_status(self):
        from pathlib import Path
        path = self.effect.background_path
        self.background_status.setText(("Missing image: " if self.effect.background_image is None else "Image: ")
                                        + Path(path).name if path else "No background image loaded")
        self.background_status.setToolTip(path)

    def _on_enable(self, on: bool):
        self.effect.enabled_param().set_base(1.0 if on else 0.0)

    def refresh(self):
        self.enable.blockSignals(True)
        self.enable.setChecked(self.effect.enabled_param().value >= 0.5)
        self.enable.blockSignals(False)
        for r in self.rows:
            r.refresh()
        if hasattr(self, "background_status"):
            self._background_status()
