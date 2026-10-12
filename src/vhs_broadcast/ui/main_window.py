"""Main application window."""
from __future__ import annotations

import os

from PySide6.QtCore import Qt, QTimer, Signal, QDateTime, QThread
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QDateTimeEdit,
    QMainWindow, QMenu, QMessageBox, QProgressDialog, QPushButton,
    QScrollArea, QSpinBox, QSplitter, QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

from .. import config as C
from ..core.automation import Recorder, Session, Timeline
from ..core.engine import Mode, PlaybackEngine
from ..core.parameters import ParameterSet
from ..core.pipeline import FileSource, Pipeline, RenderCancelled, render_to_file
from ..core.audio import load_audio, QtAudioPlayer
from ..core.audio_react import build_analysis, set_analysis, levels_at
from ..core.outputs import VirtualCamOutput, VideoRecorder
from ..core import capture as cap
from ..core.motor import motor_speed, motor_affects_audio
from ..effects.vhs import TextOverlayEffect, default_effects
from .theme import STYLESHEET
from .widgets.effect_panel import EffectPanel
from .widgets.transport import Transport
from .widgets.video_view import VideoView
from .widgets.audio_meter import AudioMeter
from .widgets.chain_list import ChainList
from .widgets.keyframes import KeyframeEditor


class RenderWorker(QThread):
    progress = Signal(float)
    done = Signal(str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, src, out, params, timeline, effects, max_width):
        super().__init__()
        self._args = (src, out, params, timeline, effects)
        self._max_width = max_width

    def run(self):
        try:
            src, out, params, timeline, effects = self._args
            render_to_file(src, out, params, timeline, effects,
                           max_width=self._max_width, with_audio=True,
                           progress=lambda f: self.progress.emit(f),
                           cancelled=self.isInterruptionRequested)
            self.done.emit(out)
        except RenderCancelled:
            self.cancelled.emit()
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{C.APP_NAME} v{C.APP_VERSION}")
        self.setStyleSheet(STYLESHEET)
        self.resize(1500, 880)

        # --- model ---
        self.params = ParameterSet()
        self.effects = default_effects(self.params)
        self.text_effect = next(e for e in self.effects
                                if isinstance(e, TextOverlayEffect))
        self.pipeline = Pipeline(self.effects)
        self.timeline = Timeline()
        self.recorder = Recorder(self.params, self.timeline)
        self.audio_player = None
        self._vcam = VirtualCamOutput()
        self._recorder = VideoRecorder()
        self.session = Session(self.params, self.timeline, self.effects)
        self.source_path: str | None = None

        # --- engine ---
        self.engine = PlaybackEngine(self.params, self.pipeline,
                                     self.recorder, self.timeline)

        self._build_ui()
        self.params.set_change_hook(self._on_parameter_changed)

        self.engine.frameReady.connect(self.video.set_image)
        self.engine.positionChanged.connect(self._on_position)
        self.engine.finished_playing.connect(self._on_finished)
        self.engine.set_audio_clock(self._audio_clock)
        self._stats_timer = QTimer(self)
        self._stats_timer.timeout.connect(self._update_stats)
        self._stats_timer.start(500)
        self.engine.start()

        # panel-sync timer (follows automation during PLAYBACK)
        self._sync = QTimer(self)
        self._sync.timeout.connect(self._sync_panels)
        self._sync.start(120)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QWidget(); root.setObjectName("Root")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(8, 6, 8, 6)

        title = QLabel(C.APP_NAME); title.setObjectName("Title")
        outer.addWidget(title)

        body = QHBoxLayout()
        outer.addLayout(body, 1)

        # The handle divides the preview/transport and timeline evenly by default.
        self.left_splitter = QSplitter(Qt.Vertical)
        self.left_splitter.setChildrenCollapsible(False)
        self.left_splitter.setHandleWidth(8)
        self.left_splitter.setToolTip("Drag the divider to resize preview and timeline")
        preview_area = QWidget()
        left = QVBoxLayout(preview_area)
        left.setContentsMargins(0, 0, 0, 0)
        self.video = VideoView()
        left.addWidget(self.video, 2)
        self.meter = AudioMeter()
        left.addWidget(self.meter)
        pv_row = QHBoxLayout()
        pv_lbl = QLabel("RESOLUTION")
        pv_lbl.setObjectName("SectionHint")
        self.preview_cb = QComboBox()
        self._preview_opts = [("480p", 480), ("720p", 720),
                              ("1080p", 1080), ("Full", 0), ("Custom", -1)]
        for name, _ in self._preview_opts:
            self.preview_cb.addItem(name)
        self.preview_cb.setCurrentIndex(1)   # 720p
        self.preview_cb.currentIndexChanged.connect(self._on_preview_res)
        self.preview_cb.setToolTip(
            "Processing resolution \u2014 drives the preview, the virtual camera, "
            "and recording (one pass). Lower = faster; raise it for higher-"
            "quality output.")
        pv_row.addWidget(pv_lbl); pv_row.addWidget(self.preview_cb, 1)
        self.custom_w = QSpinBox()
        self.custom_w.setRange(160, 7680)
        self.custom_w.setValue(1280)
        self.custom_w.setSuffix(" px")
        self.custom_w.setEnabled(False)
        self.custom_w.setToolTip("Custom processing width in pixels "
                                 "(select Custom).")
        self.custom_w.valueChanged.connect(self._on_custom_w)
        pv_row.addWidget(self.custom_w)
        pv_row.addSpacing(10)
        sc_lbl = QLabel("SCALE")
        sc_lbl.setObjectName("SectionHint")
        self.scale_spin = QSpinBox()
        self.scale_spin.setRange(5, 100)
        self.scale_spin.setValue(100)
        self.scale_spin.setSuffix(" %")
        self.scale_spin.setToolTip("Scale the processing resolution \u2014 a "
                                   "fraction of the width above (or of the "
                                   "source at Full). Never upscales.")
        self.scale_spin.valueChanged.connect(
            lambda v: self.engine.set_preview_scale(v / 100.0))
        pv_row.addWidget(sc_lbl); pv_row.addWidget(self.scale_spin)
        left.addLayout(pv_row)
        tog_row = QHBoxLayout()
        av_lbl = QLabel("A/V SYNC")
        av_lbl.setObjectName("SectionHint")
        self.sync_spin = QSpinBox()
        self.sync_spin.setRange(-500, 500)
        self.sync_spin.setSuffix(" ms")
        self.sync_spin.setValue(0)
        self.sync_spin.setToolTip("A/V trim: positive if the video lags the "
                                  "audio, negative if it leads")
        self.sync_spin.valueChanged.connect(self._on_sync)
        tog_row.addWidget(av_lbl); tog_row.addWidget(self.sync_spin)
        tog_row.addSpacing(12)
        self.drop_cb = QCheckBox("Drop frames")
        self.drop_cb.setChecked(True)
        self.drop_cb.setToolTip("On: stay locked to audio, dropping frames the "
                                "machine can't render in time. Off: show every "
                                "frame (smoother, may drift).")
        self.drop_cb.toggled.connect(self.engine.set_drop_frames)
        self.preview_on_cb = QCheckBox("Preview")
        self.preview_on_cb.setChecked(True)
        self.preview_on_cb.setToolTip("Turn off the on-screen preview to give the "
                                      "virtual camera / recorder more headroom.")
        self.preview_on_cb.toggled.connect(self._toggle_preview)
        tog_row.addWidget(self.drop_cb)
        tog_row.addWidget(self.preview_on_cb)
        tog_row.addStretch(1)
        self.timeline_btn = QPushButton("TIMELINE")
        self.timeline_btn.setCheckable(True)
        self.timeline_btn.setChecked(True)
        self.timeline_btn.setToolTip("Show or hide the keyframe editor; drag its divider to resize")
        self.timeline_btn.toggled.connect(self._show_timeline)
        tog_row.addWidget(self.timeline_btn)
        left.addLayout(tog_row)
        self.stats_lbl = QLabel("\u25cf idle")
        self.stats_lbl.setObjectName("Stats")
        left.addWidget(self.stats_lbl)
        self.transport = Transport()
        left.addWidget(self.transport)
        self.left_splitter.addWidget(preview_area)
        body.addWidget(self.left_splitter, 3)

        # right: tabbed -> CONTROLS (OSD + effect panels) and SIGNAL CHAIN
        right = QVBoxLayout()
        self.tabs = QTabWidget()

        controls = QWidget()
        cv = QVBoxLayout(controls)
        cv.setContentsMargins(6, 12, 6, 6)
        cv.addWidget(self._build_osd())
        filter_row = QHBoxLayout()
        self.filter_search = QLineEdit()
        self.filter_search.setPlaceholderText("Find an effect or control...")
        self.filter_search.setClearButtonEnabled(True)
        self.filter_search.textChanged.connect(self._filter_panels)
        self.active_only = QCheckBox("Enabled only")
        self.active_only.toggled.connect(self._filter_panels)
        filter_row.addWidget(self.filter_search, 1)
        filter_row.addWidget(self.active_only)
        cv.addLayout(filter_row)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        holder = QWidget()
        cols = QHBoxLayout(holder); cols.setSpacing(8)
        col_a = QVBoxLayout(); col_b = QVBoxLayout()
        for c in (col_a, col_b):
            cols.addLayout(c)
        self.panels: list[EffectPanel] = []
        targets = [col_a, col_b]
        i = 0
        for eff in self.effects:
            if isinstance(eff, TextOverlayEffect):
                continue  # OSD has its own editor above
            panel = EffectPanel(eff)
            self.panels.append(panel)
            targets[i % len(targets)].addWidget(panel)
            i += 1
        for c in (col_a, col_b):
            c.addStretch(1)
        scroll.setWidget(holder)
        cv.addWidget(scroll, 1)
        self.tabs.addTab(controls, "CONTROLS")

        chain_w = QWidget()
        chv = QVBoxLayout(chain_w)
        hint = QLabel("Drag to reorder the signal chain \u2014 earlier items are "
                      "applied first. Where OSD / TEXT sits decides whether the "
                      "text is degraded by the effects below it (tape vs player "
                      "vs TV). Preview and render follow this order.")
        hint.setWordWrap(True); hint.setObjectName("SectionHint")
        chv.addWidget(hint)
        self.chain_list = ChainList(self.effects, self._reorder_chain,
                                    self._toggle_chain)
        chv.addWidget(self.chain_list, 1)
        self.tabs.addTab(chain_w, "SIGNAL CHAIN")
        self.keyframes = KeyframeEditor(self.params, self.timeline, self.effects,
                                        self.engine.current_time)
        self.keyframes.changed.connect(self._keyframes_changed)
        self.keyframes.seekRequested.connect(self._seek_keyframe)
        self.keyframes.setMinimumHeight(300)
        self.left_splitter.addWidget(self.keyframes)
        self.left_splitter.setSizes([440, 440])
        self._timeline_sizes = [440, 440]
        for panel in [*self.panels, self.osd_panel]:
            for row in panel.rows:
                row.keyframeRequested.connect(self._add_parameter_key)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        right.addWidget(self.tabs)
        body.addLayout(right, 3)

        self.setCentralWidget(root)
        self._wire_transport()

    def _show_timeline(self, visible):
        if not visible:
            self._timeline_sizes = self.left_splitter.sizes()
        self.keyframes.setVisible(visible)
        if visible:
            self.left_splitter.setSizes(self._timeline_sizes)

    def _focus_timeline(self):
        self.timeline_btn.setChecked(True)
        self.keyframes.control.setFocus()

    def _filter_panels(self, *_):
        query = self.filter_search.text().strip().casefold()
        for panel in self.panels:
            effect = panel.effect
            text = " ".join([effect.label, *[p.label for p in effect.own_params()]]).casefold()
            panel.setVisible(query in text and
                             (not self.active_only.isChecked() or effect.enabled_param().value >= .5))

    def _build_osd(self) -> QGroupBox:
        box = QGroupBox("OSD / TEXT")
        lay = QVBoxLayout(box)
        r1 = QHBoxLayout()
        self.text_edit = QLineEdit(self.text_effect.custom_text)
        self.text_edit.textChanged.connect(
            lambda s: setattr(self.text_effect, "custom_text", s))
        r1.addWidget(QLabel("LABEL")); r1.addWidget(self.text_edit, 1)
        lay.addLayout(r1)
        r2 = QHBoxLayout()
        self.corner_edit = QLineEdit(self.text_effect.corner_text)
        self.corner_edit.textChanged.connect(
            lambda s: setattr(self.text_effect, "corner_text", s))
        self.clock_cb = QComboBox()
        self.clock_cb.addItems(["NONE", "12:00 BLINK", "TIME",
                                "DATE", "DATE+TIME", "TIMER"])
        self.clock_cb.setCurrentIndex(self.text_effect.clock_mode)
        self.clock_cb.currentIndexChanged.connect(
            lambda i: setattr(self.text_effect, "clock_mode", i))
        r2.addWidget(QLabel("TAG")); r2.addWidget(self.corner_edit, 1)
        r2.addWidget(QLabel("CLOCK")); r2.addWidget(self.clock_cb)
        lay.addLayout(r2)
        manual_row = QHBoxLayout()
        self.manual_clock_cb = QCheckBox("MANUAL DATE / TIME")
        self.manual_clock_cb.setChecked(self.text_effect.manual_clock)
        self.manual_datetime_edit = QDateTimeEdit()
        self.manual_datetime_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.manual_datetime_edit.setCalendarPopup(True)
        self.manual_datetime_edit.setDateTime(QDateTime.fromString(
            self.text_effect.manual_datetime, Qt.ISODate))
        self.clock_run_cb = QCheckBox("Advance with playback")
        self.clock_run_cb.setChecked(self.text_effect.clock_run)
        self.manual_datetime_edit.setEnabled(self.text_effect.manual_clock)
        self.clock_run_cb.setEnabled(self.text_effect.manual_clock)
        def manual_changed(on):
            self.text_effect.manual_clock = on
            self.manual_datetime_edit.setEnabled(on)
            self.clock_run_cb.setEnabled(on)
            if on and self.clock_cb.currentIndex() not in (2, 3, 4):
                self.clock_cb.setCurrentIndex(4)
        self.manual_clock_cb.toggled.connect(manual_changed)
        self.manual_datetime_edit.dateTimeChanged.connect(lambda dt: setattr(
            self.text_effect, "manual_datetime", dt.toString("yyyy-MM-ddTHH:mm:ss")))
        self.clock_run_cb.toggled.connect(lambda on: setattr(self.text_effect, "clock_run", on))
        for widget in (self.manual_clock_cb, self.manual_datetime_edit, self.clock_run_cb):
            manual_row.addWidget(widget)
        lay.addLayout(manual_row)
        r3 = QHBoxLayout()
        self.font_cb = QComboBox()
        self._fonts = self.text_effect.fonts()
        for name in self._fonts:
            self.font_cb.addItem(name)
        cur = next((n for n, p in self._fonts.items()
                    if p == self.text_effect.font_path), None)
        if cur:
            self.font_cb.setCurrentText(cur)
        self.font_cb.currentTextChanged.connect(self._on_font)
        r3.addWidget(QLabel("FONT")); r3.addWidget(self.font_cb, 1)
        lay.addLayout(r3)
        r4 = QHBoxLayout()
        self.glyph_btn = QToolButton()
        self.glyph_btn.setText("GLYPHS \u25be")
        self.glyph_btn.setPopupMode(QToolButton.InstantPopup)
        self.glyph_btn.setMenu(self._build_glyph_menu())
        r4.addWidget(QLabel("INSERT"))
        r4.addWidget(self.glyph_btn)
        r4.addStretch(1)
        lay.addLayout(r4)
        self.osd_panel = EffectPanel(self.text_effect, header_text="ENABLED")
        lay.addWidget(self.osd_panel)
        return box

    def _build_glyph_menu(self):
        from PySide6.QtWidgets import QGridLayout, QWidgetAction
        glyphs = [
            ("\u25b6", "Play"), ("\u23f8", "Pause"), ("\u25a0", "Stop"),
            ("\u25cf", "Record"), ("\u23ea", "Rewind"), ("\u23e9", "F.Fwd"),
            ("\u23ee", "Prev"), ("\u23ed", "Next"), ("\u23cf", "Eject"),
            ("\u25c0", "Tri Left"), ("\u25b2", "Up"), ("\u25bc", "Down"),
        ]
        menu = QMenu(self)
        wa = QWidgetAction(menu)
        holder = QWidget()
        grid = QGridLayout(holder)
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setSpacing(4)
        for i, (g, name) in enumerate(glyphs):
            b = QPushButton(g)
            b.setFixedSize(34, 30)
            b.setToolTip(f"Insert {name}")
            b.clicked.connect(lambda _=False, gg=g: (self._insert_glyph(gg),
                                                     menu.close()))
            grid.addWidget(b, i // 4, i % 4)
        wa.setDefaultWidget(holder)
        menu.addAction(wa)
        return menu

    def _insert_glyph(self, glyph: str):
        self.text_edit.insert(glyph)   # insert at cursor in the LABEL field
        self.text_edit.setFocus()

    def _on_font(self, name: str):
        path = self._fonts.get(name)
        if path:
            self.text_effect.font_path = path

    def _scrub(self, f):
        pos = f * self.engine.duration
        self.engine.seek_seconds(pos)
        if self.audio_player:
            self.audio_player.seek_seconds(pos)

    def _wire_transport(self):
        t = self.transport
        t.loadClicked.connect(self._load)
        t.input_menu.aboutToShow.connect(self._populate_input_menu)
        t.output_menu.aboutToShow.connect(self._populate_output_menu)
        t.playToggled.connect(self._toggle_play)
        t.keyframeClicked.connect(self._focus_timeline)
        t.playbackToggled.connect(self._toggle_playback)
        t.scrubbed.connect(self._scrub)
        t.saveClicked.connect(self._save_session)
        t.openSessionClicked.connect(self._open_session)
        t.renderClicked.connect(self._render)

    # --------------------------------------------------------------- actions
    def _load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load video", "", "Video (*.mp4 *.mov *.avi *.mkv)")
        if not path:
            return
        try:
            src = FileSource(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Load failed", str(e)); return
        self.source_path = path
        self.session.source_path = path
        self.engine.set_source(src)
        self._make_audio(path)
        self.engine.pause()
        self.transport.play_btn.setChecked(False)
        self.transport.set_file()
        self.keyframes.setEnabled(True)
        self.transport.set_position(0, self.engine.duration)

    # ---- input source picker -------------------------------------------
    def _populate_input_menu(self):
        m = self.transport.input_menu
        m.clear()
        act_file = QAction("Load Video File\u2026", m)
        act_file.triggered.connect(self._load)
        m.addAction(act_file)
        m.addSeparator()

        screens = cap.list_screens()
        sm = m.addMenu("Screen")
        if screens:
            for sc in screens:
                a = QAction(sc["label"], sm)
                a.triggered.connect(lambda _=False, b=sc["bbox"], l=sc["label"]:
                                    self._use_screen(b, l))
                sm.addAction(a)
        else:
            sm.setEnabled(False)

        wins = cap.list_windows()
        wm = m.addMenu("Window")
        if wins:
            for title in wins:
                a = QAction(title[:60], wm)
                a.triggered.connect(lambda _=False, tt=title: self._use_window(tt))
                wm.addAction(a)
        else:
            wm.setEnabled(False)
            wm.setTitle("Window (unavailable)")

        dm = m.addMenu("Device (Webcam / Capture Card)")
        devs = cap.list_devices()
        if devs:
            for i in devs:
                a = QAction(f"Device {i}", dm)
                a.triggered.connect(lambda _=False, idx=i: self._use_device(idx))
                dm.addAction(a)
        else:
            dm.setEnabled(False)
            dm.setTitle("Device (none found)")

    def _use_live(self, src, label):
        self.source_path = None
        if self.audio_player:
            self.audio_player.stop(); self.audio_player = None
        set_analysis(None)
        self.meter.set_active(False)
        self.engine.set_source(src)
        self.engine.set_mode(Mode.LIVE)
        self.engine.pause()
        self.transport.play_btn.setChecked(False)
        self.transport.pb_btn.setChecked(False)
        self.transport.set_live(label)
        self.keyframes.setEnabled(False)

    def _use_screen(self, bbox, label):
        try:
            self._use_live(cap.ScreenSource(bbox, label=label), label)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Screen capture failed", str(e))

    def _use_window(self, title):
        try:
            self._use_live(cap.WindowSource(title), title)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Window capture failed", str(e))

    def _use_device(self, index):
        try:
            self._use_live(cap.DeviceSource(index), f"Device {index}")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Device capture failed", str(e))

    def _on_tab_changed(self, idx):
        if self.tabs.tabText(idx) == "SIGNAL CHAIN":
            self.chain_list.rebuild(self.effects)

    def _reorder_chain(self, ids):
        by_id = {e.id: e for e in self.effects}
        new = [by_id[i] for i in ids if i in by_id]
        for e in self.effects:              # safety: keep any not listed
            if e not in new:
                new.append(e)
        self.effects = new
        self.pipeline.effects = new         # atomic rebind (engine reads this)

    def _toggle_chain(self, eid, on):
        e = next((x for x in self.effects if x.id == eid), None)
        if e is None:
            return
        e.enabled_param().set_base(1.0 if on else 0.0, notify=False)
        # keep the matching control panel checkbox in sync (OSD has no panel)
        target = next((p for p in self.panels if p.effect.id == eid), None)
        if target is not None:
            target.enable.blockSignals(True)
            target.enable.setChecked(on)
            target.enable.blockSignals(False)

    def _toggle_preview(self, on):
        self.engine.set_preview(on)
        self.video.preview_off = not on
        self.video.update()

    def _on_preview_res(self, idx):
        w = self._preview_opts[idx][1]
        custom = (w == -1)
        self.custom_w.setEnabled(custom)
        self.engine.set_preview_width(self.custom_w.value() if custom else w)

    def _on_custom_w(self, px):
        if self._preview_opts[self.preview_cb.currentIndex()][1] == -1:
            self.engine.set_preview_width(px)

    def _on_sync(self, ms):
        self.engine.set_av_offset(ms / 1000.0)

    # ---- output sinks (virtual camera / recorder) ----------------------
    def _populate_output_menu(self):
        m = self.transport.output_menu
        m.clear()
        vc = QAction("Virtual Camera", m)
        vc.setCheckable(True)
        vc.setChecked(self._vcam.enabled)
        vc.toggled.connect(self._toggle_vcam)
        if not VirtualCamOutput.available():
            vc.setText("Virtual Camera (needs OBS / v4l2loopback)")
        m.addAction(vc)
        m.addSeparator()
        if self._recorder.active():
            r = QAction("Stop Recording", m)
        else:
            r = QAction("Record Processed Video\u2026", m)
        r.triggered.connect(self._toggle_record)
        m.addAction(r)

    def _refresh_sink(self):
        active = self._vcam.enabled or self._recorder.active()
        self.engine.set_frame_sink(self._output_frame if active else None)

    def _output_frame(self, bgr):
        if self._vcam.enabled:
            self._vcam.send(bgr)
        if self._recorder.active():
            self._recorder.write(bgr)

    def _toggle_vcam(self, on):
        if on:
            self._vcam.fps = self.engine.fps or 30.0
            if not self._vcam.start():
                QMessageBox.warning(self, "Virtual Camera",
                                    "Couldn't start the virtual camera:\n\n"
                                    f"{self._vcam.error}\n\n"
                                    "On Windows/macOS install OBS (its virtual "
                                    "camera), on Linux load v4l2loopback.")
                return
            dev = self._vcam.device or "virtual camera"
            self.statusBar().showMessage(
                f"Streaming to {dev} - press Play. Pick it as the camera in "
                "OBS / Zoom / Discord.", 8000)
        else:
            self._vcam.stop()
        self._refresh_sink()

    def _toggle_record(self):
        if self._recorder.active():
            path = self._recorder.stop()
            self._refresh_sink()
            if path:
                self.statusBar().showMessage(f"Saved recording: {path}", 6000)
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Record processed video to", "vhs-capture.mp4",
                "MP4 video (*.mp4)")
            if not path:
                return
            self._recorder.start(path, self.engine.fps or 30.0)
            self._refresh_sink()
            self.statusBar().showMessage(
                "Recording processed video (no audio) - press Play; stop from "
                "the OUTPUT menu.", 8000)

    def _update_stats(self):
        st = self.engine.stats()
        w, h = st["width"], st["height"]
        res = f"{w}x{h}" if w else "-"
        if st["playing"]:
            fps = st["fps"]
            target = self.engine.fps or 30.0
            pct = (fps / target * 100.0) if target else 0.0
            warn = "  ⚠ SLOW" if pct < 70 else ""
            desync = st["desync_ms"]
            lock_thr = 1000.0 / max(1.0, target) * 0.8
            av = (f" | A/V {desync:+.0f}ms" if abs(desync) > lock_thr
                  else " | A/V lock")
            drop = st.get("dropped", 0)
            drops = (f" | drop {drop} ({st.get('drop_pct', 0):.0f}%)"
                     if drop else "")
            self.stats_lbl.setText(
                f"▶ {fps:4.1f} fps ({pct:3.0f}%) | {st['backend']} | "
                f"dec {st.get('decode','?')} | {res}{av}{drops}{warn}")
        else:
            self.stats_lbl.setText(
                f"● idle | dec {st.get('decode','?')} | {res}")

    def _audio_clock(self):
        ap = self.audio_player
        return ap.source_time() if ap is not None else None

    def _make_audio(self, path):
        if self.audio_player:
            self.audio_player.stop(); self.audio_player = None
        loaded = load_audio(path)
        if loaded:
            samples, sr = loaded
            self.audio_player = QtAudioPlayer(
                samples, sr,
                speed_at=lambda t: (motor_speed(self.params, t)
                                    if motor_affects_audio(self.params) else 1.0))
        # audio-reactive analysis (used by preview + render), aligned to frames
        n_frames = int(round(self.engine.duration * self.engine.fps))
        analysis = build_analysis(path, self.engine.fps, n_frames)
        set_analysis(analysis)
        self.meter.set_active(analysis is not None)

    def _toggle_play(self, on):
        if on:
            # resync: start the audio from the exact frame the video is paused
            # on, so they resume together (fixes drift accrued before pausing).
            if self.audio_player:
                self.audio_player.seek_seconds(self.engine.current_time())
            self.engine.play()
            if self.audio_player:
                self.audio_player.play()
        else:
            self.engine.pause()
            if self.audio_player:
                self.audio_player.pause()

    def _on_parameter_changed(self, pid, value):
        if pid.endswith(".enabled"):
            self._filter_panels()
        # Auditioning a slider is never an instruction to write automation.
        if self.transport.pb_btn.isChecked():
            self.transport.pb_btn.setChecked(False)
            self.statusBar().showMessage("Manual tweak: keyframe preview off. Use + KEY to save a keyframe.", 3500)

    def _add_parameter_key(self, pid):
        if not self.keyframes.isEnabled():
            self.statusBar().showMessage("Load a video file to edit time-based keyframes.", 3000)
            return
        self.keyframes.control.setCurrentIndex(self.keyframes.control.findData(pid))
        self.keyframes.add_key()
        self._focus_timeline()

    def _keyframes_changed(self):
        if self.transport.pb_btn.isChecked():
            self.params.reset_motion(self.timeline)
        self.keyframes.refresh()

    def _seek_keyframe(self, t):
        self.transport.play_btn.setChecked(False)
        self.engine.pause()
        self.transport.pb_btn.setChecked(True)
        self.engine.seek_seconds(t)
        if self.audio_player:
            self.audio_player.seek_seconds(t)
        self.keyframes.set_playhead(t, self.engine.duration, self.engine.fps)

    def _toggle_playback(self, on):
        if on:
            self.engine.set_mode(Mode.PLAYBACK)
        else:
            self.engine.set_mode(Mode.LIVE)

    def _render(self):
        if getattr(self, "_worker", None) and self._worker.isRunning():
            return
        if not self.source_path:
            QMessageBox.information(self, "Render", "Load a video first."); return
        out, _ = QFileDialog.getSaveFileName(
            self, "Render to", "render.mp4", "Video (*.mp4)")
        if not out:
            return
        self.engine.pause(); self.transport.play_btn.setChecked(False)
        full = QMessageBox.question(
            self, "Render", "Full resolution? (No = preview width)") \
            == QMessageBox.Yes
        source_width = self.engine._source.width
        max_w = None if full else self.engine._effective_width(source_width)
        self._dlg = QProgressDialog("Rendering\u2026", "Cancel", 0, 100, self)
        self._dlg.setWindowModality(Qt.WindowModal)
        self._worker = RenderWorker(self.source_path, out, self.params,
                                    self.timeline, self.effects, max_w)
        self._worker.progress.connect(lambda f: self._dlg.setValue(int(f * 100)))
        self._dlg.setAutoClose(False)
        self._dlg.canceled.connect(self._worker.requestInterruption)
        self._worker.done.connect(self._render_done)
        self._worker.failed.connect(self._render_failed)
        self._worker.cancelled.connect(self._render_cancelled)
        self.transport.render_btn.setEnabled(False)
        self._worker.finished.connect(lambda: self.transport.render_btn.setEnabled(bool(self.source_path)))
        self._worker.start()

    def _render_done(self, path):
        self._dlg.close()
        QMessageBox.information(self, "Render", f"Wrote:\n{path}")

    def _render_failed(self, message):
        self._dlg.close()
        QMessageBox.critical(self, "Render failed", message)

    def _render_cancelled(self):
        self._dlg.close()
        self.statusBar().showMessage("Render cancelled. Destination file unchanged.", 5000)

    def _save_session(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save take", "take.json", "Session (*.json)")
        if path:
            self.session.order = [e.id for e in self.effects]
            try:
                self.session.save(path)
            except (OSError, ValueError, TypeError) as error:
                QMessageBox.warning(self, "Save take failed", str(error))

    def _open_session(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open take", "", "Session (*.json)")
        if not path:
            return
        self.engine.pause()
        self.transport.play_btn.setChecked(False)
        self.transport.pb_btn.setChecked(False)
        try:
            self.session.load(path)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            QMessageBox.warning(self, "Open take failed", str(error))
            return
        self.pipeline.request_reset()
        self.keyframes.refresh(clear_history=True)
        for widget, value in ((self.text_edit, self.text_effect.custom_text),
                              (self.corner_edit, self.text_effect.corner_text)):
            widget.setText(value)
        self.clock_cb.setCurrentIndex(self.text_effect.clock_mode)
        self.manual_clock_cb.blockSignals(True)
        self.manual_clock_cb.setChecked(self.text_effect.manual_clock)
        self.manual_clock_cb.blockSignals(False)
        self.manual_datetime_edit.setEnabled(self.text_effect.manual_clock)
        self.clock_run_cb.setEnabled(self.text_effect.manual_clock)
        self.manual_datetime_edit.setDateTime(QDateTime.fromString(
            self.text_effect.manual_datetime, Qt.ISODate))
        self.clock_run_cb.setChecked(self.text_effect.clock_run)
        font_name = next((n for n, p in self._fonts.items()
                          if p == self.text_effect.font_path), None)
        if font_name:
            self.font_cb.setCurrentText(font_name)
        if self.session.order:
            self._reorder_chain(self.session.order)
            self.chain_list.rebuild(self.effects)
        if self.session.source_path and not os.path.exists(self.session.source_path):
            replacement, _ = QFileDialog.getOpenFileName(
                self, "Locate missing take video: " + os.path.basename(self.session.source_path),
                "", "Video (*.mp4 *.mov *.avi *.mkv)")
            if replacement:
                self.session.source_path = replacement
            else:
                QMessageBox.warning(self, "Take video missing",
                    "Loaded the controls and keys only. The current input is unchanged; load a video before exporting this take.")
        if self.session.source_path and os.path.exists(self.session.source_path):
            try:
                self.engine.set_source(FileSource(self.session.source_path))
                self._make_audio(self.session.source_path)
                self.source_path = self.session.source_path
                self.transport.set_file()
                self.keyframes.setEnabled(True)
            except Exception as error:  # noqa: BLE001
                QMessageBox.warning(self, "Take video failed", str(error))
        self._sync_panels(force=True)

    # --------------------------------------------------------------- signals
    def _on_position(self, t, index):
        self.transport.set_position(t, self.engine.duration)
        self.meter.set_levels(levels_at(t))
        self.keyframes.set_playhead(t, self.engine.duration, self.engine.fps)

    def _on_finished(self):
        self.transport.play_btn.setChecked(False)
        if self.transport.pb_btn.isChecked():
            self.transport.pb_btn.setChecked(False)

    def _sync_panels(self, force=False):
        if not force and self.engine._mode != Mode.PLAYBACK:
            return
        for p in self.panels:
            p.refresh()
        self.osd_panel.refresh()
        self._filter_panels()

    # --------------------------------------------------------------- close
    def closeEvent(self, e):
        worker = getattr(self, "_worker", None)
        if worker and worker.isRunning():
            worker.requestInterruption()
            self.statusBar().showMessage("Stopping export; close again when it finishes.", 5000)
            e.ignore()
            return
        self.engine.stop()
        if self.engine.isRunning():
            self.statusBar().showMessage("Finishing the current frame; close again in a moment.", 3000)
            e.ignore()
            return
        if self.audio_player:
            self.audio_player.stop()
        self._vcam.stop()
        self._recorder.stop()
        super().closeEvent(e)
