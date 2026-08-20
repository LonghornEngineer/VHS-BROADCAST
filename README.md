# VHS BROADCAST DECK

A retro broadcast-equipment control surface for applying live, tunable VHS
effects to video — with a **recordable, deterministic automation layer** so you
can perform the knobs like an instrument and re-render the exact take.

![before / after](demo_before_after.png)

## What it does

- Loads an `mp4` (also `mov/avi/mkv`), starts **paused** until you hit play.
- Real-time VHS effect stack: signal, **slew/misalignment**, wobble, tracking,
  **per-channel chroma**, **RF ghost**, **TV static**, **head-switching bar**,
  fuzz, scanlines/vignette, tearing, **scramble** (analog premium-channel
  sync-suppression look), **downsample** (pixelate + an **FPS %** slider that holds frames for a choppy low-framerate look), **bit-crush** (posterize),
  **interlace** (field lines + comb), **laser rot** (LaserDisc bit-rot: coloured raster-line sparkle that blends into the picture and clusters into drifting patches) — every effect **toggleable**, every
  parameter **live-adjustable**. Controls are laid out in **three compact
  columns** (no scrolling on a normal window).
- **Audio-reactive modulation**: any parameter can be driven by the audio, in
  parallel with the LFO. Pick a **band** (LOW/MID/HIGH/LEVEL), a signed
  **depth**, a **gain** (sensitivity) and **smooth** (snappy transients ↔ gentle
  swells). A 4-band meter under the video shows what the audio is doing. Because
  the band envelopes are precomputed and sampled as a pure function of time, a
  reactive take renders **bit-identically** to the preview.
- **Per-channel chroma**: independent signed Red / Green / Blue horizontal
  shift, so you can smear just one colour.
- **Tracking V-Roll** is signed and can **free-run**: set a *Roll Rate* and the
  picture rolls continuously (wraps). The rate has a quadratic response so small
  values stay gentle instead of going chaotic.
- **Audio playback** works out of the box (Qt Multimedia — no extra native libs)
  and is wobbled by the motor when *Affect Audio* is on.
- **MOTOR (wow/flutter)**: warps playback speed like a worn tape motor — slow
  wow + fast flutter — and can **optionally warp the audio** too (pitch/tempo).
- **Live inputs**: besides video files, feed the effects from a **screen /
  region**, a **window**, or a **camera / capture card** — so a real VHS deck on
  a USB dongle can run straight through the stack. Pick via the **INPUT ▾** menu.
- **OSD / text overlay** using packaged **TTF fonts** (VT323, Share Tech Mono,
  Press Start 2P — pick from a dropdown), composited *before* the analogue
  effects so the text degrades with the picture. OSD updates live even while
  paused. Transport glyphs (▶ ⏸ ■ ● ⏪ ⏩) are drawn as shapes so they show in
  any font — a **GLYPHS** picker under the LABEL field inserts them (▶ ⏸ ■ ●
  ⏪ ⏩ ⏮ ⏭ ⏏ ◀ ▲ ▼). Clock modes: **blinking 12:00** (unset-VCR, the default), **wall
  time**, **date**, **date+time**, and a deterministic **timer**.
- **"Wavy" modulation** per parameter: sine / linear / random drift, each with a
  depth and a rate (the time aspect). Random is hash-seeded, so it's identical
  every render.
- **Record your control moves** against video time (like MIDI/keyframes), then
  **PLAYBACK** to replay them, or **RENDER** to a new file — deterministically.
- Runs under **uv**, with Windows `.bat` and Linux/macOS `.sh` launchers.

## Run it

Install [uv](https://astral.sh/uv), then:

```bash
# Linux / macOS
./run.sh
# Windows
run.bat
# or directly
uv run vhs-broadcast
```

First launch, `uv` resolves PySide6 + OpenCV + numpy into an isolated env.

## The workflow (record → playback → render)

1. **LOAD** a clip (it opens paused).
2. Arm **● REC** (this clears the previous take), then **▶ PLAY**. Move sliders,
   flip effects, twist the `~` modulation strips — everything is captured with a
   timestamp.
3. Hit **⏩ PLAYBACK** to replay the take; the controls now follow your recording.
4. **SAVE TAKE** writes a `.json` (parameters + modulation + automation tracks).
5. **RENDER** bakes it to an `mp4` (offscreen, choose preview or full res).

Headless re-render, no GUI:

```bash
./render.sh take.json out.mp4 --full        # or render.bat on Windows
uv run vhs-render --session take.json -o out.mp4 --full
```

Because `Parameter.evaluate(t)` is a pure function of time, the preview and the
render produce **bit-identical** frames (the only exception is the wall-clock
OSD, which intentionally shows real time — use TIMER mode for reproducibility).

## Architecture (and where to extend)

```
core/
  parameters.py   Parameter + Modulator (LFO). Everything the user touches is a
                  Parameter with an id -> automation & modulation fall out free.
  automation.py   Recorder + Timeline (keyframe tracks) + Session (save/load).
  pipeline.py     Effect stack runner, VideoSource (FileSource), offline render.
  engine.py       PlaybackEngine QThread: LIVE (record) vs PLAYBACK (replay).
effects/
  base.py         Effect base (enable toggle + param registration).
  vhs.py          The effects, in pipeline order.
ui/               PySide6 broadcast-deck interface (custom Knob, param rows with
                  the ~ mod strip, effect panels, transport, video view).
```

Deliberate seams for the roadmap you outlined:

- **Screen / window / capture-device input** — subclass `VideoSource`
  (`ScreenSource`, `DeviceSource`) in `core/pipeline.py`; the engine and UI don't
  care where frames come from. `FileSource` is the reference implementation.
- **MIDI / USB controller** — map incoming CC messages to
  `ParameterSet.get(id).set_base(value)`. Because that path already notifies the
  recorder, hardware moves get recorded exactly like mouse moves. (Add a
  `core/midi.py` reader thread emitting `(param_id, value)`.)
- **Timeline / keyframe editor** — `Timeline.tracks` is `dict[id -> (times,
  values)]`, already sorted and JSON-serialisable; a graphical editor just needs
  to read/write those arrays. Hold-last interpolation lives in `Track.value_at`.
- **GPU backend** — effects are isolated behind `Effect.process`; a moderngl/
  shader backend can replace the numpy bodies without touching automation or UI.

## Live input

Use **INPUT ▾** in the transport to choose a source:

- **Load Video File…** — the normal file path (seekable, deterministic render).
- **Screen** — a whole display or the full desktop (via `mss`).
- **Window** — a specific window, tracked as it moves/resizes (Windows/macOS;
  needs `pygetwindow`). It grabs the window's on-screen region.
- **Device (Webcam / Capture Card)** — any OpenCV-visible capture device; a USB
  VHS dongle lets you process a real tape live.

Live sources are real-time and infinite, so **scrubbing and deterministic
render are disabled** while one is active (there are no stored frames to
re-render). PLAY starts/stops the grab, and REC still captures your control
automation over time. Recording the *processed live output* straight to a file
is the natural next step.

## Audio

- **Live playback uses Qt Multimedia**, which ships inside PySide6 (its own
  bundled FFmpeg backend) — no PortAudio or extra native libraries required.
- Decoding and render-time muxing use **PyAV**.
- The motor wow/flutter uses one shared speed curve, so picture and sound wobble
  together when *Affect Audio* is on. Offline render is sample-accurate.

### Audio reactivity

Toggle the **♪** button on any control to make it react to the sound:

- **BAND** — LOW (kick/bass), MID (vocals/snare), HIGH (hats/cymbals) or LEVEL
  (overall loudness).
- **DEP** — signed depth: how far (and which direction) the value moves.
- **GN** — gain/sensitivity applied before clamping, so quiet material still
  drives it and loud material doesn't peg.
- **SM** — smoothing: 0 punches on transients (great on the beat), 1 gives slow
  swells.

The envelopes are analysed once per clip (short-time FFT aligned to frames) and
sampled deterministically, so the render matches the preview exactly. `♪` and
`~` stack on the same knob.

## Fonts

Three OFL-licensed fonts are bundled (VT323, Share Tech Mono, Press Start 2P) and
appear in the OSD FONT dropdown.

For the authentic VCR look there's **VCR OSD Mono** (by Riciery Leal) -- a
third-party font with its own license, so it isn't bundled. The **run script
offers to fetch it on first launch** (prompts once; if you decline it won't ask
again -- delete `src/vhs_broadcast/assets/fonts/.vcr-skip` to be re-offered). You
can also fetch it directly:

```bash
python scripts/fetch_vcr_font.py            # prompts first
python scripts/fetch_vcr_font.py --yes      # no prompt
python scripts/fetch_vcr_font.py --zip vcr_osd_mono.zip   # from a local zip
```

It drops the .ttf into `src/vhs_broadcast/assets/fonts/`; the app picks it up in
the FONT dropdown automatically and uses it as the default OSD font. Source:
https://www.dafont.com/vcr-osd-mono.font

Any other `.ttf` dropped into that folder also shows up in the dropdown.

## Signal chain (reorderable)

The **SIGNAL CHAIN** tab lists every effect in processing order — including
**OSD / TEXT** as a movable item — with a checkbox for enabled state. Drag to
reorder: effects earlier in the list are applied first, so where OSD/TEXT sits
decides whether the text is part of the degraded signal or a clean overlay:

- **OSD early** → the text is a *tape/camera-generation* artifact: later effects
  (chroma, downsample, noise, wobble…) degrade it along with the picture.
- **OSD late** → the text is a *TV/monitor overlay*: it stays crisp on top of the
  already-degraded image.

The same logic lets you decide whether a given artifact enters at the tape,
player, or TV stage. The order is saved with the take. Reordering applies to the
CPU render (the canonical path); the GPU preview uses the standard chain order.

## Output (live sources -> virtual camera / recording)

File sources render deterministically offline (RENDER). Live sources (screen /
window / device) are real-time, so instead of an in-app render they go out
through the **OUTPUT ▾** menu:

- **Virtual Camera** — stream the processed picture to a system virtual camera,
  then pick it as the webcam in OBS, Zoom, Discord, Twitch, etc. Needs a
  backend: install OBS (its virtual camera) on Windows/macOS, or load
  `v4l2loopback` on Linux. Install the extra with `uv sync --extra virtualcam`.
  Video only — route audio through the host app (OBS captures desktop audio).
- **Record Processed Video…** — write the processed frames straight to an .mp4
  (video only) for a quick clip without OBS.

Both stream the preview-resolution output at the source frame rate. For a
full-resolution, sample-accurate result from a file, use RENDER instead.

## GPU (live preview)

The pipeline is **dual-path**:

- **CPU** (numpy/OpenCV) is the canonical, deterministic renderer. The offline
  **RENDER** always uses it, so re-renders stay bit-identical to what you
  designed — the whole automation/modulation determinism story is preserved.
- **GPU** (moderngl, an "uber" fragment shader) is an optional **live-preview /
  virtual-cam** backend for high frame rates and resolutions. Toggle it with the
  **GPU** button; if no usable GL context exists it stays on CPU and tells you.

The GPU path is **approximate, not bit-identical** — GLSL noise and float math
differ across drivers/GPUs — so it's live-only. OSD text is composited on the
CPU first (so it still degrades under the analog effects), then the shader
applies signal, chroma, wobble, tracking, ghost, static, grain, scanlines,
scramble and slew. The GPU path now also covers tearing, head-switch, downsample, bit-crush
and interlace; only per-speckle dropout and the motor time-warp remain CPU-only. Dependencies install by default.

If the **GPU** button says "unavailable": first check that the extra is
installed (`uv sync --extra gpu`). It is **not** related to how your displays
are connected (a dock/hub is fine) — the GPU still does the work. It means no
off-screen GL context could be created, which most often happens over Remote
Desktop or on a headless session. The app tries a native context, then EGL, then
borrows a Qt GL context before giving up, and always falls back to the CPU path.

## Performance readout

A small stats line under the meter shows, while playing: **preview FPS**, the
active **backend** (CPU/GPU), the processed **resolution**, and the FPS as a **percentage** of the source rate, and the **A/V** offset
("lock" when tight, or the residual/trim in ms). OSD rendering is cached and composited per-glyph (not full-frame), so the
overlay is nearly free. The readout flags **⚠ SLOW** if the frame rate falls
below ~70% of the source rate

A **PREVIEW** resolution selector (480p / 720p / 1080p / Full) sits above the
stats. The preview is capped for speed; at the low caps the CPU path is already
cheap, so the **GPU** button won't visibly help the *player* there -- its payoff
shows at higher preview resolutions and for the virtual-cam / record output,
which run at the chosen resolution. The preview downscale now uses a fast filter (the offline RENDER keeps the
high-quality one), which removed a large fixed per-frame cost. The remaining
per-frame floor is the video *decode* itself (software H.264) -- independent of
resolution or backend; only hardware decode would move it further. Decoding now uses **PyAV with
multithreaded (FRAME) decode across all CPU cores** (falling back to OpenCV if
needed), which is a large speedup on high-bitrate HD versus single-threaded decode. — handy for spotting when a
heavy stack wants the GPU path or a lower resolution.

## Notes

- Preview is capped to 720px wide for real-time performance
  (`config.PREVIEW_MAX_WIDTH`); rendering can use full resolution.
- CPU/numpy pipeline hits ~30fps at 720p on a typical machine. If you need
  1080p60 live, that's the moment to add the GPU backend above.
- Fonts under `assets/fonts/` are SIL OFL 1.1 (see the note there).
- **Audio is the master clock**: each frame, the video shows the position the
  speakers are actually at (buffer-compensated), so A/V stay locked even when
  heavy effects slow the video thread. A live **A/V SYNC** trim (under the
  meter, ±500 ms) nulls any fixed offset from your machine's audio/display
  latency — drag right if video lags the sound, left if it leads. With no audio
  track the video free-runs; the offline render is sample-accurate.

## Resolution & output

The **RESOLUTION** selector sets the processing resolution used for the whole pipeline in **one pass**. Besides the 480p/720p/1080p/Full presets you can pick **Custom** to type an exact width in pixels, and the **SCALE %** field scales that further (a fraction of the width, or of the source at Full); it never upscales past the source — the same processed frame feeds the on-screen preview, the **virtual camera**, and the **recorder** (the preview just scales it to the window). So raising it improves virtual-cam / recording quality (and costs more per frame); lowering it is faster. The GPU backend makes the high settings cheap, which is where it pays off.

## Hardware decode

On load, the file decoder tries the platform's **GPU video decoder** (D3D11VA / DXVA2 / QSV / CUDA on Windows, VideoToolbox on macOS, VAAPI on Linux) and uses it if it truly engages, otherwise falls back to multithreaded software decode. This offloads the heaviest per-frame cost (decoding hi-bitrate HD) to dedicated silicon — the same reason a media player can play a file your CPU struggles to software-decode. The stats line shows the active backend as `dec GPU:<type>` or `dec CPU`.


## Re-syncing A/V

Video is slaved to the audio clock, so if a drift ever creeps in, **pause and play again** — resuming re-seats the audio to the exact frame the video is paused on, so the two restart together. (When already in sync this is a no-op.) A handy recovery after a heavy moment or a resolution change.

## Drop frames & preview off (streaming)

Two toggles under RESOLUTION help when driving the virtual camera on a busy machine:

- **Drop frames** (default on): stay locked to the master clock, dropping frames that can't be rendered in time so the time domain / audio stays in sync. The stats line shows a running **drop N (X%)** counter — frames dropped and the percentage of total, reset each time you press play. Turn it off to show every frame (smoother, but it may drift). With no audio track this also holds real time via a wall clock.
- **Preview**: turn off the on-screen preview to skip the image conversion and GUI paint, leaving more headroom for the virtual camera / recorder (which keep running). The video area shows a PREVIEW OFF placeholder.
