# Effect reference


| Effect | Controls and behavior |
| --- | --- |
| Video Feedback | Previous output from this effect feeds the next frame; gain, decay, zoom, rotation, X/Y offset, mix/add/screen. |
| Datamosh / Motion Smear | Choose the original optical flow or explicit block matching against the preceding frame; block size, search radius, motion strength, persistence, held vectors and corrupted vectors. This is a frame-processing simulation, not destructive codec editing. |
| Composite Artifacts | Edge dot crawl, false color, horizontal chroma bandwidth, continuous phase drift. |
| Palette / Dither | Grayscale, green/amber CRT, CGA, EGA, Handheld, Sepia, Ice and Purple ramps, plus the multi-hue Riso Pop, Field and Dusk palettes (16–20 colours; Colors up to 32 uses them in full, fewer keeps an even dark-to-light spread); Bayer 2x2/4x4/8x8, noise, clustered dots and horizontal/diagonal lines; amount, cell size, gamma, bias and noise seed/animation. **Match: Luma** dithers brightness only and picks colours with brightness counting double, so hues stay true and the pattern reads as tone, best with the multi-hue palettes. **Block average** flattens each cell before matching for chunky pixels. **Grade** (Instant, Slide, Faded, Night) and **S-curve** recolour the matched colours afterwards, like a film stock. **Shimmer** makes each cell pulse on its own phase so the dither twinkles (Shimmer Hz sets the pace, Seed the pattern); **LED Gap** darkens the border of each cell for an LED-panel look when Cell px is 2 or more. Try `examples/purple-shimmer.json` and `examples/film-dither.json`. |
| Sync Loss | Digital signal dropout: black crush, low-bit colour with checker dither, lines smeared into solid bands (per channel with RGB Split), short frame freezes and near-black dropouts. Glitch Hz re-rolls the pattern; Burst, Freeze and Dropout set how often each tick goes heavy, holds or blacks out; Seed makes it repeatable. Try `examples/sync-loss.json`. |
| Temporal Displacement | Horizontal, vertical or radial slices of earlier frames; history seconds, depth and map animation. |
| Block Corruption | Freeze/repeat/shift rectangles; block size, density, hold time, displacement and chroma-only operation. |
| Phosphor Persistence | Highlight trails with separate R/G/B half-life seconds, threshold and strength. |
| CRT Geometry / Mask | Curved tube geometry, rounded corners, RGB triads or aperture grille, mask pitch and strength. |
| RF / Hum Bars | Structured moving dark bands and herringbone patterns; frequency, angle, drift, width and intensity. |
| Line Dither | Draws the subject as thin parallel lines that echo its outline: **Follow** measures each line in from the leading edge (default), trailing edge, or the whole outline (concentric rings), or Off for straight lines; only areas above Threshold get lines. **Lines from edge** limits how many lines grow in from the edge, so keyframing it builds the subject up line by line; 0 fills it. Bend pushes lines sideways with brightness on top. Spacing, line width, weight by light (brighter = thicker), smoothing, vertical/horizontal lines, colour (source, white, periwinkle, green, amber), background level and drift. Try `examples/line-dither.json`. |
| FMV / CD-ROM Video | Early-90s full-motion video: the picture drops to **Width px**, then is coded in 4x4 blocks like a vector-quantising CD-ROM codec. Quiet blocks collapse to one colour (**Flat blocks** sets how quiet; 1 flattens every block), detailed ones keep four 2x2 patches. **Luma levels** and **Chroma levels** give the 8-bit colour, with colour one step coarser than brightness and greys kept neutral, and **Dither** adds a fixed irregular mottle. **Skip blocks** keeps the previous frame's pixels where a block barely changed, so motion leaves sticky blocks behind, and **Frame rate** holds frames (0 = source rate). Try `examples/fmv.json`. |
| Luma Key / Threshold | Silhouette, stepped bands or keyed compositing; threshold, softness, levels, invert, and black/white/tint/original-feed/background-image fills. |

All are reorderable and disabled initially. Numeric controls retain audio/LFO
modulation and explicit keyframe buttons. Named modes use dropdowns and hold
keyframes. Resetting or tweaking a control does not automatically create keys.

### CRT raster and sync

- **CRT RASTER / SYNC**: simulates the beam drawing line by line while its
  horizontal sync keeps slipping. **Sync noise px** is how far each line's start
  wanders. It carries over from line to line, so edges form coherent jagged
  staircases rather than random fuzz. **Lock** sets how hard the sync circuit
  pulls back: low gives long swinging streaks, high fine choppy raggedness.
  **Kicks** adds rare sync spikes that leave big steps, **Crawl** moves the
  jagged pattern down (or up) the screen like a real beam, and **Churn** mixes in
  fresh noise each frame. **H-hold** makes later lines start later, so the
  picture slants and wraps into diagonal tear bars. **V-hold** rolls the picture
  with a dark **Blank bar** between frames. **Beam fade** dims lines the beam
  drew a while ago, the band you see when a CRT is filmed, with **Beam speed**
  for how fast it sweeps. Try `examples/crt-raster.json`.

### Streaks, tears and a shaking CRT

These controls were added to existing filters; their defaults keep the
original look, so older takes render unchanged (except CRT Bloom's glow, reworked below).

- **FUZZ** — **Shape** turns the grain into **Tall** vertical streaks, **Wide**
  horizontal ones or **Smear**, and runs the dropout dashes the same way. **Streak px** sets
  their length; with Tall, **Columns px** lines the dashes up in regular columns
  (0 scatters them). **Bands** swallows whole horizontal bands of the picture in
  blown-out static, from a few strips up to nearly the full frame at 1; the
  static follows Shape (Wide streaks, Tall barcode, Dots snow). **Band px** sets
  their typical height, **Band Hz** how often they jump, and **Band glow** how
  white they burn.
  With **Shape: Smear**, bands become smeared light instead of grain: long ragged streaks,
  rows of dashes, clusters of glowing dots and tapered comets, with the picture showing
  through between them;
  **Streak px** setting how long they run. Try `examples/smear-fuzz.json`.

- **TEARING** — **Direction: Vertical** shifts tall bands up and down instead of
  rows sideways. **Shape: Zigzag** steps rows sideways in a sawtooth so vertical
  edges turn into zigzags; **Zig px** sets the tooth height, **Amount** the
  swing, and **Slices** how far up from the bottom it reaches.
- **CRT BLOOM**: **Amount**, **Cutoff** and **Radius** now glow on any footage: Cutoff is relative
  to each frame's own highlights with a soft knee, the glow combines a tight core with wide
  halos, and it is screened on so bright scenes glow instead of clipping to white. **Shake px** keeps the whole picture moving slightly,
  **Line jitter px** shifts each scanline a little for ragged edges, and
  **Jumps /s** adds brief vertical-hold slips. These work even with Amount at 0,
  for shake without glow. Try `examples/analog-jitter.json`.

### Handheld and console glitches

- **LED MATRIX**: redraws the picture as a dot-matrix LED sign. Each cell of
  **LED px** becomes one round LED; **Threshold** decides which LEDs light (two
  **Levels** is pure on/off, more levels give dimmable LEDs above the threshold).
  **Dot size**, **Glow** and **Unlit LEDs** shape the panel, **Color** picks cool
  white, amber, red, green or blue, and **Invert** lights the dark areas instead.
  Try `examples/led-matrix.json`.
- **GAME BOY / POCKET VIDEO**: reduces the picture to a logical pixel grid and
  four brightness-indexed colors, then enlarges it without smoothing. Classic
  Green, Pocket Gray, Cream / Plum and Blue / Mint palettes are included. Choose
  Bayer or static-noise dithering, contrast, brightness, optional LCD trails and
  pixel-grid shading. The default fits square pixels to the source; the alternate
  160:144 grid stretches into the source frame. Start at Grid W 160, or 80 for
  chunkier pixels. Grid shading and partial mix introduce additional colors.
- **TILE SCRAMBLE**: indexed-color pixel art with displaced tiles,
  two-bit pixel damage, color-bank swaps in 2x2 tile regions, and shifted tile
  rows. Change Hz controls pattern refresh; zero holds it. Seed makes corruption
  repeatable. Zero damage retains the console color/grid treatment; Mix zero
  bypasses it. This retains the earlier NES / Tile Corruption rendering and the
  `nes_corrupt` parameter IDs so existing takes do not change.
- **BLOCK CORRUPTION** also covers repeated wrong-tile looks. **Tile Replace**
  borrows aligned tiles from the current video; **Patterns** chooses how many
  different fragments are reused, and **Clusters** groups the damage. **Lock
  Layout** holds the damaged locations and source addresses while the video keeps
  moving. Freeze, Repeat and Shift remain available with their original defaults;
  Repeat now supports multiple shared fragments. **Hold s** controls held-picture
  refresh, and unlocked layouts change on that interval. **Displace** applies to
  Freeze/Shift. Seed, Mix and Chroma Only give finer control. Untouched regions
  retain their original pixels, including with Chroma Only enabled. The separate
  Dirty Cartridge filter and its generated glyphs have been removed.
- **RGB CHANNEL TORTURE**: independently offset each output channel with wrapped
  edges, apply gain, reduce bit depth, invert it, and reroute the source RGB
  channels. Default settings leave the image unchanged; try R X at 0.02 and B X
  at -0.02, then reduce G Bits for a split-channel digital look.
- **PIXEL SORT**: sorts actual pixels inside bounded runs, horizontally or
  vertically, by brightness, hue, saturation, red, green or blue. Cutoff and High
  Cut bound the selected metric; Outside Band reverses the selection. Length caps
  each run. Full mix preserves each run's pixels/colors, while partial mix blends
  with the original. Existing takes retain their original horizontal brightness sort.

In **DATAMOSH / MOTION SMEAR**, choose **Block Matching** for block-based motion
vectors. Vector s holds detected vectors while the accumulated picture continues
moving. Bad Vectors deliberately replaces some vectors with incorrect motion.
Search px is measured in the bounded analysis image; larger searches and smaller
blocks cost more CPU time. Use RESET HISTORY (or keyed Reset pulse) to refresh the
picture. Defaults preserve the earlier optical-flow mode. Like other history
effects, seeking resets its history and paused refreshes never accumulate frames.

For dithering, start with **Grayscale**, 2–4 colors and **Bayer 4x4**. Compare
**Clustered Dots** for a printed look or **Diagonal Lines** for hatching.
**Amount** controls the pattern strength; zero still reduces the palette.
**Cell px** enlarges the pattern in processing pixels. **Gamma** shapes midtones
(1 is neutral; higher is brighter), while **Bias** shifts brightness (0 is neutral).
For **Noise**, disable **Animate Noise** to freeze its pattern and change **Seed**
for a different repeatable arrangement. Those two controls only affect Noise.
Original mode/palette indices and defaults are preserved for older takes; controls
absent from a loaded take reset to factory defaults instead of inheriting edits.

Feedback, motion smear, temporal displacement, block corruption and phosphor
persistence use frame history. **RESET HISTORY** clears an effect immediately;
to schedule a reset, manually key **Reset pulse** from off to on, then off before
the next reset. Holding the pulse on does not continuously clear the effect.
Paused preview refreshes never add frames or decay the stored history.

History clears on seek (including forward seek), source/resolution/chain changes,
disable/re-enable, or a playback gap longer than one second. Temporal displacement
keeps at most 120 frames within a 96 MiB buffer budget; high resolutions can
therefore retain less than the requested history duration. Exports start with
clean, independent buffers and replay every output frame. Skipped preview frames
or seeking into the middle of a take can produce different temporal trails from
a full export; play from the start without dropped frames to compare them.

Luma key composites into the existing BGR video output; it does not export an
alpha channel. **LOAD BACKGROUND IMAGE** selects a still image (a separate video
input is not implemented). Its absolute file path saves with the take; keep the
image available for headless renders. A missing image is labeled in the panel
and falls back to black. **Original Feed** reveals the unprocessed input under
the effect chain rather than a second file.

Preview, virtual camera, recording and offline exports use the CPU effect stack.
The optional GPU effect renderer and its dependencies have been removed.
Hardware video decoding is independent and remains available.
