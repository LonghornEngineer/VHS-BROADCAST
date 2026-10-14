# Agent API (version 1)

`vhs-agent` is a local CLI and importable Python module. No GUI application,
HTTP server, credentials, agent framework or model is required. It reads and
writes the same version-2 take format as the desktop app. It does not remote-control
a running GUI or manage live capture. Use it for repeatable file workflows.

## Discover, author, validate, render

```sh
uv run vhs-agent catalog
uv run vhs-agent create --spec examples/block-glitch.json --output take.json
uv run vhs-agent validate --session take.json
uv run vhs-agent render --session take.json --input clip.mp4 --output out.mp4 --width 1280
```

`catalog` lists stable effect IDs, parameter IDs, defaults, bounds, steps, named
choices and effect-specific settings. Use returned IDs instead of guessing from
labels. Choice values are zero-based indices; enable an effect via `<id>.enabled`.
Setting a slider does not implicitly enable its effect.

`create` accepts a partial native take with numeric parameter shorthand. It expands
all controls to a complete native take. You can pipe JSON to `--spec -` or use a
UTF-8/UTF-8-BOM file. Native takes accepted by `validate` and `render` use objects
such as `{"value": 0.5}` rather than shorthand numbers.

Set `source` to your video path, or pass `render --input` to override it. Relative
paths resolve from the process working directory, not the spec/take directory.
`create` stores the source as an absolute path. Existing source media isn't required
to author a take. Validation reports missing source as a warning; render requires it.
Validation checks structure and controls, not codec decodability or visual quality.

`order` is a list of unique effect IDs. Listed effects run first in that order;
unlisted effects follow in their default order. It does not enable effects.

## Keyframes and modulation

```json
{
  "source": "clip.mp4",
  "parameters": {
    "text.enabled": 0,
    "block_corrupt.enabled": 1,
    "block_corrupt.density": 0
  },
  "timeline": {
    "tracks": {
      "block_corrupt.density": {
        "t": [0, 2, 4],
        "v": [0, 0.6, 0],
        "interpolation": "smooth"
      }
    }
  }
}
```

Times are seconds, strictly increasing and nonnegative. Times/values must have
equal lengths. Values must be finite and within the catalog bounds. Integer
controls require integers. Interpolation is `hold`, `linear` or `smooth`; named
choices and toggles require `hold`. Endpoint values extend outside the keyed range.
Creating keys is always explicit. An empty timeline is a static take.

For LFO/audio settings, use the full parameter object, e.g.:

```json
{"value": 0.4, "mod": {"enabled": true, "shape": "sine", "depth": 0.1, "rate": 0.5}}
```

`mod` supports enabled, shape (`sine`, `linear`, `random`, `off`), depth, rate,
phase and seed. For exact audio fields/defaults, inspect the complete native take
produced by `create`. Trigger audio uses `enabled`, `mode: "trigger"`, `band`
(`low`, `mid`, `high`, `level`), `low_hz`, `high_hz`, `threshold` (0–1), `ramp`
(seconds) and `rest`; parameter value/keyframes provide the target.

`effect_settings` stores non-slider settings. Discover keys with `catalog`; for
example `text` contains text, clock and font settings, and `luma_key` stores a
background image reference. Paths refer to local files, so review untrusted specs
before executing them. Media isn't bundled inside takes.

### Changing OSD text over time

`text.text_keys` changes the OSD text partway through a video. Each key is an object
with `t` (seconds, nonnegative) plus any of `custom_text`, `corner_text`,
`clock_mode` (0–5) and `manual_datetime`. A key holds until the next key; fields it
omits keep their earlier value, and before the first key the base settings apply.
With `manual_clock` and `clock_run` on, a keyed date counts up from its own key time.

OSD positions are normal parameters and accept keyframe tracks: `text.x` / `text.y`
(label, left-anchored), `text.clock_x` / `text.clock_y` (clock, right-anchored;
defaults 0.97 / 0.06) and `text.tag_x` / `text.tag_y` (tag, left-anchored; defaults
0.03 / 0.06). Values are fractions of frame width and height.

```json
{
  "effect_settings": {
    "text": {
      "manual_clock": true, "clock_mode": 3, "manual_datetime": "1987-03-14T15:42:00",
      "text_keys": [
        {"t": 164.4, "manual_datetime": "1997-03-14T19:20:00"},
        {"t": 276.0, "manual_datetime": "2026-03-14T18:05:00", "corner_text": ""}
      ]
    }
  }
}
```

## Results and errors

Successful commands write one JSON object to stdout:

```json
{"ok": true, "api_version": 1, "result": {"path": "/absolute/path/take.json", "session_version": 2}}
```

Runtime/validation failures return exit code 1 and
`{"ok": false, "api_version": 1, "error": "message"}`. Success is code 0.
Argument syntax errors use argparse's stderr message and exit code 2. Diagnostics
go to stderr; parse stdout as JSON. Catalog's result contains its effect metadata;
validate's result contains `valid`, `warnings` and the track count.

Create/render refuse existing output paths unless `--overwrite` is passed.
Render refuses replacing its input video even with overwrite. Width is a pixel
cap of at least 16 pixels, defaults to 720, and `--width 0` requests full resolution. Use `--no-audio`
for silent export. Rendering uses a temporary destination and commits on success.
Interrupting the process does not return a structured completion result.

Always verify the returned exit code and output. Wall-clock OSD is nondeterministic;
disable it or set a manual time for reproducible output. History effects require
processing from the beginning; they do not reconstruct skipped frame history.

## Python

```python
from vhs_broadcast.api import catalog, create_take, validate_take

effects = catalog()
create_take({"parameters": {"text.enabled": 0}}, "take.json")
report = validate_take("take.json")
```

Functions return plain dictionaries and raise on invalid inputs. For rendering,
prefer a subprocess invoking `python -m vhs_broadcast.api render ...`. The render
engine uses process-global audio analysis, so parallel jobs should run in separate
processes. `python -m vhs_broadcast.api` is equivalent to the installed `vhs-agent`
command. The GUI and agent can exchange take files; neither watches them for changes.
