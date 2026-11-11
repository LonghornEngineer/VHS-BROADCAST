# MCP server

`vhs-mcp` is an optional [Model Context Protocol](https://modelcontextprotocol.io)
server over stdio. It gives agents typed tools for the same workflow as the
[Agent API](agent-api.md) CLI, plus frame previews returned as images, so an agent
can look at its result and iterate before rendering. It is a thin layer over
`vhs_broadcast.api`: validation, take normalization and rendering are the same code.

## Install

The server needs the `mcp` extra (the official Python SDK, version 2):

```sh
uv sync --extra mcp
uv run --extra mcp vhs-mcp      # speaks MCP on stdin/stdout; clients start it for you
```

Claude Code:

```sh
claude mcp add vhs-broadcast -- uv run --directory /path/to/VHS-BROADCAST --extra mcp vhs-mcp
```

Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "vhs-broadcast": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/VHS-BROADCAST", "--extra", "mcp", "vhs-mcp"]
    }
  }
}
```

Relative paths in tool arguments resolve from the server's working directory
(the repo with `--directory`); prefer absolute paths. Examples are read from the
repo's `examples/` folder, so run the server from a checkout.

## Tools

| Tool | Returns |
| --- | --- |
| `catalog(stage=None, search=None, summary=False)` | Effects, parameter IDs, bounds, choices, defaults. `stage` keeps one signal stage, `search` matches effect/parameter IDs and labels, `summary` lists parameter IDs only. |
| `list_examples()` | Example names with their enabled effects and keyed parameters. |
| `get_example(name)` | An example spec to edit. Its `source` is null. |
| `create_take(spec, output_path, overwrite=False)` | Writes a complete native take; returns `path`, `session_version`. |
| `validate_take(path)` | `valid`, `warnings`, `tracks`. |
| `preview_frame(spec_or_take, source_video_path=None, time_s=0, width=640, history_s=1)` | One PNG of the processed frame at `time_s`. |
| `preview_strip(spec_or_take, times, source_video_path=None, width=320, history_s=1)` | Up to 12 PNGs, one per time, in the given order. |
| `render(take_path, output_path, source_video_path=None, full=True, with_audio=True, overwrite=False)` | Renders an MP4 and returns its absolute `path`. Reports progress. `full=False` caps width at 720 px. |

`spec_or_take` is either a spec object (as for `create_take`; nothing is written)
or a path to a native take. Without `source_video_path`, the take's `source` is used.

Errors come back as tool errors carrying the API's message, e.g.
`Unknown parameter: made.up` or `block_corrupt.density: value must be in [0.0, 1.0]`,
never a traceback.

### Previews and history effects

A preview renders like the offline renderer: the same frame selection (including
the motor wow/flutter warp), timeline, LFOs, audio-reactive controls and FPS
hold. `width` caps the processing width just like render's width; it never
upscales. Effects with frame history (feedback, smear, temporal displacement,
block corruption, phosphor, camcorder, deck, sync loss, rolling shutter, Game Boy,
FMV, codec) start empty at a seek. A preview therefore first processes
`history_s` seconds of earlier frames (default 1 s). Temporal effects discard
history after a gap of more than 1 s, so one second gives the same frame as a full
render in the common case. Codec GOPs and datamosh, or history settings longer
than a second, can reach further back. Raise `history_s` (max 10) for those, or set it to 0
when only static effects matter. A full `render` always runs from the first frame.

Previews and renders run one at a time because the audio analysis is process-wide.

## Suggested agent loop

1. `catalog(summary=True)` for the map, then `catalog(stage=...)` or
   `catalog(search=...)` for the bounds of the effects you will use.
2. `list_examples()` and `get_example(name)` for a starting spec.
3. `preview_frame(spec, source_video_path=clip, time_s=...)` and look at it.
   Use `preview_strip` across keyframe times for animated takes.
4. Edit the spec (IDs and ranges from `catalog`) and preview again until it looks right.
5. `create_take(spec, "take.json")`, then `render("take.json", "out.mp4", clip)`.
