"""MCP server: typed agent tools over the JSON agent API, plus frame previews.

    uv sync --extra mcp
    uv run vhs-mcp            # stdio transport

A thin layer over api.py, which stays the source of truth for validation and
take normalization. Needs the optional `mcp` extra; the app does not.
"""
import functools
import inspect
import json
import math
from pathlib import Path
import tempfile
import threading

import anyio
import cv2
try:
    from mcp.server.mcpserver import Context, Image, MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError as error:  # optional extra: say how to get it
    raise ImportError("vhs-mcp needs the mcp extra: uv sync --extra mcp, "
                      "then uv run --extra mcp vhs-mcp") from error

from . import api
from .config import PREVIEW_MAX_WIDTH

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
# TemporalEffect drops its history after a gap of more than 1 s, so one second of
# prior frames reproduces it. Long codec GOPs and multi-second history settings reach
# further back; history_s lets an agent pay for more when it matters.
DEFAULT_HISTORY_S = 1.0
MAX_HISTORY_S = 10.0
# Audio-reactive analysis is process-global, so renders and previews run one at a time.
_render_lock = threading.Lock()

mcp = MCPServer("vhs-broadcast", instructions=(
    "Author VHS/broadcast effect takes. Typical loop: catalog (filter by stage or search) "
    "-> get_example -> preview_frame on a source clip -> tweak the spec -> create_take -> render. "
    "Use exact IDs from catalog; enable an effect with '<id>.enabled': 1."))


def _tool(fn):
    """Register fn as a tool, reporting anticipated failures as plain messages.

    MCPServer hides the text of any exception that isn't a ToolError, and the API's
    ValueErrors are exactly what an agent needs to correct its spec.
    """
    def clean(error):
        if isinstance(error, ToolError):
            return error
        text = str(error) if isinstance(error, (ValueError, OSError)) else f"{type(error).__name__}: {error}"
        return ToolError(text or type(error).__name__)

    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except Exception as error:
                raise clean(error) from error
    else:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as error:
                raise clean(error) from error
    mcp.tool()(wrapper)
    return wrapper


@_tool
def catalog(stage: str | None = None, search: str | None = None, summary: bool = False) -> dict:
    """List effects with their exact parameter IDs, bounds, choices and defaults.

    Args:
        stage: Only effects in this signal stage (camera, digital, tape, receiver,
            screen, stylize, composite). The `stages` list gives their order.
        search: Case-insensitive text matched against effect and parameter IDs and labels.
        summary: Return each effect's id, label, stage and parameter IDs only; call again
            without it (filtered) for bounds and defaults.

    Choice parameters take zero-based indices. Setting a slider does not enable its
    effect: set `<effect>.enabled` to 1.
    """
    result = api.catalog()
    stages = [s["id"] for s in result["stages"]]
    if stage is not None and stage not in stages:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {stages}")
    needle = (search or "").lower()

    def matches(effect):
        fields = [effect["id"], effect["label"]] + [
            f for p in effect["parameters"] for f in (p["id"], p["label"])]
        return any(needle in f.lower() for f in fields)

    effects = [e for e in result["effects"]
               if (stage is None or e["stage"] == stage) and (not needle or matches(e))]
    if summary:
        effects = [{"id": e["id"], "label": e["label"], "stage": e["stage"],
                    "parameters": [p["id"] for p in e["parameters"]]} for e in effects]
    return dict(result, effects=effects)


def _example_path(name):
    path = EXAMPLES / f"{Path(name).stem}.json"
    if not path.is_file():
        raise ValueError(f"Unknown example {name!r}; call list_examples")
    return path


@_tool
def list_examples() -> list[dict]:
    """List the bundled example specs: name, enabled effects and keyed parameters.

    Fetch one with get_example and use it as a starting point for create_take or preview.
    """
    examples = []
    for path in sorted(EXAMPLES.glob("*.json")):
        spec = json.loads(path.read_text(encoding="utf-8-sig"))
        params = spec.get("parameters", {})
        enabled = [pid.removesuffix(".enabled") for pid, v in params.items()
                   if pid.endswith(".enabled") and (v.get("value") if isinstance(v, dict) else v)]
        examples.append({"name": path.stem, "effects": enabled,
                         "tracks": sorted(spec.get("timeline", {}).get("tracks", {}))})
    return examples


@_tool
def get_example(name: str) -> dict:
    """Return an example spec by name (from list_examples), ready to edit and pass on.

    Examples have `source: null`; give a source video to preview_frame/render, or set
    `source` in the spec.
    """
    return json.loads(_example_path(name).read_text(encoding="utf-8-sig"))


@_tool
def create_take(spec: dict, output_path: str, overwrite: bool = False) -> dict:
    """Validate a spec and write it as a complete native take (the GUI's format).

    Args:
        spec: Partial take: optional source, order, parameters, effect_settings and
            timeline. Parameters accept numeric shorthand ({"vhs.enabled": 1}) or full
            objects with mod/audio settings. Timeline tracks are
            {"<param>": {"t": [...], "v": [...], "interpolation": "hold|linear|smooth"}}.
        output_path: Where to write the take JSON. Relative paths resolve from the
            server's working directory.
        overwrite: Replace an existing file.

    Returns the absolute take path and session version.
    """
    return api.create_take(spec, output_path, overwrite=overwrite)


@_tool
def validate_take(path: str) -> dict:
    """Check a native take file. Returns valid, warnings (e.g. missing source) and track count."""
    return api.validate_take(path)


def _number(value, name, lo, hi=math.inf):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f"{name} must be a number in [{lo}, {hi}]")


@functools.lru_cache(maxsize=2)
def _analysis(path, mtime, fps, frames):
    from .core import audio_react
    return audio_react.build_analysis(path, fps, frames)


def _preview(spec_or_take, source, times, width, history_s):
    """Render the frames at `times` exactly as a render would, after history_s of warm-up."""
    from .core import audio_react
    from .core.pipeline import FileSource, render_frames, source_position
    if not times:
        raise ValueError("times must not be empty")
    for t in times:
        _number(t, "time_s", 0)
    if not isinstance(width, int) or width < 16:
        raise ValueError("width must be an integer of at least 16 pixels")
    _number(history_s, "history_s", 0, MAX_HISTORY_S)
    with tempfile.TemporaryDirectory() as folder, _render_lock:
        take = spec_or_take
        if isinstance(take, dict):
            take = api.create_take(take, str(Path(folder) / "take.json"))["path"]
        params, effects, timeline, source = api.load_take(take, source)
        src = FileSource(source, read_ahead=False)
        fps = src.fps or 30.0
        previous = audio_react.set_analysis(_analysis(
            str(Path(source).resolve()), Path(source).stat().st_mtime, fps, src.frame_count))
        try:
            warmup = round(history_s * fps)
            frames, index, stream = {}, None, None
            for target in sorted({round(t * fps) for t in times}):
                # Close targets share one pass; a far one restarts with fresh history.
                if stream is None or target - index > warmup:
                    start = max(0, target - warmup)
                    stream = enumerate(render_frames(
                        src, params, timeline, effects, max_width=width, start=start,
                        start_pos=source_position(params, timeline, fps, start)), start)
                for index, frame in stream:
                    if index == target:
                        frames[target] = frame.copy()
                        break
                else:
                    raise ValueError(f"{target / fps:.3f} s is past the end of the video")
        finally:
            audio_react.set_analysis(previous)
            src.release()
    return [frames[round(t * fps)] for t in times]


def _png(frame):
    ok, data = cv2.imencode(".png", frame)
    if not ok:
        raise ValueError("Could not encode preview PNG")
    return Image(data=data.tobytes(), format="png")


@_tool
async def preview_frame(spec_or_take: dict | str, source_video_path: str | None = None,
                        time_s: float = 0.0, width: int = 640,
                        history_s: float = DEFAULT_HISTORY_S) -> Image:
    """Render one frame with the take's effects applied and return it as a PNG.

    Look at the result, adjust the spec, and preview again before rendering.

    Args:
        spec_or_take: A spec object (as for create_take) or a path to a native take.
        source_video_path: Video to process; defaults to the take's `source`.
        time_s: Output time in seconds (keyframes and LFOs are evaluated here).
        width: Processing and output width cap in pixels, like render's width. Effects
            run at this size, so it is never upscaled past the source width.
        history_s: Seconds of earlier frames processed first so history effects (feedback,
            smear, block corruption, phosphor, codec) look as in a full render. 1 s covers
            most; raise it (max 10) for long codec GOPs or long history settings, or use 0
            for speed. Static effects are exact at any value.
    """
    frames = await anyio.to_thread.run_sync(
        _preview, spec_or_take, source_video_path, [time_s], width, history_s)
    return _png(frames[0])


@_tool
async def preview_strip(spec_or_take: dict | str, times: list[float],
                        source_video_path: str | None = None, width: int = 320,
                        history_s: float = DEFAULT_HISTORY_S) -> list[Image]:
    """Render several frames in one call, e.g. across keyframes; one PNG per time, in order.

    Times within history_s of each other share one pass, so a dense strip costs little
    more than its last frame. Arguments are as for preview_frame; at most 12 times.
    """
    if len(times) > 12:
        raise ValueError("At most 12 times per strip")
    frames = await anyio.to_thread.run_sync(
        _preview, spec_or_take, source_video_path, times, width, history_s)
    return [_png(f) for f in frames]


@_tool
async def render(take_path: str, output_path: str, source_video_path: str | None = None,
                 full: bool = True, with_audio: bool = True, overwrite: bool = False,
                 ctx: Context | None = None) -> dict:
    """Render a native take (from create_take) to an MP4 and return its absolute path.

    Runs the same offline renderer as `vhs-agent render`, from the first frame, so
    history effects are exact. Can take a while; progress is reported when supported.

    Args:
        take_path: Native take JSON.
        output_path: Destination .mp4; must differ from the source.
        source_video_path: Overrides the take's `source`.
        full: Full source resolution; False caps width at 720 px for a quick check.
        with_audio: Mux the source audio (wobbled to match the motor effect).
        overwrite: Replace an existing output file.
    """
    step = [-1]

    def progress(fraction):
        # Called on the render thread; forward whole-percent steps to the client.
        percent = int(fraction * 100)
        if ctx is not None and percent != step[0]:
            step[0] = percent
            try:
                anyio.from_thread.run(ctx.report_progress, percent, 100)
            except Exception:  # noqa: BLE001 - progress is best-effort, never fatal
                pass

    def run():
        with _render_lock:
            return api.render_take(take_path, output_path, source=source_video_path,
                                   width=0 if full else PREVIEW_MAX_WIDTH,
                                   with_audio=with_audio, overwrite=overwrite,
                                   progress=progress)
    return await anyio.to_thread.run_sync(run)


def main():
    mcp.run("stdio")


if __name__ == "__main__":
    main()
