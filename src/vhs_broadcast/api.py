"""Local JSON interface for agents and scripts. No GUI or network server."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import math
from pathlib import Path
import sys
import tempfile

from .core.automation import Session, Timeline
from .core.parameters import ParameterSet
from .effects.vhs import default_effects

API_VERSION = 1


def _model():
    params = ParameterSet()
    effects = default_effects(params)
    timeline = Timeline()
    return params, effects, timeline, Session(params, timeline, effects)


def catalog() -> dict:
    """Discover exact effect/parameter IDs, bounds, choices and defaults."""
    _, effects, _, _ = _model()
    return {"api_version": API_VERSION, "session_version": Session.VERSION,
            "effects": [{"id": e.id, "label": e.label,
                         "settings": e.settings() if hasattr(e, "settings") else {},
                         "parameters": [{"id": p.id, "label": p.label, "default": p.default,
                                         "minimum": p.minimum, "maximum": p.maximum,
                                         "step": p.step, "choices": list(p.choices)}
                                        for p in [e.enabled_param(), *e.own_params()]]}
                        for e in effects]}


def _number(value, name):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name}: expected a finite number")


def _value(param, value):
    _number(value, param.id)
    if not param.minimum <= value <= param.maximum:
        raise ValueError(f"{param.id}: value must be in [{param.minimum}, {param.maximum}]")
    if param.step == 1 and value != int(value):
        raise ValueError(f"{param.id}: expected an integer")


def _validate(data):
    if not isinstance(data, dict):
        raise ValueError("Take must be a JSON object")
    allowed = {"version", "source", "order", "parameters", "effect_settings", "timeline"}
    if set(data) - allowed:
        raise ValueError(f"Unknown take fields: {sorted(set(data) - allowed)}")
    if data.get("version", Session.VERSION) not in (1, Session.VERSION):
        raise ValueError("Unsupported take version")
    if data.get("source") is not None and not isinstance(data["source"], str):
        raise ValueError("source must be a path string or null")
    params, effects, _, _ = _model()
    ids = {e.id for e in effects}
    order = data.get("order")
    if order is not None and (not isinstance(order, list) or
            any(not isinstance(i, str) or i not in ids for i in order) or len(order) != len(set(order))):
        raise ValueError("order must contain unique known effect IDs")
    controls = data.get("parameters", {})
    if not isinstance(controls, dict):
        raise ValueError("parameters must be an object")
    for pid, entry in controls.items():
        if pid not in params:
            raise ValueError(f"Unknown parameter: {pid}")
        if not isinstance(entry, dict) or "value" not in entry or set(entry) - {"value", "mod", "audio"}:
            raise ValueError(f"{pid}: expected value and optional mod/audio objects")
        _value(params.get(pid), entry["value"])
        for name in ("mod", "audio"):
            if name in entry:
                config = entry[name]
                defaults = getattr(params.get(pid), name).as_dict()
                if not isinstance(config, dict) or set(config) - set(defaults):
                    raise ValueError(f"{pid}.{name}: unknown settings")
                for key, value in config.items():
                    default = defaults[key]
                    if isinstance(default, bool) and not isinstance(value, bool):
                        raise ValueError(f"{pid}.{name}.{key}: expected a boolean")
                    if isinstance(default, str) and not isinstance(value, str):
                        raise ValueError(f"{pid}.{name}.{key}: expected a string")
                    if isinstance(default, (float, int)) and not isinstance(default, bool):
                        _number(value, f"{pid}.{name}.{key}")
                    if isinstance(value, (int, float)):
                        _number(value, f"{pid}.{name}.{key}")
                if name == "audio":
                    if config.get("band", "low") not in ("low", "mid", "high", "level"):
                        raise ValueError(f"{pid}: unknown audio band")
                    if config.get("mode", "trigger") not in ("trigger", "envelope"):
                        raise ValueError(f"{pid}: unknown audio mode")
                    if config.get("rest") is not None:
                        _value(params.get(pid), config["rest"])
    timeline = data.get("timeline", {})
    if not isinstance(timeline, dict) or set(timeline) - {"tracks"}:
        raise ValueError("timeline must contain a tracks object")
    tracks = timeline.get("tracks", {})
    if not isinstance(tracks, dict):
        raise ValueError("tracks must be an object")
    for pid, track in tracks.items():
        if pid not in params:
            raise ValueError(f"Unknown track parameter: {pid}")
        if not isinstance(track, dict) or set(track) - {"t", "v", "interpolation"}:
            raise ValueError(f"{pid}: invalid track")
        times, values = track.get("t"), track.get("v")
        if not isinstance(times, list) or not isinstance(values, list) or len(times) != len(values):
            raise ValueError(f"{pid}: t and v must be arrays of equal length")
        mode = track.get("interpolation", "hold")
        if mode not in ("hold", "linear", "smooth"):
            raise ValueError(f"{pid}: invalid interpolation")
        p = params.get(pid)
        if (p.choices or (p.step == 1 and p.minimum == 0 and p.maximum == 1)) and mode != "hold":
            raise ValueError(f"{pid}: choices and toggles require hold interpolation")
        last = -1
        for t, value in zip(times, values):
            _number(t, pid + ".time")
            if t < 0 or t <= last:
                raise ValueError(f"{pid}: key times must be nonnegative and strictly increasing")
            _value(p, value)
            last = t
    settings = data.get("effect_settings", {})
    if not isinstance(settings, dict) or set(settings) - ids:
        raise ValueError("effect_settings must use known effect IDs")
    for effect in effects:
        if effect.id in settings:
            available = effect.settings() if hasattr(effect, "settings") else {}
            supplied = settings[effect.id]
            if not isinstance(supplied, dict) or set(supplied) - set(available):
                raise ValueError(f"{effect.id}: unknown effect settings")
            for key, value in supplied.items():
                default = available[key]
                if default is not None and not isinstance(value, type(default)):
                    raise ValueError(f"{effect.id}.{key}: invalid setting type")
                if isinstance(value, (float, int)):
                    _number(value, effect.id + "." + key)
            if effect.id == "text":
                from datetime import datetime
                if "manual_datetime" in supplied:
                    datetime.fromisoformat(supplied["manual_datetime"])
                if not 0 <= supplied.get("clock_mode", 0) <= 5:
                    raise ValueError("text.clock_mode must be in [0, 5]")
    # Exercise the same model deserializers used by the GUI before writing.
    params.restore(controls)
    for effect in effects:
        if effect.id in settings and hasattr(effect, "restore_settings"):
            effect.restore_settings(settings[effect.id])
    Timeline.from_dict(timeline)


def create_take(spec: dict, output: str, *, overwrite: bool = False) -> dict:
    """Create a native take. Numeric parameter shorthand is accepted here."""
    if not isinstance(spec, dict):
        raise ValueError("Spec must be an object")
    data = dict(spec)
    if not isinstance(data.get("parameters", {}), dict):
        raise ValueError("parameters must be an object")
    data["parameters"] = {pid: {"value": v} if isinstance(v, (int, float)) else v
                          for pid, v in data.get("parameters", {}).items()}
    _validate(data)
    destination = Path(output).resolve()
    if data.get("source") and Path(data["source"]).resolve() == destination:
        raise ValueError("A take cannot overwrite its source video")
    if destination.exists() and not overwrite:
        raise ValueError("Output exists; pass --overwrite to replace it")
    # Normalize through Session so GUI and agents exchange the same take format.
    with tempfile.TemporaryDirectory() as directory:
        raw = Path(directory) / "spec.json"
        raw.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
        _, _, _, session = _model()
        session.load(str(raw))
        if session.source_path:
            session.source_path = str(Path(session.source_path).resolve())
        session.save(str(destination))
    return {"path": str(destination), "session_version": Session.VERSION}


def validate_take(path: str) -> dict:
    """Validate schema and controls; report unresolved media as warnings."""
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    _validate(data)
    warnings = []
    source = data.get("source")
    if not source or not Path(source).is_file():
        warnings.append("Source video is missing; provide a valid source or render --input.")
    return {"valid": True, "warnings": warnings,
            "tracks": len(data.get("timeline", {}).get("tracks", {}))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("catalog", help="List effects, parameter IDs and defaults as JSON")
    create = commands.add_parser("create", help="Create a GUI-compatible take from a JSON spec")
    create.add_argument("--spec", required=True, help="JSON path, or - for stdin")
    create.add_argument("--output", required=True)
    create.add_argument("--overwrite", action="store_true")
    validate = commands.add_parser("validate", help="Validate a native take")
    validate.add_argument("--session", required=True)
    render = commands.add_parser("render", help="Render a native take without launching the GUI")
    render.add_argument("--session", required=True)
    render.add_argument("--input")
    render.add_argument("--output", required=True)
    render.add_argument("--width", type=int, default=720, help="Width cap; 0 for full resolution")
    render.add_argument("--no-audio", action="store_true")
    render.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        with redirect_stdout(sys.stderr):
            if args.command == "catalog":
                result = catalog()
            elif args.command == "create":
                raw = sys.stdin.read() if args.spec == "-" else Path(args.spec).read_text(encoding="utf-8-sig")
                result = create_take(json.loads(raw), args.output, overwrite=args.overwrite)
            elif args.command == "validate":
                result = validate_take(args.session)
            else:
                from .core.pipeline import render_to_file
                validate_take(args.session)
                if args.width != 0 and args.width < 16:
                    raise ValueError("width must be zero or at least 16 pixels")
                if Path(args.output).exists() and not args.overwrite:
                    raise ValueError("Output exists; pass --overwrite to replace it")
                params, effects, timeline, session = _model()
                session.load(args.session)
                source = args.input or session.source_path
                if not source or not Path(source).is_file():
                    raise ValueError("Source video is missing")
                order = session.order or []
                by_id = {e.id: e for e in effects}
                effects = [by_id[i] for i in order] + [e for e in effects if e.id not in order]
                render_to_file(source, args.output, params, timeline, effects,
                               max_width=args.width or None, with_audio=not args.no_audio)
                result = {"path": str(Path(args.output).resolve())}
        print(json.dumps({"ok": True, "api_version": API_VERSION, "result": result}, allow_nan=False))
        return 0
    except Exception as error:
        print(json.dumps({"ok": False, "api_version": API_VERSION, "error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
