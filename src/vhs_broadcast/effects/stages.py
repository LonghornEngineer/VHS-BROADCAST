"""Signal-order stages: where each effect sits in the chain from camera to screen.

The chain runs stage by stage, so stacked looks behave like real gear: footage is
shot, digitised or compressed, dubbed to tape, received by a TV, shown on a CRT,
then stylised and composited. Within a stage the order stays adjustable.
"""
from __future__ import annotations

STAGES = (
    ("camera", "CAMERA", "Framing, exposure and colour as shot, plus the camcorder OSD."),
    ("digital", "DIGITAL", "Compression, codecs, downscaling and digital damage."),
    ("tape", "TAPE / VCR", "Videotape transport, tracking, head switching and tape noise."),
    ("receiver", "RECEIVER / TV", "Composite signal, RF, ghosts and the TV's sync."),
    ("screen", "CRT SCREEN", "Scanlines, tube geometry, glow and phosphor."),
    ("stylize", "STYLIZE", "Palettes, dithering, pixel art and colour treatments."),
    ("composite", "COMPOSITE", "Feedback and time effects over the finished picture."),
)
STAGE_INDEX = {key: i for i, (key, _, _) in enumerate(STAGES)}
STAGE_LABEL = {key: label for key, label, _ in STAGES}


def signal_order(effects):
    """Effects sorted by stage; the original order is kept within a stage."""
    return sorted(effects, key=lambda e: STAGE_INDEX[e.stage])


def in_use(effect, timeline=None):
    """True if a take turns this effect on, or moves any of its controls off default."""
    tracks = timeline.tracks if timeline is not None else {}
    enabled = effect.enabled_param()
    track = tracks.get(enabled.id)
    if enabled.value >= .5 or (track is not None and any(v >= .5 for v in track.values)):
        return True
    return any(p.id in tracks or p.value != p.default for p in effect.own_params())


def param_in_use(param, timeline=None):
    tracks = timeline.tracks if timeline is not None else {}
    return param.id in tracks or param.value != param.default
