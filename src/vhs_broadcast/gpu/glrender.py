"""GPU live-preview backend (moderngl), multi-pass.

Each effect is a full-frame shader pass reading the previous pass's texture, run
in the SAME order as the CPU signal chain (ping-pong framebuffers). The OSD is
its own pass at its chain position, so reordering the chain -- including where
OSD/TEXT sits -- is honoured on the GPU too.

This is the LIVE / virtual-cam path; it is intentionally approximate (GLSL
noise/float math differ across GPUs) and not bit-identical to the CPU renderer,
which stays the canonical deterministic offline renderer. Motor time-warp and
per-speckle dropout remain CPU-only.
"""
from __future__ import annotations

import numpy as np

from ..effects.text_render import render_osd_overlay

try:
    import moderngl
    HAVE_MODERNGL = True
except Exception:  # noqa: BLE001
    HAVE_MODERNGL = False

_VERT = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = (in_pos + 1.0) * 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_FRAG = """
#version 330
in vec2 v_uv;
out vec4 f_color;
uniform sampler2D tex;
uniform sampler2D osd;
uniform vec2 res;
uniform float t;
uniform int effect;
uniform float u_bright,u_contrast,u_sat;
uniform float u_red,u_grn,u_blu,u_bleed;
uniform float u_wamt,u_wfreq,u_wspeed;
uniform float u_band,u_jit,u_roll,u_rollsp,u_tspeed;
uniform float u_gamt,u_gdelay;
uniform float u_salpha,u_scoarse,u_stint;
uniform float u_grain;
uniform float u_strength,u_vig;
uniform float u_samt,u_swave,u_svhold,u_sinv,u_sdesat;
uniform float u_smear,u_misalign;
uniform float u_tamt,u_tcount;
uniform float u_hheight,u_hnoise,u_hshift;
uniform float u_istr,u_ioff;
uniform float u_damt;
uniform float u_bits;
uniform float u_ldens,u_lcoarse,u_lloc,u_lcolor,u_lthick;
const float PI = 3.14159265;

float hash21(vec2 p){
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
vec3 S(vec2 uv){ return texture(tex, vec2(fract(uv.x), fract(uv.y))).rgb; }

void main(){
    vec2 uv = v_uv;
    float row = uv.y * res.y;
    vec3 col;

    if(effect == 0){
        vec4 o = texture(osd, uv);
        col = mix(S(uv), o.rgb, o.a);
    } else if(effect == 1){
        col = S(uv);
        col = (col - 0.5)*u_contrast + 0.5 + u_bright*0.5;
        float g = dot(col, vec3(0.299,0.587,0.114));
        col = mix(vec3(g), col, u_sat);
    } else if(effect == 2){
        col = vec3(S(vec2(uv.x + u_red*0.02, uv.y)).r,
                   S(vec2(uv.x + u_grn*0.02, uv.y)).g,
                   S(vec2(uv.x + u_blu*0.02, uv.y)).b);
        if(u_bleed > 0.001){
            vec3 bl = vec3(0.0);
            for(int i=-3;i<=3;i++) bl += S(vec2(uv.x + float(i)/res.x, uv.y));
            col = mix(col, bl/7.0, 0.5*u_bleed);
        }
    } else if(effect == 3){
        float xo = (u_wamt*0.03)*sin(uv.y*u_wfreq*2.0*PI + t*u_wspeed);
        col = S(vec2(uv.x + xo, uv.y));
    } else if(effect == 4){
        float r = u_rollsp*abs(u_rollsp)*0.6;
        vec2 u2 = vec2(uv.x, fract(uv.y + u_roll + r*t));
        if(u_jit > 0.001){
            float c = fract(-t*u_tspeed);
            float d = abs(u2.y - c); float bh = u_band*0.5;
            if(d < bh) u2.x += sin((1.0 - d/bh)*PI*0.5)*u_jit*0.08;
        }
        col = S(u2);
    } else if(effect == 5){
        float dx = u_gdelay*0.12 + 0.004;
        col = S(uv) + u_gamt*0.6*S(vec2(uv.x-dx,uv.y))
                    + u_gamt*0.25*S(vec2(uv.x-2.0*dx,uv.y));
    } else if(effect == 6){
        col = S(uv);
        if(u_salpha > 0.001){
            float cell = 1.0 + floor(u_scoarse*7.0);
            vec2 sp = floor(gl_FragCoord.xy/cell);
            float n = hash21(sp + floor(t*1000.0)*1.7);
            vec3 snow = mix(vec3(n),
                vec3(hash21(sp+11.0),hash21(sp+29.0),hash21(sp+53.0)), u_stint);
            col = mix(col, snow, u_salpha);
        }
    } else if(effect == 7){
        col = S(uv);
        if(uv.y > 1.0 - u_hheight){
            col = S(vec2(uv.x + u_hshift*0.3, uv.y));
            if(u_hnoise > 0.001){
                float n = hash21(gl_FragCoord.xy + floor(t*1000.0)*2.3);
                col = mix(col, vec3(n), u_hnoise);
            }
        }
    } else if(effect == 8){
        col = S(uv);
        if(u_grain > 0.001)
            col += (hash21(gl_FragCoord.xy + floor(t*1000.0)) - 0.5)*u_grain*0.3;
    } else if(effect == 9){
        col = S(uv);
        if(mod(floor(row),2.0) < 0.5) col *= (1.0 - 0.6*u_strength);
        if(u_vig > 0.001){
            vec2 d = (uv - 0.5)*2.0;
            col *= clamp(1.0 - (length(d) - 0.6)*u_vig, 0.0, 1.0);
        }
    } else if(effect == 10){
        float blk = 1.0 + floor(u_damt*15.0);
        vec2 g = vec2(blk)/res;
        col = S((floor(uv/g) + 0.5)*g);
    } else if(effect == 11){
        col = S(uv);
        if(u_bits < 7.5){
            float lv = max(2.0, exp2(floor(u_bits + 0.5)));
            col = floor(col*(lv-1.0) + 0.5)/(lv-1.0);
        }
    } else if(effect == 12){
        float field = mod(floor(t*50.0), 2.0);
        vec2 u2 = uv;
        if(u_ioff > 0.001 && mod(floor(row),2.0) == field) u2.x += u_ioff*0.03;
        col = S(u2);
        if(u_istr > 0.001 && mod(floor(row),2.0) != field) col *= (1.0 - 0.5*u_istr);
    } else if(effect == 13){
        float ns = floor(u_tcount*12.0) + 1.0;
        float sl = floor(uv.y*ns);
        float xo = 0.0;
        if(hash21(vec2(sl, floor(t*10.0))) < 0.5){
            float dir = (hash21(vec2(sl, floor(t*10.0)+7.0)) < 0.5) ? -1.0 : 1.0;
            xo = dir*u_tamt*0.15;
        }
        col = S(vec2(uv.x + xo, uv.y));
    } else if(effect == 14){
        vec2 u2 = vec2(uv.x, fract(uv.y + fract(t*u_svhold*1.7)));
        float f = 1.0 + u_swave*12.0;
        u2.x += (u_samt*0.28)*sin(u2.y*f*2.0*PI + t*2.0);
        u2.x += (hash21(vec2(floor(row), floor(t*30.0))) - 0.5)*u_samt*0.04;
        col = S(u2);
        if(u_sinv > 0.001 && sin(uv.y*3.0*2.0*PI - t*3.0) > (1.0 - 2.0*u_sinv))
            col = 1.0 - col;
        if(u_sdesat > 0.001) col = mix(col, vec3(dot(col, vec3(0.333))), u_sdesat);
    } else if(effect == 15){
        float par = (mod(floor(row),2.0) < 0.5) ? 1.0 : -1.0;
        vec2 u2 = vec2(uv.x + par*u_misalign*0.04, uv.y);
        col = S(u2);
        if(u_smear > 0.001){
            vec3 sm = vec3(0.0); float w = 0.0; int L = int(u_smear*30.0);
            for(int i=0;i<30;i++){ if(i>=L) break; float wt = pow(0.7,float(i));
                sm += wt*S(vec2(u2.x - float(i)/res.x, u2.y)); w += wt; }
            if(w > 0.0) col = mix(col, sm/w, min(1.0,u_smear));
        }
    } else if(effect == 16){
        col = S(uv);
        float streak = 2.0 + u_lcoarse*10.0;
        float vthick = 1.0 + u_lthick*10.0;
        vec2 cc = vec2(floor(gl_FragCoord.x/streak), floor(gl_FragCoord.y/vthick));
        float tf = floor(t*30.0);
        vec3 c = vec3(hash21(cc+tf+1.0), hash21(cc+tf+7.0), hash21(cc+tf+13.0));
        c = mix(vec3(dot(c, vec3(0.333))), c, u_lcolor);
        vec2 pc = floor(gl_FragCoord.xy/24.0);
        float i0 = floor(t*2.0);
        float fld = mix(hash21(pc+i0*3.7), hash21(pc+(i0+1.0)*3.7), fract(t*2.0));
        float dens = pow(fld, u_lloc*8.0);
        float amp = hash21(cc + tf*2.1);
        float thr = 1.0 - u_ldens;
        float spark = clamp((amp - thr)/max(0.001, 1.0 - thr), 0.0, 1.0);
        col = mix(col, c, clamp(spark*dens, 0.0, 1.0)*0.9);
    } else {
        col = S(uv);
    }
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""

_EFFECT_ID = {
    "text": 0, "signal": 1, "chroma": 2, "wobble": 3, "tracking": 4,
    "ghost": 5, "static": 6, "headswitch": 7, "noise": 8, "scanlines": 9,
    "downsample": 10, "bitcrush": 11, "interlace": 12, "tear": 13,
    "scramble": 14, "slew": 15, "laserrot": 16,
}
_PARAM_UNIFORMS = {
    "u_bright": "signal.brightness", "u_contrast": "signal.contrast",
    "u_sat": "signal.saturation",
    "u_red": "chroma.red", "u_grn": "chroma.green", "u_blu": "chroma.blue",
    "u_bleed": "chroma.bleed",
    "u_wamt": "wobble.amount", "u_wfreq": "wobble.freq", "u_wspeed": "wobble.speed",
    "u_band": "tracking.band", "u_jit": "tracking.jitter", "u_roll": "tracking.roll",
    "u_rollsp": "tracking.rollspeed", "u_tspeed": "tracking.speed",
    "u_gamt": "ghost.amount", "u_gdelay": "ghost.delay",
    "u_salpha": "static.alpha", "u_scoarse": "static.coarse", "u_stint": "static.tint",
    "u_grain": "noise.grain",
    "u_strength": "scanlines.strength", "u_vig": "scanlines.vignette",
    "u_samt": "scramble.amount", "u_swave": "scramble.wave",
    "u_svhold": "scramble.roll", "u_sinv": "scramble.invert", "u_sdesat": "scramble.desat",
    "u_smear": "slew.smear", "u_misalign": "slew.misalign",
    "u_tamt": "tear.amount", "u_tcount": "tear.count",
    "u_hheight": "headswitch.height", "u_hnoise": "headswitch.noise",
    "u_hshift": "headswitch.shift",
    "u_istr": "interlace.strength", "u_ioff": "interlace.offset",
    "u_damt": "downsample.amount",
    "u_bits": "bitcrush.bits",
    "u_ldens": "laserrot.density", "u_lcoarse": "laserrot.coarse",
    "u_lloc": "laserrot.locality", "u_lcolor": "laserrot.color",
    "u_lthick": "laserrot.thickness",
}


class GLRenderer:
    _probe_cache = None

    @classmethod
    def available(cls):
        if cls._probe_cache is not None:
            return cls._probe_cache
        if not HAVE_MODERNGL:
            cls._probe_cache = (False, "moderngl is not installed "
                                       "(run: uv sync --extra gpu)")
            return cls._probe_cache
        ctx = keep = None
        try:
            ctx, keep = _make_context()
            cls._probe_cache = (True, ctx.info.get("GL_RENDERER", "GPU"))
        except Exception as e:  # noqa: BLE001
            cls._probe_cache = (False, str(e))
        finally:
            _release(ctx, keep)
        return cls._probe_cache

    def __init__(self):
        self.ctx, self._keep = _make_context()
        self.prog = self.ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG)
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        self.vbo = self.ctx.buffer(quad.tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, "in_pos")
        self.in_tex = None
        self.osd_tex = None
        self.rtex = [None, None]
        self.fbo = [None, None]
        self._size = None

    def _ensure(self, w, h):
        if self._size == (w, h):
            return
        for tx in (self.in_tex, self.osd_tex):
            if tx is not None:
                tx.release()
        for i in (0, 1):
            if self.fbo[i] is not None:
                self.rtex[i].release()
                self.fbo[i].release()
        self.in_tex = self.ctx.texture((w, h), 3)
        self.in_tex.repeat_x = self.in_tex.repeat_y = True
        self.in_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.osd_tex = self.ctx.texture((w, h), 4)
        self.osd_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        for i in (0, 1):
            rt = self.ctx.texture((w, h), 3)
            rt.repeat_x = rt.repeat_y = True
            rt.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.rtex[i] = rt
            self.fbo[i] = self.ctx.framebuffer(color_attachments=[rt])
        self._size = (w, h)

    def render_chain(self, frame_bgr, params, t, order, osd_items, font_path):
        h, w = frame_bgr.shape[:2]
        self._ensure(w, h)
        self.in_tex.write(np.ascontiguousarray(frame_bgr[:, :, ::-1]).tobytes())

        passes = []
        for eid in order:
            if eid == "text":
                if osd_items:
                    passes.append(0)
            elif eid in _EFFECT_ID:
                pid = f"{eid}.enabled"
                if pid in params and params.get(pid).value >= 0.5:
                    passes.append(_EFFECT_ID[eid])
        if not passes:
            return frame_bgr

        if 0 in passes:
            overlay = render_osd_overlay(h, w, osd_items, font_path)
            self.osd_tex.write(np.ascontiguousarray(overlay).tobytes())

        self.prog["res"].value = (float(w), float(h))
        self.prog["t"].value = float(t)
        for uni, pid in _PARAM_UNIFORMS.items():
            if uni in self.prog:
                self.prog[uni].value = (float(params.get(pid).evaluate(t))
                                        if pid in params else 0.0)

        cur = self.in_tex
        last = 0
        for i, eff in enumerate(passes):
            dst = i % 2
            cur.use(0)
            self.prog["tex"].value = 0
            if eff == 0:
                self.osd_tex.use(1)
                self.prog["osd"].value = 1
            self.prog["effect"].value = eff
            self.fbo[dst].use()
            self.ctx.clear(0.0, 0.0, 0.0)
            self.vao.render(moderngl.TRIANGLE_STRIP)
            cur = self.rtex[dst]
            last = dst

        data = self.fbo[last].read(components=3, alignment=1)
        out = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
        return np.ascontiguousarray(out[:, :, ::-1])

    def release(self):
        _release(self.ctx, self._keep)
        self.ctx = None
        self._keep = None


def _release(ctx, keep):
    try:
        if ctx is not None:
            ctx.release()
    except Exception:  # noqa: BLE001
        pass
    if keep is not None:
        try:
            _surf, qctx = keep
            qctx.doneCurrent()
        except Exception:  # noqa: BLE001
            pass


def _make_context():
    errs = []
    try:
        return moderngl.create_context(standalone=True), None
    except Exception as e:  # noqa: BLE001
        errs.append(f"standalone: {e}")
    try:
        return moderngl.create_context(standalone=True, backend="egl"), None
    except Exception as e:  # noqa: BLE001
        errs.append(f"egl: {e}")
    try:
        from PySide6.QtGui import (QOffscreenSurface, QOpenGLContext,
                                   QSurfaceFormat)
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        surf = QOffscreenSurface()
        surf.setFormat(fmt)
        surf.create()
        if not surf.isValid():
            raise RuntimeError("offscreen surface invalid")
        qctx = QOpenGLContext()
        qctx.setFormat(fmt)
        if not qctx.create():
            raise RuntimeError("QOpenGLContext.create() failed")
        if not qctx.makeCurrent(surf):
            raise RuntimeError("makeCurrent() failed")
        return moderngl.create_context(), (surf, qctx)
    except Exception as e:  # noqa: BLE001
        errs.append(f"qt: {e}")
    raise RuntimeError("no usable OpenGL context (" + "; ".join(errs) + ")")
