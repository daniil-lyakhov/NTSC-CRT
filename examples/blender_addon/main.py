bl_info = {
    "name": "NTSC-CRT Effect",
    "version": (2, 0, 0),
    "blender": (5, 0, 0),
    "category": "Sequencer",
    "description": "Real-time NTSC/VHS CRT television effect on video strips",
}

import os
import sys

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Operator, Panel, PropertyGroup

# ---------------------------------------------------------------------------
# Path setup — locate the NTSC-CRT Python bindings relative to this addon
# ---------------------------------------------------------------------------
_NTSC_CRT_PATH = os.environ.get("NTSC_CRT_PATH")
if not _NTSC_CRT_PATH:
    raise EnvironmentError(
        "NTSC_CRT_PATH environment variable is not set. "
        "Set it to the 'python' directory of the NTSC-CRT repo, e.g.:\n"
        "  export NTSC_CRT_PATH=/path/to/NTSC-CRT/python"
    )

for _p in (_NTSC_CRT_PATH, os.path.join(_NTSC_CRT_PATH, "lib")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from ntsc_crt import CRT, PIX_FORMAT_BGRA
except ImportError as _e:
    raise ImportError(
        "NTSC-CRT Python bindings not found. "
        "Run 'cd python && python build_crt.py' to build them first."
    ) from _e

try:
    import cv2
    import numpy as np
except ImportError as _e:
    raise ImportError(
        "Required packages missing. Install with: "
        "pip install numpy opencv-python-headless cffi"
    ) from _e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_strips(scene):
    """Return (sequence_editor, strips_collection) with Blender 4/5 compat."""
    sed = scene.sequence_editor
    if sed is None:
        return None, []
    strips = getattr(sed, "strips", None)
    if strips is None:
        strips = getattr(sed, "sequences", None)
    if strips is None:
        return sed, []
    return sed, strips


def _get_active_movie_strip(scene):
    """Return (sequence_editor, active_strip) if active strip is MOVIE, else (None, None)."""
    sed = scene.sequence_editor
    if sed is None:
        return None, None
    strip = getattr(sed, "active_strip", None)
    if strip is None or strip.type != "MOVIE":
        return None, None
    return sed, strip


def _active_movie_strip(context):
    """Return the active strip if it is a MOVIE strip, else None."""
    _, strip = _get_active_movie_strip(context.scene)
    return strip


def _output_path(source_path, output_dir, system, suffix=""):
    """Build the output video path."""
    base = os.path.splitext(os.path.basename(source_path))[0]
    name = f"{base}_ntsc_{system}{suffix}.mp4"
    if output_dir:
        directory = bpy.path.abspath(output_dir)
    else:
        directory = os.path.dirname(bpy.path.abspath(source_path))
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, name)


def _get_selected_movie_strips(context):
    """Return (strip_a, strip_b) for signal mixing.

    strip_a = the active MOVIE strip.
    strip_b = the first other selected MOVIE strip (or None).
    """
    sed = context.scene.sequence_editor
    if sed is None:
        return None, None
    active = getattr(sed, "active_strip", None)
    if active is None or active.type != "MOVIE":
        return None, None

    strips = getattr(sed, "strips", None) or getattr(sed, "sequences", None)
    if strips is None:
        return active, None

    for s in strips:
        if s != active and s.type == "MOVIE" and s.select:
            return active, s
    return active, None


def _read_video_frame(cap, strip, scene_frame):
    """Read a BGRA frame from *cap* at the scene frame position.

    Returns (frame_bgra, w, h) or (None, 0, 0) on failure.
    """
    if cap is None or not cap.isOpened():
        return None, 0, 0
    if scene_frame < strip.frame_final_start or scene_frame >= strip.frame_final_end:
        return None, 0, 0

    src_frame = scene_frame - strip.frame_final_start + strip.frame_offset_start
    cap.set(cv2.CAP_PROP_POS_FRAMES, src_frame)
    ret, frame = cap.read()
    if not ret:
        return None, 0, 0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        return None, 0, 0
    return cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA), w, h


def _mix_and_normalize(signal_a, signal_b, ratio):
    """Mix two analog NTSC signals with a weighted ratio and normalize.

    Parameters
    ----------
    signal_a, signal_b : np.ndarray (int8, shape VRES x HRES)
    ratio : int  0-100  (0 = all A, 100 = all B)

    Returns
    -------
    np.ndarray  int8, same shape – mixed & brightness-normalized signal.
    """
    SYNC_THRESHOLD = 0  # IRE; values at or below this are sync/blanking

    a = signal_a.astype(np.int16)
    b = signal_b.astype(np.int16)
    r = ratio / 100.0

    mixed = a * (1.0 - r) + b * r

    # Compute peaks of the active-video region (above sync/blank)
    mask_a = a > SYNC_THRESHOLD
    mask_b = b > SYNC_THRESHOLD
    peak_a = int(a[mask_a].max()) if mask_a.any() else 1
    peak_b = int(b[mask_b].max()) if mask_b.any() else 1
    target_peak = max(peak_a, peak_b)

    mask_mix = mixed > SYNC_THRESHOLD
    mix_peak = float(mixed[mask_mix].max()) if mask_mix.any() else 1.0

    if mix_peak > 0 and target_peak > 0:
        scale = target_peak / mix_peak
        # Only scale the active-video portion; leave sync structure intact
        mixed[mask_mix] = (mixed[mask_mix] * scale)

    return np.clip(mixed, -128, 127).astype(np.int8)


# ---------------------------------------------------------------------------
# Property update callback — refresh preview when any knob changes
# ---------------------------------------------------------------------------

def _tag_image_editors():
    """Tag all Image Editor areas for redraw."""
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "IMAGE_EDITOR":
                area.tag_redraw()


def _on_strip_knob_update(self, context):
    """Called when any per-strip CRT property changes."""
    props = context.scene.ntsc_crt
    if props.enabled or props.mix_enabled:
        _ntsc_frame_handler(context.scene)
        _tag_image_editors()


def _on_global_knob_update(self, context):
    """Called when a global property (mix_ratio) changes."""
    if self.enabled or self.mix_enabled:
        _ntsc_frame_handler(context.scene)
        _tag_image_editors()


# ---------------------------------------------------------------------------
# Per-strip CRT settings
# ---------------------------------------------------------------------------

class NTSCStripSettings(PropertyGroup):
    strip_name: StringProperty(name="Strip Name", default="")

    system: EnumProperty(
        name="System", description="CRT system variant",
        items=[
            ("ntsc", "NTSC", "Standard NTSC television"),
            ("ntscvhs", "NTSC VHS", "NTSC with VHS tape quality"),
        ],
        default="ntsc", update=_on_strip_knob_update,
    )
    hue: IntProperty(
        name="Hue", description="Color hue rotation in degrees",
        default=0, min=-360, max=360, update=_on_strip_knob_update,
    )
    brightness: IntProperty(
        name="Brightness", description="Brightness offset",
        default=0, min=-100, max=100, update=_on_strip_knob_update,
    )
    contrast: IntProperty(
        name="Contrast", description="Contrast multiplier",
        default=180, min=0, max=500, update=_on_strip_knob_update,
    )
    saturation: IntProperty(
        name="Saturation", description="Color saturation multiplier",
        default=10, min=0, max=200, update=_on_strip_knob_update,
    )
    black_point: IntProperty(
        name="Black Point", description="Black level adjustment",
        default=0, min=-50, max=50, update=_on_strip_knob_update,
    )
    white_point: IntProperty(
        name="White Point", description="White level adjustment",
        default=100, min=0, max=200, update=_on_strip_knob_update,
    )
    scanlines: BoolProperty(
        name="Scanlines", description="Visible gaps between scan lines",
        default=True, update=_on_strip_knob_update,
    )
    blend: BoolProperty(
        name="Blend", description="Blend new field onto previous image",
        default=True, update=_on_strip_knob_update,
    )
    v_fac: IntProperty(
        name="V Stretch", description="Vertical stretch factor",
        default=0, min=0, max=100, update=_on_strip_knob_update,
    )
    noise: IntProperty(
        name="Noise", description="Signal noise amount (0 = clean)",
        default=24, min=0, max=255, update=_on_strip_knob_update,
    )
    artifact_hue: IntProperty(
        name="Artifact Hue", description="Artifact color hue offset (0-359)",
        default=0, min=0, max=359, update=_on_strip_knob_update,
    )
    num_frames: IntProperty(
        name="Accumulate Frames",
        description="Frames to accumulate per output frame",
        default=4, min=1, max=16, update=_on_strip_knob_update,
    )
    progressive: BoolProperty(
        name="Progressive", description="Progressive scan (vs interlaced)",
        default=False, update=_on_strip_knob_update,
    )
    raw: BoolProperty(
        name="Raw", description="Don't scale input to fit the CRT viewport",
        default=False, update=_on_strip_knob_update,
    )
    as_color: BoolProperty(
        name="Color", description="Full color output (off = monochrome)",
        default=True, update=_on_strip_knob_update,
    )
    xoffset: IntProperty(
        name="X Offset", description="Horizontal offset in sample space",
        default=0, min=-200, max=200, update=_on_strip_knob_update,
    )
    yoffset: IntProperty(
        name="Y Offset", description="Vertical offset in scan lines",
        default=0, min=-200, max=200, update=_on_strip_knob_update,
    )
    do_aberration: BoolProperty(
        name="VHS Aberration",
        description="Signal distortion at bottom of frame (ntscvhs only)",
        default=False, update=_on_strip_knob_update,
    )


# ---------------------------------------------------------------------------
# Scene-level globals
# ---------------------------------------------------------------------------

class NTSCCRTProperties(PropertyGroup):
    enabled: BoolProperty(
        name="Enable",
        description="Enable real-time NTSC-CRT preview",
        default=False,
    )
    mix_enabled: BoolProperty(
        name="Mix Enable",
        description="Enable signal mixing of two selected movie strips",
        default=False,
    )
    mix_ratio: IntProperty(
        name="Mix Ratio",
        description="Signal mix balance: 0 = all Strip A, 100 = all Strip B",
        default=50, min=0, max=100, update=_on_global_knob_update,
    )
    output_directory: StringProperty(
        name="Output Dir",
        description="Directory for rendered video (empty = same as source)",
        subtype="DIR_PATH",
        default="",
    )
    strip_settings: CollectionProperty(type=NTSCStripSettings)


def _get_strip_settings(props, strip_name):
    """Return the per-strip CRT settings, creating an entry if needed."""
    item = _find_strip_settings(props, strip_name)
    if item is not None:
        return item
    item = props.strip_settings.add()
    item.strip_name = strip_name
    return item


def _find_strip_settings(props, strip_name):
    """Read-only lookup — returns None when the strip has no settings yet."""
    for item in props.strip_settings:
        if item.strip_name == strip_name:
            return item
    return None


# ---------------------------------------------------------------------------
# Frame handler cache — keeps CRT instance and VideoCapture alive across frames
# ---------------------------------------------------------------------------

class _HandlerState:
    """Module-level cache for the frame change handler."""

    _crts: dict[str, CRT] = {}          # slot key -> cached CRT instance
    caps = {}                           # filepath -> cv2.VideoCapture
    preview_name = "NTSC Preview"

    @classmethod
    def get_crt_slot(cls, key, system, w, h):
        cached: CRT = cls._crts.get(key)
        if (cached is None
                or cached.system != system
                or cached.out_w != w
                or cached.out_h != h):
            crt = CRT(system, out_w=w, out_h=h, out_format=PIX_FORMAT_BGRA)
            cls._crts[key] = crt
            return crt
        return cached

    @classmethod
    def get_cap(cls, filepath):
        if filepath not in cls.caps:
            cls.caps[filepath] = cv2.VideoCapture(filepath)
        return cls.caps[filepath]

    @classmethod
    def release_cap(cls, filepath=None):
        if filepath is not None:
            cap = cls.caps.pop(filepath, None)
            if cap is not None:
                cap.release()
            return
        for cap in cls.caps.values():
            cap.release()
        cls.caps.clear()

    @classmethod
    def release_all(cls):
        cls.release_cap()
        cls._crts.clear()

    @classmethod
    def get_or_create_image(cls, w, h):
        img = bpy.data.images.get(cls.preview_name)
        if img is not None:
            if img.size[0] == w and img.size[1] == h:
                return img
            bpy.data.images.remove(img)
        return bpy.data.images.new(
            cls.preview_name, width=w, height=h, alpha=True,
        )


# ---------------------------------------------------------------------------
# Frame change handler
# ---------------------------------------------------------------------------

def _apply_monitor_settings(crt, props):
    """Copy the UI monitor knobs onto a CRT instance."""
    for attr in ("hue", "brightness", "contrast", "saturation",
                 "black_point", "white_point", "scanlines", "blend", "v_fac"):
        setattr(crt, attr, getattr(props, attr))


def _write_output_to_image(output, w, h):
    """Write a BGRA output array to the Blender preview image."""
    img = _HandlerState.get_or_create_image(w, h)
    rgba = cv2.cvtColor(output, cv2.COLOR_BGRA2RGBA)
    rgba = np.flipud(rgba)
    pixels = rgba.astype(np.float32).ravel() / 255.0
    img.pixels.foreach_set(pixels)
    img.update()


def _modulate_kwargs(props):
    """Build the keyword arguments shared by modulate() calls."""
    return dict(
        hue=props.artifact_hue,
        raw=props.raw,
        as_color=props.as_color,
        in_format=PIX_FORMAT_BGRA,
        do_aberration=props.do_aberration,
        xoffset=props.xoffset,
        yoffset=props.yoffset,
    )


def _process_frame(crt, frame_bgra, settings):
    """Apply settings and process a single frame through the CRT."""
    _apply_monitor_settings(crt, settings)
    return crt.process(
        frame_bgra,
        noise=settings.noise,
        hue=settings.artifact_hue,
        num_frames=settings.num_frames,
        progressive=settings.progressive,
        raw=settings.raw,
        as_color=settings.as_color,
        in_format=PIX_FORMAT_BGRA,
        do_aberration=settings.do_aberration,
        xoffset=settings.xoffset,
        yoffset=settings.yoffset,
    )


def _process_mix_frame(crt_a, crt_b, frame_a, frame_b,
                       settings_a, settings_b, mix_ratio):
    """Modulate two frames, mix their analog signals, and demodulate."""
    _apply_monitor_settings(crt_a, settings_a)
    mkw_a = _modulate_kwargs(settings_a)
    crt_a.modulate(frame_a, field=0, frame=0, **mkw_a)
    signal_a = crt_a.get_analog_signal()

    _apply_monitor_settings(crt_b, settings_b)
    mkw_b = _modulate_kwargs(settings_b)
    crt_b.modulate(frame_b, field=0, frame=0, **mkw_b)
    signal_b = crt_b.get_analog_signal()

    mixed = _mix_and_normalize(signal_a, signal_b, mix_ratio)
    crt_a.set_analog_signal(mixed)
    return crt_a.demodulate(noise=settings_a.noise)


def _read_strip_frame(strip, scene_frame):
    """Read a BGRA frame from a strip. Returns (frame, w, h) or (None, 0, 0)."""
    path = bpy.path.abspath(strip.filepath)
    if not os.path.isfile(path):
        return None, 0, 0
    cap = _HandlerState.get_cap(path)
    return _read_video_frame(cap, strip, scene_frame)


def _single_strip_preview(strip, props, current):
    """Process a single strip through NTSC-CRT and write to the preview image."""
    frame, w, h = _read_strip_frame(strip, current)
    if frame is None:
        return

    settings = _get_strip_settings(props, strip.name)
    crt = _HandlerState.get_crt_slot("a", settings.system, w, h)
    output = _process_frame(crt, frame, settings)
    _write_output_to_image(output, w, h)


def _strip_mix_preview(sed, strip_a, props, current):
    """Attempt two-strip signal mix preview."""
    frame_a, w, h = _read_strip_frame(strip_a, current)
    if frame_a is None:
        return

    strips = getattr(sed, "strips", None) or getattr(sed, "sequences", None)
    strip_b = None
    if strips is not None:
        for s in strips:
            if s != strip_a and s.type == "MOVIE" and s.select:
                strip_b = s
                break

    if strip_b is None:
        return

    frame_b, wb, hb = _read_strip_frame(strip_b, current)
    if frame_b is None:
        return

    # Resize B to match A if needed
    if (wb, hb) != (w, h):
        frame_b = cv2.resize(frame_b, (w, h),
                             interpolation=cv2.INTER_LINEAR)

    settings_a = _get_strip_settings(props, strip_a.name)
    settings_b = _get_strip_settings(props, strip_b.name)
    crt_a = _HandlerState.get_crt_slot("a", settings_a.system, w, h)
    crt_b = _HandlerState.get_crt_slot("b", settings_b.system, w, h)
    output = _process_mix_frame(crt_a, crt_b, frame_a, frame_b,
                                settings_a, settings_b, props.mix_ratio)
    _write_output_to_image(output, w, h)


def _ntsc_frame_handler(scene):
    """Process the active strip's current frame through NTSC-CRT.

    Supports two modes:
    - Single-strip preview (props.enabled)
    - Two-strip signal mix  (props.mix_enabled)
    """
    props = scene.ntsc_crt
    if not props.enabled and not props.mix_enabled:
        return

    sed, strip_a = _get_active_movie_strip(scene)
    if strip_a is None:
        return

    current = scene.frame_current

    if props.mix_enabled:
        _strip_mix_preview(sed, strip_a, props, current)
    elif props.enabled:
        _single_strip_preview(strip_a, props, current)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class SEQUENCER_OT_ntsc_toggle(Operator):
    """Toggle real-time NTSC-CRT preview on/off"""
    bl_idname = "sequencer.ntsc_toggle"
    bl_label = "Toggle NTSC-CRT"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.ntsc_crt
        props.enabled = not props.enabled

        if props.enabled:
            if _ntsc_frame_handler not in bpy.app.handlers.frame_change_post:
                bpy.app.handlers.frame_change_post.append(_ntsc_frame_handler)
            # Process the current frame immediately
            _ntsc_frame_handler(context.scene)
            self.report({"INFO"}, "NTSC-CRT preview enabled")
        else:
            if _ntsc_frame_handler in bpy.app.handlers.frame_change_post:
                bpy.app.handlers.frame_change_post.remove(_ntsc_frame_handler)
            _HandlerState.release_all()
            self.report({"INFO"}, "NTSC-CRT preview disabled")

        return {"FINISHED"}


class SEQUENCER_OT_ntsc_refresh(Operator):
    """Refresh the NTSC-CRT preview for the current frame"""
    bl_idname = "sequencer.ntsc_refresh"
    bl_label = "Refresh NTSC-CRT"

    def execute(self, context):
        _ntsc_frame_handler(context.scene)
        return {"FINISHED"}


class SEQUENCER_OT_ntsc_reset(Operator):
    """Reset all NTSC-CRT parameters to their defaults"""
    bl_idname = "sequencer.ntsc_reset"
    bl_label = "Reset Parameters"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.ntsc_crt
        props.mix_ratio = 50
        props.strip_settings.clear()
        self.report({"INFO"}, "NTSC-CRT parameters reset")
        return {"FINISHED"}


class SEQUENCER_OT_ntsc_mix_toggle(Operator):
    """Toggle signal-mix preview of two selected movie strips"""
    bl_idname = "sequencer.ntsc_mix_toggle"
    bl_label = "Toggle Signal Mix"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.ntsc_crt
        props.mix_enabled = not props.mix_enabled

        if props.mix_enabled:
            strip_a, strip_b = _get_selected_movie_strips(context)
            if strip_a is None:
                props.mix_enabled = False
                self.report({"ERROR"}, "Select an active movie strip first")
                return {"CANCELLED"}
            if strip_b is None:
                props.mix_enabled = False
                self.report({"ERROR"},
                            "Select a second movie strip to mix with")
                return {"CANCELLED"}
            if _ntsc_frame_handler not in bpy.app.handlers.frame_change_post:
                bpy.app.handlers.frame_change_post.append(_ntsc_frame_handler)
            _ntsc_frame_handler(context.scene)
            self.report({"INFO"}, "Signal mix preview enabled")
        else:
            # Only remove handler if single-strip preview is also off
            if not props.enabled:
                if _ntsc_frame_handler in bpy.app.handlers.frame_change_post:
                    bpy.app.handlers.frame_change_post.remove(
                        _ntsc_frame_handler)
            _HandlerState.release_all()
            self.report({"INFO"}, "Signal mix preview disabled")

        return {"FINISHED"}


class SEQUENCER_OT_ntsc_render(Operator):
    """Render the active movie strip through NTSC-CRT and add as a new strip"""
    bl_idname = "sequencer.ntsc_render"
    bl_label = "Render NTSC-CRT"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _active_movie_strip(context) is not None

    def execute(self, context):
        strip = _active_movie_strip(context)
        if strip is None:
            self.report({"ERROR"}, "Select a movie strip first")
            return {"CANCELLED"}

        props = context.scene.ntsc_crt
        settings = _get_strip_settings(props, strip.name)
        source_path = bpy.path.abspath(strip.filepath)

        if not os.path.isfile(source_path):
            self.report({"ERROR"}, f"Source file not found: {source_path}")
            return {"CANCELLED"}

        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            self.report({"ERROR"}, f"Cannot open video: {source_path}")
            return {"CANCELLED"}

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if w <= 0 or h <= 0:
            cap.release()
            self.report({"ERROR"}, "Cannot read video dimensions")
            return {"CANCELLED"}

        src_start = strip.frame_offset_start
        frames_to_process = strip.frame_final_duration

        if src_start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, src_start)

        out_path = _output_path(source_path, props.output_directory,
                                settings.system)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            cap.release()
            self.report({"ERROR"}, f"Cannot create output: {out_path}")
            return {"CANCELLED"}

        wm = context.window_manager
        wm.progress_begin(0, frames_to_process)

        scene = context.scene
        original_frame = scene.frame_current
        frame_idx = 0
        try:
            while frame_idx < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break

                scene.frame_set(strip.frame_final_start + frame_idx)

                frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                crt = _HandlerState.get_crt_slot("a", settings.system, w, h)
                output = _process_frame(crt, frame_bgra, settings)
                writer.write(cv2.cvtColor(output, cv2.COLOR_BGRA2BGR))
                frame_idx += 1
                wm.progress_update(frame_idx)
        finally:
            scene.frame_set(original_frame)
            cap.release()
            writer.release()
            wm.progress_end()

        if frame_idx == 0:
            self.report({"ERROR"}, "No frames were processed")
            return {"CANCELLED"}

        sed, strips = _get_strips(context.scene)
        if sed is None:
            sed = context.scene.sequence_editor_create()
            _, strips = _get_strips(context.scene)

        new_strip = strips.new_movie(
            name=f"NTSC {settings.system.upper()}",
            filepath=out_path,
            channel=strip.channel + 1,
            frame_start=strip.frame_final_start,
        )
        sed.active_strip = new_strip

        self.report(
            {"INFO"},
            f"NTSC-CRT: rendered {frame_idx} frames -> {os.path.basename(out_path)}",
        )
        return {"FINISHED"}


class SEQUENCER_OT_ntsc_mix_render(Operator):
    """Render two selected strips mixed through analog NTSC signals"""
    bl_idname = "sequencer.ntsc_mix_render"
    bl_label = "Render Signal Mix"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        a, b = _get_selected_movie_strips(context)
        return a is not None and b is not None

    def execute(self, context):
        strip_a, strip_b = _get_selected_movie_strips(context)
        if strip_a is None or strip_b is None:
            self.report({"ERROR"}, "Select two movie strips")
            return {"CANCELLED"}

        props = context.scene.ntsc_crt
        settings_a = _get_strip_settings(props, strip_a.name)
        settings_b = _get_strip_settings(props, strip_b.name)
        path_a = bpy.path.abspath(strip_a.filepath)
        path_b = bpy.path.abspath(strip_b.filepath)

        for p in (path_a, path_b):
            if not os.path.isfile(p):
                self.report({"ERROR"}, f"File not found: {p}")
                return {"CANCELLED"}

        cap_a = cv2.VideoCapture(path_a)
        cap_b = cv2.VideoCapture(path_b)
        if not cap_a.isOpened() or not cap_b.isOpened():
            cap_a.release()
            cap_b.release()
            self.report({"ERROR"}, "Cannot open one of the video files")
            return {"CANCELLED"}

        fps = cap_a.get(cv2.CAP_PROP_FPS)
        w = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0:
            cap_a.release()
            cap_b.release()
            self.report({"ERROR"}, "Cannot read video dimensions")
            return {"CANCELLED"}

        # Timeline range: overlapping portion of both strips
        start = max(strip_a.frame_final_start, strip_b.frame_final_start)
        end = min(strip_a.frame_final_end, strip_b.frame_final_end)
        if end <= start:
            cap_a.release()
            cap_b.release()
            self.report({"ERROR"}, "Strips do not overlap on the timeline")
            return {"CANCELLED"}
        frames_to_process = end - start

        out_path = _output_path(path_a, props.output_directory,
                                settings_a.system, suffix="_mix")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            cap_a.release()
            cap_b.release()
            self.report({"ERROR"}, f"Cannot create output: {out_path}")
            return {"CANCELLED"}

        wm = context.window_manager
        wm.progress_begin(0, frames_to_process)

        scene = context.scene
        original_frame = scene.frame_current
        frame_idx = 0

        try:
            for timeline_frame in range(start, end):
                scene.frame_set(timeline_frame)

                # Read frame A
                src_a = (timeline_frame - strip_a.frame_final_start
                         + strip_a.frame_offset_start)
                cap_a.set(cv2.CAP_PROP_POS_FRAMES, src_a)
                ret_a, raw_a = cap_a.read()

                # Read frame B
                src_b = (timeline_frame - strip_b.frame_final_start
                         + strip_b.frame_offset_start)
                cap_b.set(cv2.CAP_PROP_POS_FRAMES, src_b)
                ret_b, raw_b = cap_b.read()

                if not ret_a or not ret_b:
                    break

                fa = cv2.cvtColor(raw_a, cv2.COLOR_BGR2BGRA)
                fb = cv2.cvtColor(raw_b, cv2.COLOR_BGR2BGRA)

                # Resize B to match A if needed
                bh, bw = fb.shape[:2]
                if (bw, bh) != (w, h):
                    fb = cv2.resize(fb, (w, h),
                                   interpolation=cv2.INTER_LINEAR)

                crt_a = _HandlerState.get_crt_slot("a", settings_a.system, w, h)
                crt_b = _HandlerState.get_crt_slot("b", settings_b.system, w, h)
                output = _process_mix_frame(
                    crt_a, crt_b, fa, fb,
                    settings_a, settings_b, props.mix_ratio,
                )

                writer.write(cv2.cvtColor(output, cv2.COLOR_BGRA2BGR))
                frame_idx += 1
                wm.progress_update(frame_idx)
        finally:
            scene.frame_set(original_frame)
            cap_a.release()
            cap_b.release()
            writer.release()
            wm.progress_end()

        if frame_idx == 0:
            self.report({"ERROR"}, "No frames were processed")
            return {"CANCELLED"}

        sed, strips = _get_strips(context.scene)
        if sed is None:
            sed = context.scene.sequence_editor_create()
            _, strips = _get_strips(context.scene)

        channel = max(strip_a.channel, strip_b.channel) + 1
        new_strip = strips.new_movie(
            name=f"NTSC MIX {settings_a.system.upper()}",
            filepath=out_path,
            channel=channel,
            frame_start=start,
        )
        sed.active_strip = new_strip

        self.report(
            {"INFO"},
            f"NTSC-CRT mix: rendered {frame_idx} frames "
            f"-> {os.path.basename(out_path)}",
        )
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

def _draw_strip_knobs(layout, settings, label):
    """Draw all CRT knobs for one NTSCStripSettings entry."""
    header = layout.box()
    header.label(text=label, icon="SEQ_STRIP_META")
    header.prop(settings, "system")

    box = header.box()
    box.label(text="Monitor", icon="DESKTOP")
    box.prop(settings, "hue", slider=True)
    box.prop(settings, "brightness", slider=True)
    box.prop(settings, "contrast", slider=True)
    box.prop(settings, "saturation", slider=True)
    box.prop(settings, "black_point", slider=True)
    box.prop(settings, "white_point", slider=True)

    box = header.box()
    box.label(text="Display", icon="RESTRICT_VIEW_OFF")
    row = box.row()
    row.prop(settings, "scanlines")
    row.prop(settings, "blend")
    box.prop(settings, "v_fac", slider=True)

    box = header.box()
    box.label(text="Signal", icon="FORCE_HARMONIC")
    box.prop(settings, "noise", slider=True)
    box.prop(settings, "artifact_hue", slider=True)
    box.prop(settings, "num_frames")
    row = box.row()
    row.prop(settings, "progressive")
    row.prop(settings, "as_color")
    box.prop(settings, "raw")

    box = header.box()
    box.label(text="Offsets", icon="ORIENTATION_CURSOR")
    row = box.row(align=True)
    row.prop(settings, "xoffset")
    row.prop(settings, "yoffset")

    if settings.system == "ntscvhs":
        box = header.box()
        box.label(text="VHS", icon="FILE_MOVIE")
        box.prop(settings, "do_aberration")


class SEQUENCER_PT_ntsc_crt(Panel):
    bl_label = "NTSC-CRT"
    bl_space_type = "SEQUENCE_EDITOR"
    bl_region_type = "UI"
    bl_category = "NTSC-CRT"

    def draw(self, context):
        layout = self.layout
        props = context.scene.ntsc_crt
        strip = _active_movie_strip(context)

        # Global controls
        row = layout.row(align=True)
        row.scale_y = 1.5
        icon = "PAUSE" if props.enabled else "PLAY"
        label = "Disable Preview" if props.enabled else "Enable Preview"
        row.operator("sequencer.ntsc_toggle", text=label, icon=icon)

        row = layout.row(align=True)
        sub = row.row(align=True)
        sub.operator("sequencer.ntsc_refresh", text="Refresh",
                     icon="FILE_REFRESH")
        sub.enabled = props.enabled or props.mix_enabled
        row.operator("sequencer.ntsc_reset", text="Reset", icon="LOOP_BACK")

        if props.enabled or props.mix_enabled:
            layout.label(
                text=f'View in Image Editor: "{_HandlerState.preview_name}"',
                icon="IMAGE",
            )

        # Active strip settings
        if strip:
            settings = _find_strip_settings(props, strip.name)
            if settings:
                strip_label = (f"Strip A: {strip.name}" if props.mix_enabled
                               else f"Strip: {strip.name}")
                layout.separator()
                _draw_strip_knobs(layout, settings, strip_label)
            else:
                layout.label(text=f"Strip: {strip.name} (no settings yet)",
                             icon="INFO")
        else:
            layout.label(text="Select a movie strip", icon="INFO")

        # Signal Mix
        layout.separator()
        box = layout.box()
        box.label(text="Signal Mix", icon="MOD_WAVE")

        strip_a, strip_b = _get_selected_movie_strips(context)
        if strip_a and strip_b:
            box.label(text=f"A: {strip_a.name}", icon="SEQ_STRIP_META")
            box.label(text=f"B: {strip_b.name}", icon="SEQ_STRIP_META")
        else:
            box.label(text="Select 2 movie strips", icon="INFO")

        row = box.row(align=True)
        row.scale_y = 1.3
        mix_icon = "PAUSE" if props.mix_enabled else "PLAY"
        mix_label = "Disable Mix" if props.mix_enabled else "Enable Mix"
        row.operator("sequencer.ntsc_mix_toggle", text=mix_label,
                     icon=mix_icon)
        box.prop(props, "mix_ratio", slider=True)

        if props.mix_enabled and strip_b:
            settings_b = _find_strip_settings(props, strip_b.name)
            if settings_b:
                _draw_strip_knobs(box, settings_b,
                                  f"Strip B: {strip_b.name}")

        row = box.row()
        row.scale_y = 1.3
        row.operator("sequencer.ntsc_mix_render",
                     text="Render Mix to Strip", icon="RENDER_ANIMATION")

        # Render
        layout.separator()
        box = layout.box()
        box.label(text="Render", icon="RENDER_ANIMATION")
        box.prop(props, "output_directory")
        row = box.row()
        row.scale_y = 1.3
        row.operator("sequencer.ntsc_render", text="Render to Strip",
                     icon="RENDER_ANIMATION")


# ---------------------------------------------------------------------------
# Register / Unregister
# ---------------------------------------------------------------------------

_classes = (
    NTSCStripSettings,
    NTSCCRTProperties,
    SEQUENCER_OT_ntsc_toggle,
    SEQUENCER_OT_ntsc_refresh,
    SEQUENCER_OT_ntsc_reset,
    SEQUENCER_OT_ntsc_mix_toggle,
    SEQUENCER_OT_ntsc_render,
    SEQUENCER_OT_ntsc_mix_render,
    SEQUENCER_PT_ntsc_crt,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.ntsc_crt = PointerProperty(type=NTSCCRTProperties)


def unregister():
    # Remove handler if active
    if _ntsc_frame_handler in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(_ntsc_frame_handler)
    _HandlerState.release_all()

    del bpy.types.Scene.ntsc_crt
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
