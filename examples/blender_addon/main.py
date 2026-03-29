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


def _active_movie_strip(context):
    """Return the active strip if it is a MOVIE strip, else None."""
    sed = context.scene.sequence_editor
    if sed is None:
        return None
    active = getattr(sed, "active_strip", None)
    if active and active.type == "MOVIE":
        return active
    return None


def _output_path(source_path, output_dir, system):
    """Build the output video path."""
    base = os.path.splitext(os.path.basename(source_path))[0]
    name = f"{base}_ntsc_{system}.mp4"
    if output_dir:
        directory = bpy.path.abspath(output_dir)
    else:
        directory = os.path.dirname(bpy.path.abspath(source_path))
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, name)


# ---------------------------------------------------------------------------
# Property update callback — refresh preview when any knob changes
# ---------------------------------------------------------------------------

def _on_knob_update(self, context):
    """Called when any CRT property changes; re-process current frame."""
    if self.enabled:
        # Use the global function (defined later in this module)
        _ntsc_frame_handler(context.scene)
        # Tag the image editor regions for redraw
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "IMAGE_EDITOR":
                    area.tag_redraw()


# ---------------------------------------------------------------------------
# Property group — every CRT knob
# ---------------------------------------------------------------------------

class NTSCCRTProperties(PropertyGroup):
    # Enable/disable live preview
    enabled: BoolProperty(
        name="Enable",
        description="Enable real-time NTSC-CRT preview",
        default=False,
    )

    # System
    system: EnumProperty(
        name="System",
        description="CRT system variant",
        items=[
            ("ntsc", "NTSC", "Standard NTSC television"),
            ("ntscvhs", "NTSC VHS", "NTSC with VHS tape quality"),
        ],
        default="ntsc",
        update=_on_knob_update,
    )

    # --- Monitor settings (defaults from crt_reset() in crt_core.c) ---
    hue: IntProperty(
        name="Hue", description="Color hue rotation in degrees",
        default=0, min=-360, max=360, update=_on_knob_update,
    )
    brightness: IntProperty(
        name="Brightness", description="Brightness offset",
        default=0, min=-100, max=100, update=_on_knob_update,
    )
    contrast: IntProperty(
        name="Contrast", description="Contrast multiplier",
        default=180, min=0, max=500, update=_on_knob_update,
    )
    saturation: IntProperty(
        name="Saturation", description="Color saturation multiplier",
        default=10, min=0, max=200, update=_on_knob_update,
    )
    black_point: IntProperty(
        name="Black Point", description="Black level adjustment",
        default=0, min=-50, max=50, update=_on_knob_update,
    )
    white_point: IntProperty(
        name="White Point", description="White level adjustment",
        default=100, min=0, max=200, update=_on_knob_update,
    )

    # --- Display ---
    scanlines: BoolProperty(
        name="Scanlines", description="Visible gaps between scan lines",
        default=True, update=_on_knob_update,
    )
    blend: BoolProperty(
        name="Blend", description="Blend new field onto previous image",
        default=True, update=_on_knob_update,
    )
    v_fac: IntProperty(
        name="V Stretch", description="Vertical stretch factor",
        default=0, min=0, max=100, update=_on_knob_update,
    )

    # --- Signal / process settings ---
    noise: IntProperty(
        name="Noise", description="Signal noise amount (0 = clean)",
        default=24, min=0, max=255, update=_on_knob_update,
    )
    artifact_hue: IntProperty(
        name="Artifact Hue", description="Artifact color hue offset (0-359)",
        default=0, min=0, max=359, update=_on_knob_update,
    )
    num_frames: IntProperty(
        name="Accumulate Frames",
        description="Frames to accumulate per output frame (higher = smoother but slower)",
        default=4, min=1, max=16, update=_on_knob_update,
    )
    progressive: BoolProperty(
        name="Progressive", description="Progressive scan (vs interlaced)",
        default=False, update=_on_knob_update,
    )
    raw: BoolProperty(
        name="Raw", description="Don't scale input to fit the CRT viewport",
        default=False, update=_on_knob_update,
    )
    as_color: BoolProperty(
        name="Color", description="Full color output (off = monochrome)",
        default=True, update=_on_knob_update,
    )

    # --- Offsets ---
    xoffset: IntProperty(
        name="X Offset", description="Horizontal offset in sample space",
        default=0, min=-200, max=200, update=_on_knob_update,
    )
    yoffset: IntProperty(
        name="Y Offset", description="Vertical offset in scan lines",
        default=0, min=-200, max=200, update=_on_knob_update,
    )

    # --- VHS-specific ---
    do_aberration: BoolProperty(
        name="VHS Aberration",
        description="Signal distortion at bottom of frame (ntscvhs only)",
        default=False, update=_on_knob_update,
    )

    # --- Output ---
    output_directory: StringProperty(
        name="Output Dir",
        description="Directory for rendered video (empty = same as source)",
        subtype="DIR_PATH",
        default="",
    )


# ---------------------------------------------------------------------------
# Frame handler cache — keeps CRT instance and VideoCapture alive across frames
# ---------------------------------------------------------------------------

class _HandlerState:
    """Module-level cache for the frame change handler."""
    crt = None
    crt_system = None
    crt_w = 0
    crt_h = 0
    cap = None
    cap_path = None
    preview_name = "NTSC Preview"

    @classmethod
    def get_crt(cls, system, w, h):
        if (cls.crt is None
                or cls.crt_system != system
                or cls.crt_w != w
                or cls.crt_h != h):
            cls.crt = CRT(system, out_w=w, out_h=h, out_format=PIX_FORMAT_BGRA)
            cls.crt_system = system
            cls.crt_w = w
            cls.crt_h = h
        return cls.crt

    @classmethod
    def get_cap(cls, filepath):
        if cls.cap_path != filepath:
            cls.release_cap()
            cls.cap = cv2.VideoCapture(filepath)
            cls.cap_path = filepath
        return cls.cap

    @classmethod
    def release_cap(cls):
        if cls.cap is not None:
            cls.cap.release()
            cls.cap = None
            cls.cap_path = None

    @classmethod
    def release_all(cls):
        cls.release_cap()
        cls.crt = None
        cls.crt_system = None

    @classmethod
    def get_or_create_image(cls, w, h):
        img = bpy.data.images.get(cls.preview_name)
        if img is None or img.size[0] != w or img.size[1] != h:
            if img is not None:
                bpy.data.images.remove(img)
            img = bpy.data.images.new(
                cls.preview_name, width=w, height=h, alpha=True,
            )
        return img


# ---------------------------------------------------------------------------
# Frame change handler
# ---------------------------------------------------------------------------

def _ntsc_frame_handler(scene):
    """Process the active strip's current frame through NTSC-CRT."""
    props = scene.ntsc_crt
    if not props.enabled:
        return

    sed = scene.sequence_editor
    if sed is None:
        return
    strip = getattr(sed, "active_strip", None)
    if strip is None or strip.type != "MOVIE":
        return

    source_path = bpy.path.abspath(strip.filepath)
    if not os.path.isfile(source_path):
        return

    # Compute source frame index within the video file
    current = scene.frame_current
    if current < strip.frame_final_start or current >= strip.frame_final_end:
        return
    src_frame = current - strip.frame_final_start + strip.frame_offset_start

    # Open / reuse VideoCapture
    cap = _HandlerState.get_cap(source_path)
    if cap is None or not cap.isOpened():
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        return

    # Seek and read
    cap.set(cv2.CAP_PROP_POS_FRAMES, src_frame)
    ret, frame = cap.read()
    if not ret:
        return

    frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    # Get / create CRT instance
    crt = _HandlerState.get_crt(props.system, w, h)

    # Apply monitor settings
    crt.hue = props.hue
    crt.brightness = props.brightness
    crt.contrast = props.contrast
    crt.saturation = props.saturation
    crt.black_point = props.black_point
    crt.white_point = props.white_point
    crt.scanlines = props.scanlines
    crt.blend = props.blend
    crt.v_fac = props.v_fac

    # Process
    output = crt.process(
        frame_bgra,
        noise=props.noise,
        hue=props.artifact_hue,
        num_frames=props.num_frames,
        progressive=props.progressive,
        raw=props.raw,
        as_color=props.as_color,
        in_format=PIX_FORMAT_BGRA,
        do_aberration=props.do_aberration,
        xoffset=props.xoffset,
        yoffset=props.yoffset,
    )

    # Write to Blender image (BGRA -> RGBA, float 0-1, bottom-up)
    img = _HandlerState.get_or_create_image(w, h)
    rgba = cv2.cvtColor(output, cv2.COLOR_BGRA2RGBA)
    # Blender images are bottom-up
    rgba = np.flipud(rgba)
    pixels = rgba.astype(np.float32).ravel() / 255.0
    img.pixels.foreach_set(pixels)
    img.update()


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
        props.system = "ntsc"
        props.hue = 0
        props.brightness = 0
        props.contrast = 180
        props.saturation = 10
        props.black_point = 0
        props.white_point = 100
        props.scanlines = True
        props.blend = True
        props.v_fac = 0
        props.noise = 24
        props.artifact_hue = 0
        props.num_frames = 4
        props.progressive = False
        props.raw = False
        props.as_color = True
        props.xoffset = 0
        props.yoffset = 0
        props.do_aberration = False
        self.report({"INFO"}, "NTSC-CRT parameters reset")
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

        out_path = _output_path(source_path, props.output_directory, props.system)

        crt = None
        prev_system = None

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

                # Advance Blender's frame so keyframed properties are evaluated
                scene.frame_set(strip.frame_final_start + frame_idx)

                # Recreate CRT if system changed via keyframe
                if crt is None or props.system != prev_system:
                    crt = CRT(props.system, out_w=w, out_h=h,
                              out_format=PIX_FORMAT_BGRA)
                    prev_system = props.system

                # Apply animated monitor settings
                crt.hue = props.hue
                crt.brightness = props.brightness
                crt.contrast = props.contrast
                crt.saturation = props.saturation
                crt.black_point = props.black_point
                crt.white_point = props.white_point
                crt.scanlines = props.scanlines
                crt.blend = props.blend
                crt.v_fac = props.v_fac

                frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                output = crt.process(
                    frame_bgra,
                    noise=props.noise,
                    hue=props.artifact_hue,
                    num_frames=props.num_frames,
                    progressive=props.progressive,
                    raw=props.raw,
                    as_color=props.as_color,
                    in_format=PIX_FORMAT_BGRA,
                    do_aberration=props.do_aberration,
                    xoffset=props.xoffset,
                    yoffset=props.yoffset,
                )
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
            name=f"NTSC {props.system.upper()}",
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


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class SEQUENCER_PT_ntsc_crt(Panel):
    bl_label = "NTSC-CRT"
    bl_space_type = "SEQUENCE_EDITOR"
    bl_region_type = "UI"
    bl_category = "NTSC-CRT"

    def draw(self, context):
        layout = self.layout
        props = context.scene.ntsc_crt
        strip = _active_movie_strip(context)

        # Enable toggle
        row = layout.row(align=True)
        row.scale_y = 1.5
        icon = "PAUSE" if props.enabled else "PLAY"
        label = "Disable Preview" if props.enabled else "Enable Preview"
        row.operator("sequencer.ntsc_toggle", text=label, icon=icon)

        # Refresh and Reset buttons
        row = layout.row(align=True)
        sub = row.row(align=True)
        sub.operator("sequencer.ntsc_refresh", text="Refresh", icon="FILE_REFRESH")
        sub.enabled = props.enabled
        row.operator("sequencer.ntsc_reset", text="Reset", icon="LOOP_BACK")

        if strip:
            layout.label(text=f"Active: {strip.name}", icon="SEQ_STRIP_META")
        else:
            layout.label(text="Select a movie strip", icon="INFO")

        if props.enabled:
            layout.label(
                text=f"View in Image Editor: \"{_HandlerState.preview_name}\"",
                icon="IMAGE",
            )

        layout.separator()

        # System
        layout.prop(props, "system")

        # Monitor Settings
        box = layout.box()
        box.label(text="Monitor Settings", icon="DESKTOP")
        box.prop(props, "hue", slider=True)
        box.prop(props, "brightness", slider=True)
        box.prop(props, "contrast", slider=True)
        box.prop(props, "saturation", slider=True)
        box.prop(props, "black_point", slider=True)
        box.prop(props, "white_point", slider=True)

        # Display
        box = layout.box()
        box.label(text="Display", icon="RESTRICT_VIEW_OFF")
        row = box.row()
        row.prop(props, "scanlines")
        row.prop(props, "blend")
        box.prop(props, "v_fac", slider=True)

        # Signal
        box = layout.box()
        box.label(text="Signal", icon="FORCE_HARMONIC")
        box.prop(props, "noise", slider=True)
        box.prop(props, "artifact_hue", slider=True)
        box.prop(props, "num_frames")
        row = box.row()
        row.prop(props, "progressive")
        row.prop(props, "as_color")
        box.prop(props, "raw")

        # Offsets
        box = layout.box()
        box.label(text="Offsets", icon="ORIENTATION_CURSOR")
        row = box.row(align=True)
        row.prop(props, "xoffset")
        row.prop(props, "yoffset")

        # VHS-specific (only shown for ntscvhs)
        if props.system == "ntscvhs":
            box = layout.box()
            box.label(text="VHS", icon="FILE_MOVIE")
            box.prop(props, "do_aberration")

        # Render
        layout.separator()
        box = layout.box()
        box.label(text="Render", icon="RENDER_ANIMATION")
        box.prop(props, "output_directory")
        row = box.row()
        row.scale_y = 1.3
        row.operator("sequencer.ntsc_render", text="Render to Strip", icon="RENDER_ANIMATION")


# ---------------------------------------------------------------------------
# Register / Unregister
# ---------------------------------------------------------------------------

_classes = (
    NTSCCRTProperties,
    SEQUENCER_OT_ntsc_toggle,
    SEQUENCER_OT_ntsc_refresh,
    SEQUENCER_OT_ntsc_reset,
    SEQUENCER_OT_ntsc_render,
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
