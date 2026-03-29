bl_info = {
    "name": "NTSC-CRT Effect",
    "version": (1, 0, 0),
    "blender": (5, 0, 0),
    "category": "Sequencer",
    "description": "Apply NTSC/VHS CRT television effect to video strips",
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
# Set NTSC_CRT_PATH to the directory containing the ntsc_crt module and
# compiled .so files, e.g.:
#   export NTSC_CRT_PATH=/home/user/Projects/NTSC-CRT/python
# Then launch Blender from that shell.
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
# Property group — every CRT knob
# ---------------------------------------------------------------------------

class NTSCCRTProperties(PropertyGroup):
    # System
    system: EnumProperty(
        name="System",
        description="CRT system variant",
        items=[
            ("ntsc", "NTSC", "Standard NTSC television"),
            ("ntscvhs", "NTSC VHS", "NTSC with VHS tape quality"),
        ],
        default="ntsc",
    )

    # --- Monitor settings (defaults from crt_reset() in crt_core.c) ---
    hue: IntProperty(
        name="Hue", description="Color hue rotation in degrees",
        default=0, min=-360, max=360,
    )
    brightness: IntProperty(
        name="Brightness", description="Brightness offset",
        default=0, min=-100, max=100,
    )
    contrast: IntProperty(
        name="Contrast", description="Contrast multiplier",
        default=180, min=0, max=500,
    )
    saturation: IntProperty(
        name="Saturation", description="Color saturation multiplier",
        default=10, min=0, max=200,
    )
    black_point: IntProperty(
        name="Black Point", description="Black level adjustment",
        default=0, min=-50, max=50,
    )
    white_point: IntProperty(
        name="White Point", description="White level adjustment",
        default=100, min=0, max=200,
    )

    # --- Display ---
    scanlines: BoolProperty(
        name="Scanlines", description="Visible gaps between scan lines",
        default=True,
    )
    blend: BoolProperty(
        name="Blend", description="Blend new field onto previous image",
        default=True,
    )
    v_fac: IntProperty(
        name="V Stretch", description="Vertical stretch factor",
        default=0, min=0, max=100,
    )

    # --- Signal / process settings ---
    noise: IntProperty(
        name="Noise", description="Signal noise amount (0 = clean)",
        default=24, min=0, max=255,
    )
    artifact_hue: IntProperty(
        name="Artifact Hue", description="Artifact color hue offset (0-359)",
        default=0, min=0, max=359,
    )
    num_frames: IntProperty(
        name="Accumulate Frames",
        description="Frames to accumulate per output frame (higher = smoother but slower)",
        default=4, min=1, max=16,
    )
    progressive: BoolProperty(
        name="Progressive", description="Progressive scan (vs interlaced)",
        default=False,
    )
    raw: BoolProperty(
        name="Raw", description="Don't scale input to fit the CRT viewport",
        default=False,
    )
    as_color: BoolProperty(
        name="Color", description="Full color output (off = monochrome)",
        default=True,
    )

    # --- Offsets ---
    xoffset: IntProperty(
        name="X Offset", description="Horizontal offset in sample space",
        default=0, min=-200, max=200,
    )
    yoffset: IntProperty(
        name="Y Offset", description="Vertical offset in scan lines",
        default=0, min=-200, max=200,
    )

    # --- VHS-specific ---
    do_aberration: BoolProperty(
        name="VHS Aberration",
        description="Signal distortion at bottom of frame (ntscvhs only)",
        default=False,
    )

    # --- Output ---
    output_directory: StringProperty(
        name="Output Dir",
        description="Directory for output video (empty = same as source)",
        subtype="DIR_PATH",
        default="",
    )


# ---------------------------------------------------------------------------
# Operator — batch process the selected strip
# ---------------------------------------------------------------------------

class SEQUENCER_OT_ntsc_apply(Operator):
    """Process the active movie strip through the NTSC-CRT emulator"""
    bl_idname = "sequencer.ntsc_apply"
    bl_label = "Apply NTSC-CRT"
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

        # Open source video
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            self.report({"ERROR"}, f"Cannot open video: {source_path}")
            return {"CANCELLED"}

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames <= 0 or w <= 0 or h <= 0:
            cap.release()
            self.report({"ERROR"}, "Cannot read video dimensions")
            return {"CANCELLED"}

        # Respect strip trim offsets (when the user cuts/trims a strip)
        src_start = strip.frame_offset_start
        frames_to_process = strip.frame_final_duration

        # Seek to the first visible frame
        if src_start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, src_start)

        # Build output path
        out_path = _output_path(source_path, props.output_directory, props.system)

        # Create CRT processor
        crt = CRT(props.system, out_w=w, out_h=h, out_format=PIX_FORMAT_BGRA)

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

        # Open writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            cap.release()
            self.report({"ERROR"}, f"Cannot create output: {out_path}")
            return {"CANCELLED"}

        # Process frames with progress
        wm = context.window_manager
        wm.progress_begin(0, frames_to_process)

        frame_idx = 0
        try:
            while frame_idx < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break

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

                frame_out = cv2.cvtColor(output, cv2.COLOR_BGRA2BGR)
                writer.write(frame_out)

                frame_idx += 1
                wm.progress_update(frame_idx)
        finally:
            cap.release()
            writer.release()
            wm.progress_end()

        if frame_idx == 0:
            self.report({"ERROR"}, "No frames were processed")
            return {"CANCELLED"}

        # Add the processed video as a new strip in the sequencer
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
            f"NTSC-CRT: processed {frame_idx} frames -> {os.path.basename(out_path)}",
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
        _, strips = _get_strips(context.scene)
        strip = _active_movie_strip(context)

        # Apply button
        row = layout.row(align=True)
        row.scale_y = 1.5
        row.operator("sequencer.ntsc_apply", text="Apply NTSC-CRT", icon="PLAY")

        if strip:
            layout.label(text=f"Active: {strip.name}", icon="SEQ_STRIP_META")
        else:
            layout.label(text="Select a movie strip", icon="INFO")

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

        # Output
        box = layout.box()
        box.label(text="Output", icon="OUTPUT")
        box.prop(props, "output_directory")

        # List previously generated NTSC strips
        ntsc_strips = [
            s for s in strips
            if s.type == "MOVIE" and s.name.startswith("NTSC ")
        ]
        if ntsc_strips:
            layout.separator()
            box = layout.box()
            box.label(text="Generated Strips:")
            for s in ntsc_strips:
                sed = context.scene.sequence_editor
                active = getattr(sed, "active_strip", None)
                icon = (
                    "RESTRICT_SELECT_OFF"
                    if s == active
                    else "RESTRICT_SELECT_ON"
                )
                box.label(text=s.name, icon=icon)


# ---------------------------------------------------------------------------
# Register / Unregister
# ---------------------------------------------------------------------------

_classes = (
    NTSCCRTProperties,
    SEQUENCER_OT_ntsc_apply,
    SEQUENCER_PT_ntsc_crt,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.ntsc_crt = PointerProperty(type=NTSCCRTProperties)


def unregister():
    del bpy.types.Scene.ntsc_crt
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
