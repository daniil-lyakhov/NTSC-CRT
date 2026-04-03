bl_info = {
    "name": "NTSC-CRT Effect",
    "version": (2, 0, 0),
    "blender": (5, 0, 0),
    "category": "Sequencer",
    "description": "Real-time NTSC/VHS CRT television effect on video strips",
}

import os
import sys
from dataclasses import dataclass, field

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
# Knob / Filter descriptors — single source of truth for all effects
# ---------------------------------------------------------------------------

@dataclass
class Knob:
    """Descriptor for a single Blender UI property."""
    attr: str
    label: str
    description: str = ""
    prop_type: str = "int"           # "int", "bool", "enum"
    default: object = 0
    min_val: int = 0
    max_val: int = 100
    slider: bool = True
    items: list = field(default_factory=list)   # for enum props
    group: str = ""
    group_icon: str = "NONE"
    row_with: str = ""               # attr name to share a row with


@dataclass
class MonitorKnob(Knob):
    """A global monitor property set on the CRT instance or passed as kwarg."""
    crt_attr: str = ""               # CRT attribute name (defaults to attr)
    is_demod_kwarg: bool = False     # True → passed to demodulate()/process() as kwarg


@dataclass
class StripKnob(Knob):
    """A per-strip property passed to process()/modulate()."""
    kwarg: str = ""                  # kwarg name for process()/modulate(); "" = not a kwarg
    modulate_only: bool = False      # True → only in modulate(), not in process()
    show_if_system: str = ""         # only show if strip.system == this value


@dataclass
class SignalFilterDef:
    """A signal-level filter with its UI knobs and apply function."""
    name: str
    params: list                     # list[Knob]
    toggle_attr: str = ""            # BoolProperty attr for on/off toggle
    is_active: object = None         # callable(props) -> bool
    apply: object = None             # callable(signal_int16, props) -> signal_int16


# --- Individual filter apply functions ---

def _apply_gain(s, props):
    return (s * props.filter_gain) // 100


def _apply_fuzz(s, props):
    threshold = max(1, int(127 * (1.0 - props.filter_fuzz / 100.0)))
    return np.clip(s, -threshold, threshold)


def _apply_echo(s, props):
    ghost = np.roll(s, props.filter_echo_delay, axis=1)
    return s + (ghost * props.filter_echo_amount) // 100


def _generate_pink_noise_2d(shape):
    """Generate 2D pink (1/f) noise correlated across scan lines."""
    vres, hres = shape
    white = np.random.standard_normal(shape)
    F = np.fft.rfft2(white)
    fv = np.fft.fftfreq(vres)[:, None]
    fh = np.fft.rfftfreq(hres)[None, :]
    freq_mag = np.sqrt(fv ** 2 + fh ** 2)
    freq_mag[0, 0] = 1.0  # avoid division by zero at DC
    F *= 1.0 / np.sqrt(freq_mag)
    F[0, 0] = 0.0  # zero DC component
    pink = np.fft.irfft2(F, s=shape)
    std = pink.std()
    if std > 0:
        pink /= std
    return pink


def _apply_atmospheric(s, props):
    """Apply atmospheric noise: 2D pink noise + impulse with vertical bleed."""
    strength = props.filter_atmo_strength
    impulse = props.filter_atmo_impulse
    if strength > 0:
        pink = _generate_pink_noise_2d(s.shape)
        s = s + (pink * (strength / 100.0) * 50).astype(np.int16)
    if impulse > 0:
        prob = (impulse / 100.0) * 0.10
        mask = np.random.random(s.shape) < prob
        spikes = (np.random.randint(-127, 128, size=s.shape, dtype=np.int16)
                  * mask.astype(np.int16))
        s = s + spikes
        for shift, decay in [(1, 0.4), (-1, 0.4), (2, 0.15), (-2, 0.15)]:
            s = s + (np.roll(spikes, shift, axis=0) * decay).astype(np.int16)
    return s


def _apply_h_jitter(s, props):
    """Smooth horizontal jitter — shifts all lines as a correlated wave.

    Generates 1D pink (1/f) noise along the vertical axis so neighbouring
    scan lines shift by similar amounts, producing a wobbly/warped picture
    like bad horizontal sync or a damaged VHS tape.
    """
    strength = props.filter_jitter_strength
    vres = s.shape[0]
    # 1D pink noise: correlated shifts per scan line
    white = np.fft.rfft(np.random.standard_normal(vres))
    freqs = np.fft.rfftfreq(vres)
    freqs[0] = 1.0
    white *= 1.0 / np.sqrt(freqs)
    white[0] = 0.0
    offsets = np.fft.irfft(white, n=vres)
    std = offsets.std()
    if std > 0:
        offsets /= std
    # Scale: strength 10 ≈ subtle wobble, 50 ≈ heavy warp
    offsets = (offsets * strength * 0.5).astype(np.int32)
    out = np.empty_like(s)
    for row in range(vres):
        out[row] = np.roll(s[row], offsets[row])
    return out


# Ring buffer of previous signals for the delay filter
_signal_history = []


def _apply_delay(s, props):
    """Temporal delay — blend previous frames' signals with exponential decay.

    Maintains a ring buffer of up to *tail* past signals.  Each older
    frame is weighted by  mix * decay^i  (i=0 is the most recent).
    """
    global _signal_history
    mix = props.filter_delay_mix / 100.0
    tail = props.filter_delay_tail

    if _signal_history and _signal_history[0].shape != s.shape:
        _signal_history.clear()

    if tail > 0 and mix > 0 and _signal_history:
        acc = s.astype(np.float32)
        weight_sum = 1.0
        decay = mix
        w = decay
        for past in _signal_history:
            acc += past.astype(np.float32) * w
            weight_sum += w
            w *= decay
        s = (acc / weight_sum).astype(np.int16)

    _signal_history.insert(0, s.copy())
    if len(_signal_history) > tail:
        _signal_history[:] = _signal_history[:max(tail, 1)]
    return s


# ---------------------------------------------------------------------------
# Registries — add a new effect by adding ONE entry here
# ---------------------------------------------------------------------------

# Output size presets: (width, height) — "source" means use input video dimensions
_OUTPUT_SIZE_PRESETS = {
    "source":   None,          # pass-through
    "ntsc":     (640, 480),    # NTSC standard square-pixel 4:3
    "dvd":      (720, 480),    # DVD/DV NTSC (non-square pixels)
    "half":     (320, 240),    # Quarter-frame capture card
}

# Input fps presets — "source" means keep the original frame rate
_INPUT_FPS_PRESETS = {
    "source":   None,          # pass-through
    "ntsc":     30000 / 1001,  # ≈29.97 — authentic NTSC frame rate
}

MONITOR_KNOBS = [
    MonitorKnob("output_size", "Output Size",
                "CRT output resolution preset. "
                "Source = same as input video. "
                "NTSC 640×480 = authentic square-pixel broadcast. "
                "DVD 720×480 = standard NTSC DVD. "
                "Half 320×240 = fast quarter-frame",
                prop_type="enum", default="ntsc", slider=False,
                items=[
                    ("source", "Source", "Same resolution as input video"),
                    ("ntsc", "NTSC 640×480", "Standard NTSC square-pixel 4:3 (authentic)"),
                    ("dvd", "DVD 720×480", "DVD/DV NTSC resolution (non-square pixels)"),
                    ("half", "Half 320×240", "Quarter-frame for fast preview"),
                ],
                group="Output", group_icon="OUTPUT",
                crt_attr="_skip"),
    MonitorKnob("input_fps", "Input FPS",
                "Convert source video to this frame rate before CRT "
                "processing (render only). "
                "Source = keep original frame rate. "
                "NTSC 29.97 = authentic NTSC broadcast rate, 2× faster "
                "for 60 fps sources",
                prop_type="enum", default="source", slider=False,
                items=[
                    ("source", "Source", "Keep original frame rate"),
                    ("ntsc", "NTSC 29.97", "Authentic NTSC frame rate (29.97 fps)"),
                ],
                group="Output", group_icon="OUTPUT",
                crt_attr="_skip"),
    MonitorKnob("hue", "Hue", "Color hue rotation in degrees",
                default=0, min_val=-360, max_val=360,
                group="Monitor", group_icon="DESKTOP"),
    MonitorKnob("brightness", "Brightness", "Brightness offset",
                default=0, min_val=-100, max_val=100,
                group="Monitor", group_icon="DESKTOP"),
    MonitorKnob("contrast", "Contrast", "Contrast multiplier",
                default=180, min_val=0, max_val=500,
                group="Monitor", group_icon="DESKTOP"),
    MonitorKnob("saturation", "Saturation", "Color saturation multiplier",
                default=10, min_val=0, max_val=200,
                group="Monitor", group_icon="DESKTOP"),
    MonitorKnob("black_point", "Black Point", "Black level adjustment",
                default=0, min_val=-50, max_val=50,
                group="Monitor", group_icon="DESKTOP"),
    MonitorKnob("white_point", "White Point", "White level adjustment",
                default=100, min_val=0, max_val=200,
                group="Monitor", group_icon="DESKTOP"),
    MonitorKnob("scanlines", "Scanlines", "Visible gaps between scan lines",
                prop_type="bool", default=True, slider=False,
                group="Display", group_icon="RESTRICT_VIEW_OFF",
                row_with="blend"),
    MonitorKnob("blend", "Blend", "Blend new field onto previous image",
                prop_type="bool", default=True, slider=False,
                group="Display", group_icon="RESTRICT_VIEW_OFF"),
    MonitorKnob("v_fac", "V Stretch", "Vertical stretch factor",
                default=0, min_val=0, max_val=100,
                group="Display", group_icon="RESTRICT_VIEW_OFF"),
    MonitorKnob("noise", "Noise", "Signal noise amount (0 = clean)",
                default=0, min_val=0, max_val=255,
                group="Display", group_icon="RESTRICT_VIEW_OFF",
                is_demod_kwarg=True),
]

STRIP_KNOBS = [
    StripKnob("system", "System", "CRT system variant",
              prop_type="enum", default="ntsc",
              items=[("ntsc", "NTSC", "Standard NTSC television"),
                     ("ntscvhs", "NTSC VHS", "NTSC with VHS tape quality")],
              slider=False, group="", group_icon="NONE"),
    StripKnob("artifact_hue", "Artifact Hue",
              "Artifact color hue offset (0-359)",
              default=0, min_val=0, max_val=359,
              kwarg="hue", modulate_only=True,
              group="Signal", group_icon="FORCE_HARMONIC"),
    StripKnob("num_frames", "Accumulate Frames",
              "Frames to accumulate per output frame, default is 4 for NTSC",
              default=4, min_val=1, max_val=16, slider=False,
              kwarg="num_frames",
              group="Signal", group_icon="FORCE_HARMONIC"),
    StripKnob("progressive", "Progressive",
              "Progressive scan (vs interlaced)",
              prop_type="bool", default=False, slider=False,
              kwarg="progressive",
              group="Signal", group_icon="FORCE_HARMONIC",
              row_with="as_color"),
    StripKnob("as_color", "Color",
              "Full color output (off = monochrome)",
              prop_type="bool", default=True, slider=False,
              kwarg="as_color", modulate_only=True,
              group="Signal", group_icon="FORCE_HARMONIC"),
    StripKnob("raw", "Raw",
              "Don't scale input to fit the CRT viewport",
              prop_type="bool", default=False, slider=False,
              kwarg="raw", modulate_only=True,
              group="Signal", group_icon="FORCE_HARMONIC"),
    StripKnob("xoffset", "X Offset",
              "Horizontal offset in sample space",
              default=0, min_val=-200, max_val=200, slider=False,
              kwarg="xoffset", modulate_only=True,
              group="Offsets", group_icon="ORIENTATION_CURSOR",
              row_with="yoffset"),
    StripKnob("yoffset", "Y Offset",
              "Vertical offset in scan lines",
              default=0, min_val=-200, max_val=200, slider=False,
              kwarg="yoffset", modulate_only=True,
              group="Offsets", group_icon="ORIENTATION_CURSOR"),
    StripKnob("do_aberration", "VHS Aberration",
              "Signal distortion at bottom of frame (ntscvhs only)",
              prop_type="bool", default=False, slider=False,
              kwarg="do_aberration", modulate_only=True,
              group="VHS", group_icon="FILE_MOVIE",
              show_if_system="ntscvhs"),
]

SIGNAL_FILTERS = [
    SignalFilterDef(
        "gain",
        params=[Knob("filter_gain", "Gain",
                      "Signal gain as percentage (100 = unity, 200 = 2x)",
                      default=100, min_val=0, max_val=300)],
        toggle_attr="filter_gain_enabled",
        is_active=lambda p: p.filter_gain_enabled,
        apply=_apply_gain,
    ),
    SignalFilterDef(
        "fuzz",
        params=[Knob("filter_fuzz", "Fuzz",
                      "Hard-clip distortion amount (0 = off, 100 = max)",
                      default=0, min_val=0, max_val=100)],
        toggle_attr="filter_fuzz_enabled",
        is_active=lambda p: p.filter_fuzz_enabled,
        apply=_apply_fuzz,
    ),
    SignalFilterDef(
        "echo",
        params=[
            Knob("filter_echo_delay", "Echo Delay",
                 "Ghost/echo horizontal delay in samples (0 = off)",
                 default=0, min_val=0, max_val=200),
            Knob("filter_echo_amount", "Echo Amount",
                 "Ghost/echo signal amplitude percentage",
                 default=0, min_val=0, max_val=100),
        ],
        toggle_attr="filter_echo_enabled",
        is_active=lambda p: p.filter_echo_enabled,
        apply=_apply_echo,
    ),
    SignalFilterDef(
        "atmospheric",
        toggle_attr="filter_atmo_enabled",
        params=[
            Knob("filter_atmo_strength", "Atmospheric",
                 "Pink (1/f) noise simulating atmospheric/cosmic interference "
                 "— correlated across scan lines. "
                 "Realistic: 5-15 (weak signal), 25-40 (fringe reception), "
                 "50+ (unwatchable snow)",
                 default=15, min_val=0, max_val=100),
            Knob("filter_atmo_impulse", "Impulse",
                 "Random spike bursts from lightning/electrical interference "
                 "— bleeds across adjacent scan lines. "
                 "Realistic: 2-8 (suburban), 10-20 (nearby thunderstorm), "
                 "30+ (extreme)",
                 default=8, min_val=0, max_val=100),
        ],
        is_active=lambda p: p.filter_atmo_enabled,
        apply=_apply_atmospheric,
    ),
    SignalFilterDef(
        "jitter",
        params=[
            Knob("filter_jitter_strength", "H-Jitter",
                 "Horizontal jitter — smooth correlated line displacement "
                 "simulating sync instability or tape wobble. "
                 "Realistic: 5-15 (worn tape), 25-40 (bad tracking), "
                 "50+ (unwatchable)",
                 default=0, min_val=0, max_val=100),
        ],
        toggle_attr="filter_jitter_enabled",
        is_active=lambda p: p.filter_jitter_enabled,
        apply=_apply_h_jitter,
    ),
    SignalFilterDef(
        "delay",
        params=[
            Knob("filter_delay_mix", "Delay Mix",
                 "Decay rate for each older frame — controls how quickly "
                 "past signals fade out. "
                 "Realistic: 10-30 (slight ghosting), 50-70 (heavy burn-in), "
                 "80+ (psychedelic feedback)",
                 default=50, min_val=0, max_val=100),
            Knob("filter_delay_tail", "Tail",
                 "Number of past frames to accumulate. "
                 "1 = blend with previous only, 5-10 = long trails, "
                 "20+ = extreme feedback",
                 default=3, min_val=1, max_val=30),
        ],
        toggle_attr="filter_delay_enabled",
        is_active=lambda p: p.filter_delay_enabled,
        apply=_apply_delay,
    ),
]


# ---------------------------------------------------------------------------
# Build Blender properties from descriptors
# ---------------------------------------------------------------------------

def _build_blender_prop(knob, update_fn):
    """Construct a Blender property from a Knob descriptor."""
    if knob.prop_type == "bool":
        return BoolProperty(
            name=knob.label, description=knob.description,
            default=knob.default, update=update_fn,
        )
    if knob.prop_type == "enum":
        return EnumProperty(
            name=knob.label, description=knob.description,
            items=knob.items, default=knob.default, update=update_fn,
        )
    # int (default)
    return IntProperty(
        name=knob.label, description=knob.description,
        default=knob.default, min=knob.min_val, max=knob.max_val,
        update=update_fn,
    )


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
# Per-strip CRT settings (properties generated from STRIP_KNOBS registry)
# ---------------------------------------------------------------------------

class NTSCStripSettings(PropertyGroup):
    strip_name: StringProperty(name="Strip Name", default="")


# ---------------------------------------------------------------------------
# Scene-level globals (properties generated from MONITOR_KNOBS + SIGNAL_FILTERS)
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
    signal_filters: BoolProperty(
        name="Signal Filters",
        description="Enable signal-level filters (gain, fuzz, echo)",
        default=False, update=_on_global_knob_update,
    )


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
    for k in MONITOR_KNOBS:
        if k.is_demod_kwarg or k.crt_attr == "_skip":
            continue
        setattr(crt, k.crt_attr or k.attr, getattr(props, k.attr))


def _write_output_to_image(output, w, h):
    """Write a BGRA output array to the Blender preview image."""
    img = _HandlerState.get_or_create_image(w, h)
    rgba = cv2.cvtColor(output, cv2.COLOR_BGRA2RGBA)
    rgba = np.flipud(rgba)
    pixels = rgba.astype(np.float32).ravel() / 255.0
    img.pixels.foreach_set(pixels)
    img.update()


def _apply_signal_filters(signal, props):
    """Apply all active signal filters from the SIGNAL_FILTERS registry.

    Parameters
    ----------
    signal : np.ndarray  int8 (VRES, HRES)
    props  : NTSCCRTProperties

    Returns
    -------
    np.ndarray  int8, same shape.
    """
    any_active = any(f.is_active(props) for f in SIGNAL_FILTERS)
    if not any_active:
        return signal

    s = signal.astype(np.int16)
    for f in SIGNAL_FILTERS:
        if f.is_active(props):
            s = f.apply(s, props)
    return np.clip(s, -128, 127).astype(np.int8)


def _modulate_kwargs(settings):
    """Build keyword arguments for modulate() from STRIP_KNOBS registry."""
    kw = {}
    for k in STRIP_KNOBS:
        if k.kwarg and k.modulate_only:
            kw[k.kwarg] = getattr(settings, k.attr)
    kw["in_format"] = PIX_FORMAT_BGRA
    return kw


def _demodulate_kwargs(props):
    """Build keyword arguments for demodulate() from MONITOR_KNOBS registry."""
    return {k.crt_attr or k.attr: getattr(props, k.attr)
            for k in MONITOR_KNOBS if k.is_demod_kwarg}


def _process_kwargs(strip_settings, props):
    """Build keyword arguments for process() from both registries."""
    kw = {}
    for k in STRIP_KNOBS:
        if k.kwarg:
            kw[k.kwarg] = getattr(strip_settings, k.attr)
    kw["in_format"] = PIX_FORMAT_BGRA
    for k in MONITOR_KNOBS:
        if k.is_demod_kwarg:
            kw[k.crt_attr or k.attr] = getattr(props, k.attr)
    return kw


def _process_frame(crt, frame_bgra, strip_settings, props):
    """Apply settings and process a single frame through the CRT."""
    _apply_monitor_settings(crt, props)
    return crt.process(frame_bgra, **_process_kwargs(strip_settings, props))


def _process_frame_with_filters(crt, frame_bgra, strip_settings, props):
    """Process a frame through modulate/filter/demodulate with num_frames accumulation.

    Replicates the internal process() loop but inserts signal filters
    between modulate and demodulate on every pass.
    """
    _apply_monitor_settings(crt, props)
    mkw = _modulate_kwargs(strip_settings)
    dkw = _demodulate_kwargs(props)
    num_frames = strip_settings.num_frames
    progressive = strip_settings.progressive

    field = 0
    frame = 0
    output = None

    for i in range(num_frames):
        crt.modulate(frame_bgra, field=field, frame=frame, **mkw)
        sig = crt.get_analog_signal()
        sig = _apply_signal_filters(sig, props)
        crt.set_analog_signal(sig)
        output = crt.demodulate(**dkw)

        if not progressive:
            field ^= 1
            crt.modulate(frame_bgra, field=field, frame=frame, **mkw)
            sig = crt.get_analog_signal()
            sig = _apply_signal_filters(sig, props)
            crt.set_analog_signal(sig)
            output = crt.demodulate(**dkw)
            if (i & 1) == 0:
                frame ^= 1

    return output


def _process_mix_frame(crt_a, crt_b, frame_a, frame_b,
                       settings_a, settings_b, mix_ratio, props):
    """Modulate two frames, mix their analog signals, and demodulate.

    Loops num_frames times (from settings_a, since crt_a demodulates)
    with proper field/frame advancement and progressive handling.
    """
    _apply_monitor_settings(crt_a, props)
    mkw_a = _modulate_kwargs(settings_a)
    mkw_b = _modulate_kwargs(settings_b)
    dkw = _demodulate_kwargs(props)
    num_frames = settings_a.num_frames
    progressive = settings_a.progressive

    field = 0
    frame = 0
    output = None

    for i in range(num_frames):
        crt_a.modulate(frame_a, field=field, frame=frame, **mkw_a)
        signal_a = crt_a.get_analog_signal()

        crt_b.modulate(frame_b, field=field, frame=frame, **mkw_b)
        signal_b = crt_b.get_analog_signal()

        mixed = _mix_and_normalize(signal_a, signal_b, mix_ratio)
        if props.signal_filters:
            mixed = _apply_signal_filters(mixed, props)
        crt_a.set_analog_signal(mixed)
        output = crt_a.demodulate(**dkw)

        if not progressive:
            field ^= 1
            crt_a.modulate(frame_a, field=field, frame=frame, **mkw_a)
            signal_a = crt_a.get_analog_signal()

            crt_b.modulate(frame_b, field=field, frame=frame, **mkw_b)
            signal_b = crt_b.get_analog_signal()

            mixed = _mix_and_normalize(signal_a, signal_b, mix_ratio)
            if props.signal_filters:
                mixed = _apply_signal_filters(mixed, props)
            crt_a.set_analog_signal(mixed)
            output = crt_a.demodulate(**dkw)
            if (i & 1) == 0:
                frame ^= 1

    return output


def _crt_dimensions(props, src_w, src_h):
    """Return (out_w, out_h) for CRT based on the output_size preset."""
    preset = _OUTPUT_SIZE_PRESETS.get(props.output_size)
    if preset is None:
        return src_w, src_h
    return preset


def _input_fps(props, source_fps):
    """Return the target input fps based on the input_fps preset."""
    preset = _INPUT_FPS_PRESETS.get(props.input_fps)
    if preset is None:
        return source_fps
    return preset


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

    ow, oh = _crt_dimensions(props, w, h)
    settings = _get_strip_settings(props, strip.name)
    crt = _HandlerState.get_crt_slot("a", settings.system, ow, oh)
    if props.signal_filters:
        output = _process_frame_with_filters(crt, frame, settings, props)
    else:
        output = _process_frame(crt, frame, settings, props)
    _write_output_to_image(output, ow, oh)


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

    ow, oh = _crt_dimensions(props, w, h)

    # Resize B to match A if needed
    if (wb, hb) != (w, h):
        frame_b = cv2.resize(frame_b, (w, h),
                             interpolation=cv2.INTER_LINEAR)

    settings_a = _get_strip_settings(props, strip_a.name)
    settings_b = _get_strip_settings(props, strip_b.name)
    crt_a = _HandlerState.get_crt_slot("a", settings_a.system, ow, oh)
    crt_b = _HandlerState.get_crt_slot("b", settings_b.system, ow, oh)
    output = _process_mix_frame(crt_a, crt_b, frame_a, frame_b,
                                settings_a, settings_b, props.mix_ratio,
                                props)
    _write_output_to_image(output, ow, oh)


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
        # Temporarily disable preview so property updates don't trigger
        # expensive re-renders for every single knob reset.
        was_enabled = props.enabled
        was_mix = props.mix_enabled
        props.enabled = False
        props.mix_enabled = False

        props.mix_ratio = 50
        props.strip_settings.clear()
        # Reset all registry-driven properties to defaults
        for k in MONITOR_KNOBS:
            setattr(props, k.attr, k.default)
        props.signal_filters = False
        for f in SIGNAL_FILTERS:
            if f.toggle_attr:
                setattr(props, f.toggle_attr, False)
            for p in f.params:
                setattr(props, p.attr, p.default)
        # Flush cached CRT instances so stale output buffers (used by blend)
        # and internal sync state don't bleed through after a reset.
        _HandlerState.release_all()

        # Restore preview state and render one fresh frame
        props.enabled = was_enabled
        props.mix_enabled = was_mix
        if props.enabled or props.mix_enabled:
            _ntsc_frame_handler(context.scene)
            _tag_image_editors()
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

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if src_w <= 0 or src_h <= 0:
            cap.release()
            self.report({"ERROR"}, "Cannot read video dimensions")
            return {"CANCELLED"}

        w, h = _crt_dimensions(props, src_w, src_h)
        crt_fps = _input_fps(props, src_fps)

        src_start = strip.frame_offset_start
        src_frame_count = strip.frame_final_duration

        # When input fps differs, read all source frames but only process
        # the ones that land on the target rate (nearest-neighbour drop).
        if crt_fps != src_fps and src_fps > 0:
            out_frame_count = int(src_frame_count * crt_fps / src_fps)
        else:
            out_frame_count = src_frame_count

        if src_start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, src_start)

        out_path = _output_path(source_path, props.output_directory,
                                settings.system)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, crt_fps, (w, h))
        if not writer.isOpened():
            cap.release()
            self.report({"ERROR"}, f"Cannot create output: {out_path}")
            return {"CANCELLED"}

        wm = context.window_manager
        wm.progress_begin(0, out_frame_count)

        scene = context.scene
        original_frame = scene.frame_current
        frame_idx = 0
        src_read = 0
        next_src_needed = 0
        try:
            while frame_idx < out_frame_count:
                # Advance through source frames to reach the next needed one
                while src_read <= next_src_needed:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    src_read += 1
                if not ret:
                    break

                scene.frame_set(strip.frame_final_start + frame_idx)

                frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                crt = _HandlerState.get_crt_slot("a", settings.system, w, h)
                if props.signal_filters:
                    output = _process_frame_with_filters(
                        crt, frame_bgra, settings, props)
                else:
                    output = _process_frame(crt, frame_bgra, settings, props)
                writer.write(cv2.cvtColor(output, cv2.COLOR_BGRA2BGR))
                frame_idx += 1
                wm.progress_update(frame_idx)

                # Compute which source frame is needed next
                next_src_needed = int(frame_idx * src_fps / crt_fps)
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

        # When fps was converted, add a speed effect to stretch the strip
        # so it matches the original timeline duration.
        if crt_fps != src_fps and src_fps > 0:
            speed = strips.new_effect(
                name=f"NTSC Speed",
                type="SPEED",
                channel=new_strip.channel + 1,
                frame_start=new_strip.frame_final_start,
                seq1=new_strip,
            )
            speed.speed_factor = crt_fps / src_fps
            speed.use_default_fade = False

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

        src_fps = cap_a.get(cv2.CAP_PROP_FPS)
        src_w = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if src_w <= 0 or src_h <= 0:
            cap_a.release()
            cap_b.release()
            self.report({"ERROR"}, "Cannot read video dimensions")
            return {"CANCELLED"}

        w, h = _crt_dimensions(props, src_w, src_h)
        crt_fps = _input_fps(props, src_fps)

        # Timeline range: overlapping portion of both strips
        start = max(strip_a.frame_final_start, strip_b.frame_final_start)
        end = min(strip_a.frame_final_end, strip_b.frame_final_end)
        if end <= start:
            cap_a.release()
            cap_b.release()
            self.report({"ERROR"}, "Strips do not overlap on the timeline")
            return {"CANCELLED"}
        src_frame_count = end - start

        if crt_fps != src_fps and src_fps > 0:
            out_frame_count = int(src_frame_count * crt_fps / src_fps)
        else:
            out_frame_count = src_frame_count

        out_path = _output_path(path_a, props.output_directory,
                                settings_a.system, suffix="_mix")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, crt_fps, (w, h))
        if not writer.isOpened():
            cap_a.release()
            cap_b.release()
            self.report({"ERROR"}, f"Cannot create output: {out_path}")
            return {"CANCELLED"}

        wm = context.window_manager
        wm.progress_begin(0, out_frame_count)

        scene = context.scene
        original_frame = scene.frame_current
        frame_idx = 0

        try:
            for out_i in range(out_frame_count):
                # Map output frame to source timeline frame
                if crt_fps != src_fps and src_fps > 0:
                    src_offset = int(out_i * src_fps / crt_fps)
                else:
                    src_offset = out_i
                timeline_frame = start + src_offset

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
                    settings_a, settings_b, props.mix_ratio, props,
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

        # When fps was converted, add a speed effect to stretch the strip
        # so it matches the original timeline duration.
        if crt_fps != src_fps and src_fps > 0:
            speed = strips.new_effect(
                name=f"NTSC Speed",
                type="SPEED",
                channel=new_strip.channel + 1,
                frame_start=new_strip.frame_final_start,
                seq1=new_strip,
            )
            speed.speed_factor = crt_fps / src_fps
            speed.use_default_fade = False

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
    """Draw per-strip CRT knobs from STRIP_KNOBS registry."""
    header = layout.box()
    header.label(text=label, icon="SEQ_STRIP_META")

    current_group = None
    box = header
    skip_row_attrs = set()  # attrs already drawn via row_with

    for k in STRIP_KNOBS:
        if k.attr in skip_row_attrs:
            continue
        if k.show_if_system and settings.system != k.show_if_system:
            continue

        # Start a new group box when group changes
        if k.group and k.group != current_group:
            box = header.box()
            box.label(text=k.group, icon=k.group_icon)
            current_group = k.group
        elif not k.group:
            box = header
            current_group = None

        if k.row_with:
            row = box.row()
            row.prop(settings, k.attr)
            row.prop(settings, k.row_with)
            skip_row_attrs.add(k.row_with)
        elif k.slider:
            box.prop(settings, k.attr, slider=True)
        else:
            box.prop(settings, k.attr)


def _draw_monitor_knobs(layout, props):
    """Draw global monitor / display / noise settings from MONITOR_KNOBS registry."""
    current_group = None
    box = None
    skip_row_attrs = set()

    for k in MONITOR_KNOBS:
        if k.attr in skip_row_attrs:
            continue

        if k.group != current_group:
            box = layout.box()
            box.label(text=k.group, icon=k.group_icon)
            current_group = k.group

        if k.row_with:
            row = box.row()
            row.prop(props, k.attr)
            row.prop(props, k.row_with)
            skip_row_attrs.add(k.row_with)
        elif k.slider:
            box.prop(props, k.attr, slider=True)
        else:
            box.prop(props, k.attr)


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

        # Global monitor settings
        layout.separator()
        _draw_monitor_knobs(layout, props)

        # Signal Filters
        layout.separator()
        fbox = layout.box()
        fbox.label(text="Signal Filters", icon="FORCE_CHARGE")
        fbox.prop(props, "signal_filters")
        col = fbox.column(align=True)
        col.enabled = props.signal_filters
        for f in SIGNAL_FILTERS:
            if f.toggle_attr:
                col.separator()
                col.prop(props, f.toggle_attr, text=f.name.capitalize(),
                         icon="CHECKBOX_HLT" if getattr(props, f.toggle_attr)
                         else "CHECKBOX_DEHLT")
            sub = col.column(align=True)
            sub.enabled = props.signal_filters and (
                getattr(props, f.toggle_attr) if f.toggle_attr else True)
            for p in f.params:
                sub.prop(props, p.attr, slider=p.slider)

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
    # Generate Blender properties from registries before registering classes
    for k in STRIP_KNOBS:
        NTSCStripSettings.__annotations__[k.attr] = _build_blender_prop(
            k, _on_strip_knob_update)
    for k in MONITOR_KNOBS:
        NTSCCRTProperties.__annotations__[k.attr] = _build_blender_prop(
            k, _on_global_knob_update)
    for f in SIGNAL_FILTERS:
        if f.toggle_attr:
            NTSCCRTProperties.__annotations__[f.toggle_attr] = BoolProperty(
                name=f"{f.name.capitalize()} Enabled",
                description=f"Enable/disable the {f.name} signal filter",
                default=False, update=_on_global_knob_update,
            )
        for p in f.params:
            NTSCCRTProperties.__annotations__[p.attr] = _build_blender_prop(
                p, _on_global_knob_update)

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
