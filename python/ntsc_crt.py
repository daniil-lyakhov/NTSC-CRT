"""
Python bindings for NTSC-CRT library.

Usage:
    import numpy as np
    from ntsc_crt import CRT

    # Load an image as RGBA numpy array (uint8, shape HxWx4)
    image = ...

    # Create a CRT processor for standard NTSC
    crt = CRT("ntsc", out_w=640, out_h=480)

    # Adjust CRT monitor settings
    crt.hue = 0
    crt.saturation = 10
    crt.brightness = 0
    crt.contrast = 180
    crt.scanlines = True
    crt.blend = True

    # Process the image (returns a new numpy array)
    output = crt.process(image, noise=24, hue=0, num_frames=4)

Available systems:
    "ntsc"     - Standard NTSC
    "nes"      - NES 6/9-bit pixel output
    "pv1k"     - Casio PV-1000
    "snes"     - Super Nintendo (RGB)
    "template" - Template system
    "ntscvhs"  - NTSC with VHS quality
    "nesrgb"   - NES with RGB artifacts
"""

import importlib
import os
import sys

import numpy as np

# Ensure the lib/ directory (with compiled extensions) is on sys.path
_lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
if _lib_dir not in sys.path:
    sys.path.insert(0, _lib_dir)

# Pixel format constants
PIX_FORMAT_RGB = 0
PIX_FORMAT_BGR = 1
PIX_FORMAT_ARGB = 2
PIX_FORMAT_RGBA = 3
PIX_FORMAT_ABGR = 4
PIX_FORMAT_BGRA = 5

# System names
SYSTEMS = ("ntsc", "nes", "pv1k", "snes", "template", "ntscvhs", "nesrgb")

# Systems that use unsigned char* data (RGB image input)
_RGB_SYSTEMS = {"ntsc", "pv1k", "snes", "template", "ntscvhs", "nesrgb"}
# Systems that use unsigned short* data (palette index input)
_PALETTE_SYSTEMS = {"nes"}

# Systems with specific NTSC_SETTINGS fields
_FIELD_FRAME_SYSTEMS = {"ntsc", "pv1k", "snes", "template", "ntscvhs"}
_RAW_SYSTEMS = {"ntsc", "pv1k", "snes", "template", "ntscvhs"}
_AS_COLOR_SYSTEMS = {"ntsc", "pv1k", "snes", "template", "ntscvhs"}
_DOT_CRAWL_SYSTEMS = {"nes", "pv1k", "snes", "template", "nesrgb"}
_ABERRATION_SYSTEMS = {"ntscvhs"}
_FORMAT_SYSTEMS = {"ntsc", "pv1k", "snes", "template", "ntscvhs", "nesrgb"}

# Signal dimensions per system (HRES, VRES)
_SIGNAL_DIMS = {
    "ntsc":     (910, 262),
    "nes":      (909, 262),
    "pv1k":     (1920, 262),
    "snes":     (909, 262),
    "template": (910, 262),
    "ntscvhs":  (910, 262),
    "nesrgb":   (909, 262),
}


def _load_module(system):
    """Import the cffi-compiled extension for a given system."""
    return importlib.import_module(f"_crt_{system}")


class CRT:
    """NTSC CRT emulator.

    Parameters
    ----------
    system : str
        One of: "ntsc", "nes", "pv1k", "snes", "template", "ntscvhs", "nesrgb"
    out_w : int
        Output image width.
    out_h : int
        Output image height.
    out_format : int
        Output pixel format (default: PIX_FORMAT_BGRA).
    """

    def __init__(self, system, out_w=640, out_h=480, out_format=PIX_FORMAT_BGRA):
        if system not in SYSTEMS:
            raise ValueError(
                f"Unknown system '{system}'. Choose from: {', '.join(SYSTEMS)}"
            )

        self._system = system
        self._mod = _load_module(system)
        self._ffi = self._mod.ffi
        self._lib = self._mod.lib

        bpp = self._lib.crt_bpp4fmt(out_format)
        if bpp == 0:
            raise ValueError(f"Invalid pixel format: {out_format}")

        # Allocate output buffer
        self._out_buf = self._ffi.new(
            "unsigned char[]", out_w * out_h * bpp
        )
        # Zero it out
        self._ffi.buffer(self._out_buf)[:] = b"\x00" * (out_w * out_h * bpp)

        # Create and initialize CRT struct
        self._crt = self._ffi.new("struct CRT *")
        self._lib.crt_init(self._crt, out_w, out_h, out_format, self._out_buf)

        self._out_w = out_w
        self._out_h = out_h
        self._out_format = out_format
        self._bpp = bpp

    @property
    def system(self):
        """The CRT system variant name."""
        return self._system

    @property
    def out_w(self):
        """Output width in pixels."""
        return self._out_w

    @property
    def out_h(self):
        """Output height in pixels."""
        return self._out_h

    # --- CRT monitor settings (read/write properties) ---

    @property
    def hue(self):
        return self._crt.hue

    @hue.setter
    def hue(self, v):
        self._crt.hue = int(v)

    @property
    def brightness(self):
        return self._crt.brightness

    @brightness.setter
    def brightness(self, v):
        self._crt.brightness = int(v)

    @property
    def contrast(self):
        return self._crt.contrast

    @contrast.setter
    def contrast(self, v):
        self._crt.contrast = int(v)

    @property
    def saturation(self):
        return self._crt.saturation

    @saturation.setter
    def saturation(self, v):
        self._crt.saturation = int(v)

    @property
    def black_point(self):
        return self._crt.black_point

    @black_point.setter
    def black_point(self, v):
        self._crt.black_point = int(v)

    @property
    def white_point(self):
        return self._crt.white_point

    @white_point.setter
    def white_point(self, v):
        self._crt.white_point = int(v)

    @property
    def scanlines(self):
        return bool(self._crt.scanlines)

    @scanlines.setter
    def scanlines(self, v):
        self._crt.scanlines = int(bool(v))

    @property
    def blend(self):
        return bool(self._crt.blend)

    @blend.setter
    def blend(self, v):
        self._crt.blend = int(bool(v))

    @property
    def v_fac(self):
        return self._crt.v_fac

    @v_fac.setter
    def v_fac(self, v):
        self._crt.v_fac = int(v)

    def resize(self, w, h, fmt=None, out=None):
        """Resize the output image. Reallocates the output buffer."""
        if fmt is None:
            fmt = self._out_format
        bpp = self._lib.crt_bpp4fmt(fmt)
        if bpp == 0:
            raise ValueError(f"Invalid pixel format: {fmt}")

        self._out_buf = self._ffi.new("unsigned char[]", w * h * bpp)
        self._ffi.buffer(self._out_buf)[:] = b"\x00" * (w * h * bpp)
        self._lib.crt_resize(self._crt, w, h, fmt, self._out_buf)
        self._out_w = w
        self._out_h = h
        self._out_format = fmt
        self._bpp = bpp

    @property
    def signal_hres(self):
        """Horizontal resolution of the analog signal in samples."""
        return _SIGNAL_DIMS[self._system][0]

    @property
    def signal_vres(self):
        """Vertical resolution of the analog signal in lines."""
        return _SIGNAL_DIMS[self._system][1]

    def reset(self):
        """Reset CRT settings to defaults."""
        self._lib.crt_reset(self._crt)

    def _make_ntsc_settings(
        self,
        image,
        hue=0,
        raw=False,
        as_color=True,
        in_format=PIX_FORMAT_BGRA,
        dot_crawl_offset=0,
        do_aberration=False,
        field=0,
        frame=0,
        xoffset=0,
        yoffset=0,
    ):
        """Create and populate an NTSC_SETTINGS struct from an image."""
        image = np.ascontiguousarray(image)
        sys = self._system

        ntsc = self._ffi.new("struct NTSC_SETTINGS *")
        self._ffi.buffer(ntsc)[:] = b"\x00" * self._ffi.sizeof("struct NTSC_SETTINGS")

        if sys in _PALETTE_SYSTEMS:
            if image.dtype != np.uint16:
                image = image.astype(np.uint16)
            data_ptr = self._ffi.cast(
                "const unsigned short *",
                self._ffi.from_buffer(image),
            )
            ntsc.data = data_ptr
        else:
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            data_ptr = self._ffi.cast(
                "const unsigned char *",
                self._ffi.from_buffer(image),
            )
            ntsc.data = data_ptr

        if image.ndim >= 2:
            h, w = image.shape[0], image.shape[1]
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        ntsc.w = w
        ntsc.h = h
        ntsc.hue = hue % 360
        ntsc.xoffset = xoffset
        ntsc.yoffset = yoffset

        if sys in _FORMAT_SYSTEMS:
            ntsc.format = in_format
        if sys in _RAW_SYSTEMS:
            ntsc.raw = int(raw)
        if sys in _AS_COLOR_SYSTEMS:
            ntsc.as_color = int(as_color)
        if sys in _DOT_CRAWL_SYSTEMS:
            ntsc.dot_crawl_offset = dot_crawl_offset
        if sys in _ABERRATION_SYSTEMS:
            ntsc.do_aberration = int(do_aberration)
        if sys in _FIELD_FRAME_SYSTEMS:
            ntsc.field = field
            ntsc.frame = frame

        # Keep references alive so the GC doesn't collect the image buffer
        return ntsc, image

    def modulate(
        self,
        image,
        hue=0,
        raw=False,
        as_color=True,
        in_format=PIX_FORMAT_BGRA,
        dot_crawl_offset=0,
        do_aberration=False,
        field=0,
        frame=0,
        xoffset=0,
        yoffset=0,
    ):
        """Encode an image into an analog NTSC signal.

        After calling this, read the signal with get_analog_signal().

        Parameters
        ----------
        image : numpy.ndarray
            Input image. For RGB systems: uint8 (H, W, 3) or (H, W, 4).
            For NES: uint16 (H, W).
        hue : int
            Artifact color hue offset (0-359).
        raw : bool
            Don't scale image to fit.
        as_color : bool
            True = color, False = monochrome.
        in_format : int
            Input pixel format.
        dot_crawl_offset : int
            Dot crawl phase.
        do_aberration : bool
            VHS aberration (ntscvhs only).
        field : int
            0 = even, 1 = odd.
        frame : int
            0 = even, 1 = odd.
        xoffset : int
            Horizontal offset in samples.
        yoffset : int
            Vertical offset in lines.
        """
        ntsc, _img_ref = self._make_ntsc_settings(
            image, hue, raw, as_color, in_format,
            dot_crawl_offset, do_aberration, field, frame,
            xoffset, yoffset,
        )
        self._lib.crt_modulate(self._crt, ntsc)

    def demodulate(self, noise=24):
        """Decode the analog NTSC signal to the output image buffer.

        Call modulate() first to encode a signal.

        Parameters
        ----------
        noise : int
            Amount of noise to add (0 = clean).

        Returns
        -------
        numpy.ndarray
            Output image as uint8 array (out_h, out_w, bpp).
        """
        self._lib.crt_demodulate(self._crt, noise)
        buf = self._ffi.buffer(self._out_buf, self._out_w * self._out_h * self._bpp)
        out = np.frombuffer(buf, dtype=np.uint8).copy()
        return out.reshape(self._out_h, self._out_w, self._bpp)

    def get_analog_signal(self):
        """Read the current analog NTSC signal buffer.

        Returns the raw signal after modulate() has been called.
        Values are signed chars roughly in the range -40 (sync) to +110 (white) IRE.

        Returns
        -------
        numpy.ndarray
            Signal as int8 array shaped (VRES, HRES).
        """
        hres, vres = _SIGNAL_DIMS[self._system]
        buf = self._ffi.buffer(self._crt.analog, hres * vres)
        signal = np.frombuffer(buf, dtype=np.int8).copy()
        return signal.reshape(vres, hres)

    def set_analog_signal(self, signal):
        """Write a modified analog signal back into the CRT buffer.

        Call this between modulate() and demodulate() to alter the
        intermediate NTSC waveform before it is decoded.

        Parameters
        ----------
        signal : numpy.ndarray
            int8 array shaped (VRES, HRES) matching the system dimensions.
        """
        hres, vres = _SIGNAL_DIMS[self._system]
        sig = np.asarray(signal, dtype=np.int8)
        if sig.shape != (vres, hres):
            raise ValueError(
                f"expected shape ({vres}, {hres}), got {sig.shape}"
            )
        buf = self._ffi.buffer(self._crt.analog, hres * vres)
        buf[:] = sig.tobytes()

    def process(
        self,
        image,
        noise=24,
        hue=0,
        num_frames=4,
        progressive=False,
        raw=False,
        as_color=True,
        in_format=PIX_FORMAT_BGRA,
        dot_crawl_offset=0,
        do_aberration=False,
        xoffset=0,
        yoffset=0,
    ):
        """Process an image through the CRT emulator.

        Parameters
        ----------
        image : numpy.ndarray
            Input image. For RGB systems: uint8 array (H, W, 3) or (H, W, 4).
            For NES: uint16 array (H, W) of 6/9-bit pixel values.
        noise : int
            Amount of noise to add to the signal (0 = none).
        hue : int
            Artifact/color hue offset (0-359).
        num_frames : int
            Number of frames to accumulate (default 4 for quality).
        progressive : bool
            If True, use progressive scan. If False, interlaced.
        raw : bool
            If True, don't scale image to fit. Use for artifact-color images.
        as_color : bool
            If True, full color. If False, monochrome.
        in_format : int
            Input pixel format (for RGB systems). Default: PIX_FORMAT_BGRA.
        dot_crawl_offset : int
            Dot crawl offset (system-dependent).
        do_aberration : bool
            VHS aberration effect (ntscvhs only).
        xoffset : int
            Horizontal offset in sample space.
        yoffset : int
            Vertical offset in lines.

        Returns
        -------
        numpy.ndarray
            Output image as uint8 array (out_h, out_w, bpp).
        """
        ntsc, _img_ref = self._make_ntsc_settings(
            image, hue, raw, as_color, in_format,
            dot_crawl_offset, do_aberration, 0, 0,
            xoffset, yoffset,
        )
        sys = self._system

        field = 0
        frame = 0

        for i in range(num_frames):
            if sys in _FIELD_FRAME_SYSTEMS:
                ntsc.field = field
                ntsc.frame = frame

            self._lib.crt_modulate(self._crt, ntsc)
            self._lib.crt_demodulate(self._crt, noise)

            if not progressive and sys in _FIELD_FRAME_SYSTEMS:
                ntsc.field ^= 1
                self._lib.crt_modulate(self._crt, ntsc)
                self._lib.crt_demodulate(self._crt, noise)
                if (i & 1) == 0:
                    frame ^= 1
                    if sys in _FIELD_FRAME_SYSTEMS:
                        ntsc.frame = frame

        buf = self._ffi.buffer(self._out_buf, self._out_w * self._out_h * self._bpp)
        out = np.frombuffer(buf, dtype=np.uint8).copy()
        return out.reshape(self._out_h, self._out_w, self._bpp)

    def __repr__(self):
        return (
            f"CRT(system='{self._system}', "
            f"out={self._out_w}x{self._out_h}, "
            f"format={self._out_format})"
        )
