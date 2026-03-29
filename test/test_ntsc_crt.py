"""Tests for the NTSC-CRT Python bindings (no Blender required).

Run with Blender's Python:
    NTSC_CRT_PATH=/path/to/NTSC-CRT/python \
    /path/to/blender/5.0/python/bin/python3.11 -m pytest test/ -v

Or directly:
    NTSC_CRT_PATH=/path/to/NTSC-CRT/python \
    /path/to/blender/5.0/python/bin/python3.11 test/test_ntsc_crt.py
"""

import os
import sys
import unittest

# Setup paths from env
_NTSC_CRT_PATH = os.environ.get("NTSC_CRT_PATH")
if not _NTSC_CRT_PATH:
    _NTSC_CRT_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "python")
    _NTSC_CRT_PATH = os.path.abspath(_NTSC_CRT_PATH)

for _p in (_NTSC_CRT_PATH, os.path.join(_NTSC_CRT_PATH, "lib")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from ntsc_crt import CRT, PIX_FORMAT_BGRA, PIX_FORMAT_RGB, PIX_FORMAT_RGBA, SYSTEMS


def _make_test_image(h=240, w=320, channels=4):
    """Create a synthetic BGRA test image with color bars."""
    img = np.zeros((h, w, channels), dtype=np.uint8)
    bar_w = w // 4
    img[:, :bar_w, 0] = 255             # Blue
    img[:, bar_w:2*bar_w, 1] = 255      # Green
    img[:, 2*bar_w:3*bar_w, 2] = 255    # Red
    img[:, 3*bar_w:, :3] = 255          # White
    if channels == 4:
        img[:, :, 3] = 255              # Alpha
    return img


class TestCRTCreation(unittest.TestCase):
    """Test CRT object creation and basic properties."""

    def test_create_ntsc(self):
        crt = CRT("ntsc", out_w=640, out_h=480)
        self.assertEqual(crt.system, "ntsc")

    def test_create_ntscvhs(self):
        crt = CRT("ntscvhs", out_w=640, out_h=480)
        self.assertEqual(crt.system, "ntscvhs")

    def test_invalid_system_raises(self):
        with self.assertRaises(ValueError):
            CRT("invalid_system")

    def test_default_resolution(self):
        crt = CRT("ntsc")
        # Default is 640x480 per constructor
        out = crt.process(_make_test_image(), num_frames=1)
        self.assertEqual(out.shape, (480, 640, 4))

    def test_custom_resolution(self):
        crt = CRT("ntsc", out_w=320, out_h=240)
        out = crt.process(_make_test_image(), num_frames=1)
        self.assertEqual(out.shape, (240, 320, 4))


class TestCRTDefaults(unittest.TestCase):
    """Test that CRT defaults match crt_reset() in crt_core.c."""

    def setUp(self):
        self.crt = CRT("ntsc", out_w=320, out_h=240)

    def test_hue_default(self):
        self.assertEqual(self.crt.hue, 0)

    def test_brightness_default(self):
        self.assertEqual(self.crt.brightness, 0)

    def test_contrast_default(self):
        self.assertEqual(self.crt.contrast, 180)

    def test_saturation_default(self):
        self.assertEqual(self.crt.saturation, 10)

    def test_black_point_default(self):
        self.assertEqual(self.crt.black_point, 0)

    def test_white_point_default(self):
        self.assertEqual(self.crt.white_point, 100)


class TestCRTProperties(unittest.TestCase):
    """Test setting and getting all CRT monitor properties."""

    def setUp(self):
        self.crt = CRT("ntsc", out_w=320, out_h=240)

    def test_hue(self):
        self.crt.hue = 45
        self.assertEqual(self.crt.hue, 45)

    def test_hue_negative(self):
        self.crt.hue = -90
        self.assertEqual(self.crt.hue, -90)

    def test_brightness(self):
        self.crt.brightness = 20
        self.assertEqual(self.crt.brightness, 20)

    def test_contrast(self):
        self.crt.contrast = 200
        self.assertEqual(self.crt.contrast, 200)

    def test_saturation(self):
        self.crt.saturation = 50
        self.assertEqual(self.crt.saturation, 50)

    def test_black_point(self):
        self.crt.black_point = -10
        self.assertEqual(self.crt.black_point, -10)

    def test_white_point(self):
        self.crt.white_point = 120
        self.assertEqual(self.crt.white_point, 120)

    def test_scanlines(self):
        self.crt.scanlines = True
        self.assertTrue(self.crt.scanlines)
        self.crt.scanlines = False
        self.assertFalse(self.crt.scanlines)

    def test_blend(self):
        self.crt.blend = True
        self.assertTrue(self.crt.blend)
        self.crt.blend = False
        self.assertFalse(self.crt.blend)

    def test_v_fac(self):
        self.crt.v_fac = 10
        self.assertEqual(self.crt.v_fac, 10)

    def test_reset(self):
        self.crt.hue = 99
        self.crt.contrast = 50
        self.crt.reset()
        self.assertEqual(self.crt.hue, 0)
        self.assertEqual(self.crt.contrast, 180)


class TestCRTProcess(unittest.TestCase):
    """Test the process() pipeline with various parameters."""

    def setUp(self):
        self.crt = CRT("ntsc", out_w=320, out_h=240)
        self.img = _make_test_image()

    def test_basic_process(self):
        out = self.crt.process(self.img, num_frames=1)
        self.assertEqual(out.shape, (240, 320, 4))
        self.assertEqual(out.dtype, np.uint8)

    def test_process_produces_nonzero(self):
        self.crt.blend = True
        out = self.crt.process(self.img, num_frames=4)
        self.assertGreater(out.max(), 0, "Output should not be all black")

    def test_noise_zero(self):
        out = self.crt.process(self.img, noise=0, num_frames=2)
        self.assertEqual(out.shape, (240, 320, 4))

    def test_noise_high(self):
        out = self.crt.process(self.img, noise=200, num_frames=1)
        self.assertEqual(out.shape, (240, 320, 4))

    def test_progressive(self):
        out = self.crt.process(self.img, progressive=True, num_frames=2)
        self.assertEqual(out.shape, (240, 320, 4))

    def test_monochrome(self):
        out = self.crt.process(self.img, as_color=False, num_frames=2)
        self.assertEqual(out.shape, (240, 320, 4))

    def test_raw_mode(self):
        out = self.crt.process(self.img, raw=True, num_frames=1)
        self.assertEqual(out.shape, (240, 320, 4))

    def test_artifact_hue(self):
        out = self.crt.process(self.img, hue=180, num_frames=1)
        self.assertEqual(out.shape, (240, 320, 4))

    def test_offsets(self):
        out = self.crt.process(self.img, xoffset=10, yoffset=5, num_frames=1)
        self.assertEqual(out.shape, (240, 320, 4))

    def test_num_frames_range(self):
        for n in (1, 2, 4, 8):
            out = self.crt.process(self.img, num_frames=n)
            self.assertEqual(out.shape, (240, 320, 4))

    def test_3channel_input(self):
        img_rgb = _make_test_image(channels=3)
        out = self.crt.process(img_rgb, in_format=PIX_FORMAT_RGB, num_frames=1)
        self.assertEqual(out.shape, (240, 320, 4))


class TestCRTModulateDemodulate(unittest.TestCase):
    """Test the separate modulate/demodulate pipeline."""

    def setUp(self):
        self.crt = CRT("ntsc", out_w=320, out_h=240)
        self.img = _make_test_image()

    def test_modulate_demodulate(self):
        self.crt.modulate(self.img)
        out = self.crt.demodulate(noise=24)
        self.assertEqual(out.shape, (240, 320, 4))
        self.assertEqual(out.dtype, np.uint8)

    def test_analog_signal(self):
        self.crt.modulate(self.img)
        sig = self.crt.get_analog_signal()
        self.assertEqual(sig.dtype, np.int8)
        self.assertEqual(len(sig.shape), 2)
        # Should have signal data (not all zeros after modulate)
        self.assertGreater(np.abs(sig).max(), 0)

    def test_signal_dimensions(self):
        self.crt.modulate(self.img)
        sig = self.crt.get_analog_signal()
        self.assertEqual(sig.shape, (262, 910))  # NTSC: 910 HRES, 262 VRES


class TestNTSCVHS(unittest.TestCase):
    """Test VHS-specific features."""

    def setUp(self):
        self.crt = CRT("ntscvhs", out_w=320, out_h=240)
        self.img = _make_test_image()

    def test_basic_process(self):
        out = self.crt.process(self.img, num_frames=1)
        self.assertEqual(out.shape, (240, 320, 4))

    def test_aberration(self):
        out = self.crt.process(self.img, do_aberration=True, num_frames=2)
        self.assertEqual(out.shape, (240, 320, 4))

    def test_vhs_with_all_params(self):
        self.crt.hue = 10
        self.crt.contrast = 160
        self.crt.saturation = 15
        self.crt.scanlines = True
        self.crt.blend = True
        out = self.crt.process(
            self.img,
            noise=48,
            hue=0,
            num_frames=4,
            do_aberration=True,
        )
        self.assertEqual(out.shape, (240, 320, 4))
        self.assertGreater(out.max(), 0)


class TestCRTResize(unittest.TestCase):
    """Test output buffer resizing."""

    def test_resize(self):
        crt = CRT("ntsc", out_w=320, out_h=240)
        crt.resize(640, 480)
        out = crt.process(_make_test_image(), num_frames=1)
        self.assertEqual(out.shape, (480, 640, 4))

    def test_resize_small(self):
        crt = CRT("ntsc", out_w=640, out_h=480)
        crt.resize(160, 120)
        out = crt.process(_make_test_image(), num_frames=1)
        self.assertEqual(out.shape, (120, 160, 4))


class TestVideoProcessing(unittest.TestCase):
    """Test a simulated video processing loop (like the addon does)."""

    def test_multi_frame_sequence(self):
        """Simulate processing multiple video frames sequentially."""
        crt = CRT("ntsc", out_w=320, out_h=240)
        crt.scanlines = True
        crt.blend = True

        for i in range(5):
            img = _make_test_image()
            # Shift colors slightly per frame
            img[:, :, 2] = np.clip(img[:, :, 2].astype(int) + i * 10, 0, 255).astype(np.uint8)
            out = crt.process(img, noise=24, num_frames=4)
            self.assertEqual(out.shape, (240, 320, 4))
            self.assertEqual(out.dtype, np.uint8)

    def test_opencv_roundtrip(self):
        """Test BGR->BGRA->CRT->BGRA->BGR like the addon does."""
        import cv2

        # Simulate a BGR frame from cv2.VideoCapture
        frame_bgr = np.zeros((240, 320, 3), dtype=np.uint8)
        frame_bgr[:, :, 2] = 200  # Red

        frame_bgra = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)

        crt = CRT("ntsc", out_w=320, out_h=240)
        crt.blend = True
        output = crt.process(frame_bgra, noise=24, num_frames=4)

        frame_out = cv2.cvtColor(output, cv2.COLOR_BGRA2BGR)
        self.assertEqual(frame_out.shape, (240, 320, 3))
        self.assertEqual(frame_out.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
