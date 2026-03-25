# NTSC-CRT Python Bindings — Documentation

Python bindings for the [NTSC-CRT](https://github.com/LMP88959/NTSC-CRT) library
by EMMIR, providing integer-only NTSC video signal encoding/decoding emulation.

These bindings use [cffi](https://cffi.readthedocs.io/) to wrap the C library,
exposing all 7 CRT system variants as separate compiled extensions. No file I/O
from the C library is used — images are passed in and out as numpy arrays.

---

## Table of Contents

1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [API Reference](#api-reference)
   - [Module-Level Constants](#module-level-constants)
   - [CRT Class](#crt-class)
     - [Constructor](#constructor)
     - [Properties](#properties)
     - [Methods](#methods)
5. [System Variants](#system-variants)
6. [Pixel Formats](#pixel-formats)
7. [Processing Pipeline](#processing-pipeline)
8. [Examples](#examples)
   - [Single Image Processing](#single-image-processing)
   - [Video Conversion](#video-conversion)
   - [VHS Effect](#vhs-effect)
   - [Monochrome TV](#monochrome-tv)
   - [Different Systems Side by Side](#different-systems-side-by-side)
   - [NES Pixel Data](#nes-pixel-data)
   - [Inspecting the Analog Signal](#inspecting-the-analog-signal)
   - [Manual Modulate / Demodulate Loop](#manual-modulate--demodulate-loop)
9. [Parameter Tuning Guide](#parameter-tuning-guide)
10. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.8+
- A C compiler (gcc, clang, or MSVC)
- The NTSC-CRT source code (this repository)

### Steps

```bash
# 1. Create and activate a virtual environment
python3 -m venv env
source env/bin/activate       # Linux/macOS
# env\Scripts\activate        # Windows

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Build all system extensions
cd python
python build_crt.py
```

On success, you'll see each system built:

```
Building _crt_ntsc...
  -> _crt_ntsc built successfully

Building _crt_nes...
  -> _crt_nes built successfully

...

All systems built successfully!
```

### Optional: video processing
```bash
pip install opencv-python-headless
```

---

## Project Structure

```
python/
├── build_crt.py      # Build script — compiles all cffi extensions
├── ntsc_crt.py       # Python wrapper module (import this)
├── README.md         # This documentation
└── lib/              # Compiled shared libraries (auto-generated)
    ├── _crt_ntsc.cpython-3xx-*.so
    ├── _crt_nes.cpython-3xx-*.so
    ├── _crt_pv1k.cpython-3xx-*.so
    ├── _crt_snes.cpython-3xx-*.so
    ├── _crt_template.cpython-3xx-*.so
    ├── _crt_ntscvhs.cpython-3xx-*.so
    └── _crt_nesrgb.cpython-3xx-*.so
```

---

## Quick Start

```python
import numpy as np
from ntsc_crt import CRT, PIX_FORMAT_BGRA

# Create a CRT processor
crt = CRT("ntsc", out_w=640, out_h=480)

# Load an image (H, W, 4) uint8 BGRA array — use any image library
# For example with OpenCV:
#   import cv2
#   image = cv2.cvtColor(cv2.imread("photo.png"), cv2.COLOR_BGR2BGRA)

# Or create a test pattern
image = np.zeros((240, 320, 4), dtype=np.uint8)
image[:, :, 0] = 200  # Blue channel (BGRA)
image[:, :, 3] = 255  # Alpha

# Process
output = crt.process(image, noise=24, num_frames=4)
# output.shape == (480, 640, 4), dtype == uint8
```

---

## API Reference

### Module-Level Constants

#### Pixel Formats

| Constant           | Value | Layout                      | Bytes/pixel |
|--------------------|-------|-----------------------------|-------------|
| `PIX_FORMAT_RGB`   | 0     | `[R, G, B, R, G, B, ...]`  | 3           |
| `PIX_FORMAT_BGR`   | 1     | `[B, G, R, B, G, R, ...]`  | 3           |
| `PIX_FORMAT_ARGB`  | 2     | `[A, R, G, B, A, R, G, B, ...]` | 4      |
| `PIX_FORMAT_RGBA`  | 3     | `[R, G, B, A, R, G, B, A, ...]` | 4      |
| `PIX_FORMAT_ABGR`  | 4     | `[A, B, G, R, A, B, G, R, ...]` | 4      |
| `PIX_FORMAT_BGRA`  | 5     | `[B, G, R, A, B, G, R, A, ...]` | 4      |

> **Note:** The library does not use the alpha channel. It is passed through unchanged.

#### System Names

```python
SYSTEMS = ("ntsc", "nes", "pv1k", "snes", "template", "ntscvhs", "nesrgb")
```

---

### CRT Class

The main interface. Each instance wraps a specific CRT system variant with its
own internal signal state, output buffer, and monitor settings.

#### Constructor

```python
CRT(system, out_w=640, out_h=480, out_format=PIX_FORMAT_BGRA)
```

| Parameter    | Type  | Default          | Description                              |
|-------------|-------|------------------|------------------------------------------|
| `system`    | `str` | *(required)*     | System variant name (see table below)    |
| `out_w`     | `int` | `640`            | Output image width in pixels             |
| `out_h`     | `int` | `480`            | Output image height in pixels            |
| `out_format`| `int` | `PIX_FORMAT_BGRA`| Output pixel format constant             |

**Raises:** `ValueError` if `system` is unknown or `out_format` is invalid.

```python
crt = CRT("ntsc", out_w=832, out_h=624)
```

---

#### Properties

All properties are read/write and control the simulated CRT monitor settings.
They can be changed at any time between `process()` calls.

| Property       | Type   | Range / Values          | Description                                        |
|----------------|--------|-------------------------|----------------------------------------------------|
| `system`       | `str`  | *(read-only)*           | The system variant name                            |
| `signal_hres`  | `int`  | *(read-only)*           | Analog signal horizontal resolution (samples/line) |
| `signal_vres`  | `int`  | *(read-only)*           | Analog signal vertical resolution (lines)          |
| `hue`          | `int`  | `-360` to `360`         | Color hue rotation in degrees                      |
| `brightness`   | `int`  | `-∞` to `+∞`            | Brightness offset                                  |
| `contrast`     | `int`  | `0` to `+∞`             | Contrast multiplier                                |
| `saturation`   | `int`  | `0` to `+∞`             | Color saturation multiplier                        |
| `black_point`  | `int`  | any                     | Black level adjustment                             |
| `white_point`  | `int`  | any                     | White level adjustment                             |
| `scanlines`    | `bool` | `True` / `False`        | Add visible gaps between scan lines                |
| `blend`        | `bool` | `True` / `False`        | Blend new field onto previous image                |
| `v_fac`        | `int`  | `0` to `+∞`             | Vertical stretch factor                            |

##### Signal dimensions by system

| System     | `signal_hres` | `signal_vres` | Total samples |
|------------|---------------|---------------|---------------|
| `ntsc`     | 910           | 262           | 238,420       |
| `nes`      | 909           | 262           | 238,158       |
| `pv1k`     | 1920          | 262           | 502,840       |
| `snes`     | 909           | 262           | 238,158       |
| `template` | 910           | 262           | 238,420       |
| `ntscvhs`  | 910           | 262           | 238,420       |
| `nesrgb`   | 909           | 262           | 238,158       |

```python
crt = CRT("ntsc", 640, 480)
crt.hue = 15
crt.brightness = 0
crt.contrast = 180
crt.saturation = 10
crt.scanlines = True
crt.blend = True
```

---

#### Methods

##### `process()`

```python
crt.process(
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
)
```

Encodes an image into an analog NTSC signal and decodes it back, simulating
CRT television output.

**Parameters:**

| Parameter          | Type             | Default          | Description                                              |
|--------------------|------------------|------------------|----------------------------------------------------------|
| `image`            | `numpy.ndarray`  | *(required)*     | Input image (see below for format per system)            |
| `noise`            | `int`            | `24`             | Signal noise amount. `0` = clean signal                  |
| `hue`              | `int`            | `0`              | Artifact color hue offset (`0`-`359` degrees)            |
| `num_frames`       | `int`            | `4`              | Frames to accumulate for quality / interlacing           |
| `progressive`      | `bool`           | `False`          | `True` = progressive scan, `False` = interlaced          |
| `raw`              | `bool`           | `False`          | `True` = don't scale input to fit the CRT                |
| `as_color`         | `bool`           | `True`           | `True` = color, `False` = monochrome                     |
| `in_format`        | `int`            | `PIX_FORMAT_BGRA`| Input pixel format (RGB systems only)                    |
| `dot_crawl_offset` | `int`            | `0`              | Dot crawl phase offset (system-dependent)                |
| `do_aberration`    | `bool`           | `False`          | VHS signal aberration at bottom (ntscvhs only)           |
| `xoffset`          | `int`            | `0`              | Horizontal offset in sample space                        |
| `yoffset`          | `int`            | `0`              | Vertical offset in scan lines                            |

**Input image format (by system):**

| Systems                                           | `image` dtype   | Shape              |
|---------------------------------------------------|-----------------|--------------------|
| `ntsc`, `pv1k`, `snes`, `template`, `ntscvhs`, `nesrgb` | `uint8`   | `(H, W, 3)` or `(H, W, 4)` |
| `nes`                                             | `uint16`        | `(H, W)`           |

**Returns:** `numpy.ndarray` — output image as `uint8` array with shape
`(out_h, out_w, bytes_per_pixel)`. For 4-byte formats like BGRA the shape is
`(out_h, out_w, 4)`.

**Important notes:**
- The `CRT` object is **stateful**. When `blend=True`, successive `process()`
  calls blend onto the existing output buffer, which is the intended behavior
  for video frame sequences.
- For single-image processing with `blend=True`, use `num_frames >= 4` to
  accumulate enough data for a clean image.
- The `num_frames` parameter controls how many modulate/demodulate cycles run.
  In interlaced mode, each iteration processes both even and odd fields, so
  `num_frames=4` actually runs 8 modulate/demodulate passes.

---

##### `resize()`

```python
crt.resize(w, h, fmt=None)
```

Change the output resolution. Reallocates the internal output buffer.

| Parameter | Type       | Default          | Description              |
|-----------|------------|------------------|--------------------------|
| `w`       | `int`      | *(required)*     | New output width         |
| `h`       | `int`      | *(required)*     | New output height        |
| `fmt`     | `int\|None`| `None`           | New pixel format, or keep current |

```python
crt.resize(1920, 1080)
```

---

##### `modulate()`

```python
crt.modulate(
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
)
```

Encode an image into the internal analog NTSC signal buffer. This is the
first half of the processing pipeline. After calling this, you can inspect
the raw signal with `get_analog_signal()` or decode it with `demodulate()`.

**Parameters:**

| Parameter          | Type             | Default          | Description                                     |
|--------------------|------------------|------------------|-------------------------------------------------|
| `image`            | `numpy.ndarray`  | *(required)*     | Input image (same format rules as `process()`)  |
| `hue`              | `int`            | `0`              | Artifact color hue offset (0-359)               |
| `raw`              | `bool`           | `False`          | Don't scale input to fit                        |
| `as_color`         | `bool`           | `True`           | Color / monochrome encoding                     |
| `in_format`        | `int`            | `PIX_FORMAT_BGRA`| Input pixel format                              |
| `dot_crawl_offset` | `int`            | `0`              | Dot crawl phase (system-dependent)              |
| `do_aberration`    | `bool`           | `False`          | VHS aberration (ntscvhs only)                   |
| `field`            | `int`            | `0`              | `0` = even field, `1` = odd field               |
| `frame`            | `int`            | `0`              | `0` = even frame, `1` = odd frame               |
| `xoffset`          | `int`            | `0`              | Horizontal offset in samples                    |
| `yoffset`          | `int`            | `0`              | Vertical offset in lines                        |

**Returns:** None

---

##### `demodulate()`

```python
output = crt.demodulate(noise=24)
```

Decode the analog NTSC signal (written by a prior `modulate()` call) back
into an RGB output image.

| Parameter | Type  | Default | Description                      |
|-----------|-------|---------|----------------------------------|
| `noise`   | `int` | `24`    | Signal noise amount (0 = clean)  |

**Returns:** `numpy.ndarray` — output image as `uint8` array `(out_h, out_w, bpp)`.

---

##### `get_analog_signal()`

```python
signal = crt.get_analog_signal()
```

Read the current analog NTSC signal buffer. Call after `modulate()` to
inspect the encoded signal before decoding.

Values are signed 8-bit integers roughly in the range:
- **-40** (sync tip)
- **0** (blanking level)
- **7** (black level, for most systems)
- **100** (white level)
- **110** (peak, some systems)

**Returns:** `numpy.ndarray` — `int8` array shaped `(signal_vres, signal_hres)`.

Each row is one scan line of the composite signal. The signal contains:
- Front porch, sync pulse, breezeway, color burst, back porch (horizontal blanking)
- Active video region with encoded luma + chroma

---

##### `reset()`

```python
crt.reset()
```

Reset all CRT monitor settings (hue, brightness, contrast, etc.) back to their
library defaults. Does **not** change the output resolution or format.

---

## System Variants

Each system variant emulates the NTSC signal characteristics of a specific
hardware platform. They differ in timing, color encoding, resolution, and
available settings.

| System       | ID | Description                       | Input Type       | Unique Features                     |
|--------------|----|-----------------------------------|------------------|-------------------------------------|
| `ntsc`       | 0  | Standard NTSC television          | RGB image        | Baseline — field/frame interlacing  |
| `nes`        | 1  | Nintendo Entertainment System     | 6/9-bit pixels   | PPU palette input, border color, dot crawl |
| `pv1k`       | 2  | Casio PV-1000 console             | RGB image        | 5 samples/chroma, unique timing     |
| `snes`       | 3  | Super Nintendo                    | RGB image        | RGB-mode artifacts, dot crawl       |
| `template`   | 4  | Customizable template system      | RGB image        | For experimentation, dot crawl      |
| `ntscvhs`    | 5  | NTSC with VHS tape quality        | RGB image        | Reduced bandwidth, aberration effect, noise at bottom |
| `nesrgb`     | 6  | NES with RGB output artifacts     | RGB image        | NES timing with RGB input, dot crawl |

### Per-system `process()` parameter support

Not all parameters apply to every system. Unsupported parameters are silently
ignored.

| Parameter          | ntsc | nes | pv1k | snes | template | ntscvhs | nesrgb |
|--------------------|------|-----|------|------|----------|---------|--------|
| `in_format`        | ✓    |     | ✓    | ✓    | ✓        | ✓       | ✓      |
| `raw`              | ✓    |     | ✓    | ✓    | ✓        | ✓       |        |
| `as_color`         | ✓    |     | ✓    | ✓    | ✓        | ✓       |        |
| `progressive`      | ✓    |     | ✓    | ✓    | ✓        | ✓       |        |
| `dot_crawl_offset` |      | ✓   | ✓    | ✓    | ✓        |         | ✓      |
| `do_aberration`    |      |     |      |      |          | ✓       |        |

### Dot crawl offset ranges

| System   | Range  |
|----------|--------|
| `nes`    | 0–2    |
| `pv1k`   | 0–5    |
| `snes`   | 0–3    |
| `template`| 0–5   |
| `nesrgb` | 0–2    |

---

## Pixel Formats

The pixel format controls the byte layout of both the input image and the
output buffer. The input and output formats are set independently:

- **Input format:** Set via the `in_format` parameter of `process()`
- **Output format:** Set when constructing the `CRT` object (or via `resize()`)

**Common choices:**

| Use Case                         | Recommended Format  |
|----------------------------------|---------------------|
| OpenCV (`cv2.imread`)            | `PIX_FORMAT_BGRA`   |
| Pillow (PIL)                     | `PIX_FORMAT_RGBA`   |
| Raw RGB data                     | `PIX_FORMAT_RGB`    |
| Pygame surfaces                  | `PIX_FORMAT_RGBA` or `PIX_FORMAT_ARGB` |

When using OpenCV, images are natively BGR. Convert to BGRA with:
```python
frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
```

When using Pillow:
```python
image_rgba = np.array(Image.open("photo.png").convert("RGBA"))
```

---

## Processing Pipeline

Understanding the internal pipeline helps with parameter tuning:

```
Input Image (numpy array)
        │
        ▼
┌──────────────────┐
│  crt.modulate()  │  Encode: RGB pixels → analog NTSC signal
│                  │  (writes to internal analog[] buffer)
└────────┬─────────┘
         │
         ├──→ crt.get_analog_signal()   ← inspect raw signal here
         │
         ▼
┌──────────────────┐
│ crt.demodulate() │  Decode: analog signal → RGB output
│                  │  (reads analog[], writes to output buffer)
│                  │  Applies: bandlimiting, sync, noise, scanlines
└────────┬─────────┘
         │
         ▼
   Output Image (numpy array)
```

The `process()` method runs both steps together in a loop. You can also call
`modulate()` and `demodulate()` individually to inspect or manipulate the
intermediate analog signal.

In **interlaced mode** (default), each `num_frames` iteration runs two passes:
one for the even field and one for the odd field. With `blend=True`, both
fields merge onto the same output buffer, producing a complete interlaced frame.

In **progressive mode**, only one field is processed per iteration.

The `num_frames` parameter controls how many times this cycle repeats. More
frames = more temporal accumulation = smoother result, but slower processing.

---

## Examples

### Single Image Processing

```python
import cv2
import numpy as np
from ntsc_crt import CRT, PIX_FORMAT_BGRA

# Load image
img = cv2.imread("input.png")
img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
h, w = img_bgra.shape[:2]

# Create CRT at same resolution
crt = CRT("ntsc", out_w=w, out_h=h)
crt.scanlines = True
crt.blend = True

# Process
output = crt.process(img_bgra, noise=24, num_frames=4)

# Save
cv2.imwrite("output.png", cv2.cvtColor(output, cv2.COLOR_BGRA2BGR))
```

### Video Conversion

See `examples/convert_video.py` for a full-featured command-line video converter.

Minimal video loop:

```python
import cv2
from ntsc_crt import CRT, PIX_FORMAT_BGRA

cap = cv2.VideoCapture("input.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

crt = CRT("ntsc", out_w=w, out_h=h)
crt.blend = True
crt.scanlines = True

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter("output.mp4", fourcc, fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    output = crt.process(frame_bgra, noise=24, num_frames=4)
    writer.write(cv2.cvtColor(output, cv2.COLOR_BGRA2BGR))

cap.release()
writer.release()
```

### VHS Effect

```python
crt = CRT("ntscvhs", out_w=640, out_h=480)
crt.blend = True
crt.scanlines = True

output = crt.process(
    image,
    noise=48,            # Heavy noise for VHS feel
    num_frames=4,
    do_aberration=True,  # Signal distortion at bottom of frame
)
```

### Monochrome TV

```python
crt = CRT("ntsc", out_w=640, out_h=480)
crt.scanlines = True
crt.blend = True
crt.saturation = 0

output = crt.process(
    image,
    noise=32,
    as_color=False,    # Monochrome encoding
    num_frames=4,
)
```

### Different Systems Side by Side

```python
import numpy as np
from ntsc_crt import CRT, SYSTEMS

image = ...  # Your BGRA input image

results = {}
for system in ["ntsc", "snes", "ntscvhs", "pv1k", "template"]:
    crt = CRT(system, out_w=640, out_h=480)
    crt.blend = True
    crt.scanlines = True
    results[system] = crt.process(image, noise=24, num_frames=4)

# results is a dict of system_name -> output numpy array
```

### NES Pixel Data

The NES system expects 6-bit or 9-bit palette indices as `uint16`:

```python
import numpy as np
from ntsc_crt import CRT

# 256x240 NES screen of palette indices (0-63 for 6-bit)
nes_screen = np.zeros((240, 256), dtype=np.uint16)
nes_screen[100:140, 80:176] = 0x16  # red-ish color

crt = CRT("nes", out_w=640, out_h=480)
crt.blend = True

output = crt.process(
    nes_screen,
    noise=12,
    num_frames=4,
    dot_crawl_offset=0,
)
```

### Inspecting the Analog Signal

You can encode an image and read the raw NTSC waveform:

```python
import numpy as np
import matplotlib.pyplot as plt
from ntsc_crt import CRT, PIX_FORMAT_BGRA

crt = CRT("ntsc", out_w=640, out_h=480)
image = ...  # Your BGRA uint8 image

# Encode only
crt.modulate(image)

# Get the raw analog signal
signal = crt.get_analog_signal()  # (262, 910) int8

# Plot a single scan line (e.g. line 130, middle of screen)
plt.figure(figsize=(14, 4))
plt.plot(signal[130], linewidth=0.5)
plt.axhline(y=0, color='gray', linestyle='--', label='Blanking')
plt.axhline(y=-40, color='red', linestyle='--', label='Sync')
plt.axhline(y=100, color='green', linestyle='--', label='White')
plt.xlabel('Sample')
plt.ylabel('IRE (int8)')
plt.title('NTSC Composite Signal — Line 130')
plt.legend()
plt.tight_layout()
plt.savefig('signal_line.png')

# Visualize the full signal as an image
plt.figure(figsize=(12, 8))
plt.imshow(signal, aspect='auto', cmap='gray', vmin=-40, vmax=110)
plt.xlabel('Sample (horizontal)')
plt.ylabel('Line (vertical)')
plt.title('Full NTSC Analog Signal')
plt.colorbar(label='IRE')
plt.tight_layout()
plt.savefig('signal_full.png')

# Then decode
output = crt.demodulate(noise=24)
```

### Manual Modulate / Demodulate Loop

For full control over field/frame interlacing:

```python
from ntsc_crt import CRT, PIX_FORMAT_BGRA

crt = CRT("ntsc", out_w=640, out_h=480)
crt.blend = True
crt.scanlines = True

image = ...  # BGRA uint8 image

# Manually run 2 interlaced frames (4 fields)
for frame in range(2):
    for field in range(2):
        crt.modulate(image, field=field, frame=frame)
        output = crt.demodulate(noise=24)

# output now has the accumulated result
```

---

## Parameter Tuning Guide

### Noise

| Value   | Effect                                    |
|---------|-------------------------------------------|
| `0`     | Clean signal, no static                   |
| `12`    | Subtle noise, realistic for good reception|
| `24`    | Default — moderate noise                  |
| `48+`   | Heavy static, poor reception / VHS feel   |

### num_frames

| Value | Effect                                          |
|-------|-------------------------------------------------|
| `1`   | Single field — fastest, but may show field artifacts |
| `2`   | One full interlaced frame                        |
| `4`   | Default — two full frames, good quality          |
| `8+`  | Diminishing returns, smoother but much slower    |

> In interlaced mode, each iteration renders both even and odd fields.
> So `num_frames=4` actually runs 8 modulate/demodulate cycles.

### Contrast & Brightness

These are integer values with no fixed range. The library defaults are set by
`crt_init()`. Experiment starting from the defaults:

```python
crt = CRT("ntsc", 640, 480)
print(f"Default contrast:   {crt.contrast}")
print(f"Default brightness: {crt.brightness}")
print(f"Default saturation: {crt.saturation}")
```

Typical adjustments:
- **Brighter:** increase `brightness` by 5–20
- **More vivid:** increase `saturation` by 5–20
- **Washed out:** decrease `contrast`
- **Crushed blacks:** lower `black_point`

### Progressive vs Interlaced

| Mode          | Setting               | Best For                          |
|---------------|-----------------------|-----------------------------------|
| Interlaced    | `progressive=False`   | Realistic TV look, video content  |
| Progressive   | `progressive=True`    | Game consoles, static images      |

### Raw Mode

When `raw=True`, the input image is not scaled to fit the simulated CRT viewport.
This is needed for images that use **artifact colors** — specially crafted black
and white pixel patterns that produce color through the NTSC encoding process.

---

## Troubleshooting

### Build errors

**`ModuleNotFoundError: No module named 'setuptools'`**

cffi requires setuptools on Python 3.12+:
```bash
pip install setuptools
```

**`No module named '_crt_ntsc'`**

The extensions haven't been built, or the `lib/` directory isn't on the Python
path. Run:
```bash
cd python && python build_crt.py
```

Make sure you're importing from the right location. The `lib/` directory is
automatically added to `sys.path` when you import `ntsc_crt`.

### Runtime issues

**Output is all black**

- Make sure your input image is in the correct pixel format. If using OpenCV,
  convert to BGRA: `cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)`
- Check that `as_color=True` (or `False` if you want monochrome)
- Ensure the image dtype is `uint8` for RGB systems or `uint16` for NES

**Output looks wrong / colors are off**

- Verify `in_format` matches your actual image data layout
- Try adjusting `hue` in the `process()` call (0–359)
- The CRT `hue` property and the `process(hue=...)` parameter are different:
  the property controls the monitor, the parameter controls encoding artifacts

**Video output is too slow**

- Lower `num_frames` (try `2` instead of `4`)
- Reduce output resolution
- `progressive=True` runs half as many passes as interlaced mode

**Segmentation fault**

- Ensure the input numpy array is contiguous (`np.ascontiguousarray()`)
- Ensure the input image dimensions and dtype match expectations
- Don't reuse a `CRT` object across threads without synchronization
