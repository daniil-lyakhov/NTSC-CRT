#!/usr/bin/env python3
"""
Manual Modulate/Demodulate Example
===================================

This example does exactly what CRT.process() does internally, but using
the low-level modulate() and demodulate() methods with full comments
explaining every step.

This is useful for understanding the NTSC encoding/decoding pipeline,
and for cases where you want to inspect or manipulate the intermediate
analog signal between encoding and decoding.

Usage:
    python manual_process.py input.bmp output.bmp
"""

import os
import sys

import cv2
import numpy as np

# Add the python directory to path so we can import ntsc_crt
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from ntsc_crt import CRT, PIX_FORMAT_BGRA


def process_manual(image, out_w, out_h, noise=24, num_frames=4):
    """Process an image through the CRT, step by step.

    This function replicates exactly what crt.process() does internally,
    but broken into individual modulate/demodulate calls with commentary.
    """

    # =========================================================================
    # Step 1: Create the CRT emulator
    # =========================================================================
    # Each CRT instance holds:
    #   - An internal analog signal buffer (analog[]) sized for the system
    #   - An output image buffer that we provide (out_w * out_h * bpp bytes)
    #   - Monitor settings: hue, brightness, contrast, saturation, etc.
    crt = CRT("ntsc", out_w=out_w, out_h=out_h, out_format=PIX_FORMAT_BGRA)

    # Enable blend mode. When blend=True, each demodulate() call blends onto
    # the existing output buffer rather than replacing it. This is how real CRTs
    # work — the phosphor persistence causes successive fields to accumulate.
    crt.blend = True

    # Enable scanline gaps. Real CRT electron guns paint every other line per
    # field, leaving visible dark gaps between scan lines.
    crt.scanlines = True

    # =========================================================================
    # Step 2: Interlaced processing loop
    # =========================================================================
    # Real NTSC is interlaced: each "frame" is made of two "fields".
    #   - Even field (field=0): paints even-numbered scan lines
    #   - Odd field  (field=1): paints odd-numbered scan lines
    # Together they form one complete frame.
    #
    # The "frame" counter (0 or 1) controls the chroma subcarrier phase.
    # Toggling it every 2 fields produces the "dot crawl" pattern seen on
    # real NTSC displays — a subtle crawling artifact at sharp color edges.
    #
    # For a single static image, we run multiple iterations to simulate
    # what a real TV shows after a fraction of a second of continuous signal.
    # The C library's own crt_main.c uses 4 iterations as its default.

    field = 0  # Start with the even field
    frame = 0  # Start with even frame phase

    for i in range(num_frames):
        # -----------------------------------------------------------------
        # Pass A: Encode and decode the EVEN field
        # -----------------------------------------------------------------

        # MODULATE: Convert the RGB image into an analog NTSC signal.
        # This simulates the broadcast encoder — it writes a composite
        # waveform into the CRT's internal analog[] buffer. The waveform
        # contains sync pulses, color burst, and luma+chroma video data.
        crt.modulate(
            image,
            hue=0,           # Artifact color hue (0-359 degrees)
            as_color=True,   # Full color (False = monochrome)
            raw=False,        # Scale input to fit the CRT viewport
            in_format=PIX_FORMAT_BGRA,
            field=field,     # Which field we're encoding (even=0, odd=1)
            frame=frame,     # Frame phase for dot crawl
        )

        # At this point you can inspect AND modify the raw analog signal:
        #   signal = crt.get_analog_signal()  # (262, 910) int8 array
        # Each row is one scan line. Values range from -40 (sync) to ~110 (white).
        #
        # You can manipulate the signal before decoding, for example:
        #   signal[:, 100:200] = 0                                   # blank columns
        #   signal = np.clip(signal.astype(np.int16) + 10, -128, 127).astype(np.int8)  # boost brightness
        #   signal[::2] = signal[::2] // 2                           # dim even lines
        #
        # Then write it back into the CRT buffer:
        #   crt.set_analog_signal(signal)
        #
        # The next demodulate() call will decode your modified waveform.

        # DEMODULATE: Decode the analog signal back into RGB pixels.
        # This simulates the TV tuner — it reads the analog[] buffer,
        # applies bandpass filtering to separate luma (Y) and chroma (I/Q),
        # adds noise to simulate reception quality, detects sync pulses,
        # and writes decoded RGB pixels into the output buffer.
        # Because blend=True, this accumulates onto the previous output.
        output = crt.demodulate(noise=noise)

        # -----------------------------------------------------------------
        # Pass B: Encode and decode the ODD field
        # -----------------------------------------------------------------
        # In interlaced mode, we immediately do the other field.
        # This completes one full frame (even + odd = all lines filled).

        field ^= 1  # Toggle: 0 -> 1 (now doing the odd field)

        crt.modulate(
            image,
            hue=0,
            as_color=True,
            raw=False,
            in_format=PIX_FORMAT_BGRA,
            field=field,     # Now the odd field
            frame=frame,     # Same frame phase
        )

        output = crt.demodulate(noise=noise)

        # -----------------------------------------------------------------
        # Frame phase update for dot crawl
        # -----------------------------------------------------------------
        # Every 2 iterations (i.e. every 4 fields = 2 complete frames),
        # we toggle the frame counter. This shifts the chroma subcarrier
        # phase, which is what causes the "dot crawl" pattern on real NTSC.
        # The pattern repeats every 2 frames, so toggling here cycles it.
        if (i & 1) == 0:
            frame ^= 1

    # =========================================================================
    # Step 3: Return the accumulated result
    # =========================================================================
    # After num_frames iterations (each with 2 fields), the output buffer
    # contains the blended result of all passes. For num_frames=4, that's
    # 8 modulate+demodulate cycles — enough for a stable, clean image.
    return output


def main():
    if len(sys.argv) < 2:
        print("Usage: python manual_process.py <input_image> [output_image]")
        print("       python manual_process.py photo.bmp photo_ntsc.bmp")
        return 1

    input_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_ntsc{ext}"

    # Load input image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: cannot read '{input_path}'", file=sys.stderr)
        return 1

    h, w = img.shape[:2]
    img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    print(f"Input:  {input_path} ({w}x{h})")
    print(f"Output: {output_path}")

    # Process with manual modulate/demodulate (identical to crt.process())
    output = process_manual(img_bgra, out_w=w, out_h=h, noise=24, num_frames=4)

    # Save
    cv2.imwrite(output_path, cv2.cvtColor(output, cv2.COLOR_BGRA2BGR))
    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
