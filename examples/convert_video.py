#!/usr/bin/env python3
"""
NTSC-CRT Video Converter Example
=================================

Converts a video file through the NTSC-CRT filter to produce a retro
CRT television effect. Demonstrates all major features of the Python bindings.

Requirements:
    pip install cffi numpy opencv-python-headless

Usage:
    python convert_video.py                          # uses defaults
    python convert_video.py -i input.mp4 -o out.mp4  # custom input/output
    python convert_video.py --system ntscvhs          # VHS look
    python convert_video.py --help                    # all options
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

# Add the python directory to path so we can import ntsc_crt
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from ntsc_crt import CRT, PIX_FORMAT_BGRA, SYSTEMS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a video through the NTSC-CRT filter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard NTSC with default settings
  python convert_video.py

  # VHS look with extra noise
  python convert_video.py --system ntscvhs --noise 48 --aberration

  # SNES style, progressive scan, custom resolution
  python convert_video.py --system snes --progressive --out-width 640 --out-height 480

  # Monochrome with high contrast
  python convert_video.py --monochrome --contrast 220 --noise 32

Available systems: ntsc, nes, pv1k, snes, template, ntscvhs, nesrgb
        """,
    )

    # I/O
    parser.add_argument(
        "-i", "--input",
        default=os.path.join(SCRIPT_DIR, "00029.mp4"),
        help="Input video file (default: examples/00029.mp4)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output video file (default: <input>_ntsc.<ext>)",
    )

    # System selection
    parser.add_argument(
        "--system", "-s",
        choices=SYSTEMS,
        default="ntsc",
        help="CRT system variant (default: ntsc)",
    )

    # Output resolution
    parser.add_argument("--out-width", type=int, default=None, help="Output width (default: same as input)")
    parser.add_argument("--out-height", type=int, default=None, help="Output height (default: same as input)")

    # Signal settings
    parser.add_argument("--noise", type=int, default=24, help="Signal noise amount, 0=clean (default: 24)")
    parser.add_argument("--hue", type=int, default=0, help="Artifact color hue 0-359 (default: 0)")
    parser.add_argument("--num-frames", type=int, default=4, help="Frames to accumulate per output frame (default: 4)")

    # CRT monitor settings
    parser.add_argument("--brightness", type=int, default=None, help="Brightness adjustment")
    parser.add_argument("--contrast", type=int, default=None, help="Contrast level")
    parser.add_argument("--saturation", type=int, default=None, help="Color saturation")
    parser.add_argument("--black-point", type=int, default=None, help="Black point level")
    parser.add_argument("--white-point", type=int, default=None, help="White point level")

    # Feature toggles
    parser.add_argument("--scanlines", action="store_true", default=True, help="Enable scanline gaps (default: on)")
    parser.add_argument("--no-scanlines", action="store_true", help="Disable scanline gaps")
    parser.add_argument("--blend", action="store_true", default=True, help="Blend fields (default: on)")
    parser.add_argument("--no-blend", action="store_true", help="Disable field blending")
    parser.add_argument("--progressive", action="store_true", help="Progressive scan (default: interlaced)")
    parser.add_argument("--monochrome", action="store_true", help="Monochrome output")
    parser.add_argument("--raw", action="store_true", help="Raw mode (don't scale input to fit)")
    parser.add_argument("--aberration", action="store_true", help="VHS aberration effect (ntscvhs only)")

    # Processing
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process (default: all)")

    return parser.parse_args()


def make_output_path(input_path, system):
    """Generate default output filename."""
    base, ext = os.path.splitext(input_path)
    return f"{base}_{system}{ext}"


def print_settings(args, cap):
    """Print a summary of all settings."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps if fps > 0 else 0

    print("=" * 60)
    print("  NTSC-CRT Video Converter")
    print("=" * 60)
    print(f"  Input:       {args.input}")
    print(f"  Output:      {args.output}")
    print(f"  Resolution:  {in_w}x{in_h} -> {args.out_width}x{args.out_height}")
    print(f"  FPS:         {fps:.2f}")
    print(f"  Frames:      {total} ({duration:.1f}s)")
    if args.max_frames:
        print(f"  Processing:  first {args.max_frames} frames")
    print("-" * 60)
    print(f"  System:      {args.system}")
    print(f"  Noise:       {args.noise}")
    print(f"  Hue:         {args.hue}")
    print(f"  Accumulate:  {args.num_frames} frames")
    print(f"  Scanlines:   {'on' if not args.no_scanlines else 'off'}")
    print(f"  Blend:       {'on' if not args.no_blend else 'off'}")
    print(f"  Progressive: {'yes' if args.progressive else 'no (interlaced)'}")
    print(f"  Color:       {'monochrome' if args.monochrome else 'full color'}")
    if args.system == "ntscvhs":
        print(f"  Aberration:  {'on' if args.aberration else 'off'}")
    print("=" * 60)


def convert_video(args):
    # Open input video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: cannot open '{args.input}'", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Default output resolution to match input
    if args.out_width is None:
        args.out_width = in_w
    if args.out_height is None:
        args.out_height = in_h

    # Default output path
    if args.output is None:
        args.output = make_output_path(args.input, args.system)

    print_settings(args, cap)

    # --- Initialize CRT ---
    crt = CRT(args.system, out_w=args.out_width, out_h=args.out_height, out_format=PIX_FORMAT_BGRA)

    # Apply monitor settings
    crt.scanlines = not args.no_scanlines
    crt.blend = not args.no_blend

    if args.brightness is not None:
        crt.brightness = args.brightness
    if args.contrast is not None:
        crt.contrast = args.contrast
    if args.saturation is not None:
        crt.saturation = args.saturation
    if args.black_point is not None:
        crt.black_point = args.black_point
    if args.white_point is not None:
        crt.white_point = args.white_point

    # --- Initialize video writer ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (args.out_width, args.out_height))
    if not writer.isOpened():
        print(f"Error: cannot create output '{args.output}'", file=sys.stderr)
        cap.release()
        return 1

    # --- Process frames ---
    frames_to_process = total_frames
    if args.max_frames:
        frames_to_process = min(args.max_frames, total_frames)

    t_start = time.time()
    frame_idx = 0

    try:
        while frame_idx < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            # OpenCV reads BGR, convert to BGRA for the CRT library
            frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

            # Process through CRT
            output = crt.process(
                frame_bgra,
                noise=args.noise,
                hue=args.hue,
                num_frames=args.num_frames,
                progressive=args.progressive,
                raw=args.raw,
                as_color=not args.monochrome,
                in_format=PIX_FORMAT_BGRA,
                do_aberration=args.aberration,
            )

            # Convert BGRA output back to BGR for video writing
            frame_out = cv2.cvtColor(output, cv2.COLOR_BGRA2BGR)
            writer.write(frame_out)

            frame_idx += 1

            # Progress reporting
            if frame_idx % 10 == 0 or frame_idx == frames_to_process:
                elapsed = time.time() - t_start
                fps_actual = frame_idx / elapsed if elapsed > 0 else 0
                pct = frame_idx / frames_to_process * 100
                eta = (frames_to_process - frame_idx) / fps_actual if fps_actual > 0 else 0
                print(
                    f"\r  [{pct:5.1f}%] Frame {frame_idx}/{frames_to_process}"
                    f"  |  {fps_actual:.1f} fps  |  ETA: {eta:.0f}s",
                    end="",
                    flush=True,
                )

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    # Cleanup
    cap.release()
    writer.release()

    elapsed = time.time() - t_start
    print(f"\n\nDone! Processed {frame_idx} frames in {elapsed:.1f}s")
    print(f"Output saved to: {args.output}")
    return 0


def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: input file '{args.input}' not found", file=sys.stderr)
        return 1

    return convert_video(args)


if __name__ == "__main__":
    sys.exit(main())
