# NTSC Signal — Q&A

## What is color burst?

The **color burst** is a short reference signal (~8-10 cycles of a 3.579545 MHz sine wave) placed right after the sync pulse on every scan line, during the "back porch" before the visible video starts.

Its purpose: NTSC encodes color as **phase shifts** relative to a reference oscillator. The TV needs to know "what does 0° look like?" for every line so it can decode hue correctly. The color burst IS that reference — it's a few cycles of the unshifted carrier that the TV's Phase-Locked Loop (PLL) locks onto.

```
One scan line in time:

 ──┐     ┌──╲╱╲╱╲╱──────── video data with chroma riding on top ──
   │     │  color     
   │sync │  burst     [  front porch  ][  active video region  ]
   └─────┘  (8-10     
            cycles)
```

In the signal values:
- The burst oscillates around the blanking level (0 IRE) with amplitude ~±20 IRE
- The TV's decoder compares the phase of the chroma signal in the video region against this burst to extract hue
- The amplitude of the chroma relative to the burst gives saturation

This is why guitar pedal distortion creates color artifacts — clipping the signal introduces harmonics and phase shifts in the 3.58 MHz region that the TV interprets as color changes that aren't really there.

---

## What is 0 degrees?

0° is the phase of the unshifted color burst reference. Hue is measured as a phase offset from this — e.g. 0° ≈ orange, 90° ≈ green, 180° ≈ cyan, 270° ≈ magenta in NTSC.

---

## Video data with chroma on top — it contains RGB values somehow?

Not RGB — NTSC encodes color as **YIQ**, and it's all multiplexed into a single amplitude waveform:

1. **Y (luma)** = brightness, encoded as the signal's base amplitude. This is just a weighted sum of RGB: Y = 0.30R + 0.59G + 0.11B

2. **I and Q (chroma)** = color difference signals, encoded as a **3.58 MHz sine wave** modulated in **phase and amplitude** on top of the luma:
   - **Phase** (angle) = hue. I and Q are two components at 90° to each other (like sin and cos), so any combination of I and Q maps to an angle = a specific hue.
   - **Amplitude** (how big the wave is) = saturation. No wave = gray, big wave = vivid color.

So a single sample in the signal is just one number (e.g. `47`), but it's the **sum** of:

```
signal[t] = Y(t) + I(t)·sin(2π·3.58MHz·t) + Q(t)·cos(2π·3.58MHz·t)
```

The TV separates them:
- **Low-pass filter** → extracts Y (brightness changes slowly)
- **Bandpass filter** around 3.58 MHz → extracts the chroma wave
- **Phase detection** (comparing chroma wave angle to the color burst reference) → extracts I and Q
- **Matrix math** → converts YIQ back to RGB for the electron guns

This is why NTSC is clever but fragile — luma and chroma share the same wire. Any distortion (like a guitar pedal) creates frequency content that bleeds between them, causing brightness changes to appear as color and vice versa.
