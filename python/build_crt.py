"""
Build script for NTSC-CRT Python bindings using cffi.

Usage:
    cd python
    pip install cffi
    python build_crt.py

This builds a separate cffi extension for each CRT system variant.
"""
import os
from cffi import FFI

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Common CRT struct + function declarations.
# CRT_INPUT_SIZE, CRT_CC_VPER, CRT_CC_SAMPLES are substituted per system.
CRT_STRUCT_TEMPLATE = """
struct CRT {{
    signed char analog[{input_size}];
    signed char inp[{input_size}];
    int outw, outh;
    int out_format;
    unsigned char *out;
    int hue, brightness, contrast, saturation;
    int black_point, white_point;
    int scanlines;
    int blend;
    unsigned int v_fac;
    int ccf[{cc_vper}][{cc_samples}];
    int hsync, vsync;
    int rn;
}};
"""

FUNCTIONS = """
void crt_init(struct CRT *v, int w, int h, int f, unsigned char *out);
void crt_resize(struct CRT *v, int w, int h, int f, unsigned char *out);
void crt_reset(struct CRT *v);
void crt_modulate(struct CRT *v, struct NTSC_SETTINGS *s);
void crt_demodulate(struct CRT *v, int noise);
int crt_bpp4fmt(int format);
"""

# Per-system configuration
SYSTEMS = {
    "ntsc": {
        "id": 0,
        "sources": ["crt_ntsc.c"],
        "input_size": 910 * 262,   # HRES=2275*4/10=910, VRES=262
        "cc_vper": 1,
        "cc_samples": 4,
        "settings_cdef": """
struct NTSC_SETTINGS {
    const unsigned char *data;
    int format;
    int w, h;
    int raw;
    int as_color;
    int field;
    int frame;
    int hue;
    int xoffset;
    int yoffset;
    int iirs_initialized;
};""",
    },
    "nes": {
        "id": 1,
        "sources": ["crt_nes.c"],
        "input_size": 909 * 262,   # HRES=2273*4/10=909
        "cc_vper": 3,
        "cc_samples": 4,
        "settings_cdef": """
struct NTSC_SETTINGS {
    const unsigned short *data;
    int w, h;
    unsigned int border_color;
    int dot_crawl_offset;
    int hue;
    int xoffset;
    int yoffset;
    int field_initialized;
};""",
    },
    "pv1k": {
        "id": 2,
        "sources": ["crt_pv1k.c"],
        "input_size": 1920 * 262,  # HRES=2304*5/6=1920
        "cc_vper": 5,
        "cc_samples": 5,
        "settings_cdef": """
struct NTSC_SETTINGS {
    const unsigned char *data;
    int format;
    int w, h;
    int raw;
    int as_color;
    int field;
    int frame;
    int hue;
    int xoffset;
    int yoffset;
    int dot_crawl_offset;
    int iirs_initialized;
};""",
    },
    "snes": {
        "id": 3,
        "sources": ["crt_snes.c"],
        "input_size": 909 * 262,
        "cc_vper": 3,
        "cc_samples": 4,
        "settings_cdef": """
struct NTSC_SETTINGS {
    const unsigned char *data;
    int format;
    int w, h;
    int raw;
    int as_color;
    int field;
    int frame;
    int hue;
    int xoffset;
    int yoffset;
    int dot_crawl_offset;
    int iirs_initialized;
};""",
    },
    "template": {
        "id": 4,
        "sources": ["crt_template.c"],
        "input_size": 910 * 262,
        "cc_vper": 2,
        "cc_samples": 4,
        "settings_cdef": """
struct NTSC_SETTINGS {
    const unsigned char *data;
    int format;
    int w, h;
    int raw;
    int as_color;
    int field;
    int frame;
    int hue;
    int xoffset;
    int yoffset;
    int dot_crawl_offset;
    int iirs_initialized;
};""",
    },
    "ntscvhs": {
        "id": 5,
        "sources": ["crt_ntscvhs.c"],
        "input_size": 910 * 262,
        "cc_vper": 1,
        "cc_samples": 4,
        "settings_cdef": """
struct NTSC_SETTINGS {
    const unsigned char *data;
    int format;
    int w, h;
    int raw;
    int as_color;
    int field;
    int frame;
    int hue;
    int xoffset;
    int yoffset;
    int do_aberration;
    int iirs_initialized;
};""",
    },
    "nesrgb": {
        "id": 6,
        "sources": ["crt_nesrgb.c"],
        "input_size": 909 * 262,
        "cc_vper": 3,
        "cc_samples": 4,
        "settings_cdef": """
struct NTSC_SETTINGS {
    const unsigned char *data;
    int format;
    int w, h;
    int dot_crawl_offset;
    int hue;
    int xoffset;
    int yoffset;
    int field_initialized;
};""",
    },
}


def build_system(name, cfg):
    ffi = FFI()

    # Struct and function declarations for cffi
    crt_struct = CRT_STRUCT_TEMPLATE.format(
        input_size=cfg["input_size"],
        cc_vper=cfg["cc_vper"],
        cc_samples=cfg["cc_samples"],
    )
    cdef = crt_struct + cfg["settings_cdef"] + "\n" + FUNCTIONS
    ffi.cdef(cdef)

    # C source that cffi compiles — just includes the real headers
    c_source = f'#define CRT_SYSTEM {cfg["id"]}\n#include "crt_core.h"\n'

    sources = [os.path.join(SRC_DIR, "crt_core.c")]
    for src in cfg["sources"]:
        sources.append(os.path.join(SRC_DIR, src))

    module_name = f"_crt_{name}"
    ffi.set_source(
        module_name,
        c_source,
        sources=sources,
        include_dirs=[SRC_DIR],
        define_macros=[("CRT_SYSTEM", str(cfg["id"]))],
    )

    lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
    os.makedirs(lib_dir, exist_ok=True)
    print(f"Building {module_name}...")
    ffi.compile(tmpdir=lib_dir, verbose=True)
    print(f"  -> {module_name} built successfully\n")


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for name, cfg in SYSTEMS.items():
        build_system(name, cfg)
    print("All systems built successfully!")


if __name__ == "__main__":
    main()
