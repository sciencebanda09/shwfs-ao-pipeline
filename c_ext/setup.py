import sys
from setuptools import setup, Extension

# Windows: math funcs in MSVC runtime, no separate m.lib needed.
# Linux/macOS: link -lm explicitly.
_libs = [] if sys.platform == "win32" else ["m"]

# MSVC uses /O2 not -O3; GCC/Clang use -O3.
_compile_args = ["/O2"] if sys.platform == "win32" else ["-O3"]

ext = Extension(
    "centroid_cog",
    sources=["centroid_cog.c"],
    extra_compile_args=_compile_args,
    libraries=_libs,
)

setup(
    name="centroid_cog",
    version="1.0",
    description="Fast C CoG centroiding for SH-WFS",
    ext_modules=[ext],
)
