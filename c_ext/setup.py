from setuptools import setup, Extension

ext = Extension(
    "centroid_cog",
    sources=["centroid_cog.c"],
    extra_compile_args=["-O3"],
    libraries=["m"],
)

setup(
    name="centroid_cog",
    version="1.0",
    description="Fast C CoG centroiding for SH-WFS",
    ext_modules=[ext],
)
