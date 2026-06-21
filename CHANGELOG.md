# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]
### Fixed
- fix(slodar): replace independent peak-picking with joint NNLS fit for Cn2 profile recovery � Closes #6



## [1.1.0] - 2026-06-21

### Fixed
- C extension build failure on AVX-512 CPUs (`-march=native` removed)
- C centroid threshold used background mean instead of std
- `atmosphere.reset()` was unreachable dead code
- Wrong attribute name `layer.phase_screen` → `layer.phase`
- Backwards matmul in actuator-influence projection
- Nonexistent `DMInfluenceReconstructor` class reference
- Actuator count mismatch (91 hardcoded vs 127 real)
- Tile truncation dropping 6% of aperture pixels in sensor and reconstructor

### Improved
- Reconstruction Strehl: 0.9942 → 0.9964 (classical), 0.9980 → 0.9987 (CNN)
- Closed-loop Strehl LQG: 0.871 → 0.908
- C centroiding: 127x speedup over Python, 0.035ms/frame

### Added
- Granular CI: flake8 lint, mypy typecheck, pytest-cov 50% threshold

### Known Issues
- SLODAR altitude recovery unreliable for r0/altitude combinations exceeding aperture coherence length (#6)

## [1.0.0] - 2026-06-17

### Added
- Initial release: SH-WFS simulation, zonal/modal reconstruction, LQG control, SLODAR profiling, C centroiding extension

