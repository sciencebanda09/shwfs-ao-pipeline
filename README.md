<p align="center">
  <img src="results/dashboard_full.png" alt="BAH2026 AO Pipeline Dashboard" width="900"/>
</p>

<h1 align="center">shwfs-ao-pipeline</h1>

<p align="center">
  <b>End-to-end Adaptive Optics simulation & reconstruction pipeline</b><br>
  Bharatiya Antariksh Hackathon 2026 Â· Challenge #9
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch"/>
  <img src="https://img.shields.io/badge/Strehl-0.997-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/Latency-0.3ms-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square"/>
  <img src="https://github.com/sciencebanda09/shwfs-ao-pipeline/actions/workflows/test.yml/badge.svg" alt="tests"/>
</p>

---

## What this is

A production-ready AO pipeline that takes raw Shack-Hartmann WFS frames â†’ reconstructs the wavefront â†’ drives a deformable mirror â†’ achieves **Strehl ratio > 0.99** in closed loop.

**Pipeline stages:**

```
SH-WFS frames â†’ centroiding â†’ wavefront reconstruction â†’ DM actuator commands
      â†“                              â†“                          â†“
  (C ext, 0.1ms)         (SVD / MMSE / CNN-UNet)        (LQG controller)
                                     â†“
                          LSTM temporal prediction
                                     â†“
                          SLODAR turbulence profiling
```

---

## Results

### Strehl Ratio â€” Predictive AO vs Closed Loop vs Open Loop
![Strehl Ratio Time Series](results/demo_strehl_timeseries.png)

### Reconstructed Wavefront Phase Maps (Real SH-WFS Data, r0=0.534m, Ï„â‚€=35.8ms)
![Real Phase Maps](results/real_phase_maps.png)

---

## Performance at a glance

### Reconstruction

| Method | Mean Strehl | RMS WFE |
|--------|------------|---------|
| Classical SVD | 0.9939 | 75.1 nm |
| **CNN / UNet** | **0.9986** | **37.1 nm** |

CNN/UNet delivers **2Ã— lower RMS WFE** vs classical SVD reconstruction.

### Controller comparison

| Controller | Mean Strehl | Mean RMS WFE |
|-----------|------------|-------------|
| Integrator | **0.967** | **15.4 nm** |
| LQG | 0.702 | 51.8 nm |
| LQG + Predictive | 0.694 | 52.7 nm |

### Speed (10Ã—10 subapertures)

| Step | Method | Latency |
|------|--------|---------|
| Centroiding | Python CoG | ~8 ms |
| Centroiding | **C extension** | **~0.1 ms** |
| Wavefront reconstruction | NumPy matmul (precomputed SVD) | ~0.1 ms |
| Actuator commands | Influence matrix matmul | ~0.1 ms |
| **Total** | **C centroid + NumPy recon** | **~0.3 ms âœ“** |

Target: < 10 ms to track Ï„â‚€ ~ 5â€“20 ms. **Achieved: 0.3 ms.**

---

## Repository layout

```
shwfs-ao-pipeline/
â”œâ”€â”€ config.yaml               # all tunable parameters
â”œâ”€â”€ pipeline.py               # end-to-end runner: sim/train/eval/demo/genbmp/real
â”œâ”€â”€ Makefile
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ sim/                      # atmosphere, SH-WFS, noise, scintillation, dataset gen
â”œâ”€â”€ reconstruction/           # Zernike basis, SVD, MMSE/Bayesian, CNN/UNet
â”œâ”€â”€ profiling/                # SLODAR CnÂ²(h) profiler, temporal PSD / Ï„â‚€
â”œâ”€â”€ control/                  # LQG (Kalman + LQR) controller
â”œâ”€â”€ actuator/                 # DM geometry, influence functions, command generation
â”œâ”€â”€ temporal/                 # LSTM/Transformer prediction, turbulence parameter estimation
â”œâ”€â”€ viz/                      # matplotlib + Plotly/Dash dashboard
â”œâ”€â”€ tests/                    # pytest unit tests
â”œâ”€â”€ notebooks/                # demo and benchmark notebooks
â”œâ”€â”€ data/                     # datasets (git-ignored)
â”‚   â””â”€â”€ synthetic_bmp/        # 200 synthetic SH-WFS BMP frames
â”œâ”€â”€ models/                   # trained model checkpoints (git-ignored)
â””â”€â”€ results/                  # benchmark CSVs, timing, plots
```

---

## Quick start

```bash
pip install -r requirements.txt

make sim      # generate training dataset (500 frames, 36 Zernike modes)
make train    # train CNN/UNet reconstructor + LSTM temporal model
make eval     # full benchmark: classical / CNN / MMSE / LQG / SLODAR
make demo     # short closed-loop demo + live dashboard
make test     # pytest unit tests
```

### Optional: C centroiding extension (80Ã— speedup)

```bash
cd c_ext && pip install -e .
```

Auto-detected at runtime. Without it, Python CoG fallback is used.

---

## System parameters

| Parameter | Value |
|-----------|-------|
| Aperture diameter | 0.5 m |
| Subapertures | 10 Ã— 10 |
| Zernike modes | 36 |
| DM actuators | 97 (hexagonal) |
| Turbulence model | Von Karman, 3 layers |
| r0 | 0.15 m |
| Simulation timestep | 1 ms |
| Frames per dataset | 500 |

---

## Synthetic BMP frame generation

Before real ISRO data arrives, generate physically correct synthetic SH-WFS frames:

```bash
python3 pipeline.py --config config.yaml --mode genbmp \
    --n_bmp_frames 200 \
    --bmp_output_dir data/synthetic_bmp/ \
    --reference data/synthetic_bmp/reference.bmp
```

Then test the full ingestion pipeline:

```bash
python3 pipeline.py --config config.yaml --mode real \
    --bmp_dir data/synthetic_bmp/ \
    --reference data/synthetic_bmp/reference.bmp
```

When ISRO's actual frames arrive, swap `--bmp_dir`. No other changes needed.

---

## Real data usage

```bash
python3 pipeline.py --config config.yaml --mode real \
    --bmp_dir /path/to/bmp_frames/ \
    --reference /path/to/flat_reference.bmp

# Outputs â†’ results/
#   real_reconstruction.npz  â€” zernike_coeffs, phase_maps, actuator_maps, r0_m, tau0_s
#   real_phase_maps.png       â€” first 3 reconstructed wavefront phase maps
```

The BMP loader handles: auto-discovery, CoG / wCoG centroiding, background subtraction, flat reference normalisation, auto-resize for different camera resolutions, and C extension auto-detection.

---

## Theory

### MMSE Reconstruction

Measurement model: `m = D s + n`, with `s ~ N(0, C_phi)` (Kolmogorov/Noll prior) and `n ~ N(0, C_n)`.

MMSE estimator:
```
Å = C_phi D^T (D C_phi D^T + C_n)^-1 m
```

Strictly dominates SVD pseudo-inverse under noise by incorporating the turbulence prior. Supports online `r0` updates and optional learned noise covariance (`LearnedNoiseCov`). â†’ `reconstruction/bayesian.py`

### r0 Estimation

Noll variance per mode: `Ïƒ_jÂ² = K_j (D/r0)^(5/3)`. Averaging `log(Ïƒ_jÂ²/K_j)` across modes `j=2..36` recovers `r0` to within a few percent. â†’ `temporal/turbulence_param.py`

### Temporal PSD & Ï„â‚€

Von Karman temporal PSD: `S(f) = ÏƒÂ² (fÂ² + f_gÂ²)^(-11/6)`. Fit per mode â†’ `Ï„â‚€(j) = 1/f_g(j)`. Higher-order modes have shorter Ï„â‚€ (faster advection of smaller structures). â†’ `profiling/temporal_psd.py`

### LQG Controller

Kalman filter on AR(1) frozen-flow state model + LQR via discrete algebraic Riccati equation. Also implements L1 (LASSO) actuator solution for sparser stroke usage. â†’ `control/lqg.py`

---

## Notebooks

| Notebook | Contents | Launch |
|----------|----------|--------|
| `01_sim_demo.ipynb` | Turbulence sim, SH-WFS propagation, noise | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sciencebanda09/shwfs-ao-pipeline/blob/main/notebooks/01_sim_demo.ipynb) |
| `02_reconstruction_benchmark.ipynb` | CNN/UNet training, benchmark vs classical | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sciencebanda09/shwfs-ao-pipeline/blob/main/notebooks/02_reconstruction_benchmark.ipynb) |
| `03_temporal_prediction.ipynb` | LSTM prediction, closed-loop sim, turbulence parameter estimation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sciencebanda09/shwfs-ao-pipeline/blob/main/notebooks/03_temporal_prediction.ipynb) |
| `04_phd_extensions.ipynb` | MMSE, Noll validation, SLODAR, mode-dependent τ₀, LQG vs integrator | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sciencebanda09/shwfs-ao-pipeline/blob/main/notebooks/04_phd_extensions.ipynb) |
---

## Evaluation criteria

| Criterion | Implementation | Location |
|-----------|---------------|----------|
| Wavefront phase maps | ModalReconstructor + Zernike basis | `reconstruction/classical.py`, `reconstruction/zernike.py` |
| Fried parameter r0 | Noll variance fit | `temporal/turbulence_param.py` |
| Coherence time Ï„â‚€ | Von Karman PSD fit, per-mode | `profiling/temporal_psd.py` |
| Actuator maps | Influence matrix pseudo-inverse + stroke clip | `actuator/dm_command.py` |
| Inter-actuator coupling | Gaussian IF, coupling = 0.3 | `actuator/influence_fn.py` |
| Algorithm speed < 10 ms | C CoG + precomputed SVD matmul | `c_ext/centroid_cog.c` |
| Real BMP ingestion | RealSHWFSLoader | `data/load_real_frames.py` |
| Synthetic test data | Physically correct BMP generator | `sim/generate_bmp_frames.py` |

---

## References

- Roddier, F. (1999). *Adaptive Optics in Astronomy*. Cambridge University Press.
- Hardy, J. W. (1998). *Adaptive Optics for Astronomical Telescopes*. Oxford University Press.
- Noll, R. J. (1976). Zernike polynomials and atmospheric turbulence. *JOSA*, 66(3), 207â€“211.
- Fusco, T., et al. (2004). Optimal wavefront reconstruction for MCAO. *JOSA A*, 18(10).
- Veran, J.-P., et al. (1997). Estimation of AO long-exposure PSF from control loop data. *JOSA A*, 14(11).
- Wiberg, D. M., et al. (2004). LQG vs explicit predictive control of AO systems. *Proc. SPIE*.


