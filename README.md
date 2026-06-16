# BAH2026 — Adaptive Optics Pipeline

End-to-end AO simulation and reconstruction pipeline for **Bharatiya Antariksh Hackathon 2026, Challenge #9**.

Covers: atmospheric turbulence simulation → Shack-Hartmann WFS → classical & neural wavefront reconstruction → DM control (integrator + LQG) → temporal prediction → altitude-resolved turbulence profiling (SLODAR).

---

## Repository layout

```
bah2026-ao-pipeline/
├── config.yaml               # all tunable parameters
├── pipeline.py               # end-to-end runner: sim / train / eval / demo / genbmp / real
├── Makefile
├── requirements.txt
├── sim/                      # atmosphere, SH-WFS, noise, scintillation, dataset gen
├── reconstruction/           # Zernike basis, SVD, MMSE/Bayesian, CNN/UNet
├── profiling/                # SLODAR Cn²(h) profiler, temporal PSD / τ₀
├── control/                  # LQG (Kalman + LQR) controller
├── actuator/                 # DM geometry, influence functions, command generation
├── temporal/                 # LSTM/Transformer prediction, turbulence parameter estimation
├── viz/                      # matplotlib + Plotly/Dash dashboard
├── tests/                    # pytest unit tests
├── notebooks/                # demo and benchmark notebooks
├── data/                     # datasets (git-ignored except .gitkeep)
│   └── synthetic_bmp/        # synthetic SH-WFS BMP frames
├── models/                   # trained model checkpoints (git-ignored except .gitkeep)
└── results/                  # benchmark CSVs, timing, plots (git-ignored except .gitkeep)
```

---

## Quick start

```bash
pip install -r requirements.txt

make sim      # generate training dataset
make train    # train CNN/UNet reconstructor + LSTM temporal model
make eval     # full benchmark (classical / CNN / MMSE / LQG / SLODAR)
make demo     # short closed-loop demo + dashboard
make test     # pytest unit tests
```

### Optional: C centroiding extension (80× speedup)

```bash
cd c_ext && pip install -e .
```

After building, the pipeline auto-detects and uses it. Without it, Python CoG is used (~8 ms vs ~0.1 ms per frame).

---

## Configuration

All parameters live in `config.yaml`, grouped by subsystem:

| Section | Key parameters |
|---------|---------------|
| `sim` | aperture diameter, subapertures, Zernike modes, frame count, timestep |
| `turbulence` | `r0_m`, `L0_m`, layer altitudes / weights / wind speeds |
| `noise` | photon flux, readout noise, centroiding method (`cog` / `wcog`) |
| `reconstruction` | SVD condition number, CNN architecture, training hyperparams |
| `actuator` | geometry, coupling, stroke limit |
| `temporal` | LSTM / Transformer, sequence length, predict horizon |
| `bayesian` | regularization, Kolmogorov prior, noise covariance model |
| `lqg` | process noise Q, measurement noise R, LQR weights |
| `control_method` | `"integrator"` or `"lqg"` |

---

## Synthetic BMP frame generation

Generate physically correct synthetic SH-WFS frames before real ISRO data arrives:

```bash
python3 pipeline.py --config config.yaml --mode genbmp \
    --n_bmp_frames 200 \
    --bmp_output_dir data/synthetic_bmp/ \
    --reference data/synthetic_bmp/reference.bmp
```

Each frame uses `SHWFSSensor.simulate_spot_image()` — same oversampled FFT propagation as the sim — so centroid scaling is physically consistent.

---

## Real data usage

Drop in actual SH-WFS `.bmp` frames from the ISRO lab:

```bash
python3 pipeline.py --config config.yaml --mode real \
    --bmp_dir /path/to/bmp_frames/ \
    --reference /path/to/flat_reference.bmp
```

Outputs written to `results/`:
- `real_reconstruction.npz` — zernike_coeffs, phase_maps, actuator_maps, r0_m, tau0_s
- `real_phase_maps.png` — first 3 reconstructed wavefront phase maps

The BMP loader (`data/load_real_frames.py`) handles auto-discovery, CoG/wCoG centroiding, background subtraction, flat reference normalisation, and auto-resize if camera resolution differs from config.

---

## Algorithm speed

Target: < 10 ms per frame to track τ₀ ~ 5–20 ms.

| Step | Method | Latency |
|------|--------|---------|
| Centroiding (10×10 sub-apertures) | Python CoG | ~8 ms |
| Centroiding (10×10 sub-apertures) | C CoG extension | ~0.1 ms |
| Wavefront reconstruction | NumPy matmul (precomputed SVD `W`) | ~0.1 ms |
| Actuator command generation | Influence matrix matmul | ~0.1 ms |
| **Total (C centroid + NumPy recon)** | | **~0.3 ms ✓** |

Reproduce: `python3 pipeline.py --mode eval` → `results/timing_latency.csv`.

---

## Theory notes

### MMSE reconstruction

SH-WFS measurement model: `m = D s + n`, where `s ~ N(0, C_phi)` (Zernike coefficients with Kolmogorov/Noll prior) and `n ~ N(0, C_n)`.

MMSE estimator (conditional mean for jointly Gaussian variables):

```
ŝ = C_phi D^T (D C_phi D^T + C_n)^-1 m
```

This strictly dominates SVD pseudo-inverse when measurement is noisy because it incorporates the turbulence prior `C_phi`. Implementation: `reconstruction/bayesian.py`. Supports online `r0` updates (`update_r0`) and an optional learned noise covariance network (`LearnedNoiseCov`).

### Noll variance & r0 estimation

Kolmogorov variance per Zernike mode `j`:

```
σ_j² = K_j (D / r0)^(5/3)
```

Averaging `log(σ_j² / K_j)` across modes `j = 2..36` gives a direct `r0` estimate converging to within a few percent of the true input. See `temporal/turbulence_param.py:estimate_r0_from_zernike`.

### Temporal PSD & τ₀

Von Karman temporal PSD per Zernike mode:

```
S(f) = σ² (f² + f_g²)^(-11/6)
```

`profiling/temporal_psd.py` fits this model, extracts `f_g` per mode, and reports `τ₀(j) = 1/f_g(j)`. Higher-order modes have shorter `τ₀` (smaller turbulent structures advect faster under frozen flow).

### LQG controller

Kalman filter estimates Zernike state from noisy slopes using an AR(1) frozen-flow model; LQR computes actuator commands via the discrete algebraic Riccati equation (`scipy.linalg.solve_discrete_are`). LQG achieves higher mean Strehl than the integrator by explicit noise-aware state estimation. `ActuatorStrokeMinimizer` supports L2 (least-squares) and L1 (LASSO, sparser) actuator solutions.

---

## Evaluation criteria mapping

| Criterion | Implementation | Location |
|-----------|---------------|----------|
| Wavefront phase maps W(xi,yi) | ModalReconstructor + Zernike basis | `reconstruction/classical.py`, `reconstruction/zernike.py` |
| Fried parameter r0 | Noll variance fit | `temporal/turbulence_param.py` |
| Coherence time τ₀ | Von Karman PSD fit, per-mode | `profiling/temporal_psd.py` |
| Actuator maps A(xi,yi) | Influence matrix pseudo-inverse + stroke clip | `actuator/dm_command.py` |
| Inter-actuator coupling | Gaussian IF, coupling = 0.3 | `actuator/influence_fn.py` |
| Algorithm speed (< 10 ms) | C CoG + precomputed SVD matmul | `c_ext/centroid_cog.c` |
| Real BMP ingestion | RealSHWFSLoader | `data/load_real_frames.py` |
| Synthetic test data | Physically correct BMP generator | `sim/generate_bmp_frames.py` |

---

## Notebooks

| Notebook | Contents |
|----------|----------|
| `01_sim_demo.ipynb` | Turbulence simulation, SH-WFS propagation, noise |
| `02_reconstruction_benchmark.ipynb` | CNN/UNet training, benchmark vs classical |
| `03_temporal_prediction.ipynb` | LSTM prediction, closed-loop sim, turbulence parameter estimation |
| `04_phd_extensions.ipynb` | MMSE, Noll validation, SLODAR, mode-dependent τ₀, LQG vs integrator, uncertainty-gated AO |

---

## References

- Roddier, F. (1999). *Adaptive Optics in Astronomy*. Cambridge University Press.
- Hardy, J. W. (1998). *Adaptive Optics for Astronomical Telescopes*. Oxford University Press.
- Noll, R. J. (1976). Zernike polynomials and atmospheric turbulence. *JOSA*, 66(3), 207–211.
- Fusco, T., et al. (2004). Optimal wavefront reconstruction for MCAO. *JOSA A*, 18(10).
- Veran, J.-P., et al. (1997). Estimation of AO long-exposure PSF from control loop data. *JOSA A*, 14(11).
- Wiberg, D. M., et al. (2004). LQG vs explicit predictive control of AO systems. *Proc. SPIE*.
