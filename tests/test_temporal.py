"""
tests/test_temporal.py
========================
Unit tests for temporal/lstm_model.py, predictor.py,
turbulence_param.py, train_temporal.py.
"""

import numpy as np
import torch
import pytest

from temporal.lstm_model import ZernikeTimeSeries, TemporalTransformer
from temporal.predictor import WavefrontPredictor
from temporal.turbulence_param import estimate_r0_from_zernike
from temporal.train_temporal import prepare_sequences
from reconstruction.bayesian import KolmogorovCovariance


N_ZERNIKE = 36
SEQ_LEN = 20


def test_lstm_forward():
    """ZernikeTimeSeries forward pass should produce correct output shape."""
    model = ZernikeTimeSeries(input_size=N_ZERNIKE, hidden_size=32, n_layers=2, output_size=N_ZERNIKE)
    x = torch.randn(4, SEQ_LEN, N_ZERNIKE)
    out = model(x)
    assert out.shape == (4, N_ZERNIKE)


def test_transformer_forward():
    """TemporalTransformer forward pass should produce correct output shape."""
    model = TemporalTransformer(d_model=32, nhead=4, n_encoder_layers=2, n_zernike=N_ZERNIKE, seq_len=SEQ_LEN)
    x = torch.randn(4, SEQ_LEN, N_ZERNIKE)
    out = model(x)
    assert out.shape == (4, N_ZERNIKE)


def test_predictor_cold_start():
    """Predictor should return a fallback (last frame) before buffer is filled."""
    model = ZernikeTimeSeries(input_size=N_ZERNIKE, hidden_size=32, n_layers=2, output_size=N_ZERNIKE)
    device = torch.device("cpu")
    predictor = WavefrontPredictor(model, seq_len=SEQ_LEN, device=device)

    frame = np.random.randn(N_ZERNIKE).astype(np.float32)
    predictor.update(frame)

    out = predictor.predict()
    assert out.shape == (N_ZERNIKE,)
    np.testing.assert_allclose(out, frame)


def test_predictor_predict():
    """After seq_len updates, predict() should return an array of correct shape."""
    model = ZernikeTimeSeries(input_size=N_ZERNIKE, hidden_size=32, n_layers=2, output_size=N_ZERNIKE)
    device = torch.device("cpu")
    predictor = WavefrontPredictor(model, seq_len=SEQ_LEN, device=device)

    for _ in range(SEQ_LEN):
        frame = np.random.randn(N_ZERNIKE).astype(np.float32)
        predictor.update(frame)

    out = predictor.predict()
    assert out.shape == (N_ZERNIKE,)
    assert np.all(np.isfinite(out))


def test_r0_estimation():
    """estimate_r0_from_zernike on simulated data with known r0 should have error < 20%."""
    D = 0.5
    r0_true = 0.15
    n_frames = 2000

    rng = np.random.default_rng(42)
    coeffs = np.zeros((n_frames, N_ZERNIKE))
    for idx in range(N_ZERNIKE):
        j = idx + 1
        if j == 1:
            continue
        var_j = KolmogorovCovariance.noll_variance(j, D, r0_true)
        if var_j <= 0:
            var_j = 1e-8
        coeffs[:, idx] = rng.normal(0, np.sqrt(var_j), size=n_frames)

    r0_est = estimate_r0_from_zernike(coeffs, wavelength=550e-9, D=D)

    rel_error = abs(r0_est - r0_true) / r0_true
    assert rel_error < 0.20


def test_sequence_preparation():
    """prepare_sequences should produce correctly-shaped, overlapping sequences."""
    n_frames = 100
    zernike_data = np.random.randn(n_frames, N_ZERNIKE).astype(np.float32)

    seq_len = 20
    horizon = 1
    X, y = prepare_sequences(zernike_data, seq_len, horizon)

    expected_n_samples = n_frames - seq_len - horizon + 1
    assert X.shape == (expected_n_samples, seq_len, N_ZERNIKE)
    assert y.shape == (expected_n_samples, N_ZERNIKE)

    # Check overlap: X[1, 0] should equal X[0, 1]
    np.testing.assert_allclose(X[1, 0], X[0, 1])

    # Check target alignment
    np.testing.assert_allclose(y[0], zernike_data[seq_len + horizon - 1])


def test_load_trained_temporal_model_roundtrip(tmp_path):
    """Bug 4 regression: load_trained_temporal_model round-trips a saved checkpoint."""
    import torch
    import yaml
    from temporal.lstm_model import ZernikeTimeSeries
    from temporal.train_temporal import load_trained_temporal_model

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    sim_cfg = config["sim"]
    temp_cfg = config["temporal"]
    n_zernike = sim_cfg["n_zernike"]

    model = ZernikeTimeSeries(
        input_size=n_zernike,
        hidden_size=temp_cfg["hidden_size"],
        n_layers=temp_cfg["n_layers"],
        output_size=n_zernike,
    )
    ckpt_path = str(tmp_path / "test_temporal_model.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": temp_cfg, "epoch": 0, "val_loss": 0.1}, ckpt_path)

    device = torch.device("cpu")
    loaded = load_trained_temporal_model(ckpt_path, config, device)

    # Parameters should match
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded.named_parameters()):
        assert torch.allclose(p1, p2), f"Mismatch in {n1}"

    # Should be in eval mode
    assert not loaded.training


def test_run_demo_no_checkpoint_fallback(tmp_path, monkeypatch):
    """Bug 4: run_demo/dashboard must not silently use random weights if no checkpoint exists."""
    import yaml, warnings
    from pathlib import Path

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Point models_dir at a temp dir with no checkpoint
    config["paths"]["models_dir"] = str(tmp_path)
    config["paths"]["results_dir"] = str(tmp_path / "results")

    lstm_path = Path(config["paths"]["models_dir"]) / "temporal_model.pt"
    assert not lstm_path.exists(), "Should not exist for this test"

    # Import the function and verify it warns / doesn't crash
    import importlib, pipeline as pl
    # Patch plt.subplots to avoid display
    import matplotlib
    matplotlib.use("Agg")

    printed = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(" ".join(str(x) for x in a)))

    try:
        pl.run_demo(config)
    except Exception as e:
        # May fail for other reasons (missing dataset etc) but must NOT
        # silently use random weights (we check warning was issued)
        pass

    warning_issued = any("WARNING" in s or "not found" in s for s in printed)
    assert warning_issued, f"Expected WARNING about missing checkpoint, got: {printed}"
