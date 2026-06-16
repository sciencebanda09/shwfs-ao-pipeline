# Build the C centroiding extension

```bash
cd c_ext/
pip install -e .
cd ..
python3 -c "import centroid_cog; print('C ext OK')"
```

Then the loader in `data/load_real_frames.py` auto-detects and uses it —
no code change needed. Reduces per-frame centroiding from ~8 ms to ~0.1 ms (80× speedup).

## Usage from Python

```python
import centroid_cog, numpy as np

frame_f32 = frame.astype(np.float32)
cx_list, cy_list = centroid_cog.cog_batch(
    frame_f32.tobytes(), n_sub=10, pix_per_sub=8, threshold_sigma=3.0
)
cx = np.array(cx_list).reshape(10, 10)
cy = np.array(cy_list).reshape(10, 10)
```
