/*
 * c_ext/centroid_cog.c
 * ====================
 * Fast Centre-of-Gravity centroiding for SH-WFS spot tiles.
 * Compiled as a Python C extension via c_ext/setup.py.
 *
 * Improvements over original:
 *   - Border-ring background estimate (all edge pixels, not 4 corners)
 *     → robust when spots land near subaperture corners.
 *   - Single scratch buffer allocated once outside the loop; no
 *     malloc/free per tile.
 *   - OpenMP parallelism over subapertures (compile with -fopenmp).
 *   - Separate wcog_single() for weighted CoG (Gaussian weight centred
 *     on brightest pixel) exposed as wcog_batch().
 *
 * Exposed functions:
 *   cog_batch(frame_bytes, n_sub, pix_per_sub, threshold_sigma)
 *       -> (cx_list, cy_list)
 *   wcog_batch(frame_bytes, n_sub, pix_per_sub, weight_fwhm_px)
 *       -> (cx_list, cy_list)
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

/* -----------------------------------------------------------------------
 * Border-ring background estimation.
 * Uses all pixels on the outermost ring (top row, bottom row, left col,
 * right col) so that a spot landing near any corner still gets a good
 * background estimate.  Returns mean of border pixels.
 * ---------------------------------------------------------------------- */
static double border_bg(const float *tile, int nrows, int ncols)
{
    double sum = 0.0;
    int    count = 0;

    /* top and bottom rows */
    for (int c = 0; c < ncols; c++) {
        sum += (double)tile[c];                          /* row 0 */
        sum += (double)tile[(nrows - 1) * ncols + c];   /* row nrows-1 */
        count += 2;
    }
    /* left and right columns, excluding corners already counted */
    for (int r = 1; r < nrows - 1; r++) {
        sum += (double)tile[r * ncols];                  /* col 0 */
        sum += (double)tile[r * ncols + ncols - 1];      /* col ncols-1 */
        count += 2;
    }
    return (count > 0) ? (sum / count) : 0.0;
}

/* -----------------------------------------------------------------------
 * Standard thresholded CoG on a contiguous tile buffer.
 * bg_sigma_thresh: zero out pixels below thresh * |bg|.
 * ---------------------------------------------------------------------- */
/* -----------------------------------------------------------------------
 * Border-ring background mean AND standard deviation.
 * threshold_sigma must scale the noise (std), not the background level
 * itself — using the mean here would make a brighter-but-quiet background
 * threshold too aggressively, and a near-zero-mean-but-noisy background
 * threshold almost not at all.
 * ---------------------------------------------------------------------- */
static void border_bg_stats(const float *tile, int nrows, int ncols,
                             double *bg_mean_out, double *bg_std_out)
{
    double sum = 0.0;
    int    count = 0;

    for (int c = 0; c < ncols; c++) {
        sum += (double)tile[c];
        sum += (double)tile[(nrows - 1) * ncols + c];
        count += 2;
    }
    for (int r = 1; r < nrows - 1; r++) {
        sum += (double)tile[r * ncols];
        sum += (double)tile[r * ncols + ncols - 1];
        count += 2;
    }
    double mean = (count > 0) ? (sum / count) : 0.0;

    double sq_sum = 0.0;
    for (int c = 0; c < ncols; c++) {
        double d0 = (double)tile[c] - mean;
        double d1 = (double)tile[(nrows - 1) * ncols + c] - mean;
        sq_sum += d0 * d0 + d1 * d1;
    }
    for (int r = 1; r < nrows - 1; r++) {
        double d0 = (double)tile[r * ncols] - mean;
        double d1 = (double)tile[r * ncols + ncols - 1] - mean;
        sq_sum += d0 * d0 + d1 * d1;
    }
    double std = (count > 0) ? sqrt(sq_sum / count) : 0.0;
    if (std < 1.0) std = 1.0;  /* floor, matches Python reference path */

    *bg_mean_out = mean;
    *bg_std_out  = std;
}

/* -----------------------------------------------------------------------
 * Standard thresholded CoG on a contiguous tile buffer.
 * bg_sigma_thresh: zero out pixels below thresh * background_std.
 * ---------------------------------------------------------------------- */
static void cog_single(
    const float *tile, int nrows, int ncols,
    double bg_sigma_thresh,
    double *cx_out, double *cy_out)
{
    double bg, bg_std;
    border_bg_stats(tile, nrows, ncols, &bg, &bg_std);
    double thresh = bg_sigma_thresh * bg_std;

    double sum = 0.0, sum_x = 0.0, sum_y = 0.0;
    for (int r = 0; r < nrows; r++) {
        for (int c = 0; c < ncols; c++) {
            double v = (double)tile[r * ncols + c] - bg;
            if (v < thresh || v < 0.0) v = 0.0;
            sum   += v;
            sum_x += v * (double)c;
            sum_y += v * (double)r;
        }
    }
    if (sum <= 0.0) {
        *cx_out = (ncols - 1) * 0.5;
        *cy_out = (nrows - 1) * 0.5;
    } else {
        *cx_out = sum_x / sum;
        *cy_out = sum_y / sum;
    }
}

/* -----------------------------------------------------------------------
 * Weighted CoG: Gaussian weight centred on the brightest pixel.
 * weight_fwhm_px: FWHM of the Gaussian weight in pixels.
 * More robust than plain CoG at low SNR.
 * ---------------------------------------------------------------------- */
static void wcog_single(
    const float *tile, int nrows, int ncols,
    double weight_fwhm_px,
    double *cx_out, double *cy_out)
{
    /* find brightest pixel */
    int    peak_r = 0, peak_c = 0;
    double peak_val = -1e30;
    for (int r = 0; r < nrows; r++) {
        for (int c = 0; c < ncols; c++) {
            double v = (double)tile[r * ncols + c];
            if (v > peak_val) { peak_val = v; peak_r = r; peak_c = c; }
        }
    }

    double sigma2 = (weight_fwhm_px / 2.3548) * (weight_fwhm_px / 2.3548);
    double bg     = border_bg(tile, nrows, ncols);

    double sum = 0.0, sum_x = 0.0, sum_y = 0.0;
    for (int r = 0; r < nrows; r++) {
        for (int c = 0; c < ncols; c++) {
            double dr = r - peak_r;
            double dc = c - peak_c;
            double w  = exp(-(dr * dr + dc * dc) / (2.0 * sigma2));
            double v  = ((double)tile[r * ncols + c] - bg) * w;
            if (v < 0.0) v = 0.0;
            sum   += v;
            sum_x += v * (double)c;
            sum_y += v * (double)r;
        }
    }
    if (sum <= 0.0) {
        *cx_out = (ncols - 1) * 0.5;
        *cy_out = (nrows - 1) * 0.5;
    } else {
        *cx_out = sum_x / sum;
        *cy_out = sum_y / sum;
    }
}

/* -----------------------------------------------------------------------
 * Copy one subaperture tile from the full frame into dst.
 * Inlined so the compiler can optimise it alongside the caller.
 * ---------------------------------------------------------------------- */
static inline void extract_tile(
    const float *frame, int total_w,
    int i, int j, int pps,
    float *dst)
{
    for (int r = 0; r < pps; r++) {
        int src_row = i * pps + r;
        memcpy(dst + r * pps,
               frame + src_row * total_w + j * pps,
               pps * sizeof(float));
    }
}

/* -----------------------------------------------------------------------
 * cog_batch: thresholded CoG over all subapertures.
 *
 * Python signature:
 *   cog_batch(frame_bytes: bytes, n_sub: int, pix_per_sub: int,
 *             threshold_sigma: float) -> (cx_list, cy_list)
 *
 * frame_bytes must be a flat float32 array of length (n_sub*pps)^2.
 * ---------------------------------------------------------------------- */
static PyObject *
py_cog_batch(PyObject *self, PyObject *args)
{
    Py_buffer view;
    int    n_sub, pps;
    double thresh;

    if (!PyArg_ParseTuple(args, "y*iid", &view, &n_sub, &pps, &thresh))
        return NULL;

    int total_w    = n_sub * pps;
    int frame_size = total_w * total_w;

    if (view.len != (Py_ssize_t)(frame_size * sizeof(float))) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "frame_flat length mismatch");
        return NULL;
    }

    const float *frame = (const float *)view.buf;
    int n_tiles = n_sub * n_sub;

    /* Allocate result arrays up front */
    double *cx_arr = (double *)malloc(n_tiles * sizeof(double));
    double *cy_arr = (double *)malloc(n_tiles * sizeof(double));
    if (!cx_arr || !cy_arr) {
        free(cx_arr); free(cy_arr);
        PyBuffer_Release(&view);
        return PyErr_NoMemory();
    }

    /* One scratch tile buffer per thread (OpenMP) or one total (serial) */
#ifdef _OPENMP
    #pragma omp parallel
    {
        float *tile = (float *)malloc(pps * pps * sizeof(float));
        #pragma omp for schedule(static)
        for (int k = 0; k < n_tiles; k++) {
            int i = k / n_sub, j = k % n_sub;
            extract_tile(frame, total_w, i, j, pps, tile);
            cog_single(tile, pps, pps, thresh, &cx_arr[k], &cy_arr[k]);
        }
        free(tile);
    }
#else
    {
        float *tile = (float *)malloc(pps * pps * sizeof(float));
        for (int k = 0; k < n_tiles; k++) {
            int i = k / n_sub, j = k % n_sub;
            extract_tile(frame, total_w, i, j, pps, tile);
            cog_single(tile, pps, pps, thresh, &cx_arr[k], &cy_arr[k]);
        }
        free(tile);
    }
#endif

    PyBuffer_Release(&view);

    /* Pack into Python lists */
    PyObject *cx_list = PyList_New(n_tiles);
    PyObject *cy_list = PyList_New(n_tiles);
    for (int k = 0; k < n_tiles; k++) {
        PyList_SET_ITEM(cx_list, k, PyFloat_FromDouble(cx_arr[k]));
        PyList_SET_ITEM(cy_list, k, PyFloat_FromDouble(cy_arr[k]));
    }
    free(cx_arr); free(cy_arr);

    return PyTuple_Pack(2, cx_list, cy_list);
}

/* -----------------------------------------------------------------------
 * wcog_batch: Gaussian-weighted CoG over all subapertures.
 *
 * Python signature:
 *   wcog_batch(frame_bytes: bytes, n_sub: int, pix_per_sub: int,
 *              weight_fwhm_px: float) -> (cx_list, cy_list)
 * ---------------------------------------------------------------------- */
static PyObject *
py_wcog_batch(PyObject *self, PyObject *args)
{
    Py_buffer view;
    int    n_sub, pps;
    double fwhm;

    if (!PyArg_ParseTuple(args, "y*iid", &view, &n_sub, &pps, &fwhm))
        return NULL;

    int total_w    = n_sub * pps;
    int frame_size = total_w * total_w;

    if (view.len != (Py_ssize_t)(frame_size * sizeof(float))) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "frame_flat length mismatch");
        return NULL;
    }

    const float *frame = (const float *)view.buf;
    int n_tiles = n_sub * n_sub;

    double *cx_arr = (double *)malloc(n_tiles * sizeof(double));
    double *cy_arr = (double *)malloc(n_tiles * sizeof(double));
    if (!cx_arr || !cy_arr) {
        free(cx_arr); free(cy_arr);
        PyBuffer_Release(&view);
        return PyErr_NoMemory();
    }

#ifdef _OPENMP
    #pragma omp parallel
    {
        float *tile = (float *)malloc(pps * pps * sizeof(float));
        #pragma omp for schedule(static)
        for (int k = 0; k < n_tiles; k++) {
            int i = k / n_sub, j = k % n_sub;
            extract_tile(frame, total_w, i, j, pps, tile);
            wcog_single(tile, pps, pps, fwhm, &cx_arr[k], &cy_arr[k]);
        }
        free(tile);
    }
#else
    {
        float *tile = (float *)malloc(pps * pps * sizeof(float));
        for (int k = 0; k < n_tiles; k++) {
            int i = k / n_sub, j = k % n_sub;
            extract_tile(frame, total_w, i, j, pps, tile);
            wcog_single(tile, pps, pps, fwhm, &cx_arr[k], &cy_arr[k]);
        }
        free(tile);
    }
#endif

    PyBuffer_Release(&view);

    PyObject *cx_list = PyList_New(n_tiles);
    PyObject *cy_list = PyList_New(n_tiles);
    for (int k = 0; k < n_tiles; k++) {
        PyList_SET_ITEM(cx_list, k, PyFloat_FromDouble(cx_arr[k]));
        PyList_SET_ITEM(cy_list, k, PyFloat_FromDouble(cy_arr[k]));
    }
    free(cx_arr); free(cy_arr);

    return PyTuple_Pack(2, cx_list, cy_list);
}

/* -----------------------------------------------------------------------
 * Module table
 * ---------------------------------------------------------------------- */
static PyMethodDef CentroidMethods[] = {
    {"cog_batch",  py_cog_batch,  METH_VARARGS,
     "cog_batch(frame_bytes, n_sub, pix_per_sub, threshold_sigma)"
     " -> (cx_list, cy_list)\n"
     "Thresholded CoG centroiding.  frame_bytes: flat float32 frame."},
    {"wcog_batch", py_wcog_batch, METH_VARARGS,
     "wcog_batch(frame_bytes, n_sub, pix_per_sub, weight_fwhm_px)"
     " -> (cx_list, cy_list)\n"
     "Gaussian-weighted CoG centroiding (robust at low SNR)."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef centroidmodule = {
    PyModuleDef_HEAD_INIT, "centroid_cog", NULL, -1, CentroidMethods
};

PyMODINIT_FUNC
PyInit_centroid_cog(void) {
    return PyModule_Create(&centroidmodule);
}
