/*
 * c_ext/centroid_cog.c
 * ====================
 * Fast Centre-of-Gravity centroiding for SH-WFS spot tiles.
 * Compiled as a Python C extension via c_ext/setup.py.
 *
 * Exposed function:
 *   cog_batch(frame_bytes, n_sub, pix_per_sub, threshold_sigma) -> (cx_list, cy_list)
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdlib.h>

static void cog_single(
    const float *tile, int nrows, int ncols,
    double bg_sigma_thresh,
    double *cx_out, double *cy_out
) {
    double bg = 0.25 * (
        (double)tile[0] +
        (double)tile[ncols - 1] +
        (double)tile[(nrows - 1) * ncols] +
        (double)tile[nrows * ncols - 1]
    );

    double sum = 0.0, sum_x = 0.0, sum_y = 0.0;
    double thresh = bg_sigma_thresh * fabs(bg);
    for (int r = 0; r < nrows; r++) {
        for (int c = 0; c < ncols; c++) {
            double v = (double)tile[r * ncols + c] - bg;
            if (v < thresh) v = 0.0;
            if (v < 0.0)   v = 0.0;
            sum   += v;
            sum_x += v * c;
            sum_y += v * r;
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

static PyObject *
py_cog_batch(PyObject *self, PyObject *args)
{
    Py_buffer view;
    int n_sub, pps;
    double thresh;

    if (!PyArg_ParseTuple(args, "y*iid", &view, &n_sub, &pps, &thresh))
        return NULL;

    int total_w = n_sub * pps;
    int frame_size = total_w * total_w;

    if (view.len != (Py_ssize_t)(frame_size * sizeof(float))) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "frame_flat length mismatch");
        return NULL;
    }

    const float *frame = (const float *)view.buf;

    PyObject *cx_list = PyList_New(n_sub * n_sub);
    PyObject *cy_list = PyList_New(n_sub * n_sub);

    for (int i = 0; i < n_sub; i++) {
        for (int j = 0; j < n_sub; j++) {
            float *tile = (float *)malloc(pps * pps * sizeof(float));
            for (int r = 0; r < pps; r++) {
                int src_row = i * pps + r;
                for (int c = 0; c < pps; c++) {
                    int src_col = j * pps + c;
                    tile[r * pps + c] = frame[src_row * total_w + src_col];
                }
            }
            double cx, cy;
            cog_single(tile, pps, pps, thresh, &cx, &cy);
            free(tile);

            int idx = i * n_sub + j;
            PyList_SET_ITEM(cx_list, idx, PyFloat_FromDouble(cx));
            PyList_SET_ITEM(cy_list, idx, PyFloat_FromDouble(cy));
        }
    }

    PyBuffer_Release(&view);
    return PyTuple_Pack(2, cx_list, cy_list);
}

static PyMethodDef CentroidMethods[] = {
    {"cog_batch", py_cog_batch, METH_VARARGS,
     "cog_batch(frame_bytes, n_sub, pix_per_sub, threshold_sigma) -> (cx_list, cy_list)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef centroidmodule = {
    PyModuleDef_HEAD_INIT, "centroid_cog", NULL, -1, CentroidMethods
};

PyMODINIT_FUNC
PyInit_centroid_cog(void) {
    return PyModule_Create(&centroidmodule);
}
