/* Low-level CUDA backend for ADDA MatVec and CUDA-resident iterative solvers.
 *
 * This header deliberately uses only C ABI/POD types. The main ADDA code uses
 * C99 complex arrays (float or double, matching the selected backend), while
 * CUDA sources are compiled as C++. Complex vectors
 * are passed as opaque host-pointer identities; complex scalar values cross
 * the DLL boundary as separate real/imaginary doubles.
 */
#ifndef ADDA_CUDAMATVEC_BACKEND_H
#define ADDA_CUDAMATVEC_BACKEND_H

#include <stddef.h>

/* Number of kz slices processed together by adda_cuda_slice.  The C wrapper
 * uses the same value for memory reporting, while the CUDA backend creates one
 * 2-D cuFFT plan with 3*ADDA_CUDA_SLICE_BATCH transforms. */
#ifndef ADDA_CUDA_SLICE_BATCH
#define ADDA_CUDA_SLICE_BATCH 4
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Size, in bytes, of one real value in the compiled CUDA numerical path.
 * Used by the MinGW wrapper to reject accidental double/single DLL mismatch. */
int adda_cuda_backend_real_bytes(void);

typedef struct AddaCudaMatVecConfig {
    size_t gridX;
    size_t gridY;
    size_t gridZ;
    size_t DsizeY;
    size_t DsizeZ;
    size_t RsizeY;
    size_t ndip;
    size_t nrows;
    int surface;
    int reduced_fft;
    int device; /* < 0: use ADDA_CUDA_DEVICE or CUDA device 0 */
} AddaCudaMatVecConfig;

int adda_cuda_matvec_init(const AddaCudaMatVecConfig *cfg,
                          const void *Dmatrix,
                          const void *Rmatrix,
                          const unsigned char *material,
                          const unsigned short *position);

/* Slice backend initialization. Keeps the legacy AddaCudaMatVecConfig ABI intact
 * and supplies the physical non-zero-padded box separately. */
int adda_cuda_matvec_init_slice(const AddaCudaMatVecConfig *cfg,
                                size_t boxX, size_t boxY, size_t boxZ,
                                const void *Dmatrix,
                                const void *Rmatrix,
                                const unsigned char *material,
                                const unsigned short *position);

/* Low-memory slice backend. Stores only x=0..gridX/2 Green planes on GPU and
 * reconstructs the negative-kx octant by exact parity of the free-space Green
 * dyadic. Requires reduced_fft and does not support surface mode. */
int adda_cuda_matvec_init_low_mem(const AddaCudaMatVecConfig *cfg,
                                  size_t boxX, size_t boxY, size_t boxZ,
                                  const void *Dmatrix,
                                  const void *Rmatrix,
                                  const unsigned char *material,
                                  const unsigned short *position);

int adda_cuda_matvec_update_cc(const void *cc_sqrt, size_t complex_count);

typedef struct AddaCudaMemoryInfo {
    size_t device_total_bytes;
    size_t free_after_context_bytes;
    size_t free_after_libraries_bytes;
    size_t current_free_bytes;

    /* Exact live allocations owned by the ADDA CUDA backend. */
    size_t green_tensor_bytes;        /* Dmatrix */
    size_t surface_tensor_bytes;      /* Rmatrix */
    size_t fft_grid_bytes;            /* full-mode d_grid */
    size_t surface_fft_grid_bytes;    /* full-mode d_gridR */
    size_t slice_z_bytes;             /* slice-mode compact Z buffer */
    size_t slice_xy_bytes;            /* slice-mode batched XY buffer */
    size_t fft_workspace_bytes;       /* explicit shared cuFFT workspace (slice mode) */
    size_t reduction_scratch_bytes;   /* float32 backend: chunked FP64 cuBLAS reduction scratch */
    size_t matvec_vector_bytes;       /* d_arg + d_result */
    size_t iterative_vector_bytes;    /* solver-specific resident vectors */
    size_t cc_bytes;                  /* cc_sqrt */
    size_t material_bytes;            /* material */
    size_t position_bytes;            /* position */

    size_t explicit_current_bytes;
    size_t explicit_peak_bytes;
} AddaCudaMemoryInfo;

/* Return CUDA memory information.  explicit_* counts only allocations made
 * explicitly by the ADDA CUDA backend.  cudaMemGetInfo-based fields also
 * include CUDA context/library usage and any memory used by other processes. */
int adda_cuda_memory_info(AddaCudaMemoryInfo *info);

/* CPU-facing MatVec: upload input, execute the GPU core, download output. */
int adda_cuda_matvec_execute(const void *argvec, void *resultvec, int her,
                             double *inprod, double *elapsed_ms);

/* CUDA-resident MatVec. Arguments are registered host-pointer identities.
 * No H2D/D2H vector copy is performed by this call. */
int adda_cuda_matvec_execute_gpu(const void *argvec_id, const void *resultvec_id,
                                 int her, double *inprod, double *elapsed_ms);

/* Register vectors used by the selected iterative solver and upload current
 * host values. The legacy entry point covers vec1..vec7; the list entry point
 * supports memory-reduced higher-order solvers. In list mode host_ids[0:2]
 * must be xvec,rvec and are mapped onto the existing MatVec staging buffers. */
int adda_cuda_iter_init(const void *xvec, const void *rvec, const void *pvec,
                        const void *vec1, const void *vec2, const void *vec3,
                        const void *vec4, const void *vec5, const void *vec6,
                        const void *vec7, const void *Avecbuffer);
int adda_cuda_iter_init_list(const void * const *host_ids, size_t count);
int adda_cuda_iter_upload_all(void);
int adda_cuda_iter_download_all(void);
int adda_cuda_iter_upload_one(const void *id);
int adda_cuda_iter_download_one(const void *id);
/* Release only solver-specific resident vectors. MatVec staging buffers and all
 * static CUDA/FFT data remain allocated, so ordinary MatVec() stays usable. */
int adda_cuda_iter_release(void);

/* cuBLAS reductions. dotu = sum(a*b). dotc matches ADDA nDotProd(a,b),
 * i.e. sum(a*conj(b)). Norm functions return the squared Hermitian norm. */
int adda_cuda_iter_dotu(const void *a_id, const void *b_id,
                        double *out_re, double *out_im);
int adda_cuda_iter_dotc(const void *a_id, const void *b_id,
                        double *out_re, double *out_im);
int adda_cuda_iter_dotu_self_norm2(const void *a_id,
                                   double *out_re, double *out_im,
                                   double *norm2);
int adda_cuda_iter_norm2(const void *a_id, double *norm2);

/* Vector kernels. Optional norm2 pointers may be NULL. Complex coefficients
 * cross the ABI as real/imaginary pairs. */
int adda_cuda_iter_copy(const void *a_id, const void *b_id);
int adda_cuda_iter_mult(const void *a_id, const void *b_id,
                        double cr, double ci);
int adda_cuda_iter_mult_self(const void *a_id, double cr, double ci);
int adda_cuda_iter_mult_self_conj(const void *a_id, double c);

int adda_cuda_iter_lincomb1(const void *a_id, const void *b_id, const void *c_id,
                            double c1r, double c1i, double *norm2);
int adda_cuda_iter_lincomb(const void *a_id, const void *b_id, const void *c_id,
                           double c1r, double c1i, double c2r, double c2i,
                           double *norm2);
int adda_cuda_iter_lincomb1_conj(const void *a_id, const void *b_id, const void *c_id,
                                 double c1r, double c1i, double *norm2);

int adda_cuda_iter_increm01(const void *a_id, const void *b_id,
                            double cr, double ci, double *norm2);
int adda_cuda_iter_increm10(const void *a_id, const void *b_id,
                            double cr, double ci, double *norm2);
int adda_cuda_iter_increm011(const void *a_id, const void *b_id, const void *c_id,
                             double c1r, double c1i, double c2r, double c2i,
                             double *norm2);
int adda_cuda_iter_increm110(const void *a_id, const void *b_id, const void *c_id,
                             double c1r, double c1i, double c2r, double c2i);
int adda_cuda_iter_increm111(const void *a_id, const void *b_id, const void *c_id,
                             double c1r, double c1i, double c2r, double c2i,
                             double c3r, double c3i);
int adda_cuda_iter_increm11_d_c(const void *a_id, const void *b_id,
                                double c1, double c2r, double c2i,
                                double *norm2);
int adda_cuda_iter_increm110_d_c_conj(const void *a_id, const void *b_id, const void *c_id,
                                      double c1, double c2r, double c2i,
                                      double *norm2);

void adda_cuda_matvec_free(void);
const char *adda_cuda_matvec_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* ADDA_CUDAMATVEC_BACKEND_H */
