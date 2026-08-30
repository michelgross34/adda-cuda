/* Low-level CUDA backend for ADDA MatVec and CUDA-resident iterative solvers.
 *
 * This header deliberately uses only C ABI/POD types. The main ADDA code uses
 * C99 double complex, while CUDA sources are compiled as C++. Complex vectors
 * are passed as opaque host-pointer identities; complex scalar values cross
 * the DLL boundary as separate real/imaginary doubles.
 */
#ifndef ADDA_CUDAMATVEC_BACKEND_H
#define ADDA_CUDAMATVEC_BACKEND_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

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

int adda_cuda_matvec_update_cc(const void *cc_sqrt, size_t complex_count);

/* CPU-facing MatVec: upload input, execute the GPU core, download output. */
int adda_cuda_matvec_execute(const void *argvec, void *resultvec, int her,
                             double *inprod, double *elapsed_ms);

/* CUDA-resident MatVec. Arguments are registered host-pointer identities.
 * No H2D/D2H vector copy is performed by this call. */
int adda_cuda_matvec_execute_gpu(const void *argvec_id, const void *resultvec_id,
                                 int her, double *inprod, double *elapsed_ms);

/* Register all vectors used by the selected iterative solver and upload their
 * current host values. vec1..vec4 may be NULL when that solver does not use
 * them; xvec/rvec/pvec/Avecbuffer are mandatory. */
int adda_cuda_iter_init(const void *xvec, const void *rvec, const void *pvec,
                        const void *vec1, const void *vec2, const void *vec3,
                        const void *vec4, const void *Avecbuffer);
int adda_cuda_iter_upload_all(void);
int adda_cuda_iter_download_all(void);

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
