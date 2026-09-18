/* C-facing integration of the CUDA MatVec/iterative backend with ADDA globals. */
#ifndef ADDA_CUDAMATVEC_H
#define ADDA_CUDAMATVEC_H

#ifdef ADDA_CUDA

#include "timing.h"
#include "types.h"
#include <stdbool.h>

void CudaMatVecInit(void);
void CudaMatVecUpdateCC(void);
void CudaMatVecFree(void);

/* MatVec on registered CUDA-resident iterative vectors. Host pointer values
 * identify device allocations; no H2D/D2H vector copy is performed here. */
void MatVec_GPU(doublecomplex * restrict argvec,
                doublecomplex * restrict resultvec,
                double *inprod,
                bool her,
                TIME_TYPE *timing,
                TIME_TYPE *comm_timing);

/* Lanier CUDA preconditioner integration. The validated reference 1x/1.5x
 * modes remain BCGS2-only. LANIER_FULL, LANIER_NESTED, and additive
 * LANIER_MULTIZONE retain the all-solver integration. LANIER_MULTIZONE_SCHWARZ, LANIER_MULTIZONE_SCHWARZ_REVERSE, LANIER_MULTIZONE_SCHWARZ_SYM, and LANIER_MULTIZONE_SCHUR
 * V1 are restricted to the supported right-preconditioned nonsymmetric solvers.
 * All modes are available in the CUDA executable variants; slice/low-memory
 * physical MatVec modes still use independent full 3-D auxiliary FFTs for the
 * Lanier blocks. */
void CudaLanierInit(void);
void CudaLanierRelease(void);
/* LANIER_PARTITION_V2: restrict physical MatVec rows to one material.
 * Pass -1 to restore the complete physical operator. */
void CudaLanierPartitionProjection(int material_id);
void CudaLanierMatVec(doublecomplex * restrict argvec,
                      doublecomplex * restrict resultvec,
                      double *inprod,
                      bool her,
                      TIME_TYPE *timing,
                      TIME_TYPE *comm_timing);
/* Direct P application, used for residual-space conversion of the
 * complex-symmetric Krylov family. src==dst is supported. */
void CudaLanierApply(doublecomplex *src,doublecomplex *dst);
/* Congruence operator P*A*P. Since both A and P are complex symmetric, this
 * preserves the matrix symmetry required by BiCG_CS/CSYM/QMR_CS. */
void CudaLanierMatVecCongruence(doublecomplex * restrict argvec,
                                doublecomplex * restrict resultvec,
                                double *inprod,
                                TIME_TYPE *timing,
                                TIME_TYPE *comm_timing);
void CudaLanierAxpy(doublecomplex * restrict dst,
                    const doublecomplex * restrict src,
                    double complex alpha);

/* Resident-vector lifecycle for any current iterative solver. `method` is an
 * enum iter value, accepted as int here to keep this header independent of
 * const.h include ordering. */
void CudaIterInit(int method);
void CudaIterInitList(const void * const *host_ids,size_t count,const char *method_name);
void CudaIterPrintMemoryBeforeLoop(void);
void CudaIterSyncToHost(void);
void CudaIterUploadOne(const doublecomplex *host_id);
void CudaIterDownloadOne(doublecomplex *host_id);
void CudaIterRelease(void);

/* cuBLAS reductions. Names intentionally mirror linalg.c. */
double CudaIterNorm2(const doublecomplex * restrict a,TIME_TYPE *comm_timing);
double complex CudaIterDotProd(const doublecomplex * restrict a,
                              const doublecomplex * restrict b,
                              TIME_TYPE *comm_timing);
double complex CudaIterDotProd64(const doublecomplex * restrict a,
                                const doublecomplex * restrict b,
                                TIME_TYPE *comm_timing);
double complex CudaIterDotProd_conj(const doublecomplex * restrict a,
                                   const doublecomplex * restrict b,
                                   TIME_TYPE *comm_timing);
double complex CudaIterDotProdSelf_conj(const doublecomplex * restrict a,
                                       TIME_TYPE *comm_timing);
double complex CudaIterDotProdSelf_conj_Norm2(const doublecomplex * restrict a,
                                             double * restrict norm,
                                             TIME_TYPE *comm_timing);

/* CUDA vector primitives mirroring the subset of linalg.c used by the
 * iterative solvers. Optional inprod arguments return squared norms. */
void CudaIterCopy(doublecomplex * restrict a,const doublecomplex * restrict b);
void CudaIterIncrem(doublecomplex * restrict a,const doublecomplex * restrict b,
                    double * restrict inprod,TIME_TYPE *comm_timing);
void CudaIterIncrem01(doublecomplex * restrict a,const doublecomplex * restrict b,double c,
                      double * restrict inprod,TIME_TYPE *comm_timing);
void CudaIterIncrem10(doublecomplex * restrict a,const doublecomplex * restrict b,double c,
                      double * restrict inprod,TIME_TYPE *comm_timing);
void CudaIterIncrem01_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,double complex c,
                            double * restrict inprod,TIME_TYPE *comm_timing);
void CudaIterIncrem10_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,double complex c,
                            double * restrict inprod,TIME_TYPE *comm_timing);
void CudaIterIncrem011_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,
                             const doublecomplex * restrict c,double complex c1,double complex c2);
void CudaIterIncrem110_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,
                             const doublecomplex * restrict c,double complex c1,double complex c2);
void CudaIterIncrem111_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,
                             const doublecomplex * restrict c,double complex c1,double complex c2,double complex c3);
void CudaIterIncrem11_d_c(doublecomplex * restrict a,const doublecomplex * restrict b,double c1,double complex c2,
                          double * restrict inprod,TIME_TYPE *comm_timing);
void CudaIterIncrem110_d_c_conj(doublecomplex * restrict a,const doublecomplex * restrict b,
                                const doublecomplex * restrict c,double c1,double complex c2,
                                double * restrict inprod,TIME_TYPE *comm_timing);

void CudaIterLinComb_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,
                           const doublecomplex * restrict c,double complex c1,double complex c2,
                           double * restrict inprod,TIME_TYPE *comm_timing);
void CudaIterLinComb1_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,
                            const doublecomplex * restrict c,double complex c1,
                            double * restrict inprod,TIME_TYPE *comm_timing);
void CudaIterLinComb1_cmplx_conj(doublecomplex * restrict a,const doublecomplex * restrict b,
                                 const doublecomplex * restrict c,double complex c1,
                                 double * restrict inprod,TIME_TYPE *comm_timing);

void CudaIterMult(doublecomplex * restrict a,const doublecomplex * restrict b,double c);
void CudaIterMult_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,double complex c);
void CudaIterMultSelf(doublecomplex * restrict a,double c);
void CudaIterMultSelf_conj(doublecomplex * restrict a,double c);
void CudaIterMultSelf_cmplx(doublecomplex * restrict a,double complex c);

#endif /* ADDA_CUDA */

#endif /* ADDA_CUDAMATVEC_H */
