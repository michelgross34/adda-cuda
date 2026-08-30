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
doublecomplex CudaIterDotProd(const doublecomplex * restrict a,
                              const doublecomplex * restrict b,
                              TIME_TYPE *comm_timing);
double complex CudaIterDotProd64(const doublecomplex * restrict a,
                                const doublecomplex * restrict b,
                                TIME_TYPE *comm_timing);
doublecomplex CudaIterDotProd_conj(const doublecomplex * restrict a,
                                   const doublecomplex * restrict b,
                                   TIME_TYPE *comm_timing);
doublecomplex CudaIterDotProdSelf_conj(const doublecomplex * restrict a,
                                       TIME_TYPE *comm_timing);
doublecomplex CudaIterDotProdSelf_conj_Norm2(const doublecomplex * restrict a,
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
void CudaIterIncrem01_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,doublecomplex c,
                            double * restrict inprod,TIME_TYPE *comm_timing);
void CudaIterIncrem10_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,doublecomplex c,
                            double * restrict inprod,TIME_TYPE *comm_timing);
void CudaIterIncrem011_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,
                             const doublecomplex * restrict c,doublecomplex c1,doublecomplex c2);
void CudaIterIncrem110_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,
                             const doublecomplex * restrict c,doublecomplex c1,doublecomplex c2);
void CudaIterIncrem111_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,
                             const doublecomplex * restrict c,doublecomplex c1,doublecomplex c2,doublecomplex c3);
void CudaIterIncrem11_d_c(doublecomplex * restrict a,const doublecomplex * restrict b,double c1,doublecomplex c2,
                          double * restrict inprod,TIME_TYPE *comm_timing);
void CudaIterIncrem110_d_c_conj(doublecomplex * restrict a,const doublecomplex * restrict b,
                                const doublecomplex * restrict c,double c1,doublecomplex c2,
                                double * restrict inprod,TIME_TYPE *comm_timing);

void CudaIterLinComb_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,
                           const doublecomplex * restrict c,doublecomplex c1,doublecomplex c2,
                           double * restrict inprod,TIME_TYPE *comm_timing);
void CudaIterLinComb1_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,
                            const doublecomplex * restrict c,doublecomplex c1,
                            double * restrict inprod,TIME_TYPE *comm_timing);
void CudaIterLinComb1_cmplx_conj(doublecomplex * restrict a,const doublecomplex * restrict b,
                                 const doublecomplex * restrict c,doublecomplex c1,
                                 double * restrict inprod,TIME_TYPE *comm_timing);

void CudaIterMult(doublecomplex * restrict a,const doublecomplex * restrict b,double c);
void CudaIterMult_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,doublecomplex c);
void CudaIterMultSelf(doublecomplex * restrict a,double c);
void CudaIterMultSelf_conj(doublecomplex * restrict a,double c);
void CudaIterMultSelf_cmplx(doublecomplex * restrict a,doublecomplex c);

#endif /* ADDA_CUDA */

#endif /* ADDA_CUDAMATVEC_H */
