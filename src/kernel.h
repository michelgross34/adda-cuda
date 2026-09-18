/* CUDA kernel-launch wrapper ABI for the split ADDA CUDA backend.
 * This is a private ABI between wrappermatvec_backend.cpp and kernel.cu.
 * The public ADDA ABI remains src/cudamatvec_backend.h.
 */
#ifndef ADDA_CUDA_KERNEL_H
#define ADDA_CUDA_KERNEL_H

#include <stddef.h>
#include <cuComplex.h>

typedef cuDoubleComplex AddaCudaKernelDoubleComplex;
#ifdef ADDA_CUDA_SINGLE_BACKEND
typedef cuFloatComplex AddaCudaKernelComplex;
typedef float AddaCudaKernelReal;
#else
typedef cuDoubleComplex AddaCudaKernelComplex;
typedef double AddaCudaKernelReal;
#endif

#ifdef __cplusplus
extern "C" {
#endif

void adda_kernel_scatterKernel(unsigned int dipBlocks, unsigned int threads, const AddaCudaKernelComplex *arg, AddaCudaKernelComplex *grid, const unsigned char *material, const unsigned short *position, const AddaCudaKernelComplex *cc, size_t ndip, size_t gridX, size_t gridY, size_t gridN, int her);
void adda_kernel_spectralMultiplyKernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *grid, const AddaCudaKernelComplex *gridR, const AddaCudaKernelComplex *D, const AddaCudaKernelComplex *R, size_t gridX, size_t gridY, size_t gridZ, size_t gridN, size_t DsizeY, size_t DsizeZ, size_t RsizeY, int reduced, int transposed, int surface);
void adda_kernel_gatherKernel(unsigned int dipBlocks, unsigned int threads, const AddaCudaKernelComplex *arg, AddaCudaKernelComplex *result, const AddaCudaKernelComplex *grid, const unsigned char *material, const unsigned short *position, const AddaCudaKernelComplex *cc, size_t ndip, size_t gridX, size_t gridY, size_t gridN, int her);
void adda_kernel_lanierBoxScatterKernel(unsigned int dipBlocks, unsigned int threads, const AddaCudaKernelComplex *arg, AddaCudaKernelComplex *grid, const unsigned short *position, size_t ndip,size_t nx,size_t ny,size_t n);
void adda_kernel_lanierBoxGatherKernel(unsigned int dipBlocks, unsigned int threads, const AddaCudaKernelComplex *grid, AddaCudaKernelComplex *out, const unsigned short *position, size_t ndip,size_t nx,size_t ny,size_t n);
void adda_kernel_lanierFullBoxScatterKernel(unsigned int dipBlocks, unsigned int threads, const AddaCudaKernelComplex *arg, AddaCudaKernelComplex *grid, const unsigned short *position, const unsigned char *material, const AddaCudaKernelComplex *cc, size_t ndip,size_t nx,size_t ny,size_t n,AddaCudaKernelReal sqrt_dipvol, int active_material,size_t x0,size_t y0,size_t z0);
void adda_kernel_lanierFullBoxGatherKernel(unsigned int dipBlocks, unsigned int threads, const AddaCudaKernelComplex *grid, AddaCudaKernelComplex *out, const unsigned short *position, const unsigned char *material, const AddaCudaKernelComplex *cc, size_t ndip,size_t nx,size_t ny,size_t n,AddaCudaKernelReal sqrt_dipvol, int active_material,size_t x0,size_t y0,size_t z0);
void adda_kernel_materialProjectionKernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *v,const unsigned char *material, size_t ndip,int active_material);
void adda_kernel_lanierReducedMultiplyKernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *grid, const AddaCudaKernelComplex *coef, size_t nx,size_t ny,size_t nz,size_t n, size_t rx,size_t ry,size_t rz,size_t nred);
void adda_kernel_scatterSliceKernel(unsigned int dipBlocks, unsigned int threads, const AddaCudaKernelComplex *arg, AddaCudaKernelComplex *slice, const unsigned char *material, const unsigned short *position, const AddaCudaKernelComplex *cc, size_t ndip, size_t boxX, size_t boxY, size_t sliceN, int her);
void adda_kernel_sliceToPlaneBatchKernel(unsigned int dipBlocks, unsigned int threads, const AddaCudaKernelComplex *slice, AddaCudaKernelComplex *plane, size_t kz0, size_t activeSlices, size_t boxX, size_t boxY, size_t gridX, size_t boxXY, size_t sliceN, size_t planeN);
void adda_kernel_spectralMultiplySliceBatchKernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *plane, const AddaCudaKernelComplex *D, size_t kz0, size_t activeSlices, size_t gridX, size_t gridY, size_t gridZ, size_t planeN, size_t DsizeX, size_t DsizeY, size_t DsizeZ, int reduced, int transposed, int lowMemGreen);
void adda_kernel_planeToSliceBatchKernel(unsigned int dipBlocks, unsigned int threads, const AddaCudaKernelComplex *plane, AddaCudaKernelComplex *slice, size_t kz0, size_t activeSlices, size_t boxX, size_t gridX, size_t boxXY, size_t sliceN, size_t planeN);
void adda_kernel_gatherSliceKernel(unsigned int dipBlocks, unsigned int threads, const AddaCudaKernelComplex *arg, AddaCudaKernelComplex *result, const AddaCudaKernelComplex *slice, const unsigned char *material, const unsigned short *position, const AddaCudaKernelComplex *cc, size_t ndip, size_t boxX, size_t boxY, size_t sliceN, int her);
void adda_kernel_qmrMultKernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *a, const AddaCudaKernelComplex *b, AddaCudaKernelComplex c, size_t n);
void adda_kernel_qmrMultSelfKernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *a, AddaCudaKernelComplex c, size_t n);
void adda_kernel_qmrLinComb1Kernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *a, const AddaCudaKernelComplex *b, const AddaCudaKernelComplex *c, AddaCudaKernelComplex c1, size_t n);
void adda_kernel_qmrLinCombKernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *a, const AddaCudaKernelComplex *b, const AddaCudaKernelComplex *c, AddaCudaKernelComplex c1, AddaCudaKernelComplex c2, size_t n);
void adda_kernel_qmrIncrem110Kernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *a, const AddaCudaKernelComplex *b, const AddaCudaKernelComplex *c, AddaCudaKernelComplex c1, AddaCudaKernelComplex c2, size_t n);
void adda_kernel_qmrIncrem111Kernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *a, const AddaCudaKernelComplex *b, const AddaCudaKernelComplex *c, AddaCudaKernelComplex c1, AddaCudaKernelComplex c2, AddaCudaKernelComplex c3, size_t n);
void adda_kernel_qmrIncrem01Kernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *a, const AddaCudaKernelComplex *b, AddaCudaKernelComplex c, size_t n);
void adda_kernel_qmrIncrem11DCKernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *a, const AddaCudaKernelComplex *b, AddaCudaKernelReal c1, AddaCudaKernelComplex c2, size_t n);
void adda_kernel_iterCopyKernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *a, const AddaCudaKernelComplex *b, size_t n);
void adda_kernel_iterIncrem10Kernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *a, const AddaCudaKernelComplex *b, AddaCudaKernelComplex c, size_t n);
void adda_kernel_iterIncrem011Kernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *a, const AddaCudaKernelComplex *b, const AddaCudaKernelComplex *c, AddaCudaKernelComplex c1, AddaCudaKernelComplex c2, size_t n);
void adda_kernel_iterLinComb1ConjKernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *a, const AddaCudaKernelComplex *b, const AddaCudaKernelComplex *c, AddaCudaKernelComplex c1, size_t n);
void adda_kernel_iterIncrem110DCConjKernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *a, const AddaCudaKernelComplex *b, const AddaCudaKernelComplex *c, AddaCudaKernelReal c1, AddaCudaKernelComplex c2, size_t n);
void adda_kernel_iterMultSelfConjKernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *a, AddaCudaKernelReal c, size_t n);
void adda_kernel_convertComplexF32ToF64Kernel(unsigned int dipBlocks, unsigned int threads, const cuFloatComplex *src, AddaCudaKernelDoubleComplex *dst, size_t n);
void adda_kernel_convertComplexPairF32ToF64Kernel(unsigned int dipBlocks, unsigned int threads, const cuFloatComplex *src_a, const cuFloatComplex *src_b, AddaCudaKernelDoubleComplex *dst_a, AddaCudaKernelDoubleComplex *dst_b, size_t n);

#ifdef __cplusplus
}
#endif

#endif
