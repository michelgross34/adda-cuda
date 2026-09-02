/* CUDA implementation of ADDA's sequential FFT MatVec.
 *
 * Two execution modes share the same C ABI:
 *   - full mode: the existing three-component zero-padded 3-D grid and cuFFT 3-D plan;
 *   - slice mode: a compact [3][gridZ][boxY][boxX] buffer, a batched 1-D Z
 *     transform, and ADDA_CUDA_SLICE_BATCH zero-padded XY slices processed
 *     together.  With the default batch=4, one cuFFT call performs 12 2-D
 *     transforms (4 kz slices x 3 vector components).
 *
 * In slice mode the transforms are
 *     FFT_Z -> for kz batches { FFT2_XY -> D(k)* -> IFFT2_XY } -> IFFT_Z .
 *
 * 1-D and 2-D transforms commute, so this is the same unnormalised 3-D DFT as
 * the full plan. Dmatrix already contains ADDA's -1/Ngrid normalization.
 */

#include "cudamatvec_backend.h"

#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <cuComplex.h>

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <stdint.h>

/* The same CUDA source is compiled twice: once as the historical double
 * backend, and once with ADDA_CUDA_SINGLE_BACKEND for float32 data/FFT/BLAS.
 * The C ABI deliberately remains identical (scalar results cross it as
 * doubles), so MinGW C code needs no CUDA headers and both backends can share
 * cudamatvec.c. */
#ifdef ADDA_CUDA_SINGLE_BACKEND
/* Preserve access to the real FP64 cuBLAS entry points before the historical
 * token aliases below remap the main numerical data path to float32.  The
 * single-precision backend stores all large vectors as cuFloatComplex, but
 * scalar reductions are performed chunk-by-chunk on temporary
 * cuDoubleComplex buffers and accumulated on the CPU in double precision. */
typedef cuDoubleComplex AddaCudaDoubleComplex;
static inline cublasStatus_t addaCublasZdotu(cublasHandle_t h,int n,
                                              const AddaCudaDoubleComplex *x,int incx,
                                              const AddaCudaDoubleComplex *y,int incy,
                                              AddaCudaDoubleComplex *result)
{
    return cublasZdotu(h,n,x,incx,y,incy,result);
}
static inline cublasStatus_t addaCublasZdotc(cublasHandle_t h,int n,
                                              const AddaCudaDoubleComplex *x,int incx,
                                              const AddaCudaDoubleComplex *y,int incy,
                                              AddaCudaDoubleComplex *result)
{
    return cublasZdotc(h,n,x,incx,y,incy,result);
}
static inline cublasStatus_t addaCublasDznrm2(cublasHandle_t h,int n,
                                               const AddaCudaDoubleComplex *x,int incx,
                                               double *result)
{
    return cublasDznrm2(h,n,x,incx,result);
}

#  define cuDoubleComplex cuFloatComplex
#  define make_cuDoubleComplex make_cuFloatComplex
#  define cufftDoubleComplex cufftComplex
#  define cufftExecZ2Z cufftExecC2C
#  define CUFFT_Z2Z CUFFT_C2C
#  define cublasZdotu cublasCdotu
#  define cublasZdotc cublasCdotc
#  define cublasDznrm2 cublasScnrm2
typedef float AddaCudaReal;
#else
typedef cuDoubleComplex AddaCudaDoubleComplex;
typedef double AddaCudaReal;
#endif

namespace {

static_assert(sizeof(cuDoubleComplex) == 2 * sizeof(AddaCudaReal),
              "CUDA complex type must be two packed scalar values");

struct Context {
    bool initialized = false;
    AddaCudaMatVecConfig cfg{};
    bool slice_fft = false;
    bool low_mem_green = false; /* store only 0..Nx/2 Green tensor planes on GPU */
    size_t boxX = 0, boxY = 0, boxZ = 0;
    size_t DsizeX = 0;
    size_t gridYZ = 0;
    size_t gridN = 0;
    size_t gridComplexCount = 0;
    size_t boxXY = 0;
    size_t sliceN = 0;
    size_t sliceComplexCount = 0;
    size_t planeN = 0;
    size_t planeComplexCount = 0;
    size_t dComplexCount = 0;
    size_t rComplexCount = 0;

    cuDoubleComplex *d_D = nullptr;
    cuDoubleComplex *d_R = nullptr;
    cuDoubleComplex *d_cc = nullptr;
    size_t ccCapacity = 0;
    unsigned char *d_material = nullptr;
    unsigned short *d_position = nullptr;
    cuDoubleComplex *d_arg = nullptr;
    cuDoubleComplex *d_result = nullptr;
    cuDoubleComplex *d_grid = nullptr;
    cuDoubleComplex *d_gridR = nullptr;
    cuDoubleComplex *d_slice = nullptr; /* [3][gridZ][boxY][boxX] */
    cuDoubleComplex *d_plane = nullptr; /* [batch][3][gridY][gridX] */
#ifdef ADDA_CUDA_SINGLE_BACKEND
    /* FP64 scratch used only for mixed-precision scalar reductions.
     * Large Krylov vectors remain float32. */
    AddaCudaDoubleComplex *d_reduce_a = nullptr;
    AddaCudaDoubleComplex *d_reduce_b = nullptr;
    size_t reduceCapacity = 0;
#endif
    void *d_fftWork = nullptr;          /* slice mode: shared Z/XY cuFFT workspace */
    size_t fftWorkSize = 0;

    /* Memory accounting. cudaMemGetInfo snapshots are device-wide; the
     * explicit counters describe only allocations owned by this backend. */
    size_t deviceTotalBytes = 0;
    size_t freeAfterContextBytes = 0;
    size_t freeAfterLibrariesBytes = 0;
    size_t explicitPeakBytes = 0;

    cufftHandle plan3d = 0; /* three component-major full 3-D transforms */
    cufftHandle planXY = 0; /* surface: 2-D XY transform for every z/component */
    cufftHandle planZ = 0;  /* surface: Z transform, executed once per component */
    cudaEvent_t matvec_start = nullptr;
    cudaEvent_t matvec_stop = nullptr;
    cublasHandle_t cublas = nullptr;

    /* CUDA-resident iterative-solver vectors. Host addresses are stable
     * identities. d_arg and d_result are reused for xvec and rvec; only the
     * remaining registered vectors require extra allocations. GPBiCGStab(2)
     * needs x/r/p + vec1..vec7 + Avecbuffer = 11 device vectors. */
    static const int ITER_VECTOR_MAX = 32;
    const void *iter_host[ITER_VECTOR_MAX] = {};
    cuDoubleComplex *iter_device[ITER_VECTOR_MAX] = {};
    size_t iter_count = 0;
    cuDoubleComplex *d_iter_extra = nullptr;
    size_t iter_extra_capacity = 0; /* number of complete nrows vectors */
    bool iter_initialized = false;
};

Context g;
char g_error[512] = {0};

__host__ __device__ inline cuDoubleComplex cmul(const cuDoubleComplex a, const cuDoubleComplex b)
{
    return make_cuDoubleComplex(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

__host__ __device__ inline cuDoubleComplex cadd_hd(const cuDoubleComplex a, const cuDoubleComplex b)
{
    return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
}

__host__ __device__ inline cuDoubleComplex cneg_hd(const cuDoubleComplex a)
{
    return make_cuDoubleComplex(-a.x, -a.y);
}

__host__ __device__ inline cuDoubleComplex cconj_hd(const cuDoubleComplex a)
{
    return make_cuDoubleComplex(a.x, -a.y);
}

void setError(const char *msg)
{
    std::snprintf(g_error,sizeof(g_error),"%s",msg != nullptr ? msg : "unknown CUDA MatVec error");
}

void setDetailedError(const char *what,const char *detail)
{
    std::snprintf(g_error,sizeof(g_error),"%s: %s",what,detail);
}

int failCuda(cudaError_t err, const char *what)
{
    if (err == cudaSuccess) return 0;
    setDetailedError(what,cudaGetErrorString(err));
    return -1;
}

const char *cufftErrorString(cufftResult err)
{
    switch (err) {
        case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
#if defined(CUFFT_INCOMPLETE_PARAMETER_LIST)
        case CUFFT_INCOMPLETE_PARAMETER_LIST: return "CUFFT_INCOMPLETE_PARAMETER_LIST";
#endif
#if defined(CUFFT_INVALID_DEVICE)
        case CUFFT_INVALID_DEVICE: return "CUFFT_INVALID_DEVICE";
#endif
#if defined(CUFFT_PARSE_ERROR)
        case CUFFT_PARSE_ERROR: return "CUFFT_PARSE_ERROR";
#endif
#if defined(CUFFT_NO_WORKSPACE)
        case CUFFT_NO_WORKSPACE: return "CUFFT_NO_WORKSPACE";
#endif
#if defined(CUFFT_NOT_IMPLEMENTED)
        case CUFFT_NOT_IMPLEMENTED: return "CUFFT_NOT_IMPLEMENTED";
#endif
#if defined(CUFFT_LICENSE_ERROR)
        case CUFFT_LICENSE_ERROR: return "CUFFT_LICENSE_ERROR";
#endif
#if defined(CUFFT_NOT_SUPPORTED)
        case CUFFT_NOT_SUPPORTED: return "CUFFT_NOT_SUPPORTED";
#endif
        default: return "unknown cuFFT error";
    }
}

int failCufft(cufftResult err, const char *what)
{
    if (err == CUFFT_SUCCESS) return 0;
    setDetailedError(what,cufftErrorString(err));
    return -1;
}

const char *cublasErrorString(cublasStatus_t err)
{
    switch (err) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
#if defined(CUBLAS_STATUS_NOT_SUPPORTED)
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if defined(CUBLAS_STATUS_LICENSE_ERROR)
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
        default: return "unknown cuBLAS error";
    }
}

int failCublas(cublasStatus_t err, const char *what)
{
    if (err == CUBLAS_STATUS_SUCCESS) return 0;
    setDetailedError(what,cublasErrorString(err));
    return -1;
}

bool checkedMul(size_t a, size_t b, size_t &out)
{
    if (a != 0 && b > SIZE_MAX / a) return false;
    out = a * b;
    return true;
}

bool checkedAddBytes(size_t a, size_t b, size_t &out)
{
    if (b > SIZE_MAX-a) return false;
    out=a+b;
    return true;
}

size_t explicitAllocationBytes()
{
    size_t total=0, bytes=0;
#define ADD_BYTES_IF(ptr,count,type) \
    do { \
        if ((ptr) != nullptr) { \
            if (!checkedMul((count),sizeof(type),bytes) || !checkedAddBytes(total,bytes,total)) return SIZE_MAX; \
        } \
    } while (0)
    ADD_BYTES_IF(g.d_D,g.dComplexCount,cuDoubleComplex);
    ADD_BYTES_IF(g.d_R,g.rComplexCount,cuDoubleComplex);
    ADD_BYTES_IF(g.d_cc,g.ccCapacity,cuDoubleComplex);
    ADD_BYTES_IF(g.d_material,g.cfg.ndip,unsigned char);
    ADD_BYTES_IF(g.d_position,g.cfg.nrows,unsigned short);
    ADD_BYTES_IF(g.d_arg,g.cfg.nrows,cuDoubleComplex);
    ADD_BYTES_IF(g.d_result,g.cfg.nrows,cuDoubleComplex);
    ADD_BYTES_IF(g.d_grid,g.gridComplexCount,cuDoubleComplex);
    ADD_BYTES_IF(g.d_gridR,g.gridComplexCount,cuDoubleComplex);
    ADD_BYTES_IF(g.d_slice,g.sliceComplexCount,cuDoubleComplex);
    ADD_BYTES_IF(g.d_plane,g.planeComplexCount,cuDoubleComplex);
#ifdef ADDA_CUDA_SINGLE_BACKEND
    ADD_BYTES_IF(g.d_reduce_a,g.reduceCapacity,AddaCudaDoubleComplex);
    ADD_BYTES_IF(g.d_reduce_b,g.reduceCapacity,AddaCudaDoubleComplex);
#endif
    if (g.d_fftWork != nullptr) {
        if (!checkedAddBytes(total,g.fftWorkSize,total)) return SIZE_MAX;
    }
    if (g.d_iter_extra != nullptr) {
        size_t vectors=0;
        if (!checkedMul(g.iter_extra_capacity,g.cfg.nrows,vectors) ||
            !checkedMul(vectors,sizeof(cuDoubleComplex),bytes) ||
            !checkedAddBytes(total,bytes,total)) return SIZE_MAX;
    }
#undef ADD_BYTES_IF
    return total;
}

void updateExplicitPeak()
{
    const size_t current=explicitAllocationBytes();
    if (current != SIZE_MAX && current > g.explicitPeakBytes) g.explicitPeakBytes=current;
}

int toInt(size_t x, const char *name)
{
    if (x > static_cast<size_t>(INT_MAX)) {
        std::snprintf(g_error,sizeof(g_error),"cuFFT dimension exceeds INT_MAX: %s",name);
        return -1;
    }
    return static_cast<int>(x);
}

int requestedDevice(int configured)
{
    if (configured >= 0) return configured;
    const char *env = std::getenv("ADDA_CUDA_DEVICE");
    if (env == nullptr || *env == '\0') return 0;
    errno = 0;
    char *end = nullptr;
    long v = std::strtol(env, &end, 10);
    if (errno != 0 || end == env || *end != '\0' || v < 0 || v > INT_MAX) {
        setError("ADDA_CUDA_DEVICE must be a non-negative integer");
        return -1;
    }
    return static_cast<int>(v);
}

void freeContext()
{
    if (g.cublas) { cublasDestroy(g.cublas); g.cublas = nullptr; }
    if (g.matvec_stop) { cudaEventDestroy(g.matvec_stop); g.matvec_stop = nullptr; }
    if (g.matvec_start) { cudaEventDestroy(g.matvec_start); g.matvec_start = nullptr; }
    if (g.planZ) { cufftDestroy(g.planZ); g.planZ = 0; }
    if (g.planXY) { cufftDestroy(g.planXY); g.planXY = 0; }
    if (g.plan3d) { cufftDestroy(g.plan3d); g.plan3d = 0; }

    /* Do not call cudaFree(NULL) here. On CUDA 11.8 a null cudaFree is the
     * conventional way to force runtime/context initialization; doing that in
     * freeContext() before cudaSetDevice() would initialize the wrong/default
     * device. Only free allocations that actually exist. */
    if (g.d_iter_extra) { cudaFree(g.d_iter_extra); g.d_iter_extra = nullptr; }
#ifdef ADDA_CUDA_SINGLE_BACKEND
    if (g.d_reduce_b) { cudaFree(g.d_reduce_b); g.d_reduce_b = nullptr; }
    if (g.d_reduce_a) { cudaFree(g.d_reduce_a); g.d_reduce_a = nullptr; }
#endif
    if (g.d_fftWork) { cudaFree(g.d_fftWork); g.d_fftWork = nullptr; }
    if (g.d_plane) { cudaFree(g.d_plane); g.d_plane = nullptr; }
    if (g.d_slice) { cudaFree(g.d_slice); g.d_slice = nullptr; }
    if (g.d_gridR) { cudaFree(g.d_gridR); g.d_gridR = nullptr; }
    if (g.d_grid) { cudaFree(g.d_grid); g.d_grid = nullptr; }
    if (g.d_result) { cudaFree(g.d_result); g.d_result = nullptr; }
    if (g.d_arg) { cudaFree(g.d_arg); g.d_arg = nullptr; }
    if (g.d_position) { cudaFree(g.d_position); g.d_position = nullptr; }
    if (g.d_material) { cudaFree(g.d_material); g.d_material = nullptr; }
    if (g.d_cc) { cudaFree(g.d_cc); g.d_cc = nullptr; }
    if (g.d_R) { cudaFree(g.d_R); g.d_R = nullptr; }
    if (g.d_D) { cudaFree(g.d_D); g.d_D = nullptr; }

    g = Context{};
}

cuDoubleComplex *iterDeviceVector(const void *host_id)
{
    if (!g.iter_initialized || host_id == nullptr) {
        setError("CUDA iterative-solver vectors are not initialized");
        return nullptr;
    }
    for (size_t i=0; i<g.iter_count; ++i) {
        if (g.iter_host[i] != nullptr && g.iter_host[i] == host_id) return g.iter_device[i];
    }
    setError("unknown host vector passed to CUDA iterative backend");
    return nullptr;
}

__global__ void scatterKernel(const cuDoubleComplex *arg,
                              cuDoubleComplex *grid,
                              const unsigned char *material,
                              const unsigned short *position,
                              const cuDoubleComplex *cc,
                              size_t ndip,
                              size_t gridX,
                              size_t gridY,
                              size_t gridN,
                              int her)
{
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i >= ndip) return;

    const size_t p = 3 * i;
    const size_t x = position[p];
    const size_t y = position[p + 1];
    const size_t z = position[p + 2];
    const size_t index = (z * gridY + y) * gridX + x;
    const size_t mat = material[i];

#pragma unroll
    for (int c = 0; c < 3; ++c) {
        cuDoubleComplex a = arg[p + static_cast<size_t>(c)];
        if (her) a = cconj_hd(a);
        grid[static_cast<size_t>(c) * gridN + index] =
            cmul(cc[3 * mat + static_cast<size_t>(c)], a);
    }
}

__device__ inline void symMatVec(const cuDoubleComplex f[6],
                                 const cuDoubleComplex x[3],
                                 cuDoubleComplex y[3])
{
    y[0] = cadd_hd(cadd_hd(cmul(f[0],x[0]), cmul(f[1],x[1])), cmul(f[2],x[2]));
    y[1] = cadd_hd(cadd_hd(cmul(f[1],x[0]), cmul(f[3],x[1])), cmul(f[4],x[2]));
    y[2] = cadd_hd(cadd_hd(cmul(f[2],x[0]), cmul(f[4],x[1])), cmul(f[5],x[2]));
}

__device__ inline void reflMatVec(const cuDoubleComplex f[6],
                                  const cuDoubleComplex x[3],
                                  cuDoubleComplex y[3])
{
    y[0] = cadd_hd(cadd_hd(cmul(f[0],x[0]), cmul(f[1],x[1])), cmul(f[2],x[2]));
    y[1] = cadd_hd(cadd_hd(cmul(f[1],x[0]), cmul(f[3],x[1])), cmul(f[4],x[2]));
    y[2] = cadd_hd(cadd_hd(cneg_hd(cmul(f[2],x[0])), cneg_hd(cmul(f[4],x[1]))), cmul(f[5],x[2]));
}

__global__ void spectralMultiplyKernel(cuDoubleComplex *grid,
                                       const cuDoubleComplex *gridR,
                                       const cuDoubleComplex *D,
                                       const cuDoubleComplex *R,
                                       size_t gridX,
                                       size_t gridY,
                                       size_t gridZ,
                                       size_t gridN,
                                       size_t DsizeY,
                                       size_t DsizeZ,
                                       size_t RsizeY,
                                       int reduced,
                                       int transposed,
                                       int surface)
{
    const size_t index = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (index >= gridN) return;

    const size_t x0 = index % gridX;
    const size_t yz = index / gridX;
    const size_t y0 = yz % gridY;
    const size_t z0 = yz / gridY;

    cuDoubleComplex xv[3], yv[3], f[6];
#pragma unroll
    for (int c=0; c<3; ++c) xv[c] = grid[static_cast<size_t>(c)*gridN + index];

    size_t x=x0, y=y0, z=z0;
    if (transposed) {
        if (x>0) x=gridX-x;
        if (y>0) y=gridY-y;
        if (z>0) z=gridZ-z;
    }
    else {
        if (y>=DsizeY) y=gridY-y;
        if (z>=DsizeZ) z=gridZ-z;
    }
    const size_t dbase = 6 * ((x*DsizeZ + z)*DsizeY + y);
#pragma unroll
    for (int k=0; k<6; ++k) f[k]=D[dbase + static_cast<size_t>(k)];

    if (reduced) {
        if (y0>=DsizeY) {
            f[1]=cneg_hd(f[1]);
            if (z0>=DsizeZ) f[2]=cneg_hd(f[2]);
            else f[4]=cneg_hd(f[4]);
        }
        else if (z0>=DsizeZ) {
            f[2]=cneg_hd(f[2]);
            f[4]=cneg_hd(f[4]);
        }
    }
    symMatVec(f,xv,yv);

    if (surface) {
        cuDoubleComplex xr[3], yr[3];
#pragma unroll
        for (int c=0; c<3; ++c) xr[c] = gridR[static_cast<size_t>(c)*gridN + index];

        x=x0; y=y0; z=z0;
        if (transposed) {
            if (x>0) x=gridX-x;
            if (y>0) y=gridY-y;
        }
        else if (y>=RsizeY) y=gridY-y;

        const size_t rbase = 6 * ((x*gridZ + z)*RsizeY + y);
#pragma unroll
        for (int k=0; k<6; ++k) f[k]=R[rbase + static_cast<size_t>(k)];
        if (reduced && y0>=RsizeY) {
            f[1]=cneg_hd(f[1]);
            f[4]=cneg_hd(f[4]);
        }
        if (transposed) {
            f[2]=cneg_hd(f[2]);
            f[4]=cneg_hd(f[4]);
        }
        reflMatVec(f,xr,yr);
#pragma unroll
        for (int c=0; c<3; ++c) yv[c]=cadd_hd(yv[c],yr[c]);
    }

#pragma unroll
    for (int c=0; c<3; ++c) grid[static_cast<size_t>(c)*gridN + index]=yv[c];
}

__global__ void gatherKernel(const cuDoubleComplex *arg,
                             cuDoubleComplex *result,
                             const cuDoubleComplex *grid,
                             const unsigned char *material,
                             const unsigned short *position,
                             const cuDoubleComplex *cc,
                             size_t ndip,
                             size_t gridX,
                             size_t gridY,
                             size_t gridN,
                             int her)
{
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i >= ndip) return;

    const size_t p = 3 * i;
    const size_t x = position[p];
    const size_t y = position[p + 1];
    const size_t z = position[p + 2];
    const size_t index = (z * gridY + y) * gridX + x;
    const size_t mat = material[i];

#pragma unroll
    for (int c=0; c<3; ++c) {
        cuDoubleComplex a = arg[p + static_cast<size_t>(c)];
        if (her) a = cconj_hd(a);
        cuDoubleComplex r = cadd_hd(a,
            cmul(cc[3*mat + static_cast<size_t>(c)],
                 grid[static_cast<size_t>(c)*gridN + index]));
        if (her) r = cconj_hd(r);
        result[p + static_cast<size_t>(c)] = r;
    }
}



/* --------------------------- Slice FFT MatVec --------------------------- */
__global__ void scatterSliceKernel(const cuDoubleComplex *arg, cuDoubleComplex *slice,
                                   const unsigned char *material, const unsigned short *position,
                                   const cuDoubleComplex *cc, size_t ndip, size_t boxX, size_t boxY,
                                   size_t sliceN, int her)
{
    const size_t i=blockIdx.x*static_cast<size_t>(blockDim.x)+threadIdx.x;
    if (i>=ndip) return;
    const size_t p=3*i, x=position[p], y=position[p+1], z=position[p+2];
    const size_t index=z*(boxX*boxY)+y*boxX+x, mat=material[i];
#pragma unroll
    for (int c=0;c<3;++c) {
        cuDoubleComplex a=arg[p+static_cast<size_t>(c)];
        if (her) a=cconj_hd(a);
        slice[static_cast<size_t>(c)*sliceN+index]=cmul(cc[3*mat+static_cast<size_t>(c)],a);
    }
}

__global__ void sliceToPlaneBatchKernel(const cuDoubleComplex *slice, cuDoubleComplex *plane,
                                        size_t kz0, size_t activeSlices,
                                        size_t boxX, size_t boxY, size_t gridX,
                                        size_t boxXY, size_t sliceN, size_t planeN)
{
    const size_t i=blockIdx.x*static_cast<size_t>(blockDim.x)+threadIdx.x;
    const size_t batchPlaneN=static_cast<size_t>(ADDA_CUDA_SLICE_BATCH)*planeN;
    if (i>=batchPlaneN) return;
    const size_t b=i/planeN;
    const size_t local=i-b*planeN;
    const size_t x=local%gridX, y=local/gridX;
    const bool valid=(b<activeSlices && x<boxX && y<boxY);
    const size_t src=valid ? (kz0+b)*boxXY+y*boxX+x : 0;
#pragma unroll
    for (int c=0;c<3;++c) {
        const size_t dst=(b*3+static_cast<size_t>(c))*planeN+local;
        plane[dst]=valid ? slice[static_cast<size_t>(c)*sliceN+src]
                         : make_cuDoubleComplex(0.0,0.0);
    }
}

__global__ void spectralMultiplySliceBatchKernel(cuDoubleComplex *plane, const cuDoubleComplex *D,
                                                 size_t kz0, size_t activeSlices,
                                                 size_t gridX, size_t gridY, size_t gridZ,
                                                 size_t planeN, size_t DsizeX, size_t DsizeY, size_t DsizeZ,
                                                 int reduced, int transposed, int lowMemGreen)
{
    const size_t index=blockIdx.x*static_cast<size_t>(blockDim.x)+threadIdx.x;
    const size_t activeN=activeSlices*planeN;
    if (index>=activeN) return;
    const size_t b=index/planeN;
    const size_t local=index-b*planeN;
    const size_t x0=local%gridX, y0=local/gridX, z0=kz0+b;
    cuDoubleComplex xv[3],yv[3],f[6];
#pragma unroll
    for (int c=0;c<3;++c)
        xv[c]=plane[(b*3+static_cast<size_t>(c))*planeN+local];
    size_t x=x0,y=y0,z=z0;
    bool rx=false,ry=false,rz=false;
    if (transposed) {
        if (x>0) x=gridX-x;
        if (y>0) y=gridY-y;
        if (z>0) z=gridZ-z;
    } else {
        if (lowMemGreen && x>=DsizeX) { x=gridX-x; rx=true; }
        if (y>=DsizeY) { y=gridY-y; ry=true; }
        if (z>=DsizeZ) { z=gridZ-z; rz=true; }
    }
    const size_t dbase=6*((x*DsizeZ+z)*DsizeY+y);
#pragma unroll
    for (int j=0;j<6;++j) f[j]=D[dbase+static_cast<size_t>(j)];
    if (reduced) {
        /* G=A I+B rr is even on diagonal and changes sign on an off-diagonal
         * component iff exactly one of its coordinate axes is reflected.
         * Component order is xx,xy,xz,yy,yz,zz. */
        if (rx != ry) f[1]=cneg_hd(f[1]);
        if (rx != rz) f[2]=cneg_hd(f[2]);
        if (ry != rz) f[4]=cneg_hd(f[4]);
    }
    symMatVec(f,xv,yv);
#pragma unroll
    for (int c=0;c<3;++c)
        plane[(b*3+static_cast<size_t>(c))*planeN+local]=yv[c];
}

__global__ void planeToSliceBatchKernel(const cuDoubleComplex *plane, cuDoubleComplex *slice,
                                        size_t kz0, size_t activeSlices,
                                        size_t boxX, size_t gridX, size_t boxXY,
                                        size_t sliceN, size_t planeN)
{
    const size_t i=blockIdx.x*static_cast<size_t>(blockDim.x)+threadIdx.x;
    const size_t activeN=activeSlices*boxXY;
    if (i>=activeN) return;
    const size_t b=i/boxXY;
    const size_t local=i-b*boxXY;
    const size_t x=local%boxX, y=local/boxX;
    const size_t src=y*gridX+x, dst=(kz0+b)*boxXY+local;
#pragma unroll
    for (int c=0;c<3;++c)
        slice[static_cast<size_t>(c)*sliceN+dst]=plane[(b*3+static_cast<size_t>(c))*planeN+src];
}

__global__ void gatherSliceKernel(const cuDoubleComplex *arg, cuDoubleComplex *result,
                                  const cuDoubleComplex *slice, const unsigned char *material,
                                  const unsigned short *position, const cuDoubleComplex *cc,
                                  size_t ndip, size_t boxX, size_t boxY, size_t sliceN, int her)
{
    const size_t i=blockIdx.x*static_cast<size_t>(blockDim.x)+threadIdx.x;
    if (i>=ndip) return;
    const size_t p=3*i, x=position[p], y=position[p+1], z=position[p+2];
    const size_t index=z*(boxX*boxY)+y*boxX+x, mat=material[i];
#pragma unroll
    for (int c=0;c<3;++c) {
        cuDoubleComplex a=arg[p+static_cast<size_t>(c)];
        if (her) a=cconj_hd(a);
        cuDoubleComplex r=cadd_hd(a,cmul(cc[3*mat+static_cast<size_t>(c)],slice[static_cast<size_t>(c)*sliceN+index]));
        if (her) r=cconj_hd(r);
        result[p+static_cast<size_t>(c)]=r;
    }
}

__global__ void qmrMultKernel(cuDoubleComplex *a, const cuDoubleComplex *b,
                              cuDoubleComplex c, size_t n)
{
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < n) a[i] = cmul(c,b[i]);
}

__global__ void qmrMultSelfKernel(cuDoubleComplex *a, cuDoubleComplex c, size_t n)
{
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < n) a[i] = cmul(c,a[i]);
}

__global__ void qmrLinComb1Kernel(cuDoubleComplex *a, const cuDoubleComplex *b,
                                  const cuDoubleComplex *c, cuDoubleComplex c1, size_t n)
{
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < n) a[i] = cadd_hd(cmul(c1,b[i]),c[i]);
}

__global__ void qmrLinCombKernel(cuDoubleComplex *a, const cuDoubleComplex *b,
                                 const cuDoubleComplex *c, cuDoubleComplex c1,
                                 cuDoubleComplex c2, size_t n)
{
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < n) a[i] = cadd_hd(cmul(c1,b[i]),cmul(c2,c[i]));
}

__global__ void qmrIncrem110Kernel(cuDoubleComplex *a, const cuDoubleComplex *b,
                                   const cuDoubleComplex *c, cuDoubleComplex c1,
                                   cuDoubleComplex c2, size_t n)
{
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < n) a[i] = cadd_hd(cadd_hd(cmul(c1,a[i]),cmul(c2,b[i])),c[i]);
}

__global__ void qmrIncrem111Kernel(cuDoubleComplex *a, const cuDoubleComplex *b,
                                   const cuDoubleComplex *c, cuDoubleComplex c1,
                                   cuDoubleComplex c2, cuDoubleComplex c3, size_t n)
{
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < n) a[i] = cadd_hd(cadd_hd(cmul(c1,a[i]),cmul(c2,b[i])),cmul(c3,c[i]));
}

__global__ void qmrIncrem01Kernel(cuDoubleComplex *a, const cuDoubleComplex *b,
                                  cuDoubleComplex c, size_t n)
{
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < n) a[i] = cadd_hd(a[i],cmul(c,b[i]));
}

__global__ void qmrIncrem11DCKernel(cuDoubleComplex *a, const cuDoubleComplex *b,
                                    AddaCudaReal c1, cuDoubleComplex c2, size_t n)
{
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < n) {
        const cuDoubleComplex r1 = make_cuDoubleComplex(c1*a[i].x,c1*a[i].y);
        a[i] = cadd_hd(r1,cmul(c2,b[i]));
    }
}

__global__ void iterCopyKernel(cuDoubleComplex *a, const cuDoubleComplex *b, size_t n)
{
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < n) a[i]=b[i];
}

__global__ void iterIncrem10Kernel(cuDoubleComplex *a, const cuDoubleComplex *b,
                                   cuDoubleComplex c, size_t n)
{
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < n) a[i]=cadd_hd(cmul(c,a[i]),b[i]);
}

__global__ void iterIncrem011Kernel(cuDoubleComplex *a, const cuDoubleComplex *b,
                                    const cuDoubleComplex *c, cuDoubleComplex c1,
                                    cuDoubleComplex c2, size_t n)
{
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < n) a[i]=cadd_hd(a[i],cadd_hd(cmul(c1,b[i]),cmul(c2,c[i])));
}

__global__ void iterLinComb1ConjKernel(cuDoubleComplex *a, const cuDoubleComplex *b,
                                       const cuDoubleComplex *c, cuDoubleComplex c1, size_t n)
{
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < n) a[i]=cadd_hd(cmul(c1,cconj_hd(b[i])),c[i]);
}

__global__ void iterIncrem110DCConjKernel(cuDoubleComplex *a, const cuDoubleComplex *b,
                                          const cuDoubleComplex *c, AddaCudaReal c1,
                                          cuDoubleComplex c2, size_t n)
{
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < n) {
        const cuDoubleComplex ca=cconj_hd(a[i]);
        const cuDoubleComplex r1=make_cuDoubleComplex(c1*ca.x,c1*ca.y);
        a[i]=cadd_hd(cadd_hd(r1,cmul(c2,cconj_hd(b[i]))),c[i]);
    }
}

__global__ void iterMultSelfConjKernel(cuDoubleComplex *a, AddaCudaReal c, size_t n)
{
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < n) a[i]=make_cuDoubleComplex(c*a[i].x,-c*a[i].y);
}

int launchChecked(const char *name)
{
    return failCuda(cudaGetLastError(),name);
}

#ifdef ADDA_CUDA_SINGLE_BACKEND
static const size_t ADDA_FP64_REDUCTION_CHUNK = static_cast<size_t>(1) << 20; /* 1,048,576 */

__global__ void convertComplexF32ToF64Kernel(const cuFloatComplex *src,
                                             AddaCudaDoubleComplex *dst,
                                             size_t n)
{
    const size_t i=blockIdx.x*static_cast<size_t>(blockDim.x)+threadIdx.x;
    if (i<n) {
        dst[i].x=static_cast<double>(src[i].x);
        dst[i].y=static_cast<double>(src[i].y);
    }
}

__global__ void convertComplexPairF32ToF64Kernel(const cuFloatComplex *src_a,
                                                 const cuFloatComplex *src_b,
                                                 AddaCudaDoubleComplex *dst_a,
                                                 AddaCudaDoubleComplex *dst_b,
                                                 size_t n)
{
    const size_t i=blockIdx.x*static_cast<size_t>(blockDim.x)+threadIdx.x;
    if (i<n) {
        dst_a[i].x=static_cast<double>(src_a[i].x);
        dst_a[i].y=static_cast<double>(src_a[i].y);
        dst_b[i].x=static_cast<double>(src_b[i].x);
        dst_b[i].y=static_cast<double>(src_b[i].y);
    }
}

int convertReductionChunk(const cuDoubleComplex *src,AddaCudaDoubleComplex *dst,
                          size_t count,const char *what)
{
    if (count==0) return 0;
    const unsigned int threads=256;
    const size_t b=(count+threads-1)/threads;
    if (b>static_cast<size_t>(UINT_MAX)) {
        setError("FP64 reduction conversion launch exceeds CUDA grid limit");
        return -1;
    }
    convertComplexF32ToF64Kernel<<<static_cast<unsigned int>(b),threads>>>(
        reinterpret_cast<const cuFloatComplex*>(src),dst,count);
    return launchChecked(what);
}

int convertReductionPairChunk(const cuDoubleComplex *src_a,const cuDoubleComplex *src_b,
                              AddaCudaDoubleComplex *dst_a,AddaCudaDoubleComplex *dst_b,
                              size_t count)
{
    if (count==0) return 0;
    const unsigned int threads=256;
    const size_t blocks=(count+threads-1)/threads;
    if (blocks>static_cast<size_t>(UINT_MAX)) {
        setError("FP64 pair-conversion launch exceeds CUDA grid limit");
        return -1;
    }
    convertComplexPairF32ToF64Kernel<<<static_cast<unsigned int>(blocks),threads>>>(
        reinterpret_cast<const cuFloatComplex*>(src_a),
        reinterpret_cast<const cuFloatComplex*>(src_b),
        dst_a,dst_b,count);
    return launchChecked("float32->float64 reduction pair conversion");
}

int mixedDotF32(const cuDoubleComplex *a,const cuDoubleComplex *b,size_t n,
                bool conjugate_first,double *out_re,double *out_im,const char *what)
{
    double total_re=0.0,total_im=0.0;
    for (size_t offset=0; offset<n; offset+=g.reduceCapacity) {
        const size_t count=(n-offset<g.reduceCapacity) ? n-offset : g.reduceCapacity;
        if (convertReductionPairChunk(a+offset,b+offset,g.d_reduce_a,g.d_reduce_b,count)) return -1;

        AddaCudaDoubleComplex partial{};
        const int icount=static_cast<int>(count); /* reduceCapacity <= INT_MAX */
        const cublasStatus_t st=conjugate_first
            ? addaCublasZdotc(g.cublas,icount,g.d_reduce_a,1,g.d_reduce_b,1,&partial)
            : addaCublasZdotu(g.cublas,icount,g.d_reduce_a,1,g.d_reduce_b,1,&partial);
        if (failCublas(st,what)) return -1;
        /* CUBLAS_POINTER_MODE_HOST makes partial available here.  Only one
         * complex<double> per chunk crosses to the CPU. */
        total_re+=partial.x;
        total_im+=partial.y;
    }
    *out_re=total_re;
    *out_im=total_im;
    return 0;
}

int mixedNorm2F32(const cuDoubleComplex *a,size_t n,double *norm2,const char *what)
{
    double total=0.0;
    for (size_t offset=0; offset<n; offset+=g.reduceCapacity) {
        const size_t count=(n-offset<g.reduceCapacity) ? n-offset : g.reduceCapacity;
        if (convertReductionChunk(a+offset,g.d_reduce_a,count,
                                  "float32->float64 norm conversion")) return -1;
        double partial_norm=0.0;
        if (failCublas(addaCublasDznrm2(g.cublas,static_cast<int>(count),
                                        g.d_reduce_a,1,&partial_norm),what)) return -1;
        total+=partial_norm*partial_norm;
    }
    *norm2=total;
    return 0;
}

int mixedSelfDotuNorm2F32(const cuDoubleComplex *a,size_t n,
                          double *out_re,double *out_im,double *norm2)
{
    double total_re=0.0,total_im=0.0,total_norm2=0.0;
    for (size_t offset=0; offset<n; offset+=g.reduceCapacity) {
        const size_t count=(n-offset<g.reduceCapacity) ? n-offset : g.reduceCapacity;
        if (convertReductionChunk(a+offset,g.d_reduce_a,count,
                                  "float32->float64 self reduction conversion")) return -1;

        AddaCudaDoubleComplex partial_dot{};
        double partial_norm=0.0;
        const int icount=static_cast<int>(count);
        if (failCublas(addaCublasZdotu(g.cublas,icount,g.d_reduce_a,1,
                                       g.d_reduce_a,1,&partial_dot),
                       "cublasZdotu(FP64 mixed self)") ||
            failCublas(addaCublasDznrm2(g.cublas,icount,g.d_reduce_a,1,&partial_norm),
                       "cublasDznrm2(FP64 mixed self)")) return -1;
        total_re+=partial_dot.x;
        total_im+=partial_dot.y;
        total_norm2+=partial_norm*partial_norm;
    }
    *out_re=total_re;
    *out_im=total_im;
    *norm2=total_norm2;
    return 0;
}
#endif

int backendNorm2(const cuDoubleComplex *a,size_t n,double *norm2,const char *what)
{
#ifdef ADDA_CUDA_SINGLE_BACKEND
    return mixedNorm2F32(a,n,norm2,what);
#else
    if (n>static_cast<size_t>(INT_MAX)) {
        setError("cuBLAS FP64 norm requires vector length <= INT_MAX");
        return -1;
    }
    double norm=0.0;
    if (failCublas(cublasDznrm2(g.cublas,static_cast<int>(n),a,1,&norm),what)) return -1;
    *norm2=norm*norm;
    return 0;
#endif
}

int backendDotu(const cuDoubleComplex *a,const cuDoubleComplex *b,size_t n,
                double *out_re,double *out_im,const char *what)
{
#ifdef ADDA_CUDA_SINGLE_BACKEND
    return mixedDotF32(a,b,n,false,out_re,out_im,what);
#else
    if (n>static_cast<size_t>(INT_MAX)) {
        setError("cuBLAS FP64 dotu requires vector length <= INT_MAX");
        return -1;
    }
    cuDoubleComplex out{};
    if (failCublas(cublasZdotu(g.cublas,static_cast<int>(n),a,1,b,1,&out),what)) return -1;
    *out_re=out.x;
    *out_im=out.y;
    return 0;
#endif
}

int backendDotc(const cuDoubleComplex *a,const cuDoubleComplex *b,size_t n,
                double *out_re,double *out_im,const char *what)
{
#ifdef ADDA_CUDA_SINGLE_BACKEND
    return mixedDotF32(a,b,n,true,out_re,out_im,what);
#else
    if (n>static_cast<size_t>(INT_MAX)) {
        setError("cuBLAS FP64 dotc requires vector length <= INT_MAX");
        return -1;
    }
    cuDoubleComplex out{};
    if (failCublas(cublasZdotc(g.cublas,static_cast<int>(n),a,1,b,1,&out),what)) return -1;
    *out_re=out.x;
    *out_im=out.y;
    return 0;
#endif
}

int backendSelfDotuNorm2(const cuDoubleComplex *a,size_t n,
                         double *out_re,double *out_im,double *norm2)
{
#ifdef ADDA_CUDA_SINGLE_BACKEND
    return mixedSelfDotuNorm2F32(a,n,out_re,out_im,norm2);
#else
    if (backendDotu(a,a,n,out_re,out_im,"cublasZdotu(iterative self)")) return -1;
    return backendNorm2(a,n,norm2,"cublasDznrm2(iterative self)");
#endif
}


int createPlans()
{
    const int gx=toInt(g.cfg.gridX,"gridX"); if (gx<0) return -1;
    const int gy=toInt(g.cfg.gridY,"gridY"); if (gy<0) return -1;
    const int gz=toInt(g.cfg.gridZ,"gridZ"); if (gz<0) return -1;
    if (g.slice_fft) {
        const int boxXY=toInt(g.boxXY,"boxX*boxY"); if (boxXY<0) return -1;
        const int planeN=toInt(g.planeN,"gridX*gridY"); if (planeN<0) return -1;
        size_t workZ=0,workXY=0;
        int dimZ[1]={gz},embedZ[1]={gz};
        if (failCufft(cufftCreate(&g.planZ),"cufftCreate(slice Z)") ||
            failCufft(cufftSetAutoAllocation(g.planZ,0),"cufftSetAutoAllocation(slice Z)") ||
            failCufft(cufftMakePlanMany(g.planZ,1,dimZ,embedZ,boxXY,1,embedZ,boxXY,1,
                                        CUFFT_Z2Z,boxXY,&workZ),"cufftMakePlanMany(slice Z)")) return -1;
        int dims2[2]={gy,gx};
        if (failCufft(cufftCreate(&g.planXY),"cufftCreate(slice XY)") ||
            failCufft(cufftSetAutoAllocation(g.planXY,0),"cufftSetAutoAllocation(slice XY)") ||
            failCufft(cufftMakePlanMany(g.planXY,2,dims2,nullptr,1,planeN,nullptr,1,planeN,
                                        CUFFT_Z2Z,3*ADDA_CUDA_SLICE_BATCH,&workXY),
                      "cufftMakePlanMany(slice XY batch)")) return -1;
        g.fftWorkSize=(workZ>workXY)?workZ:workXY;
        return 0;
    }
    const int gridN=toInt(g.gridN,"gridX*gridY*gridZ"); if (gridN<0) return -1;
    int dims3[3]={gz,gy,gx};
    if (failCufft(cufftPlanMany(&g.plan3d,3,dims3,nullptr,1,gridN,nullptr,1,gridN,CUFFT_Z2Z,3),
                  "cufftPlanMany(3D)")) return -1;
    if (g.cfg.surface) {
        const size_t xySizeSt=g.cfg.gridX*g.cfg.gridY;
        const int xySize=toInt(xySizeSt,"gridX*gridY"); if (xySize<0) return -1;
        size_t xyBatchSt;
        if (!checkedMul(3,g.cfg.gridZ,xyBatchSt)) { setError("overflow while creating surface XY cuFFT batch"); return -1; }
        const int xyBatch=toInt(xyBatchSt,"3*gridZ"); if (xyBatch<0) return -1;
        int dims2[2]={gy,gx};
        if (failCufft(cufftPlanMany(&g.planXY,2,dims2,nullptr,1,xySize,nullptr,1,xySize,CUFFT_Z2Z,xyBatch),
                      "cufftPlanMany(surface XY)")) return -1;
        const int zBatch=toInt(xySizeSt,"gridX*gridY"); if (zBatch<0) return -1;
        int dimZ[1]={gz},embedZ[1]={gz};
        if (failCufft(cufftPlanMany(&g.planZ,1,dimZ,embedZ,xySize,1,embedZ,xySize,1,CUFFT_Z2Z,zBatch),
                      "cufftPlanMany(surface Z)")) return -1;
    }
    return 0;
}

int executeMixedSurfaceForward()
{
    const size_t bytes = g.gridComplexCount * sizeof(cuDoubleComplex);
    if (failCuda(cudaMemcpy(g.d_gridR,g.d_grid,bytes,cudaMemcpyDeviceToDevice),
                 "copy grid for surface transform")) return -1;

    if (failCufft(cufftExecZ2Z(g.planXY,
                               reinterpret_cast<cufftDoubleComplex*>(g.d_gridR),
                               reinterpret_cast<cufftDoubleComplex*>(g.d_gridR),
                               CUFFT_FORWARD),
                  "cufftExecZ2Z(surface XY forward)")) return -1;

    for (int c=0; c<3; ++c) {
        cufftDoubleComplex *ptr = reinterpret_cast<cufftDoubleComplex*>(g.d_gridR + static_cast<size_t>(c)*g.gridN);
        if (failCufft(cufftExecZ2Z(g.planZ,ptr,ptr,CUFFT_INVERSE),
                      "cufftExecZ2Z(surface Z inverse)")) return -1;
    }
    return 0;
}

} // namespace

static int matvecInitCommon(const AddaCudaMatVecConfig *cfg,
                            const size_t boxX,const size_t boxY,const size_t boxZ,
                            const bool slice_fft,const bool low_mem_green,
                            const void *Dmatrix,
                            const void *Rmatrix,
                            const unsigned char *material,
                            const unsigned short *position)
{
    g_error[0] = '\0';
    if (cfg == nullptr || Dmatrix == nullptr || material == nullptr || position == nullptr) {
        setError("invalid null pointer passed to CUDA MatVec initialization");
        return -1;
    }
    if (cfg->surface && Rmatrix == nullptr) {
        setError("surface CUDA MatVec requires Rmatrix");
        return -1;
    }
    if (cfg->gridX == 0 || cfg->gridY == 0 || cfg->gridZ == 0 || cfg->ndip == 0) {
        setError("CUDA MatVec received a zero grid dimension or zero dipole count");
        return -1;
    }
    if (slice_fft) {
        if (boxX == 0 || boxY == 0 || boxZ == 0 ||
            boxX > cfg->gridX || boxY > cfg->gridY || boxZ > cfg->gridZ) {
            setError("invalid physical box dimensions for CUDA slice MatVec");
            return -1;
        }
        if (cfg->surface) {
            setError("CUDA slice MatVec v1 does not support surface mode; use adda_cuda.exe");
            return -1;
        }
    }
    if (low_mem_green) {
        if (!slice_fft) {
            setError("low-memory Green symmetry requires CUDA slice MatVec");
            return -1;
        }
        if (!cfg->reduced_fft) {
            setError("adda_low_mem requires reduced FFT symmetry; remove -no_reduced_fft");
            return -1;
        }
        if (cfg->DsizeY != cfg->gridY/2 + 1 || cfg->DsizeZ != cfg->gridZ/2 + 1) {
            setError("adda_low_mem received inconsistent reduced Dmatrix dimensions");
            return -1;
        }
        if (cfg->surface) {
            setError("adda_low_mem does not support surface mode; use adda_cuda.exe");
            return -1;
        }
    }
    size_t expectedRows;
    if (!checkedMul(static_cast<size_t>(3),cfg->ndip,expectedRows) || cfg->nrows != expectedRows) {
        setError("CUDA MatVec requires nrows == 3*ndip in sequential mode");
        return -1;
    }

    freeContext();
    g.cfg = *cfg;
    g.slice_fft = slice_fft;
    g.low_mem_green = low_mem_green;
    g.boxX = boxX; g.boxY = boxY; g.boxZ = boxZ;
    g.DsizeX = low_mem_green ? (cfg->gridX/2 + 1) : cfg->gridX;
    if (!checkedMul(cfg->gridY,cfg->gridZ,g.gridYZ) ||
        !checkedMul(cfg->gridX,g.gridYZ,g.gridN) ||
        !checkedMul(static_cast<size_t>(3),g.gridN,g.gridComplexCount)) {
        setError("overflow while calculating CUDA FFT grid size");
        freeContext();
        return -1;
    }
    if (slice_fft) {
        if (!checkedMul(boxX,boxY,g.boxXY) ||
            !checkedMul(g.boxXY,cfg->gridZ,g.sliceN) ||
            !checkedMul(static_cast<size_t>(3),g.sliceN,g.sliceComplexCount) ||
            !checkedMul(cfg->gridX,cfg->gridY,g.planeN) ||
            !checkedMul(static_cast<size_t>(3*ADDA_CUDA_SLICE_BATCH),g.planeN,g.planeComplexCount)) {
            setError("overflow while calculating CUDA slice FFT workspace");
            freeContext();
            return -1;
        }
    }
    size_t dBlocks;
    if (!checkedMul(g.DsizeX,cfg->DsizeZ,dBlocks) ||
        !checkedMul(dBlocks,cfg->DsizeY,dBlocks) ||
        !checkedMul(dBlocks,static_cast<size_t>(6),g.dComplexCount)) {
        setError("overflow while calculating Dmatrix device size");
        freeContext();
        return -1;
    }
    if (cfg->surface) {
        size_t rBlocks;
        if (!checkedMul(cfg->gridX,cfg->gridZ,rBlocks) ||
            !checkedMul(rBlocks,cfg->RsizeY,rBlocks) ||
            !checkedMul(rBlocks,static_cast<size_t>(6),g.rComplexCount)) {
            setError("overflow while calculating Rmatrix device size");
            freeContext();
            return -1;
        }
    }

    const int device = requestedDevice(cfg->device);
    if (device < 0) { freeContext(); return -1; }
    int count = 0;
    if (failCuda(cudaGetDeviceCount(&count),"cudaGetDeviceCount")) { freeContext(); return -1; }
    if (device >= count) {
        setError("requested CUDA device does not exist");
        freeContext();
        return -1;
    }
    if (failCuda(cudaSetDevice(device),"cudaSetDevice")) { freeContext(); return -1; }

    /* CUDA 11.8: cudaSetDevice() selects the device but does not necessarily
     * create the primary context immediately.  Force runtime/context creation
     * before any sizeable ADDA allocation. */
    if (failCuda(cudaFree(nullptr),"CUDA 11.8 context initialization (cudaFree(0))")) {
        freeContext();
        return -1;
    }
    if (failCuda(cudaMemGetInfo(&g.freeAfterContextBytes,&g.deviceTotalBytes),
                 "cudaMemGetInfo(after CUDA context)")) {
        freeContext();
        return -1;
    }

    cudaDeviceProp prop{};
    if (failCuda(cudaGetDeviceProperties(&prop,device),"cudaGetDeviceProperties")) { freeContext(); return -1; }
#ifndef ADDA_CUDA_SINGLE_BACKEND
    if (prop.major == 1 && prop.minor < 3) {
        setError("CUDA device does not support double precision");
        freeContext();
        return -1;
    }
#endif

    if (failCuda(cudaEventCreate(&g.matvec_start),"cudaEventCreate(matvec start)") ||
        failCuda(cudaEventCreate(&g.matvec_stop),"cudaEventCreate(matvec stop)")) {
        freeContext();
        return -1;
    }

    /* Create library contexts/workspaces before the large ADDA allocations so
     * CUDA/cuBLAS/cuFFT reserve what they need first. */
    if (failCublas(cublasCreate(&g.cublas),"cublasCreate")) { freeContext(); return -1; }
    if (failCublas(cublasSetPointerMode(g.cublas,CUBLAS_POINTER_MODE_HOST),
                   "cublasSetPointerMode(HOST)")) { freeContext(); return -1; }
    if (createPlans()) { freeContext(); return -1; }
    {
        size_t total_now=0;
        if (failCuda(cudaMemGetInfo(&g.freeAfterLibrariesBytes,&total_now),
                     "cudaMemGetInfo(after cuBLAS/cuFFT initialization)")) { freeContext(); return -1; }
        if (g.deviceTotalBytes == 0) g.deviceTotalBytes=total_now;
    }

#define CUDA_ALLOC(ptr,count,type,label) \
    do { if (failCuda(cudaMalloc(reinterpret_cast<void**>(&(ptr)),(count)*sizeof(type)),label)) { freeContext(); return -1; } } while (0)

    /* Reserve the single shared cuFFT work area before the large ADDA arrays. */
    if (slice_fft && g.fftWorkSize != 0) {
        if (failCuda(cudaMalloc(&g.d_fftWork,g.fftWorkSize),"cudaMalloc(shared slice cuFFT workspace)")) { freeContext(); return -1; }
        if (failCufft(cufftSetWorkArea(g.planZ,g.d_fftWork),"cufftSetWorkArea(slice Z)") ||
            failCufft(cufftSetWorkArea(g.planXY,g.d_fftWork),"cufftSetWorkArea(slice XY)")) { freeContext(); return -1; }
    }

    CUDA_ALLOC(g.d_D,g.dComplexCount,cuDoubleComplex,"cudaMalloc(Dmatrix)");
    if (cfg->surface) CUDA_ALLOC(g.d_R,g.rComplexCount,cuDoubleComplex,"cudaMalloc(Rmatrix)");
    CUDA_ALLOC(g.d_material,cfg->ndip,unsigned char,"cudaMalloc(material)");
    CUDA_ALLOC(g.d_position,cfg->nrows,unsigned short,"cudaMalloc(position)");
    CUDA_ALLOC(g.d_arg,cfg->nrows,cuDoubleComplex,"cudaMalloc(argvec/iterative xvec)");
    CUDA_ALLOC(g.d_result,cfg->nrows,cuDoubleComplex,"cudaMalloc(resultvec/iterative rvec)");
#ifdef ADDA_CUDA_SINGLE_BACKEND
    g.reduceCapacity=(cfg->nrows<ADDA_FP64_REDUCTION_CHUNK) ? cfg->nrows : ADDA_FP64_REDUCTION_CHUNK;
    if (g.reduceCapacity==0) {
        setError("invalid zero-sized FP64 reduction scratch");
        freeContext();
        return -1;
    }
    CUDA_ALLOC(g.d_reduce_a,g.reduceCapacity,AddaCudaDoubleComplex,
               "cudaMalloc(FP64 reduction scratch A)");
    CUDA_ALLOC(g.d_reduce_b,g.reduceCapacity,AddaCudaDoubleComplex,
               "cudaMalloc(FP64 reduction scratch B)");
#endif
    if (slice_fft) {
        CUDA_ALLOC(g.d_slice,g.sliceComplexCount,cuDoubleComplex,"cudaMalloc(slice Z workspace)");
        CUDA_ALLOC(g.d_plane,g.planeComplexCount,cuDoubleComplex,"cudaMalloc(slice XY plane workspace)");
    } else {
        CUDA_ALLOC(g.d_grid,g.gridComplexCount,cuDoubleComplex,"cudaMalloc(full FFT grid)");
        if (cfg->surface) CUDA_ALLOC(g.d_gridR,g.gridComplexCount,cuDoubleComplex,"cudaMalloc(surface FFT grid)");
    }
#undef CUDA_ALLOC
    updateExplicitPeak();

    if (failCuda(cudaMemcpy(g.d_D,Dmatrix,g.dComplexCount*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice),
                 "upload Dmatrix") ||
        failCuda(cudaMemcpy(g.d_material,material,cfg->ndip*sizeof(unsigned char),cudaMemcpyHostToDevice),
                 "upload material") ||
        failCuda(cudaMemcpy(g.d_position,position,cfg->nrows*sizeof(unsigned short),cudaMemcpyHostToDevice),
                 "upload position")) {
        freeContext(); return -1;
    }
    if (cfg->surface && failCuda(cudaMemcpy(g.d_R,Rmatrix,g.rComplexCount*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice),
                                 "upload Rmatrix")) {
        freeContext(); return -1;
    }

    g.initialized = true;
    return 0;
}

extern "C" int adda_cuda_backend_real_bytes(void)
{
    return static_cast<int>(sizeof(AddaCudaReal));
}

extern "C" int adda_cuda_matvec_init(const AddaCudaMatVecConfig *cfg,
                                      const void *Dmatrix,
                                      const void *Rmatrix,
                                      const unsigned char *material,
                                      const unsigned short *position)
{
    return matvecInitCommon(cfg,0,0,0,false,false,Dmatrix,Rmatrix,material,position);
}

extern "C" int adda_cuda_matvec_init_slice(const AddaCudaMatVecConfig *cfg,
                                            size_t boxX,size_t boxY,size_t boxZ,
                                            const void *Dmatrix,
                                            const void *Rmatrix,
                                            const unsigned char *material,
                                            const unsigned short *position)
{
    return matvecInitCommon(cfg,boxX,boxY,boxZ,true,false,Dmatrix,Rmatrix,material,position);
}

extern "C" int adda_cuda_matvec_init_low_mem(const AddaCudaMatVecConfig *cfg,
                                              size_t boxX,size_t boxY,size_t boxZ,
                                              const void *Dmatrix,
                                              const void *Rmatrix,
                                              const unsigned char *material,
                                              const unsigned short *position)
{
    /* Dmatrix host layout is [x][z][y][6]. The first gridX/2+1 x-planes
     * are contiguous, therefore matvecInitCommon can upload the compact first
     * octant directly without a host repack buffer. */
    return matvecInitCommon(cfg,boxX,boxY,boxZ,true,true,Dmatrix,Rmatrix,material,position);
}

extern "C" int adda_cuda_memory_info(AddaCudaMemoryInfo *info)
{
    g_error[0]='\0';
    if (info == nullptr) { setError("null CUDA memory info pointer"); return -1; }
    if (!g.initialized) { setError("CUDA backend is not initialized"); return -1; }
    size_t free_now=0,total_now=0;
    if (failCuda(cudaMemGetInfo(&free_now,&total_now),"cudaMemGetInfo(current)")) return -1;
    const size_t current=explicitAllocationBytes();
    if (current == SIZE_MAX) { setError("overflow while calculating CUDA allocation size"); return -1; }
    if (current > g.explicitPeakBytes) g.explicitPeakBytes=current;
    info->device_total_bytes = g.deviceTotalBytes ? g.deviceTotalBytes : total_now;
    info->free_after_context_bytes = g.freeAfterContextBytes;
    info->free_after_libraries_bytes = g.freeAfterLibrariesBytes;
    info->current_free_bytes = free_now;

    /* Per-category exact accounting.  These values describe only live
     * allocations explicitly owned by ADDA; internal cuFFT/cuBLAS/runtime
     * allocations are reflected by cudaMemGetInfo deltas below the C ABI. */
    info->green_tensor_bytes = g.d_D ? g.dComplexCount*sizeof(cuDoubleComplex) : 0;
    info->surface_tensor_bytes = g.d_R ? g.rComplexCount*sizeof(cuDoubleComplex) : 0;
    info->fft_grid_bytes = g.d_grid ? g.gridComplexCount*sizeof(cuDoubleComplex) : 0;
    info->surface_fft_grid_bytes = g.d_gridR ? g.gridComplexCount*sizeof(cuDoubleComplex) : 0;
    info->slice_z_bytes = g.d_slice ? g.sliceComplexCount*sizeof(cuDoubleComplex) : 0;
    info->slice_xy_bytes = g.d_plane ? g.planeComplexCount*sizeof(cuDoubleComplex) : 0;
    info->fft_workspace_bytes = g.d_fftWork ? g.fftWorkSize : 0;
    info->reduction_scratch_bytes = 0;
#ifdef ADDA_CUDA_SINGLE_BACKEND
    if (g.d_reduce_a) info->reduction_scratch_bytes += g.reduceCapacity*sizeof(AddaCudaDoubleComplex);
    if (g.d_reduce_b) info->reduction_scratch_bytes += g.reduceCapacity*sizeof(AddaCudaDoubleComplex);
#endif
    info->matvec_vector_bytes = 0;
    if (g.d_arg) info->matvec_vector_bytes += g.cfg.nrows*sizeof(cuDoubleComplex);
    if (g.d_result) info->matvec_vector_bytes += g.cfg.nrows*sizeof(cuDoubleComplex);
    info->iterative_vector_bytes = 0;
    if (g.d_iter_extra) {
        size_t vectors=0;
        if (!checkedMul(g.iter_extra_capacity,g.cfg.nrows,vectors) ||
            !checkedMul(vectors,sizeof(cuDoubleComplex),info->iterative_vector_bytes)) {
            setError("overflow while calculating iterative CUDA vector memory");
            return -1;
        }
    }
    info->cc_bytes = g.d_cc ? g.ccCapacity*sizeof(cuDoubleComplex) : 0;
    info->material_bytes = g.d_material ? g.cfg.ndip*sizeof(unsigned char) : 0;
    info->position_bytes = g.d_position ? g.cfg.nrows*sizeof(unsigned short) : 0;

    info->explicit_current_bytes = current;
    info->explicit_peak_bytes = g.explicitPeakBytes;
    return 0;
}

extern "C" int adda_cuda_matvec_update_cc(const void *cc_sqrt, size_t complex_count)
{
    g_error[0] = '\0';
    if (!g.initialized) {
        setError("CUDA MatVec is not initialized before cc_sqrt update");
        return -1;
    }
    if (cc_sqrt == nullptr || complex_count == 0) {
        setError("invalid cc_sqrt passed to CUDA MatVec");
        return -1;
    }
    if (complex_count > g.ccCapacity) {
        if (g.d_cc) cudaFree(g.d_cc);
        g.d_cc = nullptr;
        if (failCuda(cudaMalloc(reinterpret_cast<void**>(&g.d_cc),complex_count*sizeof(cuDoubleComplex)),
                     "cudaMalloc(cc_sqrt)")) return -1;
        g.ccCapacity = complex_count;
        updateExplicitPeak();
    }
    return failCuda(cudaMemcpy(g.d_cc,cc_sqrt,complex_count*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice),
                    "upload cc_sqrt");
}


int matvecGpuCoreSlice(const cuDoubleComplex *d_arg, cuDoubleComplex *d_result, int her,
                       double *inprod, double *elapsed_ms)
{
    if (failCuda(cudaEventRecord(g.matvec_start,0),"cudaEventRecord(slice matvec start)") ||
        failCuda(cudaMemset(g.d_slice,0,g.sliceComplexCount*sizeof(cuDoubleComplex)),
                 "clear CUDA slice Z workspace")) return -1;
    const int threads=256;
    const unsigned int dipBlocks=static_cast<unsigned int>((g.cfg.ndip+threads-1)/threads);
    scatterSliceKernel<<<dipBlocks,threads>>>(d_arg,g.d_slice,g.d_material,g.d_position,g.d_cc,
                                              g.cfg.ndip,g.boxX,g.boxY,g.sliceN,her!=0);
    if (failCuda(cudaGetLastError(),"scatterSliceKernel launch")) return -1;
    for (int c=0;c<3;++c) {
        cufftDoubleComplex *ptr=reinterpret_cast<cufftDoubleComplex*>(g.d_slice+static_cast<size_t>(c)*g.sliceN);
        if (failCufft(cufftExecZ2Z(g.planZ,ptr,ptr,CUFFT_FORWARD),"cufftExecZ2Z(slice Z forward)")) return -1;
    }
    const size_t batchPlaneN=static_cast<size_t>(ADDA_CUDA_SLICE_BATCH)*g.planeN;
    const unsigned int planeBatchBlocks=static_cast<unsigned int>((batchPlaneN+threads-1)/threads);
    const int transposed=(!g.cfg.reduced_fft && her)?1:0;
    for (size_t kz0=0;kz0<g.cfg.gridZ;kz0+=ADDA_CUDA_SLICE_BATCH) {
        const size_t remaining=g.cfg.gridZ-kz0;
        const size_t active=(remaining<static_cast<size_t>(ADDA_CUDA_SLICE_BATCH))
                          ? remaining : static_cast<size_t>(ADDA_CUDA_SLICE_BATCH);
        sliceToPlaneBatchKernel<<<planeBatchBlocks,threads>>>(g.d_slice,g.d_plane,kz0,active,
            g.boxX,g.boxY,g.cfg.gridX,g.boxXY,g.sliceN,g.planeN);
        if (failCuda(cudaGetLastError(),"sliceToPlaneBatchKernel launch")) return -1;
        if (failCufft(cufftExecZ2Z(g.planXY,reinterpret_cast<cufftDoubleComplex*>(g.d_plane),
                                   reinterpret_cast<cufftDoubleComplex*>(g.d_plane),CUFFT_FORWARD),
                      "cufftExecZ2Z(slice XY batch forward)")) return -1;
        const size_t activePlaneN=active*g.planeN;
        const unsigned int activePlaneBlocks=static_cast<unsigned int>((activePlaneN+threads-1)/threads);
        spectralMultiplySliceBatchKernel<<<activePlaneBlocks,threads>>>(g.d_plane,g.d_D,kz0,active,
            g.cfg.gridX,g.cfg.gridY,g.cfg.gridZ,g.planeN,g.DsizeX,g.cfg.DsizeY,g.cfg.DsizeZ,
            g.cfg.reduced_fft,transposed,g.low_mem_green ? 1 : 0);
        if (failCuda(cudaGetLastError(),"spectralMultiplySliceBatchKernel launch")) return -1;
        if (failCufft(cufftExecZ2Z(g.planXY,reinterpret_cast<cufftDoubleComplex*>(g.d_plane),
                                   reinterpret_cast<cufftDoubleComplex*>(g.d_plane),CUFFT_INVERSE),
                      "cufftExecZ2Z(slice XY batch inverse)")) return -1;
        const size_t activeBoxN=active*g.boxXY;
        const unsigned int activeBoxBlocks=static_cast<unsigned int>((activeBoxN+threads-1)/threads);
        planeToSliceBatchKernel<<<activeBoxBlocks,threads>>>(g.d_plane,g.d_slice,kz0,active,
            g.boxX,g.cfg.gridX,g.boxXY,g.sliceN,g.planeN);
        if (failCuda(cudaGetLastError(),"planeToSliceBatchKernel launch")) return -1;
    }
    for (int c=0;c<3;++c) {
        cufftDoubleComplex *ptr=reinterpret_cast<cufftDoubleComplex*>(g.d_slice+static_cast<size_t>(c)*g.sliceN);
        if (failCufft(cufftExecZ2Z(g.planZ,ptr,ptr,CUFFT_INVERSE),"cufftExecZ2Z(slice Z inverse)")) return -1;
    }
    gatherSliceKernel<<<dipBlocks,threads>>>(d_arg,d_result,g.d_slice,g.d_material,g.d_position,g.d_cc,
                                             g.cfg.ndip,g.boxX,g.boxY,g.sliceN,her!=0);
    if (failCuda(cudaGetLastError(),"gatherSliceKernel launch")) return -1;
    if (inprod != nullptr) {
        if (backendNorm2(d_result,g.cfg.nrows,inprod,
                         "cuBLAS FP64 norm(slice MatVec result)")) return -1;
    }
    if (failCuda(cudaEventRecord(g.matvec_stop,0),"cudaEventRecord(slice matvec stop)") ||
        failCuda(cudaEventSynchronize(g.matvec_stop),"cudaEventSynchronize(slice matvec stop)")) return -1;
    if (elapsed_ms != nullptr) {
        float ms=0.0f;
        if (failCuda(cudaEventElapsedTime(&ms,g.matvec_start,g.matvec_stop),"cudaEventElapsedTime(slice matvec)")) return -1;
        *elapsed_ms=static_cast<double>(ms);
    }
    return 0;
}

int matvecGpuCore(const cuDoubleComplex *d_arg, cuDoubleComplex *d_result, int her,
                  double *inprod, double *elapsed_ms)
{
    if (g.slice_fft) return matvecGpuCoreSlice(d_arg,d_result,her,inprod,elapsed_ms);
    const size_t gridBytes = g.gridComplexCount*sizeof(cuDoubleComplex);
    if (failCuda(cudaEventRecord(g.matvec_start,0),"cudaEventRecord(matvec start)") ||
        failCuda(cudaMemset(g.d_grid,0,gridBytes),"clear CUDA FFT grid")) return -1;

    const int threads = 256;
    const unsigned int dipBlocks = static_cast<unsigned int>((g.cfg.ndip + threads - 1)/threads);
    scatterKernel<<<dipBlocks,threads>>>(d_arg,g.d_grid,g.d_material,g.d_position,g.d_cc,
                                        g.cfg.ndip,g.cfg.gridX,g.cfg.gridY,g.gridN,her != 0);
    if (failCuda(cudaGetLastError(),"scatterKernel launch")) return -1;

    if (g.cfg.surface && executeMixedSurfaceForward()) return -1;

    if (failCufft(cufftExecZ2Z(g.plan3d,
                               reinterpret_cast<cufftDoubleComplex*>(g.d_grid),
                               reinterpret_cast<cufftDoubleComplex*>(g.d_grid),
                               CUFFT_FORWARD),
                  "cufftExecZ2Z(3D forward)")) return -1;

    const unsigned int gridBlocks = static_cast<unsigned int>((g.gridN + threads - 1)/threads);
    const int transposed = (!g.cfg.reduced_fft && her) ? 1 : 0;
    spectralMultiplyKernel<<<gridBlocks,threads>>>(g.d_grid,g.d_gridR,g.d_D,g.d_R,
                                                   g.cfg.gridX,g.cfg.gridY,g.cfg.gridZ,g.gridN,
                                                   g.cfg.DsizeY,g.cfg.DsizeZ,g.cfg.RsizeY,
                                                   g.cfg.reduced_fft,transposed,g.cfg.surface);
    if (failCuda(cudaGetLastError(),"spectralMultiplyKernel launch")) return -1;

    if (failCufft(cufftExecZ2Z(g.plan3d,
                               reinterpret_cast<cufftDoubleComplex*>(g.d_grid),
                               reinterpret_cast<cufftDoubleComplex*>(g.d_grid),
                               CUFFT_INVERSE),
                  "cufftExecZ2Z(3D inverse)")) return -1;

    gatherKernel<<<dipBlocks,threads>>>(d_arg,d_result,g.d_grid,g.d_material,g.d_position,g.d_cc,
                                       g.cfg.ndip,g.cfg.gridX,g.cfg.gridY,g.gridN,her != 0);
    if (failCuda(cudaGetLastError(),"gatherKernel launch")) return -1;

    if (inprod != nullptr) {
        if (backendNorm2(d_result,g.cfg.nrows,inprod,
                         "cuBLAS FP64 norm(MatVec result)")) return -1;
    }

    if (failCuda(cudaEventRecord(g.matvec_stop,0),"cudaEventRecord(matvec stop)") ||
        failCuda(cudaEventSynchronize(g.matvec_stop),"cudaEventSynchronize(matvec stop)")) return -1;
    if (elapsed_ms != nullptr) {
        float elapsed_gpu_ms = 0.0f;
        if (failCuda(cudaEventElapsedTime(&elapsed_gpu_ms,g.matvec_start,g.matvec_stop),
                     "cudaEventElapsedTime(matvec)")) return -1;
        *elapsed_ms = static_cast<double>(elapsed_gpu_ms);
    }
    return 0;
}

extern "C" int adda_cuda_matvec_execute(const void *argvec, void *resultvec, int her,
                                         double *inprod, double *elapsed_ms)
{
    g_error[0] = '\0';
    if (!g.initialized || g.d_cc == nullptr) {
        setError("CUDA MatVec or cc_sqrt is not initialized");
        return -1;
    }
    if (argvec == nullptr || resultvec == nullptr) {
        setError("null argument/result vector passed to CUDA MatVec");
        return -1;
    }

    const size_t vectorBytes = g.cfg.nrows*sizeof(cuDoubleComplex);
    if (failCuda(cudaMemcpy(g.d_arg,argvec,vectorBytes,cudaMemcpyHostToDevice),
                 "upload MatVec argument")) return -1;
    if (matvecGpuCore(g.d_arg,g.d_result,her,inprod,elapsed_ms)) return -1;
    return failCuda(cudaMemcpy(resultvec,g.d_result,vectorBytes,cudaMemcpyDeviceToHost),
                    "download MatVec result");
}

extern "C" int adda_cuda_matvec_execute_gpu(const void *argvec_id, const void *resultvec_id, int her,
                                             double *inprod, double *elapsed_ms)
{
    g_error[0] = '\0';
    if (!g.initialized || g.d_cc == nullptr) {
        setError("CUDA MatVec or cc_sqrt is not initialized");
        return -1;
    }
    cuDoubleComplex *d_arg = iterDeviceVector(argvec_id);
    if (d_arg == nullptr) return -1;
    cuDoubleComplex *d_result = iterDeviceVector(resultvec_id);
    if (d_result == nullptr) return -1;
    if (d_arg == d_result) {
        setError("MatVec_GPU input and output vectors must not alias");
        return -1;
    }
    return matvecGpuCore(d_arg,d_result,her,inprod,elapsed_ms);
}

static int iterInitListCommon(const void * const *host_ids,size_t count)
{
    g_error[0] = '\0';
    if (!g.initialized || g.cublas == nullptr) {
        setError("CUDA backend is not initialized before iterative-solver initialization");
        return -1;
    }
    if (g.cfg.nrows > static_cast<size_t>(INT_MAX)) {
        setError("CUDA iterative solvers require nrows <= INT_MAX for cuBLAS");
        return -1;
    }
    if (host_ids == nullptr || count < 2 || count > static_cast<size_t>(Context::ITER_VECTOR_MAX)) {
        setError("invalid CUDA iterative vector list");
        return -1;
    }
    if (host_ids[0] == nullptr || host_ids[1] == nullptr) {
        setError("the first two CUDA iterative identities must be xvec and rvec");
        return -1;
    }
    for (size_t i=0;i<count;++i) {
        if (host_ids[i] == nullptr) { setError("null identity in CUDA iterative vector list"); return -1; }
        for (size_t j=0;j<i;++j) {
            if (host_ids[i] == host_ids[j]) { setError("CUDA iterative vector identities must be distinct"); return -1; }
        }
    }
    const size_t extra_count=count-2; /* x/r reuse d_arg/d_result */
    if (extra_count != g.iter_extra_capacity) {
        if (g.d_iter_extra) {
            if (failCuda(cudaFree(g.d_iter_extra),"cudaFree(previous iterative extra vectors)")) return -1;
        }
        g.d_iter_extra=nullptr;
        g.iter_extra_capacity=0;
        size_t total=0;
        if (!checkedMul(extra_count,g.cfg.nrows,total)) {
            setError("overflow while allocating CUDA iterative vectors");
            return -1;
        }
        if (total != 0 && failCuda(cudaMalloc(reinterpret_cast<void**>(&g.d_iter_extra),
                                               total*sizeof(cuDoubleComplex)),
                                   "cudaMalloc(exact iterative extra vectors)")) return -1;
        g.iter_extra_capacity=extra_count;
        updateExplicitPeak();
    }
    for (int i=0;i<Context::ITER_VECTOR_MAX;++i) { g.iter_host[i]=nullptr; g.iter_device[i]=nullptr; }
    g.iter_count=count;
    g.iter_host[0]=host_ids[0]; g.iter_device[0]=g.d_arg;
    g.iter_host[1]=host_ids[1]; g.iter_device[1]=g.d_result;
    for (size_t i=2;i<count;++i) {
        g.iter_host[i]=host_ids[i];
        g.iter_device[i]=g.d_iter_extra+(i-2)*g.cfg.nrows;
    }
    g.iter_initialized=true;
    return adda_cuda_iter_upload_all();
}

extern "C" int adda_cuda_iter_init_list(const void * const *host_ids,size_t count)
{
    return iterInitListCommon(host_ids,count);
}

extern "C" int adda_cuda_iter_init(const void *xvec, const void *rvec, const void *pvec,
                                    const void *vec1, const void *vec2, const void *vec3,
                                    const void *vec4, const void *vec5, const void *vec6,
                                    const void *vec7, const void *Avecbuffer)
{
    const void *raw[11]={xvec,rvec,pvec,vec1,vec2,vec3,vec4,vec5,vec6,vec7,Avecbuffer};
    if (xvec == nullptr || rvec == nullptr || pvec == nullptr || Avecbuffer == nullptr) {
        setError("xvec, rvec, pvec and Avecbuffer must be registered for CUDA iterative solvers");
        return -1;
    }
    const void *compact[11]; size_t count=0;
    compact[count++]=xvec; compact[count++]=rvec;
    for (size_t i=2;i<11;++i) if (raw[i]!=nullptr) compact[count++]=raw[i];
    return iterInitListCommon(compact,count);
}

extern "C" int adda_cuda_iter_upload_all(void)
{
    g_error[0] = '\0';
    if (!g.iter_initialized) { setError("CUDA iterative vectors are not initialized"); return -1; }
    const size_t bytes=g.cfg.nrows*sizeof(cuDoubleComplex);
    for (size_t i=0; i<g.iter_count; ++i) {
        if (g.iter_host[i] == nullptr) continue;
        if (failCuda(cudaMemcpy(g.iter_device[i],g.iter_host[i],bytes,cudaMemcpyHostToDevice),
                     "upload iterative vector")) return -1;
    }
    return 0;
}

extern "C" int adda_cuda_iter_download_all(void)
{
    g_error[0] = '\0';
    if (!g.iter_initialized) { setError("CUDA iterative vectors are not initialized"); return -1; }
    const size_t bytes=g.cfg.nrows*sizeof(cuDoubleComplex);
    for (size_t i=0; i<g.iter_count; ++i) {
        if (g.iter_host[i] == nullptr) continue;
        if (failCuda(cudaMemcpy(const_cast<void*>(g.iter_host[i]),g.iter_device[i],bytes,
                                cudaMemcpyDeviceToHost),"download iterative vector")) return -1;
    }
    return 0;
}

extern "C" int adda_cuda_iter_upload_one(const void *id)
{
    g_error[0] = '\0';
    if (!g.iter_initialized || id == nullptr) {
        setError("CUDA iterative vector is not initialized");
        return -1;
    }
    const size_t bytes=g.cfg.nrows*sizeof(cuDoubleComplex);
    for (size_t i=0; i<g.iter_count; ++i) {
        if (g.iter_host[i] == id) {
            return failCuda(cudaMemcpy(g.iter_device[i],id,bytes,cudaMemcpyHostToDevice),
                            "upload one iterative vector");
        }
    }
    setError("unknown host vector passed to CUDA iterative upload");
    return -1;
}

extern "C" int adda_cuda_iter_download_one(const void *id)
{
    g_error[0] = '\0';
    if (!g.iter_initialized || id == nullptr) {
        setError("CUDA iterative vector is not initialized");
        return -1;
    }
    const size_t bytes=g.cfg.nrows*sizeof(cuDoubleComplex);
    for (size_t i=0; i<g.iter_count; ++i) {
        if (g.iter_host[i] == id) {
            return failCuda(cudaMemcpy(const_cast<void*>(id),g.iter_device[i],bytes,cudaMemcpyDeviceToHost),
                            "download one iterative vector");
        }
    }
    setError("unknown host vector passed to CUDA iterative download");
    return -1;
}

static int iterBlocks(unsigned int *blocks)
{
    const int threads=256;
    const size_t b=(g.cfg.nrows + static_cast<size_t>(threads)-1)/static_cast<size_t>(threads);
    if (b > static_cast<size_t>(UINT_MAX)) {
        setError("iterative vector is too large for CUDA launch grid");
        return -1;
    }
    *blocks=static_cast<unsigned int>(b);
    return 0;
}

static int iterNorm2(cuDoubleComplex *a,double *norm2,const char *what)
{
    if (norm2 == nullptr) return 0;
    return backendNorm2(a,g.cfg.nrows,norm2,what);
}

extern "C" int adda_cuda_iter_norm2(const void *a_id,double *norm2)
{
    g_error[0]='\0';
    if (norm2 == nullptr) { setError("null iterative norm result pointer"); return -1; }
    cuDoubleComplex *a=iterDeviceVector(a_id); if(!a)return -1;
    return iterNorm2(a,norm2,"cublasDznrm2(iterative vector)");
}

extern "C" int adda_cuda_iter_release(void)
{
    g_error[0] = '\0';
    if (!g.initialized) {
        setError("CUDA backend is not initialized");
        return -1;
    }

    if (g.d_iter_extra) {
        if (failCuda(cudaFree(g.d_iter_extra),"cudaFree(iterative extra vectors)")) return -1;
        g.d_iter_extra=nullptr;
    }
    g.iter_extra_capacity=0;
    for (int i=0; i<Context::ITER_VECTOR_MAX; ++i) {
        g.iter_host[i]=nullptr;
        g.iter_device[i]=nullptr;
    }
    g.iter_count=0;
    g.iter_initialized=false;
    return 0;
}

extern "C" int adda_cuda_iter_dotu(const void *a_id,const void *b_id,double *out_re,double *out_im)
{
    g_error[0]='\0';
    if (out_re == nullptr || out_im == nullptr) { setError("null iterative dotu result pointer"); return -1; }
    cuDoubleComplex *a=iterDeviceVector(a_id); if(!a)return -1;
    cuDoubleComplex *b=iterDeviceVector(b_id); if(!b)return -1;
    return backendDotu(a,b,g.cfg.nrows,out_re,out_im,
                       "cublasZdotu(FP64 mixed iterative)");
}

extern "C" int adda_cuda_iter_dotc(const void *a_id,const void *b_id,double *out_re,double *out_im)
{
    g_error[0]='\0';
    if (out_re == nullptr || out_im == nullptr) { setError("null iterative dotc result pointer"); return -1; }
    cuDoubleComplex *a=iterDeviceVector(a_id); if(!a)return -1;
    cuDoubleComplex *b=iterDeviceVector(b_id); if(!b)return -1;
    /* ADDA nDotProd(a,b) = sum a*conj(b). cuBLAS dotc conjugates its first
     * operand, hence dotc(b,a) gives exactly conj(b)*a. */
    return backendDotc(b,a,g.cfg.nrows,out_re,out_im,
                       "cublasZdotc(FP64 mixed iterative)");
}

extern "C" int adda_cuda_iter_dotu_self_norm2(const void *a_id,double *out_re,double *out_im,double *norm2)
{
    g_error[0]='\0';
    if (out_re == nullptr || out_im == nullptr || norm2 == nullptr) {
        setError("null iterative dotu/norm result pointer"); return -1;
    }
    cuDoubleComplex *a=iterDeviceVector(a_id); if(!a)return -1;
    return backendSelfDotuNorm2(a,g.cfg.nrows,out_re,out_im,norm2);
}

extern "C" int adda_cuda_iter_copy(const void *a_id,const void *b_id)
{
    g_error[0]='\0'; cuDoubleComplex *a=iterDeviceVector(a_id); if(!a)return -1;
    cuDoubleComplex *b=iterDeviceVector(b_id); if(!b)return -1;
    unsigned int blocks; if(iterBlocks(&blocks))return -1;
    iterCopyKernel<<<blocks,256>>>(a,b,g.cfg.nrows);
    return launchChecked("iterCopyKernel launch");
}

extern "C" int adda_cuda_iter_mult(const void *a_id,const void *b_id,double cr,double ci)
{
    g_error[0]='\0'; cuDoubleComplex *a=iterDeviceVector(a_id); if(!a)return -1;
    cuDoubleComplex *b=iterDeviceVector(b_id); if(!b)return -1;
    unsigned int blocks; if(iterBlocks(&blocks))return -1;
    qmrMultKernel<<<blocks,256>>>(a,b,make_cuDoubleComplex(cr,ci),g.cfg.nrows);
    return launchChecked("iterMultKernel launch");
}

extern "C" int adda_cuda_iter_mult_self(const void *a_id,double cr,double ci)
{
    g_error[0]='\0'; cuDoubleComplex *a=iterDeviceVector(a_id); if(!a)return -1;
    unsigned int blocks; if(iterBlocks(&blocks))return -1;
    qmrMultSelfKernel<<<blocks,256>>>(a,make_cuDoubleComplex(cr,ci),g.cfg.nrows);
    return launchChecked("iterMultSelfKernel launch");
}

extern "C" int adda_cuda_iter_mult_self_conj(const void *a_id,double c)
{
    g_error[0]='\0'; cuDoubleComplex *a=iterDeviceVector(a_id); if(!a)return -1;
    unsigned int blocks; if(iterBlocks(&blocks))return -1;
    iterMultSelfConjKernel<<<blocks,256>>>(a,c,g.cfg.nrows);
    return launchChecked("iterMultSelfConjKernel launch");
}

extern "C" int adda_cuda_iter_lincomb1(const void *a_id,const void *b_id,const void *c_id,
                                        double c1r,double c1i,double *norm2)
{
    g_error[0]='\0'; cuDoubleComplex *a=iterDeviceVector(a_id); if(!a)return -1;
    cuDoubleComplex *b=iterDeviceVector(b_id); if(!b)return -1;
    cuDoubleComplex *c=iterDeviceVector(c_id); if(!c)return -1;
    unsigned int blocks; if(iterBlocks(&blocks))return -1;
    qmrLinComb1Kernel<<<blocks,256>>>(a,b,c,make_cuDoubleComplex(c1r,c1i),g.cfg.nrows);
    if (launchChecked("iterLinComb1Kernel launch")) return -1;
    return iterNorm2(a,norm2,"cublasDznrm2(iterLinComb1)");
}

extern "C" int adda_cuda_iter_lincomb(const void *a_id,const void *b_id,const void *c_id,
                                       double c1r,double c1i,double c2r,double c2i,double *norm2)
{
    g_error[0]='\0'; cuDoubleComplex *a=iterDeviceVector(a_id); if(!a)return -1;
    cuDoubleComplex *b=iterDeviceVector(b_id); if(!b)return -1;
    cuDoubleComplex *c=iterDeviceVector(c_id); if(!c)return -1;
    unsigned int blocks; if(iterBlocks(&blocks))return -1;
    qmrLinCombKernel<<<blocks,256>>>(a,b,c,make_cuDoubleComplex(c1r,c1i),make_cuDoubleComplex(c2r,c2i),g.cfg.nrows);
    if (launchChecked("iterLinCombKernel launch")) return -1;
    return iterNorm2(a,norm2,"cublasDznrm2(iterLinComb)");
}

extern "C" int adda_cuda_iter_lincomb1_conj(const void *a_id,const void *b_id,const void *c_id,
                                             double c1r,double c1i,double *norm2)
{
    g_error[0]='\0'; cuDoubleComplex *a=iterDeviceVector(a_id); if(!a)return -1;
    cuDoubleComplex *b=iterDeviceVector(b_id); if(!b)return -1;
    cuDoubleComplex *c=iterDeviceVector(c_id); if(!c)return -1;
    unsigned int blocks; if(iterBlocks(&blocks))return -1;
    iterLinComb1ConjKernel<<<blocks,256>>>(a,b,c,make_cuDoubleComplex(c1r,c1i),g.cfg.nrows);
    if (launchChecked("iterLinComb1ConjKernel launch")) return -1;
    return iterNorm2(a,norm2,"cublasDznrm2(iterLinComb1Conj)");
}

extern "C" int adda_cuda_iter_increm01(const void *a_id,const void *b_id,double cr,double ci,double *norm2)
{
    g_error[0]='\0'; cuDoubleComplex *a=iterDeviceVector(a_id); if(!a)return -1;
    cuDoubleComplex *b=iterDeviceVector(b_id); if(!b)return -1;
    unsigned int blocks; if(iterBlocks(&blocks))return -1;
    qmrIncrem01Kernel<<<blocks,256>>>(a,b,make_cuDoubleComplex(cr,ci),g.cfg.nrows);
    if (launchChecked("iterIncrem01Kernel launch")) return -1;
    return iterNorm2(a,norm2,"cublasDznrm2(iterIncrem01)");
}

extern "C" int adda_cuda_iter_increm10(const void *a_id,const void *b_id,double cr,double ci,double *norm2)
{
    g_error[0]='\0'; cuDoubleComplex *a=iterDeviceVector(a_id); if(!a)return -1;
    cuDoubleComplex *b=iterDeviceVector(b_id); if(!b)return -1;
    unsigned int blocks; if(iterBlocks(&blocks))return -1;
    iterIncrem10Kernel<<<blocks,256>>>(a,b,make_cuDoubleComplex(cr,ci),g.cfg.nrows);
    if (launchChecked("iterIncrem10Kernel launch")) return -1;
    return iterNorm2(a,norm2,"cublasDznrm2(iterIncrem10)");
}

extern "C" int adda_cuda_iter_increm011(const void *a_id,const void *b_id,const void *c_id,
                                         double c1r,double c1i,double c2r,double c2i,double *norm2)
{
    g_error[0]='\0'; cuDoubleComplex *a=iterDeviceVector(a_id); if(!a)return -1;
    cuDoubleComplex *b=iterDeviceVector(b_id); if(!b)return -1;
    cuDoubleComplex *c=iterDeviceVector(c_id); if(!c)return -1;
    unsigned int blocks; if(iterBlocks(&blocks))return -1;
    iterIncrem011Kernel<<<blocks,256>>>(a,b,c,make_cuDoubleComplex(c1r,c1i),make_cuDoubleComplex(c2r,c2i),g.cfg.nrows);
    if (launchChecked("iterIncrem011Kernel launch")) return -1;
    return iterNorm2(a,norm2,"cublasDznrm2(iterIncrem011)");
}

extern "C" int adda_cuda_iter_increm110(const void *a_id,const void *b_id,const void *c_id,
                                         double c1r,double c1i,double c2r,double c2i)
{
    g_error[0]='\0'; cuDoubleComplex *a=iterDeviceVector(a_id); if(!a)return -1;
    cuDoubleComplex *b=iterDeviceVector(b_id); if(!b)return -1;
    cuDoubleComplex *c=iterDeviceVector(c_id); if(!c)return -1;
    unsigned int blocks; if(iterBlocks(&blocks))return -1;
    qmrIncrem110Kernel<<<blocks,256>>>(a,b,c,make_cuDoubleComplex(c1r,c1i),make_cuDoubleComplex(c2r,c2i),g.cfg.nrows);
    return launchChecked("iterIncrem110Kernel launch");
}

extern "C" int adda_cuda_iter_increm111(const void *a_id,const void *b_id,const void *c_id,
                                         double c1r,double c1i,double c2r,double c2i,double c3r,double c3i)
{
    g_error[0]='\0'; cuDoubleComplex *a=iterDeviceVector(a_id); if(!a)return -1;
    cuDoubleComplex *b=iterDeviceVector(b_id); if(!b)return -1;
    cuDoubleComplex *c=iterDeviceVector(c_id); if(!c)return -1;
    unsigned int blocks; if(iterBlocks(&blocks))return -1;
    qmrIncrem111Kernel<<<blocks,256>>>(a,b,c,make_cuDoubleComplex(c1r,c1i),make_cuDoubleComplex(c2r,c2i),
                                      make_cuDoubleComplex(c3r,c3i),g.cfg.nrows);
    return launchChecked("iterIncrem111Kernel launch");
}

extern "C" int adda_cuda_iter_increm11_d_c(const void *a_id,const void *b_id,double c1,
                                             double c2r,double c2i,double *norm2)
{
    g_error[0]='\0'; cuDoubleComplex *a=iterDeviceVector(a_id); if(!a)return -1;
    cuDoubleComplex *b=iterDeviceVector(b_id); if(!b)return -1;
    unsigned int blocks; if(iterBlocks(&blocks))return -1;
    qmrIncrem11DCKernel<<<blocks,256>>>(a,b,c1,make_cuDoubleComplex(c2r,c2i),g.cfg.nrows);
    if (launchChecked("iterIncrem11DCKernel launch")) return -1;
    return iterNorm2(a,norm2,"cublasDznrm2(iterIncrem11D_C)");
}

extern "C" int adda_cuda_iter_increm110_d_c_conj(const void *a_id,const void *b_id,const void *c_id,
                                                   double c1,double c2r,double c2i,double *norm2)
{
    g_error[0]='\0'; cuDoubleComplex *a=iterDeviceVector(a_id); if(!a)return -1;
    cuDoubleComplex *b=iterDeviceVector(b_id); if(!b)return -1;
    cuDoubleComplex *c=iterDeviceVector(c_id); if(!c)return -1;
    unsigned int blocks; if(iterBlocks(&blocks))return -1;
    iterIncrem110DCConjKernel<<<blocks,256>>>(a,b,c,c1,make_cuDoubleComplex(c2r,c2i),g.cfg.nrows);
    if (launchChecked("iterIncrem110DCConjKernel launch")) return -1;
    return iterNorm2(a,norm2,"cublasDznrm2(iterIncrem110D_C_Conj)");
}

extern "C" void adda_cuda_matvec_free(void)
{
    freeContext();
    g_error[0] = '\0';
}

extern "C" const char *adda_cuda_matvec_last_error(void)
{
    return g_error;
}
