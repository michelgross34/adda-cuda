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
#include "kernel.h"

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
#  ifdef cublasZdotu
#    undef cublasZdotu
#  endif
#  ifdef cublasZdotc
#    undef cublasZdotc
#  endif
#  ifdef cublasDznrm2
#    undef cublasDznrm2
#  endif
#  define cublasZdotu cublasCdotu
#  define cublasZdotc cublasCdotc
#  define cublasDznrm2 cublasScnrm2
typedef float AddaCudaReal;
#else
typedef cuDoubleComplex AddaCudaDoubleComplex;
typedef double AddaCudaReal;
#endif

#include "lanier_full_ai_cuda.inc"

namespace {

static_assert(sizeof(cuDoubleComplex) == 2 * sizeof(AddaCudaReal),
              "CUDA complex type must be two packed scalar values");

/* LANIER_NESTED / LANIER_MULTIZONE keep one independent FULL6 coefficient set
 * and one lightweight cuFFT plan per ADDA domain block.  V2 shares a single
 * FFT grid and a single manually managed cuFFT work area across all blocks.
 * MAX_NMAT is 60 in ADDA; keep the backend limit equal to that public domain
 * limit without changing the C ABI. */
static const int LANIER_NESTED_MAX_SLOTS=60;
static const char *LANIER_MULTIZONE_BACKEND_VERSION="V2 shared-grid/shared-workspace";
/* Optional post-correction for reverse multiplicative Schwarz.
 * beta=0 preserves the historical V1 operator exactly.
 * beta!=0 applies
 *   P_beta r = P_R r + beta P_BJ (r - A P_R r),
 * where P_R is reverse multiplicative Schwarz and P_BJ is the existing
 * strict-mask additive multizone (block-Jacobi) operator.  This adds no
 * persistent GPU vectors; the existing nested scratch is reused. */
static double schwarzReverseGlobalBeta()
{
    static int initialized=0;
    static double beta=0.0;
    if(!initialized){
        const char *env=std::getenv("ADDA_LANIER_REVERSE_GLOBAL_BETA");
        if(env!=nullptr && *env!='\0'){
            char *end=nullptr;
            const double value=std::strtod(env,&end);
            if(end!=env && end!=nullptr && *end=='\0' && value==value){
                beta=value;
                if(beta < -1.0) beta=-1.0;
                if(beta >  1.0) beta= 1.0;
            }
        }
        initialized=1;
    }
    return beta;
}
struct NestedLanierState {
    bool initialized=false;
    int active_material=-1;
    size_t origin_x=0,origin_y=0,origin_z=0;
    size_t nx=0,ny=0,nz=0,n=0;
    size_t rx=0,ry=0,rz=0,nred=0;
    size_t gridComplexCount=0,coeffComplexCount=0;
    size_t fftWorkSize=0;
    cufftHandle plan=0;
    cuDoubleComplex *coeff=nullptr;
};

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
    /* ADDA_LANIER_REFERENCE15X_V4: independent box-sized circulant. */
    cuDoubleComplex *d_lanier_tmp = nullptr;   /* compact M^-1*v */
    cuDoubleComplex *d_lanier_grid = nullptr;  /* [3][nz][ny][nx] */
    cuDoubleComplex *d_lanier_coeff = nullptr; /* six reduced inverse spectra */
    size_t lanier_nx=0,lanier_ny=0,lanier_nz=0,lanierN=0;
    size_t lanier_rx=0,lanier_ry=0,lanier_rz=0,lanierNred=0;
    size_t lanierGridComplexCount=0,lanierCoeffComplexCount=0;
    cufftHandle planLanier3d = 0;
    bool lanier_initialized = false;
    AddaCudaReal lanier_full_sqrt_dipvol = static_cast<AddaCudaReal>(1);
    /* LANIER_PARTITION_V2: when >=0, FULL6 addresses only this material
     * inside its local bounding box. The normal global FULL6 path keeps -1. */
    int lanier_active_material = -1;
    size_t lanier_origin_x = 0, lanier_origin_y = 0, lanier_origin_z = 0;
    /* Row projection for local partition solves. -1 means full physical A. */
    int project_material = -1;
    /* LANIER_NESTED / LANIER_MULTIZONE strict-mask block-Jacobi states and compact scratch. */
    NestedLanierState nested[LANIER_NESTED_MAX_SLOTS]{};
    int nested_count=0;
    bool nested_initialized=false;
    cuDoubleComplex *d_nested_grid_shared=nullptr; /* max [3][nz][ny][nx] over zones */
    size_t nestedGridCapacity=0;
    void *d_nested_fft_work=nullptr; /* one cuFFT work area shared by all zone plans */
    size_t nestedFftWorkCapacity=0;
    cuDoubleComplex *d_nested_tmp1=nullptr; /* in-place input preservation */
    cuDoubleComplex *d_nested_tmp2=nullptr; /* per-slot accumulation */
    /* LANIER_MULTIZONE_SCHUR V2.1: two extra compact vectors allocated lazily.
     * The three legacy nested scratch vectors are reused for interface actions. */
    cuDoubleComplex *d_schur_corr=nullptr;  /* one-zone Schur/LDU correction */
    cuDoubleComplex *d_schur_input=nullptr; /* preserves src only for in-place application */
    double schur_omega=0.10; /* V2.1 signed damping of the first-order Schur feedback term */
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

/* cmul moved to kernel.cu */


/* cadd_hd moved to kernel.cu */


/* csub_hd moved to kernel.cu */


/* cscale_hd moved to kernel.cu */


/* cdiv_hd moved to kernel.cu */


/* cabs2_hd moved to kernel.cu */


/* cneg_hd moved to kernel.cu */


/* cconj_hd moved to kernel.cu */


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
    ADD_BYTES_IF(g.d_lanier_tmp,g.cfg.nrows,cuDoubleComplex);
    ADD_BYTES_IF(g.d_lanier_grid,g.lanierGridComplexCount,cuDoubleComplex);
    ADD_BYTES_IF(g.d_lanier_coeff,g.lanierCoeffComplexCount,cuDoubleComplex);
    ADD_BYTES_IF(g.d_nested_tmp1,g.cfg.nrows,cuDoubleComplex);
    ADD_BYTES_IF(g.d_nested_tmp2,g.cfg.nrows,cuDoubleComplex);
    ADD_BYTES_IF(g.d_schur_corr,g.cfg.nrows,cuDoubleComplex);
    ADD_BYTES_IF(g.d_schur_input,g.cfg.nrows,cuDoubleComplex);
    ADD_BYTES_IF(g.d_nested_grid_shared,g.nestedGridCapacity,cuDoubleComplex);
    if (g.d_nested_fft_work != nullptr) {
        if (!checkedAddBytes(total,g.nestedFftWorkCapacity,total)) return SIZE_MAX;
    }
    for(int ns=0;ns<LANIER_NESTED_MAX_SLOTS;ns++) {
        ADD_BYTES_IF(g.nested[ns].coeff,g.nested[ns].coeffComplexCount,cuDoubleComplex);
    }
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
    if (g.planLanier3d) { cufftDestroy(g.planLanier3d); g.planLanier3d = 0; }
    for(int ns=0;ns<LANIER_NESTED_MAX_SLOTS;ns++) {
        if(g.nested[ns].plan){cufftDestroy(g.nested[ns].plan);g.nested[ns].plan=0;}
    }
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
    if (g.d_lanier_coeff) { cudaFree(g.d_lanier_coeff); g.d_lanier_coeff = nullptr; }
    if (g.d_lanier_grid) { cudaFree(g.d_lanier_grid); g.d_lanier_grid = nullptr; }
    for(int ns=0;ns<LANIER_NESTED_MAX_SLOTS;ns++) {
        if(g.nested[ns].coeff){cudaFree(g.nested[ns].coeff);g.nested[ns].coeff=nullptr;}
    }
    if (g.d_nested_fft_work) { cudaFree(g.d_nested_fft_work); g.d_nested_fft_work = nullptr; }
    if (g.d_nested_grid_shared) { cudaFree(g.d_nested_grid_shared); g.d_nested_grid_shared = nullptr; }
    if (g.d_schur_input) { cudaFree(g.d_schur_input); g.d_schur_input = nullptr; }
    if (g.d_schur_corr) { cudaFree(g.d_schur_corr); g.d_schur_corr = nullptr; }
    if (g.d_nested_tmp2) { cudaFree(g.d_nested_tmp2); g.d_nested_tmp2 = nullptr; }
    if (g.d_nested_tmp1) { cudaFree(g.d_nested_tmp1); g.d_nested_tmp1 = nullptr; }
    if (g.d_lanier_tmp) { cudaFree(g.d_lanier_tmp); g.d_lanier_tmp = nullptr; }
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

/* scatterKernel moved to kernel.cu */


/* symMatVec moved to kernel.cu */


/* reflMatVec moved to kernel.cu */


/* spectralMultiplyKernel moved to kernel.cu */


/* gatherKernel moved to kernel.cu */




/* ---------------- ADDA_LANIER_REFERENCE15X_V4 ----------------
 * Source-faithful three-level circulant application.
 * The CPU builder (lanier_precon.c) has already performed axis-wise
 * Chan/Lanier averaging, Fourier transformation, 3x3 inversion, and the
 * 1/N inverse-FFT normalization.  CUDA only scatters, FFTs, multiplies by the
 * reduced inverse spectrum, inverse FFTs, and gathers.
 */
/* lanierBoxScatterKernel moved to kernel.cu */


/* lanierBoxGatherKernel moved to kernel.cu */


/* LANIER_FULL uses the DDSCAT lattice-unit circulant M_hat ~= alpha_opt^-1-G_hat.
 * ADDA solves a symmetrically transformed system. With alpha_hat=cc/dipvol,
 * the corresponding right preconditioner is P=S_hat^-1 M_hat^-1 S_hat^-1,
 * S_hat=sqrt(alpha_hat). These kernels therefore multiply each 1/cc_sqrt by
 * sqrt(dipvol), without modifying the validated reference Lanier path. */
/* lanierFullBoxScatterKernel moved to kernel.cu */


/* lanierFullBoxGatherKernel moved to kernel.cu */


/* Project a compact physical MatVec result onto one material partition. */
/* materialProjectionKernel moved to kernel.cu */


/* lanierReducedMultiplyKernel moved to kernel.cu */

/* --------------------------- Slice FFT MatVec --------------------------- */
/* scatterSliceKernel moved to kernel.cu */


/* sliceToPlaneBatchKernel moved to kernel.cu */


/* spectralMultiplySliceBatchKernel moved to kernel.cu */


/* planeToSliceBatchKernel moved to kernel.cu */


/* gatherSliceKernel moved to kernel.cu */


/* qmrMultKernel moved to kernel.cu */


/* qmrMultSelfKernel moved to kernel.cu */


/* qmrLinComb1Kernel moved to kernel.cu */


/* qmrLinCombKernel moved to kernel.cu */


/* qmrIncrem110Kernel moved to kernel.cu */


/* qmrIncrem111Kernel moved to kernel.cu */


/* qmrIncrem01Kernel moved to kernel.cu */


/* qmrIncrem11DCKernel moved to kernel.cu */


/* iterCopyKernel moved to kernel.cu */


/* iterIncrem10Kernel moved to kernel.cu */


/* iterIncrem011Kernel moved to kernel.cu */


/* iterLinComb1ConjKernel moved to kernel.cu */


/* iterIncrem110DCConjKernel moved to kernel.cu */


/* iterMultSelfConjKernel moved to kernel.cu */


int launchChecked(const char *name)
{
    return failCuda(cudaGetLastError(),name);
}

#ifdef ADDA_CUDA_SINGLE_BACKEND
static const size_t ADDA_FP64_REDUCTION_CHUNK = static_cast<size_t>(1) << 20; /* 1,048,576 */

/* convertComplexF32ToF64Kernel moved to kernel.cu */


/* convertComplexPairF32ToF64Kernel moved to kernel.cu */


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
    adda_kernel_convertComplexF32ToF64Kernel(static_cast<unsigned int>(b),threads,
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
    adda_kernel_convertComplexPairF32ToF64Kernel(static_cast<unsigned int>(blocks),threads,
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
    info->lanier_workspace_bytes = 0;
    if (g.d_lanier_tmp) info->lanier_workspace_bytes += g.cfg.nrows*sizeof(cuDoubleComplex);
    if (g.d_lanier_grid) info->lanier_workspace_bytes += g.lanierGridComplexCount*sizeof(cuDoubleComplex);
    if (g.d_lanier_coeff) info->lanier_workspace_bytes += g.lanierCoeffComplexCount*sizeof(cuDoubleComplex);
    if (g.d_nested_tmp1) info->lanier_workspace_bytes += g.cfg.nrows*sizeof(cuDoubleComplex);
    if (g.d_nested_tmp2) info->lanier_workspace_bytes += g.cfg.nrows*sizeof(cuDoubleComplex);
    if (g.d_schur_corr) info->lanier_workspace_bytes += g.cfg.nrows*sizeof(cuDoubleComplex);
    if (g.d_schur_input) info->lanier_workspace_bytes += g.cfg.nrows*sizeof(cuDoubleComplex);
    if (g.d_nested_grid_shared) info->lanier_workspace_bytes += g.nestedGridCapacity*sizeof(cuDoubleComplex);
    if (g.d_nested_fft_work) info->lanier_workspace_bytes += g.nestedFftWorkCapacity;
    for(int ns=0;ns<LANIER_NESTED_MAX_SLOTS;ns++) {
        if(g.nested[ns].coeff) info->lanier_workspace_bytes += g.nested[ns].coeffComplexCount*sizeof(cuDoubleComplex);
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
    adda_kernel_scatterSliceKernel(dipBlocks,threads,d_arg,g.d_slice,g.d_material,g.d_position,g.d_cc,
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
        adda_kernel_sliceToPlaneBatchKernel(planeBatchBlocks,threads,g.d_slice,g.d_plane,kz0,active,
            g.boxX,g.boxY,g.cfg.gridX,g.boxXY,g.sliceN,g.planeN);
        if (failCuda(cudaGetLastError(),"sliceToPlaneBatchKernel launch")) return -1;
        if (failCufft(cufftExecZ2Z(g.planXY,reinterpret_cast<cufftDoubleComplex*>(g.d_plane),
                                   reinterpret_cast<cufftDoubleComplex*>(g.d_plane),CUFFT_FORWARD),
                      "cufftExecZ2Z(slice XY batch forward)")) return -1;
        const size_t activePlaneN=active*g.planeN;
        const unsigned int activePlaneBlocks=static_cast<unsigned int>((activePlaneN+threads-1)/threads);
        adda_kernel_spectralMultiplySliceBatchKernel(activePlaneBlocks,threads,g.d_plane,g.d_D,kz0,active,
            g.cfg.gridX,g.cfg.gridY,g.cfg.gridZ,g.planeN,g.DsizeX,g.cfg.DsizeY,g.cfg.DsizeZ,
            g.cfg.reduced_fft,transposed,g.low_mem_green ? 1 : 0);
        if (failCuda(cudaGetLastError(),"spectralMultiplySliceBatchKernel launch")) return -1;
        if (failCufft(cufftExecZ2Z(g.planXY,reinterpret_cast<cufftDoubleComplex*>(g.d_plane),
                                   reinterpret_cast<cufftDoubleComplex*>(g.d_plane),CUFFT_INVERSE),
                      "cufftExecZ2Z(slice XY batch inverse)")) return -1;
        const size_t activeBoxN=active*g.boxXY;
        const unsigned int activeBoxBlocks=static_cast<unsigned int>((activeBoxN+threads-1)/threads);
        adda_kernel_planeToSliceBatchKernel(activeBoxBlocks,threads,g.d_plane,g.d_slice,kz0,active,
            g.boxX,g.cfg.gridX,g.boxXY,g.sliceN,g.planeN);
        if (failCuda(cudaGetLastError(),"planeToSliceBatchKernel launch")) return -1;
    }
    for (int c=0;c<3;++c) {
        cufftDoubleComplex *ptr=reinterpret_cast<cufftDoubleComplex*>(g.d_slice+static_cast<size_t>(c)*g.sliceN);
        if (failCufft(cufftExecZ2Z(g.planZ,ptr,ptr,CUFFT_INVERSE),"cufftExecZ2Z(slice Z inverse)")) return -1;
    }
    adda_kernel_gatherSliceKernel(dipBlocks,threads,d_arg,d_result,g.d_slice,g.d_material,g.d_position,g.d_cc,
                                             g.cfg.ndip,g.boxX,g.boxY,g.sliceN,her!=0);
    if (failCuda(cudaGetLastError(),"gatherSliceKernel launch")) return -1;
    if (g.project_material>=0) {
        adda_kernel_materialProjectionKernel(dipBlocks,threads,d_result,g.d_material,g.cfg.ndip,g.project_material);
        if (failCuda(cudaGetLastError(),"materialProjectionKernel(slice) launch")) return -1;
    }
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
    adda_kernel_scatterKernel(dipBlocks,threads,d_arg,g.d_grid,g.d_material,g.d_position,g.d_cc,
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
    adda_kernel_spectralMultiplyKernel(gridBlocks,threads,g.d_grid,g.d_gridR,g.d_D,g.d_R,
                                                   g.cfg.gridX,g.cfg.gridY,g.cfg.gridZ,g.gridN,
                                                   g.cfg.DsizeY,g.cfg.DsizeZ,g.cfg.RsizeY,
                                                   g.cfg.reduced_fft,transposed,g.cfg.surface);
    if (failCuda(cudaGetLastError(),"spectralMultiplyKernel launch")) return -1;

    if (failCufft(cufftExecZ2Z(g.plan3d,
                               reinterpret_cast<cufftDoubleComplex*>(g.d_grid),
                               reinterpret_cast<cufftDoubleComplex*>(g.d_grid),
                               CUFFT_INVERSE),
                  "cufftExecZ2Z(3D inverse)")) return -1;

    adda_kernel_gatherKernel(dipBlocks,threads,d_arg,d_result,g.d_grid,g.d_material,g.d_position,g.d_cc,
                                       g.cfg.ndip,g.cfg.gridX,g.cfg.gridY,g.gridN,her != 0);
    if (failCuda(cudaGetLastError(),"gatherKernel launch")) return -1;
    if (g.project_material>=0) {
        adda_kernel_materialProjectionKernel(dipBlocks,threads,d_result,g.d_material,g.cfg.ndip,g.project_material);
        if (failCuda(cudaGetLastError(),"materialProjectionKernel(full) launch")) return -1;
    }

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

static void lanierReleaseNoFail()
{
    if(g.planLanier3d){cufftDestroy(g.planLanier3d);g.planLanier3d=0;}
    if(g.d_lanier_coeff){cudaFree(g.d_lanier_coeff);g.d_lanier_coeff=nullptr;}
    if(g.d_lanier_grid){cudaFree(g.d_lanier_grid);g.d_lanier_grid=nullptr;}
    if(g.d_lanier_tmp){cudaFree(g.d_lanier_tmp);g.d_lanier_tmp=nullptr;}
    g.lanier_nx=g.lanier_ny=g.lanier_nz=g.lanierN=0;
    g.lanier_rx=g.lanier_ry=g.lanier_rz=g.lanierNred=0;
    g.lanierGridComplexCount=g.lanierCoeffComplexCount=0;
    g.lanier_initialized=false;
    g.lanier_active_material=-1;
    g.lanier_origin_x=g.lanier_origin_y=g.lanier_origin_z=0;
}

static void nestedReleaseNoFail()
{
    for(int ns=0;ns<LANIER_NESTED_MAX_SLOTS;ns++) {
        NestedLanierState &st=g.nested[ns];
        if(st.plan){cufftDestroy(st.plan);st.plan=0;}
        if(st.coeff){cudaFree(st.coeff);st.coeff=nullptr;}
        st=NestedLanierState{};
    }
    if(g.d_nested_fft_work){cudaFree(g.d_nested_fft_work);g.d_nested_fft_work=nullptr;}
    if(g.d_nested_grid_shared){cudaFree(g.d_nested_grid_shared);g.d_nested_grid_shared=nullptr;}
    g.nestedFftWorkCapacity=0;
    g.nestedGridCapacity=0;
    if(g.d_schur_input){cudaFree(g.d_schur_input);g.d_schur_input=nullptr;}
    if(g.d_schur_corr){cudaFree(g.d_schur_corr);g.d_schur_corr=nullptr;}
    if(g.d_nested_tmp2){cudaFree(g.d_nested_tmp2);g.d_nested_tmp2=nullptr;}
    if(g.d_nested_tmp1){cudaFree(g.d_nested_tmp1);g.d_nested_tmp1=nullptr;}
    /* d_lanier_tmp is shared as the compact P*v scratch in nested mode. */
    if(g.d_lanier_tmp){cudaFree(g.d_lanier_tmp);g.d_lanier_tmp=nullptr;}
    g.nested_count=0;g.nested_initialized=false;
}

static int lanierApplyDeviceMode(const cuDoubleComplex *src,cuDoubleComplex *dst,const bool full_mode)
{
    if(!g.lanier_initialized || g.d_lanier_tmp==nullptr || g.d_lanier_grid==nullptr ||
       g.d_lanier_coeff==nullptr || g.planLanier3d==0){
        setError(full_mode ? "Lanier full preconditioner is not initialized" : "Lanier reference preconditioner is not initialized"); return -1;
    }
    if(full_mode && (g.d_cc==nullptr || g.d_material==nullptr)){
        setError("Lanier full requires cc_sqrt and material on the GPU"); return -1;
    }
    if(failCuda(cudaMemset(g.d_lanier_grid,0,g.lanierGridComplexCount*sizeof(cuDoubleComplex)),
                full_mode ? "clear Lanier full grid" : "clear Lanier reference grid")) return -1;
    const int threads=256;
    const unsigned int db=static_cast<unsigned int>((g.cfg.ndip+threads-1)/threads);
    if(full_mode)
        adda_kernel_lanierFullBoxScatterKernel(db,threads,src,g.d_lanier_grid,g.d_position,g.d_material,g.d_cc,g.cfg.ndip,
                                                   g.lanier_nx,g.lanier_ny,g.lanierN,g.lanier_full_sqrt_dipvol,
                                                   g.lanier_active_material,g.lanier_origin_x,g.lanier_origin_y,g.lanier_origin_z);
    else
        adda_kernel_lanierBoxScatterKernel(db,threads,src,g.d_lanier_grid,g.d_position,g.cfg.ndip,
                                               g.lanier_nx,g.lanier_ny,g.lanierN);
    if(launchChecked(full_mode ? "lanierFullBoxScatterKernel" : "lanierBoxScatterKernel")) return -1;
    if(failCufft(cufftExecZ2Z(g.planLanier3d,
                              reinterpret_cast<cufftDoubleComplex*>(g.d_lanier_grid),
                              reinterpret_cast<cufftDoubleComplex*>(g.d_lanier_grid),CUFFT_FORWARD),
                 full_mode ? "cufftExecZ2Z(Lanier full forward)" : "cufftExecZ2Z(Lanier reference forward)")) return -1;
    const unsigned int gb=static_cast<unsigned int>((g.lanierN+threads-1)/threads);
    adda_kernel_lanierReducedMultiplyKernel(gb,threads,g.d_lanier_grid,g.d_lanier_coeff,
        g.lanier_nx,g.lanier_ny,g.lanier_nz,g.lanierN,
        g.lanier_rx,g.lanier_ry,g.lanier_rz,g.lanierNred);
    if(launchChecked("lanierReducedMultiplyKernel")) return -1;
    if(failCufft(cufftExecZ2Z(g.planLanier3d,
                              reinterpret_cast<cufftDoubleComplex*>(g.d_lanier_grid),
                              reinterpret_cast<cufftDoubleComplex*>(g.d_lanier_grid),CUFFT_INVERSE),
                 full_mode ? "cufftExecZ2Z(Lanier full inverse)" : "cufftExecZ2Z(Lanier reference inverse)")) return -1;
    if(full_mode)
        adda_kernel_lanierFullBoxGatherKernel(db,threads,g.d_lanier_grid,dst,g.d_position,g.d_material,g.d_cc,g.cfg.ndip,
                                                  g.lanier_nx,g.lanier_ny,g.lanierN,g.lanier_full_sqrt_dipvol,
                                                  g.lanier_active_material,g.lanier_origin_x,g.lanier_origin_y,g.lanier_origin_z);
    else
        adda_kernel_lanierBoxGatherKernel(db,threads,g.d_lanier_grid,dst,g.d_position,g.cfg.ndip,
                                              g.lanier_nx,g.lanier_ny,g.lanierN);
    return launchChecked(full_mode ? "lanierFullBoxGatherKernel" : "lanierBoxGatherKernel");
}

static int lanierApplyDevice(const cuDoubleComplex *src,cuDoubleComplex *dst)
{
    return lanierApplyDeviceMode(src,dst,false);
}

static int lanierFullApplyDevice(const cuDoubleComplex *src,cuDoubleComplex *dst)
{
    return lanierApplyDeviceMode(src,dst,true);
}

static int lanierConjugateSelf(cuDoubleComplex *v,const char *label);

/* Apply one cached material block. Scatter and gather both enforce the same
 * material mask, i.e. M_i P_i^-1 M_i. This avoids the output leakage present
 * in some earlier multi-region implementations. */
static int nestedEnsureSharedGrid(size_t complex_count)
{
    if(complex_count<=g.nestedGridCapacity && g.d_nested_grid_shared!=nullptr)return 0;
    cuDoubleComplex *replacement=nullptr;
    /* Release the smaller buffer first. During initialization no zone FFT is
     * executing, and avoiding old+new coexistence preserves the low-memory
     * purpose of this implementation. */
    if(g.d_nested_grid_shared){cudaFree(g.d_nested_grid_shared);g.d_nested_grid_shared=nullptr;g.nestedGridCapacity=0;}
    if(failCuda(cudaMalloc(reinterpret_cast<void**>(&replacement),complex_count*sizeof(cuDoubleComplex)),
                "cudaMalloc(Lanier multizone shared FFT grid)"))return -1;
    g.d_nested_grid_shared=replacement;
    g.nestedGridCapacity=complex_count;
    return 0;
}

static int nestedEnsureSharedFftWork(size_t bytes)
{
    if(bytes>g.nestedFftWorkCapacity || (bytes>0 && g.d_nested_fft_work==nullptr)) {
        void *replacement=nullptr;
        if(g.d_nested_fft_work){cudaFree(g.d_nested_fft_work);g.d_nested_fft_work=nullptr;g.nestedFftWorkCapacity=0;}
        if(bytes>0 && failCuda(cudaMalloc(&replacement,bytes),
                               "cudaMalloc(Lanier multizone shared cuFFT workspace)"))return -1;
        g.d_nested_fft_work=replacement;
        g.nestedFftWorkCapacity=bytes;
    }
    /* A newly allocated shared work area must be rebound to every already
     * created no-auto-allocation plan. If cuFFT reports zero required bytes,
     * no work-area binding is needed. */
    if(g.d_nested_fft_work!=nullptr) {
        for(int ns=0;ns<g.nested_count;ns++) {
            if(g.nested[ns].plan && failCufft(cufftSetWorkArea(g.nested[ns].plan,g.d_nested_fft_work),
                                              "cufftSetWorkArea(Lanier multizone shared)"))return -1;
        }
    }
    return 0;
}

static int nestedSlotApplyDevice(NestedLanierState &st,const cuDoubleComplex *src,cuDoubleComplex *dst)
{
    if(!st.initialized || g.d_nested_grid_shared==nullptr || st.coeff==nullptr || st.plan==0 ||
       st.gridComplexCount>g.nestedGridCapacity){
        setError("Lanier nested/multizone slot or shared FFT grid is not initialized");return -1;
    }
    cuDoubleComplex *grid=g.d_nested_grid_shared;
    if(failCuda(cudaMemset(grid,0,st.gridComplexCount*sizeof(cuDoubleComplex)),
                "clear Lanier multizone shared FFT grid"))return -1;
    const int threads=256;
    const unsigned int db=static_cast<unsigned int>((g.cfg.ndip+threads-1)/threads);
    adda_kernel_lanierFullBoxScatterKernel(db,threads,src,grid,g.d_position,g.d_material,g.d_cc,g.cfg.ndip,
        st.nx,st.ny,st.n,g.lanier_full_sqrt_dipvol,st.active_material,st.origin_x,st.origin_y,st.origin_z);
    if(launchChecked("lanierNestedBoxScatterKernel"))return -1;
    if(failCufft(cufftExecZ2Z(st.plan,reinterpret_cast<cufftDoubleComplex*>(grid),
                              reinterpret_cast<cufftDoubleComplex*>(grid),CUFFT_FORWARD),
                 "cufftExecZ2Z(Lanier nested forward)"))return -1;
    const unsigned int gb=static_cast<unsigned int>((st.n+threads-1)/threads);
    adda_kernel_lanierReducedMultiplyKernel(gb,threads,grid,st.coeff,st.nx,st.ny,st.nz,st.n,
                                                st.rx,st.ry,st.rz,st.nred);
    if(launchChecked("lanierNestedReducedMultiplyKernel"))return -1;
    if(failCufft(cufftExecZ2Z(st.plan,reinterpret_cast<cufftDoubleComplex*>(grid),
                              reinterpret_cast<cufftDoubleComplex*>(grid),CUFFT_INVERSE),
                 "cufftExecZ2Z(Lanier nested inverse)"))return -1;
    adda_kernel_lanierFullBoxGatherKernel(db,threads,grid,dst,g.d_position,g.d_material,g.d_cc,g.cfg.ndip,
        st.nx,st.ny,st.n,g.lanier_full_sqrt_dipvol,st.active_material,st.origin_x,st.origin_y,st.origin_z);
    return launchChecked("lanierNestedBoxGatherKernel");
}

static int nestedApplyDevice(const cuDoubleComplex *src,cuDoubleComplex *dst)
{
    if(!g.nested_initialized || g.nested_count<2 || g.d_nested_tmp1==nullptr ||
       g.d_nested_tmp2==nullptr || g.d_lanier_tmp==nullptr){
        setError("Lanier nested preconditioner is not initialized");return -1;
    }
    const cuDoubleComplex *input=src;
    const size_t bytes=g.cfg.nrows*sizeof(cuDoubleComplex);
    if(src==dst){
        if(failCuda(cudaMemcpy(g.d_nested_tmp1,src,bytes,cudaMemcpyDeviceToDevice),
                    "preserve Lanier nested in-place input"))return -1;
        input=g.d_nested_tmp1;
    }
    if(nestedSlotApplyDevice(g.nested[0],input,dst))return -1;
    const int threads=256;
    const unsigned int blocks=static_cast<unsigned int>((g.cfg.nrows+threads-1)/threads);
    const cuDoubleComplex one=make_cuDoubleComplex(static_cast<AddaCudaReal>(1),static_cast<AddaCudaReal>(0));
    for(int ns=1;ns<g.nested_count;ns++) {
        if(nestedSlotApplyDevice(g.nested[ns],input,g.d_nested_tmp2))return -1;
        adda_kernel_qmrIncrem01Kernel(blocks,threads,dst,g.d_nested_tmp2,one,g.cfg.nrows);
        if(launchChecked("Lanier nested block accumulation"))return -1;
    }
    return 0;
}

/* Fixed-order multiplicative Schwarz / forward block Gauss--Seidel.
 * For residual r and correction z:
 *   r_0 = r, z_0 = 0
 *   delta_i = P_i^{-1} M_i r_{i-1}
 *   z_i     = z_{i-1} + delta_i
 *   r_i     = r_{i-1} - A delta_i
 * Each P_i is the same strict-mask FULL6 regional inverse used by the
 * block-Jacobi multizone mode. A is always the exact global ADDA MatVec, so
 * cross-zone coupling enters after every regional correction. The fixed
 * material/domain order makes this a linear but generally nonsymmetric
 * preconditioner. */
static int schwarzApplyDevice(const cuDoubleComplex *src,cuDoubleComplex *dst)
{
    if(!g.nested_initialized || g.nested_count<2 || g.d_nested_tmp1==nullptr ||
       g.d_nested_tmp2==nullptr || g.d_lanier_tmp==nullptr){
        setError("Lanier multizone Schwarz preconditioner is not initialized");return -1;
    }
    const size_t bytes=g.cfg.nrows*sizeof(cuDoubleComplex);
    /* tmp1 is the evolving residual. Copy before clearing dst so in-place
     * application (src==dst) remains valid. */
    if(failCuda(cudaMemcpy(g.d_nested_tmp1,src,bytes,cudaMemcpyDeviceToDevice),
                "initialize Lanier Schwarz residual"))return -1;
    if(failCuda(cudaMemset(dst,0,bytes),"clear Lanier Schwarz accumulated correction"))return -1;

    const int threads=256;
    const unsigned int blocks=static_cast<unsigned int>((g.cfg.nrows+threads-1)/threads);
    const cuDoubleComplex one=make_cuDoubleComplex(static_cast<AddaCudaReal>(1),static_cast<AddaCudaReal>(0));
    const cuDoubleComplex minus_one=make_cuDoubleComplex(static_cast<AddaCudaReal>(-1),static_cast<AddaCudaReal>(0));

    for(int ns=0;ns<g.nested_count;ns++) {
        /* tmp2 = M_i P_i^{-1} M_i r_{i-1}. */
        if(nestedSlotApplyDevice(g.nested[ns],g.d_nested_tmp1,g.d_nested_tmp2))return -1;
        adda_kernel_qmrIncrem01Kernel(blocks,threads,dst,g.d_nested_tmp2,one,g.cfg.nrows);
        if(launchChecked("Lanier Schwarz correction accumulation"))return -1;

        /* No residual update is needed after the last block. */
        if(ns+1<g.nested_count) {
            if(matvecGpuCore(g.d_nested_tmp2,g.d_lanier_tmp,0,nullptr,nullptr))return -1;
            adda_kernel_qmrIncrem01Kernel(blocks,threads,g.d_nested_tmp1,g.d_lanier_tmp,minus_one,g.cfg.nrows);
            if(launchChecked("Lanier Schwarz residual update"))return -1;
        }
    }
    return 0;
}

static int schwarzMatvecRightDevice(const cuDoubleComplex *src,cuDoubleComplex *dst,double *inprod)
{
    /* Use dst itself as the temporary preconditioned direction.  After the
     * Schwarz sweep tmp1 is no longer needed as a residual, so it can hold
     * A*(P_MS*src), which is finally copied back to dst. */
    if(schwarzApplyDevice(src,dst))return -1;
    if(matvecGpuCore(dst,g.d_nested_tmp1,0,inprod,nullptr))return -1;
    return failCuda(cudaMemcpy(dst,g.d_nested_tmp1,g.cfg.nrows*sizeof(cuDoubleComplex),cudaMemcpyDeviceToDevice),
                    "copy Lanier Schwarz right-preconditioned MatVec result");
}

static int schwarzAxpyDevice(cuDoubleComplex *dst,const cuDoubleComplex *src,cuDoubleComplex alpha)
{
    if(!g.nested_initialized || g.nested_count<2 || g.d_nested_tmp1==nullptr ||
       g.d_nested_tmp2==nullptr || g.d_lanier_tmp==nullptr){
        setError("Lanier multizone Schwarz preconditioner is not initialized");return -1;
    }
    const size_t bytes=g.cfg.nrows*sizeof(cuDoubleComplex);
    if(failCuda(cudaMemcpy(g.d_nested_tmp1,src,bytes,cudaMemcpyDeviceToDevice),
                "initialize Lanier Schwarz axpy residual"))return -1;
    const int threads=256;
    const unsigned int blocks=static_cast<unsigned int>((g.cfg.nrows+threads-1)/threads);
    const cuDoubleComplex minus_one=make_cuDoubleComplex(static_cast<AddaCudaReal>(-1),static_cast<AddaCudaReal>(0));
    for(int ns=0;ns<g.nested_count;ns++) {
        if(nestedSlotApplyDevice(g.nested[ns],g.d_nested_tmp1,g.d_nested_tmp2))return -1;
        adda_kernel_qmrIncrem01Kernel(blocks,threads,dst,g.d_nested_tmp2,alpha,g.cfg.nrows);
        if(launchChecked("Lanier Schwarz physical-solution update"))return -1;
        if(ns+1<g.nested_count) {
            if(matvecGpuCore(g.d_nested_tmp2,g.d_lanier_tmp,0,nullptr,nullptr))return -1;
            adda_kernel_qmrIncrem01Kernel(blocks,threads,g.d_nested_tmp1,g.d_lanier_tmp,minus_one,g.cfg.nrows);
            if(launchChecked("Lanier Schwarz axpy residual update"))return -1;
        }
    }
    return 0;
}


/* Reverse-order multiplicative Schwarz / reverse block Gauss--Seidel.
 * For N regional FULL6 blocks the sequence is N-1,N-2,...,0. The
 * mathematical recurrence is otherwise identical to the forward Schwarz
 * mode and uses the exact global ADDA operator A after every correction
 * except the final one. This is a fixed linear, generally nonsymmetric,
 * right preconditioner. */
static int schwarzReverseApplyDevice(const cuDoubleComplex *src,cuDoubleComplex *dst)
{
    if(!g.nested_initialized || g.nested_count<2 || g.d_nested_tmp1==nullptr ||
       g.d_nested_tmp2==nullptr || g.d_lanier_tmp==nullptr){
        setError("Lanier multizone reverse Schwarz preconditioner is not initialized");return -1;
    }
    const double beta=schwarzReverseGlobalBeta();
    const size_t bytes=g.cfg.nrows*sizeof(cuDoubleComplex);
    if(failCuda(cudaMemcpy(g.d_nested_tmp1,src,bytes,cudaMemcpyDeviceToDevice),
                "initialize Lanier reverse Schwarz residual"))return -1;
    if(failCuda(cudaMemset(dst,0,bytes),"clear Lanier reverse Schwarz accumulated correction"))return -1;

    const int threads=256;
    const unsigned int blocks=static_cast<unsigned int>((g.cfg.nrows+threads-1)/threads);
    const cuDoubleComplex one=make_cuDoubleComplex(static_cast<AddaCudaReal>(1),static_cast<AddaCudaReal>(0));
    const cuDoubleComplex minus_one=make_cuDoubleComplex(static_cast<AddaCudaReal>(-1),static_cast<AddaCudaReal>(0));

    for(int ns=g.nested_count-1;ns>=0;ns--) {
        if(nestedSlotApplyDevice(g.nested[ns],g.d_nested_tmp1,g.d_nested_tmp2))return -1;
        adda_kernel_qmrIncrem01Kernel(blocks,threads,dst,g.d_nested_tmp2,one,g.cfg.nrows);
        if(launchChecked("Lanier reverse Schwarz correction accumulation"))return -1;
        /* beta!=0 needs the exact post-sweep residual, including the last zone. */
        if(ns>0 || beta!=0.0) {
            if(matvecGpuCore(g.d_nested_tmp2,g.d_lanier_tmp,0,nullptr,nullptr))return -1;
            adda_kernel_qmrIncrem01Kernel(blocks,threads,g.d_nested_tmp1,g.d_lanier_tmp,minus_one,g.cfg.nrows);
            if(launchChecked("Lanier reverse Schwarz residual update"))return -1;
        }
    }

    if(beta!=0.0) {
        /* tmp1 = r - A*P_reverse*r. Reuse the additive multizone operator as
         * a global residual post-correction; no extra persistent vector. */
        if(nestedApplyDevice(g.d_nested_tmp1,g.d_lanier_tmp))return -1;
        const cuDoubleComplex b=make_cuDoubleComplex(static_cast<AddaCudaReal>(beta),static_cast<AddaCudaReal>(0));
        adda_kernel_qmrIncrem01Kernel(blocks,threads,dst,g.d_lanier_tmp,b,g.cfg.nrows);
        if(launchChecked("Lanier reverse Schwarz global residual post-correction"))return -1;
    }
    return 0;
}

static int schwarzReverseMatvecRightDevice(const cuDoubleComplex *src,cuDoubleComplex *dst,double *inprod)
{
    if(schwarzReverseApplyDevice(src,dst))return -1;
    if(matvecGpuCore(dst,g.d_nested_tmp1,0,inprod,nullptr))return -1;
    return failCuda(cudaMemcpy(dst,g.d_nested_tmp1,g.cfg.nrows*sizeof(cuDoubleComplex),cudaMemcpyDeviceToDevice),
                    "copy Lanier reverse Schwarz right-preconditioned MatVec result");
}

static int schwarzReverseAxpyDevice(cuDoubleComplex *dst,const cuDoubleComplex *src,cuDoubleComplex alpha)
{
    if(!g.nested_initialized || g.nested_count<2 || g.d_nested_tmp1==nullptr ||
       g.d_nested_tmp2==nullptr || g.d_lanier_tmp==nullptr){
        setError("Lanier multizone reverse Schwarz preconditioner is not initialized");return -1;
    }
    const double beta=schwarzReverseGlobalBeta();
    const size_t bytes=g.cfg.nrows*sizeof(cuDoubleComplex);
    if(failCuda(cudaMemcpy(g.d_nested_tmp1,src,bytes,cudaMemcpyDeviceToDevice),
                "initialize Lanier reverse Schwarz axpy residual"))return -1;
    const int threads=256;
    const unsigned int blocks=static_cast<unsigned int>((g.cfg.nrows+threads-1)/threads);
    const cuDoubleComplex minus_one=make_cuDoubleComplex(static_cast<AddaCudaReal>(-1),static_cast<AddaCudaReal>(0));
    for(int ns=g.nested_count-1;ns>=0;ns--) {
        if(nestedSlotApplyDevice(g.nested[ns],g.d_nested_tmp1,g.d_nested_tmp2))return -1;
        adda_kernel_qmrIncrem01Kernel(blocks,threads,dst,g.d_nested_tmp2,alpha,g.cfg.nrows);
        if(launchChecked("Lanier reverse Schwarz physical-solution update"))return -1;
        if(ns>0 || beta!=0.0) {
            if(matvecGpuCore(g.d_nested_tmp2,g.d_lanier_tmp,0,nullptr,nullptr))return -1;
            adda_kernel_qmrIncrem01Kernel(blocks,threads,g.d_nested_tmp1,g.d_lanier_tmp,minus_one,g.cfg.nrows);
            if(launchChecked("Lanier reverse Schwarz axpy residual update"))return -1;
        }
    }
    if(beta!=0.0) {
        if(nestedApplyDevice(g.d_nested_tmp1,g.d_lanier_tmp))return -1;
        const cuDoubleComplex alpha_beta=make_cuDoubleComplex(
            static_cast<AddaCudaReal>(alpha.x*beta),static_cast<AddaCudaReal>(alpha.y*beta));
        adda_kernel_qmrIncrem01Kernel(blocks,threads,dst,g.d_lanier_tmp,alpha_beta,g.cfg.nrows);
        if(launchChecked("Lanier reverse Schwarz global physical-solution update"))return -1;
    }
    return 0;
}


/* LANIER_MULTIZONE_SCHUR V2.1 (signed damped) --------------------------------
 *
 * Ordered nearest-neighbor approximate Schur-LDU preconditioner.  Let
 *   A_ij = M_i A M_j
 * and let B_i be the strict-mask FULL6 regional inverse already cached for
 * zone i.  The first-order inverse of the local Schur complement generated
 * by eliminating the immediately preceding (outer) zone is approximated by
 *
 *   Shat_i^{-1} q = B_i q
 *                   + omega B_i A_{i,i-1} B_{i-1} A_{i-1,i} B_i q, i > 0,
 *   Shat_0^{-1} q = B_0 q.
 *
 * omega in [-1,1] scales the feedback term. omega=1 reproduces V1 exactly;
 * omega=0 removes the feedback term while retaining the nearest-neighbor LDU
 * forward/backward coupling. This diagnostic endpoint is useful to separate
 * instability of the Schur feedback from instability of the LDU chain itself.
 * For omega=1 this is the first two terms of the Neumann expansion of
 * (A_ii - A_{i,i-1} B_{i-1} A_{i-1,i})^{-1}.  Every A_ij action below is
 * formed with the exact global ADDA MatVec followed by a strict row mask;
 * no Green interaction is approximated or dropped inside an interface
 * action.  Only non-neighbor block couplings are omitted from the Schur
 * factorization itself.
 *
 * The corrected diagonal blocks are then used in a block-LDU chain:
 *
 *   p_0 = Shat_0^{-1} r_0
 *   p_i = Shat_i^{-1}(r_i - A_{i,i-1} p_{i-1})
 *   x_{N-1} = p_{N-1}
 *   x_i = p_i - Shat_i^{-1} A_{i,i+1} x_{i+1}.
 *
 * Domain IDs therefore define the chain ordering.  The intended V1 use is
 * an ordered layered particle (especially concentric onion/core-shell
 * domains).  For arbitrary disconnected domains, adjacency by domain ID is
 * only an algebraic approximation and should not be interpreted as a
 * geometric-neighbor detector.
 */
static int schurEnsureScratch()
{
    if(g.d_schur_corr!=nullptr && g.d_schur_input!=nullptr)return 0;
    const size_t bytes=g.cfg.nrows*sizeof(cuDoubleComplex);
    if(g.d_schur_corr==nullptr &&
       failCuda(cudaMalloc(reinterpret_cast<void**>(&g.d_schur_corr),bytes),
                "cudaMalloc(Lanier multizone Schur correction scratch)"))return -1;
    if(g.d_schur_input==nullptr &&
       failCuda(cudaMalloc(reinterpret_cast<void**>(&g.d_schur_input),bytes),
                "cudaMalloc(Lanier multizone Schur input scratch)"))return -1;
    updateExplicitPeak();
    return 0;
}

static int schurProjectInPlace(cuDoubleComplex *v,int active_material,const char *label)
{
    const int threads=256;
    const unsigned int dipBlocks=static_cast<unsigned int>((g.cfg.ndip+threads-1)/threads);
    adda_kernel_materialProjectionKernel(dipBlocks,threads,v,g.d_material,g.cfg.ndip,active_material);
    return launchChecked(label);
}

static int schurCopyProjected(const cuDoubleComplex *src,cuDoubleComplex *dst,
                              int active_material,const char *label)
{
    const size_t bytes=g.cfg.nrows*sizeof(cuDoubleComplex);
    if(failCuda(cudaMemcpy(dst,src,bytes,cudaMemcpyDeviceToDevice),label))return -1;
    return schurProjectInPlace(dst,active_material,"Lanier Schur material projection");
}

static int schurEffectiveInverseDevice(int ns,const cuDoubleComplex *src,cuDoubleComplex *dst)
{
    if(ns<0 || ns>=g.nested_count){setError("invalid Lanier Schur zone index");return -1;}
    if(nestedSlotApplyDevice(g.nested[ns],src,dst))return -1;
    if(ns==0 || g.schur_omega==0.0)return 0;

    /* t1 = A_{ns-1,ns} B_ns src. */
    if(matvecGpuCore(dst,g.d_nested_tmp1,0,nullptr,nullptr))return -1;
    if(schurProjectInPlace(g.d_nested_tmp1,g.nested[ns-1].active_material,
                           "Lanier Schur outer-neighbor projection"))return -1;

    /* t2 = B_{ns-1} A_{ns-1,ns} B_ns src. */
    if(nestedSlotApplyDevice(g.nested[ns-1],g.d_nested_tmp1,g.d_nested_tmp2))return -1;

    /* t3 = A_{ns,ns-1} t2. */
    if(matvecGpuCore(g.d_nested_tmp2,g.d_lanier_tmp,0,nullptr,nullptr))return -1;
    if(schurProjectInPlace(g.d_lanier_tmp,g.nested[ns].active_material,
                           "Lanier Schur inner-zone feedback projection"))return -1;

    /* t1 = B_ns A_{ns,ns-1} B_{ns-1} A_{ns-1,ns} B_ns src. */
    if(nestedSlotApplyDevice(g.nested[ns],g.d_lanier_tmp,g.d_nested_tmp1))return -1;
    const int threads=256;
    const unsigned int blocks=static_cast<unsigned int>((g.cfg.nrows+threads-1)/threads);
    const cuDoubleComplex omega=make_cuDoubleComplex(static_cast<AddaCudaReal>(g.schur_omega),
                                                       static_cast<AddaCudaReal>(0));
    adda_kernel_qmrIncrem01Kernel(blocks,threads,dst,g.d_nested_tmp1,omega,g.cfg.nrows);
    return launchChecked("Lanier Schur damped effective-diagonal correction");
}

static int schurApplyDevice(const cuDoubleComplex *src,cuDoubleComplex *dst)
{
    if(!g.nested_initialized || g.nested_count<2 || g.d_nested_tmp1==nullptr ||
       g.d_nested_tmp2==nullptr || g.d_lanier_tmp==nullptr){
        setError("Lanier multizone Schur preconditioner is not initialized");return -1;
    }
    if(schurEnsureScratch())return -1;
    const size_t bytes=g.cfg.nrows*sizeof(cuDoubleComplex);
    const cuDoubleComplex *input=src;
    if(src==dst){
        if(failCuda(cudaMemcpy(g.d_schur_input,src,bytes,cudaMemcpyDeviceToDevice),
                    "preserve Lanier Schur in-place input"))return -1;
        input=g.d_schur_input;
    }
    if(failCuda(cudaMemset(dst,0,bytes),"clear Lanier Schur LDU accumulator"))return -1;

    const int threads=256;
    const unsigned int blocks=static_cast<unsigned int>((g.cfg.nrows+threads-1)/threads);
    const cuDoubleComplex one=make_cuDoubleComplex(static_cast<AddaCudaReal>(1),static_cast<AddaCudaReal>(0));
    const cuDoubleComplex minus_one=make_cuDoubleComplex(static_cast<AddaCudaReal>(-1),static_cast<AddaCudaReal>(0));

    /* Forward elimination / diagonal solve. p_0 is written directly into dst. */
    if(schurEffectiveInverseDevice(0,input,dst))return -1;
    for(int ns=1;ns<g.nested_count;ns++) {
        /* tmp1 = p_{ns-1}, tmp2 = A p_{ns-1}. Only the previous-zone rows
         * survive in tmp1, therefore the subsequent Shat_ns mask sees exactly
         * A_{ns,ns-1} p_{ns-1}. */
        if(schurCopyProjected(dst,g.d_nested_tmp1,g.nested[ns-1].active_material,
                              "copy Lanier Schur previous forward zone"))return -1;
        if(matvecGpuCore(g.d_nested_tmp1,g.d_nested_tmp2,0,nullptr,nullptr))return -1;
        if(failCuda(cudaMemcpy(g.d_nested_tmp1,input,bytes,cudaMemcpyDeviceToDevice),
                    "restore Lanier Schur forward rhs"))return -1;
        adda_kernel_qmrIncrem01Kernel(blocks,threads,g.d_nested_tmp1,g.d_nested_tmp2,minus_one,g.cfg.nrows);
        if(launchChecked("Lanier Schur forward reduced rhs"))return -1;
        if(schurEffectiveInverseDevice(ns,g.d_nested_tmp1,g.d_schur_corr))return -1;
        adda_kernel_qmrIncrem01Kernel(blocks,threads,dst,g.d_schur_corr,one,g.cfg.nrows);
        if(launchChecked("Lanier Schur forward block accumulation"))return -1;
    }

    /* Back substitution.  At step ns, zone ns+1 in dst has already been
     * converted from p to final x, while zone ns still contains p_ns. */
    for(int ns=g.nested_count-2;ns>=0;ns--) {
        if(schurCopyProjected(dst,g.d_nested_tmp1,g.nested[ns+1].active_material,
                              "copy Lanier Schur next backward zone"))return -1;
        if(matvecGpuCore(g.d_nested_tmp1,g.d_nested_tmp2,0,nullptr,nullptr))return -1;
        if(schurEffectiveInverseDevice(ns,g.d_nested_tmp2,g.d_schur_corr))return -1;
        adda_kernel_qmrIncrem01Kernel(blocks,threads,dst,g.d_schur_corr,minus_one,g.cfg.nrows);
        if(launchChecked("Lanier Schur backward substitution"))return -1;
    }
    return 0;
}

static int schurMatvecRightDevice(const cuDoubleComplex *src,cuDoubleComplex *dst,double *inprod)
{
    if(schurApplyDevice(src,dst))return -1;
    if(matvecGpuCore(dst,g.d_nested_tmp1,0,inprod,nullptr))return -1;
    return failCuda(cudaMemcpy(dst,g.d_nested_tmp1,g.cfg.nrows*sizeof(cuDoubleComplex),cudaMemcpyDeviceToDevice),
                    "copy Lanier Schur right-preconditioned MatVec result");
}

static int schurAxpyDevice(cuDoubleComplex *dst,const cuDoubleComplex *src,cuDoubleComplex alpha)
{
    if(schurEnsureScratch())return -1;
    /* Use the dedicated input scratch as the complete P_Schur*src output.
     * src is a registered Krylov vector and therefore distinct from this
     * backend-owned buffer. */
    if(schurApplyDevice(src,g.d_schur_input))return -1;
    const int threads=256;
    const unsigned int blocks=static_cast<unsigned int>((g.cfg.nrows+threads-1)/threads);
    adda_kernel_qmrIncrem01Kernel(blocks,threads,dst,g.d_schur_input,alpha,g.cfg.nrows);
    return launchChecked("Lanier Schur physical-solution update");
}


/* Symmetric multiplicative Schwarz / symmetric block Gauss--Seidel.
 * Sequence for N zones:
 *   0,1,...,N-1,N-2,...,1,0
 * Each correction uses the current exact physical residual. Thus for N
 * zones the preconditioner contains (2*N-1) regional solves and (2*N-2)
 * exact global A*delta residual updates. This is a fixed linear operator.
 *
 * V1 is exposed as a right preconditioner only. Although the forward/backward
 * construction removes the one-way ordering bias, numerical complex symmetry
 * of the complete approximate inverse is not assumed here. */
static int schwarzSymApplyDevice(const cuDoubleComplex *src,cuDoubleComplex *dst)
{
    if(!g.nested_initialized || g.nested_count<2 || g.d_nested_tmp1==nullptr ||
       g.d_nested_tmp2==nullptr || g.d_lanier_tmp==nullptr){
        setError("Lanier multizone symmetric Schwarz preconditioner is not initialized");return -1;
    }
    const size_t bytes=g.cfg.nrows*sizeof(cuDoubleComplex);
    if(failCuda(cudaMemcpy(g.d_nested_tmp1,src,bytes,cudaMemcpyDeviceToDevice),
                "initialize Lanier symmetric Schwarz residual"))return -1;
    if(failCuda(cudaMemset(dst,0,bytes),"clear Lanier symmetric Schwarz accumulated correction"))return -1;

    const int threads=256;
    const unsigned int blocks=static_cast<unsigned int>((g.cfg.nrows+threads-1)/threads);
    const cuDoubleComplex one=make_cuDoubleComplex(static_cast<AddaCudaReal>(1),static_cast<AddaCudaReal>(0));
    const cuDoubleComplex minus_one=make_cuDoubleComplex(static_cast<AddaCudaReal>(-1),static_cast<AddaCudaReal>(0));

    for(int ns=0;ns<g.nested_count;ns++) {
        if(nestedSlotApplyDevice(g.nested[ns],g.d_nested_tmp1,g.d_nested_tmp2))return -1;
        adda_kernel_qmrIncrem01Kernel(blocks,threads,dst,g.d_nested_tmp2,one,g.cfg.nrows);
        if(launchChecked("Lanier symmetric Schwarz forward correction accumulation"))return -1;
        if(matvecGpuCore(g.d_nested_tmp2,g.d_lanier_tmp,0,nullptr,nullptr))return -1;
        adda_kernel_qmrIncrem01Kernel(blocks,threads,g.d_nested_tmp1,g.d_lanier_tmp,minus_one,g.cfg.nrows);
        if(launchChecked("Lanier symmetric Schwarz forward residual update"))return -1;
    }

    for(int ns=g.nested_count-2;ns>=0;ns--) {
        if(nestedSlotApplyDevice(g.nested[ns],g.d_nested_tmp1,g.d_nested_tmp2))return -1;
        adda_kernel_qmrIncrem01Kernel(blocks,threads,dst,g.d_nested_tmp2,one,g.cfg.nrows);
        if(launchChecked("Lanier symmetric Schwarz backward correction accumulation"))return -1;
        if(ns>0) {
            if(matvecGpuCore(g.d_nested_tmp2,g.d_lanier_tmp,0,nullptr,nullptr))return -1;
            adda_kernel_qmrIncrem01Kernel(blocks,threads,g.d_nested_tmp1,g.d_lanier_tmp,minus_one,g.cfg.nrows);
            if(launchChecked("Lanier symmetric Schwarz backward residual update"))return -1;
        }
    }
    return 0;
}

static int schwarzSymMatvecRightDevice(const cuDoubleComplex *src,cuDoubleComplex *dst,double *inprod)
{
    if(schwarzSymApplyDevice(src,dst))return -1;
    if(matvecGpuCore(dst,g.d_nested_tmp1,0,inprod,nullptr))return -1;
    return failCuda(cudaMemcpy(dst,g.d_nested_tmp1,g.cfg.nrows*sizeof(cuDoubleComplex),cudaMemcpyDeviceToDevice),
                    "copy Lanier symmetric Schwarz right-preconditioned MatVec result");
}

static int schwarzSymAxpyDevice(cuDoubleComplex *dst,const cuDoubleComplex *src,cuDoubleComplex alpha)
{
    if(!g.nested_initialized || g.nested_count<2 || g.d_nested_tmp1==nullptr ||
       g.d_nested_tmp2==nullptr || g.d_lanier_tmp==nullptr){
        setError("Lanier multizone symmetric Schwarz preconditioner is not initialized");return -1;
    }
    const size_t bytes=g.cfg.nrows*sizeof(cuDoubleComplex);
    if(failCuda(cudaMemcpy(g.d_nested_tmp1,src,bytes,cudaMemcpyDeviceToDevice),
                "initialize Lanier symmetric Schwarz axpy residual"))return -1;

    const int threads=256;
    const unsigned int blocks=static_cast<unsigned int>((g.cfg.nrows+threads-1)/threads);
    const cuDoubleComplex minus_one=make_cuDoubleComplex(static_cast<AddaCudaReal>(-1),static_cast<AddaCudaReal>(0));

    for(int ns=0;ns<g.nested_count;ns++) {
        if(nestedSlotApplyDevice(g.nested[ns],g.d_nested_tmp1,g.d_nested_tmp2))return -1;
        adda_kernel_qmrIncrem01Kernel(blocks,threads,dst,g.d_nested_tmp2,alpha,g.cfg.nrows);
        if(launchChecked("Lanier symmetric Schwarz forward physical-solution update"))return -1;
        if(matvecGpuCore(g.d_nested_tmp2,g.d_lanier_tmp,0,nullptr,nullptr))return -1;
        adda_kernel_qmrIncrem01Kernel(blocks,threads,g.d_nested_tmp1,g.d_lanier_tmp,minus_one,g.cfg.nrows);
        if(launchChecked("Lanier symmetric Schwarz forward axpy residual update"))return -1;
    }
    for(int ns=g.nested_count-2;ns>=0;ns--) {
        if(nestedSlotApplyDevice(g.nested[ns],g.d_nested_tmp1,g.d_nested_tmp2))return -1;
        adda_kernel_qmrIncrem01Kernel(blocks,threads,dst,g.d_nested_tmp2,alpha,g.cfg.nrows);
        if(launchChecked("Lanier symmetric Schwarz backward physical-solution update"))return -1;
        if(ns>0) {
            if(matvecGpuCore(g.d_nested_tmp2,g.d_lanier_tmp,0,nullptr,nullptr))return -1;
            adda_kernel_qmrIncrem01Kernel(blocks,threads,g.d_nested_tmp1,g.d_lanier_tmp,minus_one,g.cfg.nrows);
            if(launchChecked("Lanier symmetric Schwarz backward axpy residual update"))return -1;
        }
    }
    return 0;
}

static int nestedMatvecRightDevice(const cuDoubleComplex *src,cuDoubleComplex *dst,int her,double *inprod)
{
    if(!her){
        if(nestedApplyDevice(src,g.d_lanier_tmp))return -1;
        return matvecGpuCore(g.d_lanier_tmp,dst,0,inprod,nullptr);
    }
    if(matvecGpuCore(src,g.d_lanier_tmp,1,nullptr,nullptr))return -1;
    if(lanierConjugateSelf(g.d_lanier_tmp,"Lanier nested adjoint input conjugation"))return -1;
    if(nestedApplyDevice(g.d_lanier_tmp,dst))return -1;
    if(lanierConjugateSelf(dst,"Lanier nested adjoint output conjugation"))return -1;
    if(inprod!=nullptr && backendNorm2(dst,g.cfg.nrows,inprod,
        "cuBLAS FP64 norm(Lanier nested adjoint result)"))return -1;
    return 0;
}

static int nestedMatvecCongruenceDevice(const cuDoubleComplex *src,cuDoubleComplex *dst,double *inprod)
{
    if(nestedApplyDevice(src,g.d_lanier_tmp))return -1;
    if(matvecGpuCore(g.d_lanier_tmp,dst,0,nullptr,nullptr))return -1;
    if(nestedApplyDevice(dst,dst))return -1;
    if(inprod!=nullptr && backendNorm2(dst,g.cfg.nrows,inprod,
        "cuBLAS FP64 norm(Lanier nested congruence result)"))return -1;
    return 0;
}

static int lanierConjugateSelf(cuDoubleComplex *v,const char *label)
{
    const int threads=256;
    const unsigned int blocks=static_cast<unsigned int>((g.cfg.nrows+threads-1)/threads);
    adda_kernel_iterMultSelfConjKernel(blocks,threads,v,static_cast<AddaCudaReal>(1),g.cfg.nrows);
    return launchChecked(label);
}

/* Apply the right-preconditioned operator B=A*P or its Hermitian adjoint.
 * P is complex symmetric (P^T=P), hence P^H*v = conj(P*conj(v)).
 * This gives B^H=P^H*A^H without storing a second coefficient set. */
static int lanierMatvecRightDevice(const cuDoubleComplex *src,cuDoubleComplex *dst,
                                   const bool full_mode,const bool her,double *inprod)
{
    if(!her){
        if(lanierApplyDeviceMode(src,g.d_lanier_tmp,full_mode))return -1;
        return matvecGpuCore(g.d_lanier_tmp,dst,0,inprod,nullptr);
    }
    if(matvecGpuCore(src,g.d_lanier_tmp,1,nullptr,nullptr))return -1;
    if(lanierConjugateSelf(g.d_lanier_tmp,"Lanier adjoint input conjugation"))return -1;
    if(lanierApplyDeviceMode(g.d_lanier_tmp,dst,full_mode))return -1;
    if(lanierConjugateSelf(dst,"Lanier adjoint output conjugation"))return -1;
    if(inprod!=nullptr && backendNorm2(dst,g.cfg.nrows,inprod,
        "cuBLAS FP64 norm(Lanier adjoint result)"))return -1;
    return 0;
}

/* Congruence operator P*A*P used by the complex-symmetric Krylov methods.
 * Since A^T=A and P^T=P, (P*A*P)^T=P*A*P, so BiCG_CS/CSYM/QMR_CS
 * retain the symmetry property their recurrences require. */
static int lanierMatvecCongruenceDevice(const cuDoubleComplex *src,cuDoubleComplex *dst,
                                        const bool full_mode,double *inprod)
{
    if(lanierApplyDeviceMode(src,g.d_lanier_tmp,full_mode))return -1;
    if(matvecGpuCore(g.d_lanier_tmp,dst,0,nullptr,nullptr))return -1;
    /* In-place application is safe: scatter completes before gather writes dst. */
    if(lanierApplyDeviceMode(dst,dst,full_mode))return -1;
    if(inprod!=nullptr && backendNorm2(dst,g.cfg.nrows,inprod,
        "cuBLAS FP64 norm(Lanier congruence result)"))return -1;
    return 0;
}

extern "C" int adda_cuda_lanier_init(size_t nx,size_t ny,size_t nz,
                                      size_t rx,size_t ry,size_t rz,
                                      const void *inverse6_reduced,size_t complex_count)
{
    g_error[0]='\0';
    if(!g.initialized){setError("CUDA MatVec must be initialized before Lanier reference");return -1;}
    if(g.cfg.surface){setError("Lanier reference does not yet support surface mode");return -1;}
    if(g.lanier_initialized){setError("Lanier reference is already initialized");return -1;}
    if(nx==0||ny==0||nz==0||rx!=nx/2+1||ry!=ny/2+1||rz!=nz/2+1||inverse6_reduced==nullptr){
        setError("invalid Lanier reference dimensions or coefficient pointer");return -1;
    }
    size_t nxy=0,n=0,rxy=0,nred=0,threeN=0,sixNred=0;
    if(!checkedMul(nx,ny,nxy)||!checkedMul(nxy,nz,n)||
       !checkedMul(rx,ry,rxy)||!checkedMul(rxy,rz,nred)||
       !checkedMul(static_cast<size_t>(3),n,threeN)||
       !checkedMul(static_cast<size_t>(6),nred,sixNred)||complex_count!=sixNred){
        setError("Lanier reference size overflow or coefficient-count mismatch");return -1;
    }
    g.lanier_nx=nx;g.lanier_ny=ny;g.lanier_nz=nz;g.lanierN=n;
    g.lanier_rx=rx;g.lanier_ry=ry;g.lanier_rz=rz;g.lanierNred=nred;
    g.lanierGridComplexCount=threeN;g.lanierCoeffComplexCount=sixNred;

    int dims[3],dist;
    dims[0]=toInt(nz,"Lanier nz");dims[1]=toInt(ny,"Lanier ny");dims[2]=toInt(nx,"Lanier nx");
    dist=toInt(n,"Lanier volume");
    if(dims[0]<0||dims[1]<0||dims[2]<0||dist<0){lanierReleaseNoFail();return -1;}
    if(failCufft(cufftPlanMany(&g.planLanier3d,3,dims,nullptr,1,dist,nullptr,1,dist,CUFFT_Z2Z,3),
                 "cufftPlanMany(Lanier reference batch=3)")){lanierReleaseNoFail();return -1;}
    if(failCuda(cudaMalloc(reinterpret_cast<void**>(&g.d_lanier_tmp),g.cfg.nrows*sizeof(cuDoubleComplex)),
                "cudaMalloc(Lanier compact workspace)")){lanierReleaseNoFail();return -1;}
    if(failCuda(cudaMalloc(reinterpret_cast<void**>(&g.d_lanier_grid),threeN*sizeof(cuDoubleComplex)),
                "cudaMalloc(Lanier box FFT grid)")){lanierReleaseNoFail();return -1;}
    if(failCuda(cudaMalloc(reinterpret_cast<void**>(&g.d_lanier_coeff),sixNred*sizeof(cuDoubleComplex)),
                "cudaMalloc(Lanier reduced inverse coefficients)")){lanierReleaseNoFail();return -1;}
    if(failCuda(cudaMemcpy(g.d_lanier_coeff,inverse6_reduced,sixNred*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice),
                "upload Lanier reduced inverse coefficients")){lanierReleaseNoFail();return -1;}
    g.lanier_initialized=true;updateExplicitPeak();return 0;
}

extern "C" int adda_cuda_lanier_release(void)
{
    g_error[0]='\0';
    if(g.planLanier3d){if(failCufft(cufftDestroy(g.planLanier3d),"cufftDestroy(Lanier reference)"))return -1;g.planLanier3d=0;}
    if(g.d_lanier_coeff){if(failCuda(cudaFree(g.d_lanier_coeff),"cudaFree(Lanier coefficients)"))return -1;g.d_lanier_coeff=nullptr;}
    if(g.d_lanier_grid){if(failCuda(cudaFree(g.d_lanier_grid),"cudaFree(Lanier FFT grid)"))return -1;g.d_lanier_grid=nullptr;}
    if(g.d_lanier_tmp){if(failCuda(cudaFree(g.d_lanier_tmp),"cudaFree(Lanier compact workspace)"))return -1;g.d_lanier_tmp=nullptr;}
    g.lanier_nx=g.lanier_ny=g.lanier_nz=g.lanierN=0;
    g.lanier_rx=g.lanier_ry=g.lanier_rz=g.lanierNred=0;
    g.lanierGridComplexCount=g.lanierCoeffComplexCount=0;g.lanier_initialized=false;
    g.lanier_active_material=-1;g.lanier_origin_x=g.lanier_origin_y=g.lanier_origin_z=0;return 0;
}

extern "C" int adda_cuda_lanier_apply(const void *src_id,const void *dst_id)
{
    g_error[0]='\0';
    cuDoubleComplex *src=iterDeviceVector(src_id);if(src==nullptr)return -1;
    cuDoubleComplex *dst=iterDeviceVector(dst_id);if(dst==nullptr)return -1;
    return lanierApplyDevice(src,dst);
}

extern "C" int adda_cuda_lanier_matvec_right(const void *src_id,const void *dst_id,int her,double *inprod)
{
    g_error[0]='\0';
    cuDoubleComplex *src=iterDeviceVector(src_id);if(src==nullptr)return -1;
    cuDoubleComplex *dst=iterDeviceVector(dst_id);if(dst==nullptr)return -1;
    return lanierMatvecRightDevice(src,dst,false,her!=0,inprod);
}

extern "C" int adda_cuda_lanier_matvec_congruence(const void *src_id,const void *dst_id,double *inprod)
{
    g_error[0]='\0';
    cuDoubleComplex *src=iterDeviceVector(src_id);if(src==nullptr)return -1;
    cuDoubleComplex *dst=iterDeviceVector(dst_id);if(dst==nullptr)return -1;
    return lanierMatvecCongruenceDevice(src,dst,false,inprod);
}

extern "C" int adda_cuda_lanier_axpy(const void *dst_id,const void *src_id,double ar,double ai)
{
    g_error[0]='\0';
    cuDoubleComplex *dst=iterDeviceVector(dst_id);if(dst==nullptr)return -1;
    cuDoubleComplex *src=iterDeviceVector(src_id);if(src==nullptr)return -1;
    if(lanierApplyDevice(src,g.d_lanier_tmp))return -1;
    const int threads=256;
    const unsigned int blocks=static_cast<unsigned int>((g.cfg.nrows+threads-1)/threads);
    const cuDoubleComplex alpha=make_cuDoubleComplex(static_cast<AddaCudaReal>(ar),static_cast<AddaCudaReal>(ai));
    adda_kernel_qmrIncrem01Kernel(blocks,threads,dst,g.d_lanier_tmp,alpha,g.cfg.nrows);
    return launchChecked("Lanier reference solution axpy");
}

extern "C" int adda_cuda_lanier_set_region(int active_material,size_t x0,size_t y0,size_t z0)
{
    g_error[0]='\0';
    if(!g.initialized){setError("CUDA MatVec must be initialized before Lanier partition region setup");return -1;}
    if(g.lanier_initialized){setError("Lanier region must be selected before Lanier coefficient initialization");return -1;}
    if(active_material < -1 || active_material >= 255){setError("invalid Lanier partition material index");return -1;}
    g.lanier_active_material=active_material;
    g.lanier_origin_x=x0;g.lanier_origin_y=y0;g.lanier_origin_z=z0;
    return 0;
}

extern "C" int adda_cuda_set_material_projection(int active_material)
{
    g_error[0]='\0';
    if(!g.initialized){setError("CUDA MatVec must be initialized before material projection setup");return -1;}
    if(active_material < -1 || active_material >= 255){setError("invalid material projection index");return -1;}
    g.project_material=active_material;
    return 0;
}

extern "C" int adda_cuda_lanier_full_predict(const void *cc_sqrt,size_t complex_count,
                                                const unsigned char *material,
                                                const unsigned short *position,size_t ndip,
                                                int boxX,int boxY,int boxZ,double kd,double dipvol,
                                                const char *actor_path,
                                                AddaCudaLanierFullPrediction *out)
{
    g_error[0]='\0';
    if(adda_cuda_backend_real_bytes()!=(int)sizeof(AddaCudaReal)){setError("LANIER_FULL backend precision mismatch");return -1;}
    const int rc=lanier_full_ai::predict(reinterpret_cast<const cuDoubleComplex*>(cc_sqrt),complex_count,
                                         material,position,ndip,boxX,boxY,boxZ,kd,dipvol,actor_path,out);
    if(rc!=0){setError(lanier_full_ai::last_error());return -1;}
    g.lanier_full_sqrt_dipvol=static_cast<AddaCudaReal>(std::sqrt(dipvol));
    return 0;
}

extern "C" int adda_cuda_lanier_full_apply(const void *src_id,const void *dst_id)
{
    g_error[0]='\0';
    cuDoubleComplex *src=iterDeviceVector(src_id);if(src==nullptr)return -1;
    cuDoubleComplex *dst=iterDeviceVector(dst_id);if(dst==nullptr)return -1;
    return lanierFullApplyDevice(src,dst);
}

extern "C" int adda_cuda_lanier_full_matvec_right(const void *src_id,const void *dst_id,int her,double *inprod)
{
    g_error[0]='\0';
    cuDoubleComplex *src=iterDeviceVector(src_id);if(src==nullptr)return -1;
    cuDoubleComplex *dst=iterDeviceVector(dst_id);if(dst==nullptr)return -1;
    return lanierMatvecRightDevice(src,dst,true,her!=0,inprod);
}

extern "C" int adda_cuda_lanier_full_matvec_congruence(const void *src_id,const void *dst_id,double *inprod)
{
    g_error[0]='\0';
    cuDoubleComplex *src=iterDeviceVector(src_id);if(src==nullptr)return -1;
    cuDoubleComplex *dst=iterDeviceVector(dst_id);if(dst==nullptr)return -1;
    return lanierMatvecCongruenceDevice(src,dst,true,inprod);
}

extern "C" int adda_cuda_lanier_full_axpy(const void *dst_id,const void *src_id,double ar,double ai)
{
    g_error[0]='\0';
    cuDoubleComplex *dst=iterDeviceVector(dst_id);if(dst==nullptr)return -1;
    cuDoubleComplex *src=iterDeviceVector(src_id);if(src==nullptr)return -1;
    if(lanierFullApplyDevice(src,g.d_lanier_tmp))return -1;
    const int threads=256;
    const unsigned int blocks=static_cast<unsigned int>((g.cfg.nrows+threads-1)/threads);
    const cuDoubleComplex alpha=make_cuDoubleComplex(static_cast<AddaCudaReal>(ar),static_cast<AddaCudaReal>(ai));
    adda_kernel_qmrIncrem01Kernel(blocks,threads,dst,g.d_lanier_tmp,alpha,g.cfg.nrows);
    return launchChecked("Lanier full solution axpy");
}
extern "C" int adda_cuda_lanier_nested_init_slot(int slot,int active_material,
                                                       size_t x0,size_t y0,size_t z0,
                                                       size_t nx,size_t ny,size_t nz,
                                                       size_t rx,size_t ry,size_t rz,
                                                       const void *inverse6_reduced,size_t complex_count)
{
    g_error[0]='\0';
    (void)LANIER_MULTIZONE_BACKEND_VERSION;
    if(!g.initialized){setError("CUDA MatVec must be initialized before Lanier nested");return -1;}
    if(g.cfg.surface){setError("Lanier nested does not yet support surface mode");return -1;}
    if(slot<0 || slot>=LANIER_NESTED_MAX_SLOTS){setError("invalid Lanier nested slot");return -1;}
    if(active_material<0 || active_material>=255){setError("invalid Lanier nested material index");return -1;}
    NestedLanierState &st=g.nested[slot];
    if(st.initialized){setError("Lanier nested slot already initialized");return -1;}
    if(nx==0||ny==0||nz==0||rx!=nx/2+1||ry!=ny/2+1||rz!=nz/2+1||inverse6_reduced==nullptr){
        setError("invalid Lanier nested dimensions or coefficient pointer");return -1;
    }
    size_t nxy=0,n=0,rxy=0,nred=0,threeN=0,sixNred=0;
    if(!checkedMul(nx,ny,nxy)||!checkedMul(nxy,nz,n)||
       !checkedMul(rx,ry,rxy)||!checkedMul(rxy,rz,nred)||
       !checkedMul(static_cast<size_t>(3),n,threeN)||
       !checkedMul(static_cast<size_t>(6),nred,sixNred)||complex_count!=sixNred){
        setError("Lanier nested size overflow or coefficient-count mismatch");return -1;
    }
    st.active_material=active_material;st.origin_x=x0;st.origin_y=y0;st.origin_z=z0;
    st.nx=nx;st.ny=ny;st.nz=nz;st.n=n;st.rx=rx;st.ry=ry;st.rz=rz;st.nred=nred;
    st.gridComplexCount=threeN;st.coeffComplexCount=sixNred;
    int dims[3],dist;
    dims[0]=toInt(nz,"Lanier nested nz");dims[1]=toInt(ny,"Lanier nested ny");dims[2]=toInt(nx,"Lanier nested nx");
    dist=toInt(n,"Lanier nested volume");
    if(dims[0]<0||dims[1]<0||dims[2]<0||dist<0){nestedReleaseNoFail();return -1;}
    size_t workSize=0;
    if(failCufft(cufftCreate(&st.plan),"cufftCreate(Lanier multizone zone plan)") ||
       failCufft(cufftSetAutoAllocation(st.plan,0),"cufftSetAutoAllocation(Lanier multizone zone plan)") ||
       failCufft(cufftMakePlanMany(st.plan,3,dims,nullptr,1,dist,nullptr,1,dist,CUFFT_Z2Z,3,&workSize),
                 "cufftMakePlanMany(Lanier multizone batch=3)")){nestedReleaseNoFail();return -1;}
    st.fftWorkSize=workSize;
    if(nestedEnsureSharedGrid(threeN)){nestedReleaseNoFail();return -1;}
    /* Include this newly created plan when rebinding the shared work area. */
    if(slot+1>g.nested_count)g.nested_count=slot+1;
    if(nestedEnsureSharedFftWork(workSize)){nestedReleaseNoFail();return -1;}
    if(g.d_nested_fft_work!=nullptr &&
       failCufft(cufftSetWorkArea(st.plan,g.d_nested_fft_work),
                 "cufftSetWorkArea(Lanier multizone zone plan)")){nestedReleaseNoFail();return -1;}
    if(failCuda(cudaMalloc(reinterpret_cast<void**>(&st.coeff),sixNred*sizeof(cuDoubleComplex)),
                "cudaMalloc(Lanier nested coefficients)")){nestedReleaseNoFail();return -1;}
    if(failCuda(cudaMemcpy(st.coeff,inverse6_reduced,sixNred*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice),
                "upload Lanier nested coefficients")){nestedReleaseNoFail();return -1;}
    if(g.d_lanier_tmp==nullptr && failCuda(cudaMalloc(reinterpret_cast<void**>(&g.d_lanier_tmp),g.cfg.nrows*sizeof(cuDoubleComplex)),
                                           "cudaMalloc(Lanier nested compact scratch)")){nestedReleaseNoFail();return -1;}
    if(g.d_nested_tmp1==nullptr && failCuda(cudaMalloc(reinterpret_cast<void**>(&g.d_nested_tmp1),g.cfg.nrows*sizeof(cuDoubleComplex)),
                                            "cudaMalloc(Lanier nested input scratch)")){nestedReleaseNoFail();return -1;}
    if(g.d_nested_tmp2==nullptr && failCuda(cudaMalloc(reinterpret_cast<void**>(&g.d_nested_tmp2),g.cfg.nrows*sizeof(cuDoubleComplex)),
                                            "cudaMalloc(Lanier nested accumulation scratch)")){nestedReleaseNoFail();return -1;}
    st.initialized=true;
    g.nested_initialized=(g.nested_count>=2);
    updateExplicitPeak();return 0;
}

extern "C" int adda_cuda_lanier_nested_release(void)
{
    g_error[0]='\0';nestedReleaseNoFail();return 0;
}

extern "C" int adda_cuda_lanier_nested_apply(const void *src_id,const void *dst_id)
{
    g_error[0]='\0';
    cuDoubleComplex *src=iterDeviceVector(src_id);if(src==nullptr)return -1;
    cuDoubleComplex *dst=iterDeviceVector(dst_id);if(dst==nullptr)return -1;
    return nestedApplyDevice(src,dst);
}

extern "C" int adda_cuda_lanier_nested_matvec_right(const void *src_id,const void *dst_id,int her,double *inprod)
{
    g_error[0]='\0';
    cuDoubleComplex *src=iterDeviceVector(src_id);if(src==nullptr)return -1;
    cuDoubleComplex *dst=iterDeviceVector(dst_id);if(dst==nullptr)return -1;
    return nestedMatvecRightDevice(src,dst,her,inprod);
}

extern "C" int adda_cuda_lanier_nested_matvec_congruence(const void *src_id,const void *dst_id,double *inprod)
{
    g_error[0]='\0';
    cuDoubleComplex *src=iterDeviceVector(src_id);if(src==nullptr)return -1;
    cuDoubleComplex *dst=iterDeviceVector(dst_id);if(dst==nullptr)return -1;
    return nestedMatvecCongruenceDevice(src,dst,inprod);
}

extern "C" int adda_cuda_lanier_nested_axpy(const void *dst_id,const void *src_id,double ar,double ai)
{
    g_error[0]='\0';
    cuDoubleComplex *dst=iterDeviceVector(dst_id);if(dst==nullptr)return -1;
    cuDoubleComplex *src=iterDeviceVector(src_id);if(src==nullptr)return -1;
    if(nestedApplyDevice(src,g.d_lanier_tmp))return -1;
    const int threads=256;
    const unsigned int blocks=static_cast<unsigned int>((g.cfg.nrows+threads-1)/threads);
    const cuDoubleComplex alpha=make_cuDoubleComplex(static_cast<AddaCudaReal>(ar),static_cast<AddaCudaReal>(ai));
    adda_kernel_qmrIncrem01Kernel(blocks,threads,dst,g.d_lanier_tmp,alpha,g.cfg.nrows);
    return launchChecked("Lanier nested solution axpy");
}
extern "C" int adda_cuda_lanier_schwarz_apply(const void *src_id,const void *dst_id)
{
    g_error[0]='\0';
    cuDoubleComplex *src=iterDeviceVector(src_id);if(src==nullptr)return -1;
    cuDoubleComplex *dst=iterDeviceVector(dst_id);if(dst==nullptr)return -1;
    return schwarzApplyDevice(src,dst);
}

extern "C" int adda_cuda_lanier_schwarz_matvec_right(const void *src_id,const void *dst_id,double *inprod)
{
    g_error[0]='\0';
    cuDoubleComplex *src=iterDeviceVector(src_id);if(src==nullptr)return -1;
    cuDoubleComplex *dst=iterDeviceVector(dst_id);if(dst==nullptr)return -1;
    return schwarzMatvecRightDevice(src,dst,inprod);
}

extern "C" int adda_cuda_lanier_schwarz_axpy(const void *dst_id,const void *src_id,double ar,double ai)
{
    g_error[0]='\0';
    cuDoubleComplex *dst=iterDeviceVector(dst_id);if(dst==nullptr)return -1;
    cuDoubleComplex *src=iterDeviceVector(src_id);if(src==nullptr)return -1;
    const cuDoubleComplex alpha=make_cuDoubleComplex(static_cast<AddaCudaReal>(ar),static_cast<AddaCudaReal>(ai));
    return schwarzAxpyDevice(dst,src,alpha);
}


extern "C" int adda_cuda_lanier_schwarz_reverse_apply(const void *src_id,const void *dst_id)
{
    g_error[0]='\0';
    cuDoubleComplex *src=iterDeviceVector(src_id);if(src==nullptr)return -1;
    cuDoubleComplex *dst=iterDeviceVector(dst_id);if(dst==nullptr)return -1;
    return schwarzReverseApplyDevice(src,dst);
}

extern "C" int adda_cuda_lanier_schwarz_reverse_matvec_right(const void *src_id,const void *dst_id,double *inprod)
{
    g_error[0]='\0';
    cuDoubleComplex *src=iterDeviceVector(src_id);if(src==nullptr)return -1;
    cuDoubleComplex *dst=iterDeviceVector(dst_id);if(dst==nullptr)return -1;
    return schwarzReverseMatvecRightDevice(src,dst,inprod);
}

extern "C" int adda_cuda_lanier_schwarz_reverse_axpy(const void *dst_id,const void *src_id,double ar,double ai)
{
    g_error[0]='\0';
    cuDoubleComplex *dst=iterDeviceVector(dst_id);if(dst==nullptr)return -1;
    cuDoubleComplex *src=iterDeviceVector(src_id);if(src==nullptr)return -1;
    const cuDoubleComplex alpha=make_cuDoubleComplex(static_cast<AddaCudaReal>(ar),static_cast<AddaCudaReal>(ai));
    return schwarzReverseAxpyDevice(dst,src,alpha);
}

extern "C" int adda_cuda_lanier_schur_set_omega(double omega)
{
    if(!(omega>=-1.0 && omega<=1.0)){
        setError("LANIER_MULTIZONE_SCHUR V2.1 omega must satisfy -1 <= omega <= 1");
        return -1;
    }
    g.schur_omega=omega;
    return 0;
}

extern "C" int adda_cuda_lanier_schur_prepare(void)
{
    g_error[0]='\0';
    return schurEnsureScratch();
}

extern "C" int adda_cuda_lanier_schur_apply(const void *src_id,const void *dst_id)
{
    g_error[0]='\0';
    cuDoubleComplex *src=iterDeviceVector(src_id);if(src==nullptr)return -1;
    cuDoubleComplex *dst=iterDeviceVector(dst_id);if(dst==nullptr)return -1;
    return schurApplyDevice(src,dst);
}

extern "C" int adda_cuda_lanier_schur_matvec_right(const void *src_id,const void *dst_id,double *inprod)
{
    g_error[0]='\0';
    cuDoubleComplex *src=iterDeviceVector(src_id);if(src==nullptr)return -1;
    cuDoubleComplex *dst=iterDeviceVector(dst_id);if(dst==nullptr)return -1;
    return schurMatvecRightDevice(src,dst,inprod);
}

extern "C" int adda_cuda_lanier_schur_axpy(const void *dst_id,const void *src_id,double ar,double ai)
{
    g_error[0]='\0';
    cuDoubleComplex *dst=iterDeviceVector(dst_id);if(dst==nullptr)return -1;
    cuDoubleComplex *src=iterDeviceVector(src_id);if(src==nullptr)return -1;
    const cuDoubleComplex alpha=make_cuDoubleComplex(static_cast<AddaCudaReal>(ar),static_cast<AddaCudaReal>(ai));
    return schurAxpyDevice(dst,src,alpha);
}

extern "C" int adda_cuda_lanier_schwarz_sym_apply(const void *src_id,const void *dst_id)
{
    g_error[0]='\0';
    cuDoubleComplex *src=iterDeviceVector(src_id);if(src==nullptr)return -1;
    cuDoubleComplex *dst=iterDeviceVector(dst_id);if(dst==nullptr)return -1;
    return schwarzSymApplyDevice(src,dst);
}

extern "C" int adda_cuda_lanier_schwarz_sym_matvec_right(const void *src_id,const void *dst_id,double *inprod)
{
    g_error[0]='\0';
    cuDoubleComplex *src=iterDeviceVector(src_id);if(src==nullptr)return -1;
    cuDoubleComplex *dst=iterDeviceVector(dst_id);if(dst==nullptr)return -1;
    return schwarzSymMatvecRightDevice(src,dst,inprod);
}

extern "C" int adda_cuda_lanier_schwarz_sym_axpy(const void *dst_id,const void *src_id,double ar,double ai)
{
    g_error[0]='\0';
    cuDoubleComplex *dst=iterDeviceVector(dst_id);if(dst==nullptr)return -1;
    cuDoubleComplex *src=iterDeviceVector(src_id);if(src==nullptr)return -1;
    const cuDoubleComplex alpha=make_cuDoubleComplex(static_cast<AddaCudaReal>(ar),static_cast<AddaCudaReal>(ai));
    return schwarzSymAxpyDevice(dst,src,alpha);
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
    adda_kernel_iterCopyKernel(blocks,256,a,b,g.cfg.nrows);
    return launchChecked("iterCopyKernel launch");
}

extern "C" int adda_cuda_iter_mult(const void *a_id,const void *b_id,double cr,double ci)
{
    g_error[0]='\0'; cuDoubleComplex *a=iterDeviceVector(a_id); if(!a)return -1;
    cuDoubleComplex *b=iterDeviceVector(b_id); if(!b)return -1;
    unsigned int blocks; if(iterBlocks(&blocks))return -1;
    adda_kernel_qmrMultKernel(blocks,256,a,b,make_cuDoubleComplex(cr,ci),g.cfg.nrows);
    return launchChecked("iterMultKernel launch");
}

extern "C" int adda_cuda_iter_mult_self(const void *a_id,double cr,double ci)
{
    g_error[0]='\0'; cuDoubleComplex *a=iterDeviceVector(a_id); if(!a)return -1;
    unsigned int blocks; if(iterBlocks(&blocks))return -1;
    adda_kernel_qmrMultSelfKernel(blocks,256,a,make_cuDoubleComplex(cr,ci),g.cfg.nrows);
    return launchChecked("iterMultSelfKernel launch");
}

extern "C" int adda_cuda_iter_mult_self_conj(const void *a_id,double c)
{
    g_error[0]='\0'; cuDoubleComplex *a=iterDeviceVector(a_id); if(!a)return -1;
    unsigned int blocks; if(iterBlocks(&blocks))return -1;
    adda_kernel_iterMultSelfConjKernel(blocks,256,a,c,g.cfg.nrows);
    return launchChecked("iterMultSelfConjKernel launch");
}

extern "C" int adda_cuda_iter_lincomb1(const void *a_id,const void *b_id,const void *c_id,
                                        double c1r,double c1i,double *norm2)
{
    g_error[0]='\0'; cuDoubleComplex *a=iterDeviceVector(a_id); if(!a)return -1;
    cuDoubleComplex *b=iterDeviceVector(b_id); if(!b)return -1;
    cuDoubleComplex *c=iterDeviceVector(c_id); if(!c)return -1;
    unsigned int blocks; if(iterBlocks(&blocks))return -1;
    adda_kernel_qmrLinComb1Kernel(blocks,256,a,b,c,make_cuDoubleComplex(c1r,c1i),g.cfg.nrows);
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
    adda_kernel_qmrLinCombKernel(blocks,256,a,b,c,make_cuDoubleComplex(c1r,c1i),make_cuDoubleComplex(c2r,c2i),g.cfg.nrows);
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
    adda_kernel_iterLinComb1ConjKernel(blocks,256,a,b,c,make_cuDoubleComplex(c1r,c1i),g.cfg.nrows);
    if (launchChecked("iterLinComb1ConjKernel launch")) return -1;
    return iterNorm2(a,norm2,"cublasDznrm2(iterLinComb1Conj)");
}

extern "C" int adda_cuda_iter_increm01(const void *a_id,const void *b_id,double cr,double ci,double *norm2)
{
    g_error[0]='\0'; cuDoubleComplex *a=iterDeviceVector(a_id); if(!a)return -1;
    cuDoubleComplex *b=iterDeviceVector(b_id); if(!b)return -1;
    unsigned int blocks; if(iterBlocks(&blocks))return -1;
    adda_kernel_qmrIncrem01Kernel(blocks,256,a,b,make_cuDoubleComplex(cr,ci),g.cfg.nrows);
    if (launchChecked("iterIncrem01Kernel launch")) return -1;
    return iterNorm2(a,norm2,"cublasDznrm2(iterIncrem01)");
}

extern "C" int adda_cuda_iter_increm10(const void *a_id,const void *b_id,double cr,double ci,double *norm2)
{
    g_error[0]='\0'; cuDoubleComplex *a=iterDeviceVector(a_id); if(!a)return -1;
    cuDoubleComplex *b=iterDeviceVector(b_id); if(!b)return -1;
    unsigned int blocks; if(iterBlocks(&blocks))return -1;
    adda_kernel_iterIncrem10Kernel(blocks,256,a,b,make_cuDoubleComplex(cr,ci),g.cfg.nrows);
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
    adda_kernel_iterIncrem011Kernel(blocks,256,a,b,c,make_cuDoubleComplex(c1r,c1i),make_cuDoubleComplex(c2r,c2i),g.cfg.nrows);
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
    adda_kernel_qmrIncrem110Kernel(blocks,256,a,b,c,make_cuDoubleComplex(c1r,c1i),make_cuDoubleComplex(c2r,c2i),g.cfg.nrows);
    return launchChecked("iterIncrem110Kernel launch");
}

extern "C" int adda_cuda_iter_increm111(const void *a_id,const void *b_id,const void *c_id,
                                         double c1r,double c1i,double c2r,double c2i,double c3r,double c3i)
{
    g_error[0]='\0'; cuDoubleComplex *a=iterDeviceVector(a_id); if(!a)return -1;
    cuDoubleComplex *b=iterDeviceVector(b_id); if(!b)return -1;
    cuDoubleComplex *c=iterDeviceVector(c_id); if(!c)return -1;
    unsigned int blocks; if(iterBlocks(&blocks))return -1;
    adda_kernel_qmrIncrem111Kernel(blocks,256,a,b,c,make_cuDoubleComplex(c1r,c1i),make_cuDoubleComplex(c2r,c2i),
                                      make_cuDoubleComplex(c3r,c3i),g.cfg.nrows);
    return launchChecked("iterIncrem111Kernel launch");
}

extern "C" int adda_cuda_iter_increm11_d_c(const void *a_id,const void *b_id,double c1,
                                             double c2r,double c2i,double *norm2)
{
    g_error[0]='\0'; cuDoubleComplex *a=iterDeviceVector(a_id); if(!a)return -1;
    cuDoubleComplex *b=iterDeviceVector(b_id); if(!b)return -1;
    unsigned int blocks; if(iterBlocks(&blocks))return -1;
    adda_kernel_qmrIncrem11DCKernel(blocks,256,a,b,c1,make_cuDoubleComplex(c2r,c2i),g.cfg.nrows);
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
    adda_kernel_iterIncrem110DCConjKernel(blocks,256,a,b,c,c1,make_cuDoubleComplex(c2r,c2i),g.cfg.nrows);
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
