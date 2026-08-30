/* CUDA implementation of ADDA's sequential FFT MatVec.
 *
 * Mapping
 * -------
 * ADDA's CPU implementation expands the compact Xmatrix one x-slice at a time,
 * performs X/Z/Y FFTs, multiplies by the compact spectral Green tensor, then
 * reverses the transforms.  CUDA keeps a full zero-padded 3-D grid resident on
 * the device instead.  For component c the linear layout is
 *
 *     grid[c * (gridX*gridY*gridZ) + ((z*gridY + y)*gridX + x)] .
 *
 * Therefore a batched cuFFT with dimensions {gridZ,gridY,gridX} is exactly the
 * same transform as the CPU X -> Z -> Y sequence (the 1-D transforms commute).
 * The surface term uses Fx * Fy * Fz^{-1}; a 2-D XY plan plus a strided Z plan
 * reproduces that mixed transform without host-side transposes.
 *
 * cuFFT, like FFTW as configured by ADDA, leaves forward and inverse transforms
 * unnormalised.  Dmatrix/Rmatrix already contain the -1/Ngrid factor produced
 * by InitDmatrix, so no additional scaling is applied here.
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

namespace {

static_assert(sizeof(cuDoubleComplex) == 2 * sizeof(double),
              "cuDoubleComplex must be two packed doubles");

struct Context {
    bool initialized = false;
    AddaCudaMatVecConfig cfg{};
    size_t gridYZ = 0;
    size_t gridN = 0;
    size_t gridComplexCount = 0;
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

    cufftHandle plan3d = 0; /* three component-major full 3-D transforms */
    cufftHandle planXY = 0; /* surface: 2-D XY transform for every z/component */
    cufftHandle planZ = 0;  /* surface: Z transform, executed once per component */
    cudaEvent_t matvec_start = nullptr;
    cudaEvent_t matvec_stop = nullptr;
    cublasHandle_t cublas = nullptr;

    /* CUDA-resident iterative-solver vectors. Host addresses are stable
     * identities. d_arg and d_result are reused for xvec and rvec; only the
     * remaining registered vectors require extra allocations. BCGS2 is the
     * largest current solver and needs x/r/p + vec1..vec4 + Avecbuffer = 8
     * device vectors, i.e. at most six extras. */
    static const int ITER_VECTOR_COUNT = 8;
    const void *iter_host[ITER_VECTOR_COUNT] = {};
    cuDoubleComplex *iter_device[ITER_VECTOR_COUNT] = {};
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
    for (int i=0; i<Context::ITER_VECTOR_COUNT; ++i) {
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
                                    double c1, cuDoubleComplex c2, size_t n)
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
                                          const cuDoubleComplex *c, double c1,
                                          cuDoubleComplex c2, size_t n)
{
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < n) {
        const cuDoubleComplex ca=cconj_hd(a[i]);
        const cuDoubleComplex r1=make_cuDoubleComplex(c1*ca.x,c1*ca.y);
        a[i]=cadd_hd(cadd_hd(r1,cmul(c2,cconj_hd(b[i]))),c[i]);
    }
}

__global__ void iterMultSelfConjKernel(cuDoubleComplex *a, double c, size_t n)
{
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < n) a[i]=make_cuDoubleComplex(c*a[i].x,-c*a[i].y);
}

int launchChecked(const char *name)
{
    return failCuda(cudaGetLastError(),name);
}


int createPlans()
{
    const int gx = toInt(g.cfg.gridX,"gridX"); if (gx < 0) return -1;
    const int gy = toInt(g.cfg.gridY,"gridY"); if (gy < 0) return -1;
    const int gz = toInt(g.cfg.gridZ,"gridZ"); if (gz < 0) return -1;
    const int gridN = toInt(g.gridN,"gridX*gridY*gridZ"); if (gridN < 0) return -1;

    int dims3[3] = {gz,gy,gx};
    if (failCufft(cufftPlanMany(&g.plan3d,3,dims3,
                                nullptr,1,gridN,
                                nullptr,1,gridN,
                                CUFFT_Z2Z,3),
                  "cufftPlanMany(3D)")) return -1;

    if (g.cfg.surface) {
        const size_t xySizeSt = g.cfg.gridX * g.cfg.gridY;
        const int xySize = toInt(xySizeSt,"gridX*gridY"); if (xySize < 0) return -1;
        size_t xyBatchSt;
        if (!checkedMul(3,g.cfg.gridZ,xyBatchSt)) {
            setError("overflow while creating surface XY cuFFT batch"); return -1;
        }
        const int xyBatch = toInt(xyBatchSt,"3*gridZ"); if (xyBatch < 0) return -1;
        int dims2[2] = {gy,gx};
        if (failCufft(cufftPlanMany(&g.planXY,2,dims2,
                                    nullptr,1,xySize,
                                    nullptr,1,xySize,
                                    CUFFT_Z2Z,xyBatch),
                      "cufftPlanMany(surface XY)")) return -1;

        const int zBatch = toInt(xySizeSt,"gridX*gridY"); if (zBatch < 0) return -1;
        int dimZ[1] = {gz};
        int embedZ[1] = {gz};
        /* For one component, every (x,y) transform starts one element after the
         * previous one and advances by gridX*gridY between z samples.  A
         * non-NULL embed is required by cuFFT for the advanced stride/dist
         * parameters to take effect. */
        if (failCufft(cufftPlanMany(&g.planZ,1,dimZ,
                                    embedZ,xySize,1,
                                    embedZ,xySize,1,
                                    CUFFT_Z2Z,zBatch),
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

extern "C" int adda_cuda_matvec_init(const AddaCudaMatVecConfig *cfg,
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
    size_t expectedRows;
    if (!checkedMul(static_cast<size_t>(3),cfg->ndip,expectedRows) || cfg->nrows != expectedRows) {
        setError("CUDA MatVec requires nrows == 3*ndip in sequential mode");
        return -1;
    }

    freeContext();
    g.cfg = *cfg;
    if (!checkedMul(cfg->gridY,cfg->gridZ,g.gridYZ) ||
        !checkedMul(cfg->gridX,g.gridYZ,g.gridN) ||
        !checkedMul(static_cast<size_t>(3),g.gridN,g.gridComplexCount)) {
        setError("overflow while calculating CUDA FFT grid size");
        freeContext();
        return -1;
    }
    size_t dBlocks;
    if (!checkedMul(cfg->gridX,cfg->DsizeZ,dBlocks) ||
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

    cudaDeviceProp prop{};
    if (failCuda(cudaGetDeviceProperties(&prop,device),"cudaGetDeviceProperties")) { freeContext(); return -1; }
    if (prop.major == 1 && prop.minor < 3) {
        setError("CUDA device does not support double precision");
        freeContext();
        return -1;
    }

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

#define CUDA_ALLOC(ptr,count,type,label) \
    do { if (failCuda(cudaMalloc(reinterpret_cast<void**>(&(ptr)),(count)*sizeof(type)),label)) { freeContext(); return -1; } } while (0)

    CUDA_ALLOC(g.d_D,g.dComplexCount,cuDoubleComplex,"cudaMalloc(Dmatrix)");
    if (cfg->surface) CUDA_ALLOC(g.d_R,g.rComplexCount,cuDoubleComplex,"cudaMalloc(Rmatrix)");
    CUDA_ALLOC(g.d_material,cfg->ndip,unsigned char,"cudaMalloc(material)");
    CUDA_ALLOC(g.d_position,cfg->nrows,unsigned short,"cudaMalloc(position)");
    CUDA_ALLOC(g.d_arg,cfg->nrows,cuDoubleComplex,"cudaMalloc(argvec/iterative xvec)");
    CUDA_ALLOC(g.d_result,cfg->nrows,cuDoubleComplex,"cudaMalloc(resultvec/iterative rvec)");
    CUDA_ALLOC(g.d_grid,g.gridComplexCount,cuDoubleComplex,"cudaMalloc(full FFT grid)");
    if (cfg->surface) CUDA_ALLOC(g.d_gridR,g.gridComplexCount,cuDoubleComplex,"cudaMalloc(surface FFT grid)");
#undef CUDA_ALLOC

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
    }
    return failCuda(cudaMemcpy(g.d_cc,cc_sqrt,complex_count*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice),
                    "upload cc_sqrt");
}

int matvecGpuCore(const cuDoubleComplex *d_arg, cuDoubleComplex *d_result, int her,
                  double *inprod, double *elapsed_ms)
{
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
        if (g.cfg.nrows > static_cast<size_t>(INT_MAX)) {
            setError("MatVec norm via cuBLAS requires nrows <= INT_MAX");
            return -1;
        }
        double norm = 0.0;
        if (failCublas(cublasDznrm2(g.cublas,static_cast<int>(g.cfg.nrows),d_result,1,&norm),
                       "cublasDznrm2(MatVec result)")) return -1;
        *inprod = norm*norm;
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

extern "C" int adda_cuda_iter_init(const void *xvec, const void *rvec, const void *pvec,
                                    const void *vec1, const void *vec2, const void *vec3,
                                    const void *vec4, const void *Avecbuffer)
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

    const void *host[Context::ITER_VECTOR_COUNT] = {xvec,rvec,pvec,vec1,vec2,vec3,vec4,Avecbuffer};
    if (xvec == nullptr || rvec == nullptr || pvec == nullptr || Avecbuffer == nullptr) {
        setError("xvec, rvec, pvec and Avecbuffer must be registered for CUDA iterative solvers");
        return -1;
    }
    int nonnull=0;
    for (int i=0; i<Context::ITER_VECTOR_COUNT; ++i) {
        if (host[i] == nullptr) continue;
        ++nonnull;
        for (int j=0; j<i; ++j) {
            if (host[j] != nullptr && host[i] == host[j]) {
                setError("CUDA iterative vector identities must be distinct");
                return -1;
            }
        }
    }
    const size_t extra_count=static_cast<size_t>(nonnull-2); /* x/r reuse d_arg/d_result */
    if (extra_count > g.iter_extra_capacity) {
        if (g.d_iter_extra) cudaFree(g.d_iter_extra);
        g.d_iter_extra=nullptr;
        g.iter_extra_capacity=0;
        size_t total=0;
        if (!checkedMul(extra_count,g.cfg.nrows,total)) {
            setError("overflow while allocating CUDA iterative vectors");
            return -1;
        }
        if (total != 0 && failCuda(cudaMalloc(reinterpret_cast<void**>(&g.d_iter_extra),
                                               total*sizeof(cuDoubleComplex)),
                                   "cudaMalloc(iterative extra vectors)")) return -1;
        g.iter_extra_capacity=extra_count;
    }

    for (int i=0; i<Context::ITER_VECTOR_COUNT; ++i) {
        g.iter_host[i]=host[i];
        g.iter_device[i]=nullptr;
    }
    g.iter_device[0]=g.d_arg;    /* xvec */
    g.iter_device[1]=g.d_result; /* rvec */
    size_t extra=0;
    for (int i=2; i<Context::ITER_VECTOR_COUNT; ++i) {
        if (host[i] != nullptr) {
            g.iter_device[i]=g.d_iter_extra + extra*g.cfg.nrows;
            ++extra;
        }
    }
    g.iter_initialized=true;
    return adda_cuda_iter_upload_all();
}

extern "C" int adda_cuda_iter_upload_all(void)
{
    g_error[0] = '\0';
    if (!g.iter_initialized) { setError("CUDA iterative vectors are not initialized"); return -1; }
    const size_t bytes=g.cfg.nrows*sizeof(cuDoubleComplex);
    for (int i=0; i<Context::ITER_VECTOR_COUNT; ++i) {
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
    for (int i=0; i<Context::ITER_VECTOR_COUNT; ++i) {
        if (g.iter_host[i] == nullptr) continue;
        if (failCuda(cudaMemcpy(const_cast<void*>(g.iter_host[i]),g.iter_device[i],bytes,
                                cudaMemcpyDeviceToHost),"download iterative vector")) return -1;
    }
    return 0;
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
    double norm=0.0;
    if (failCublas(cublasDznrm2(g.cublas,static_cast<int>(g.cfg.nrows),a,1,&norm),what)) return -1;
    *norm2=norm*norm;
    return 0;
}

extern "C" int adda_cuda_iter_norm2(const void *a_id,double *norm2)
{
    g_error[0]='\0';
    if (norm2 == nullptr) { setError("null iterative norm result pointer"); return -1; }
    cuDoubleComplex *a=iterDeviceVector(a_id); if(!a)return -1;
    return iterNorm2(a,norm2,"cublasDznrm2(iterative vector)");
}

extern "C" int adda_cuda_iter_dotu(const void *a_id,const void *b_id,double *out_re,double *out_im)
{
    g_error[0]='\0';
    if (out_re == nullptr || out_im == nullptr) { setError("null iterative dotu result pointer"); return -1; }
    cuDoubleComplex *a=iterDeviceVector(a_id); if(!a)return -1;
    cuDoubleComplex *b=iterDeviceVector(b_id); if(!b)return -1;
    cuDoubleComplex out{};
    if (failCublas(cublasZdotu(g.cublas,static_cast<int>(g.cfg.nrows),a,1,b,1,&out),
                   "cublasZdotu(iterative)")) return -1;
    *out_re=out.x; *out_im=out.y;
    return 0;
}

extern "C" int adda_cuda_iter_dotc(const void *a_id,const void *b_id,double *out_re,double *out_im)
{
    g_error[0]='\0';
    if (out_re == nullptr || out_im == nullptr) { setError("null iterative dotc result pointer"); return -1; }
    cuDoubleComplex *a=iterDeviceVector(a_id); if(!a)return -1;
    cuDoubleComplex *b=iterDeviceVector(b_id); if(!b)return -1;
    cuDoubleComplex out{};
    /* ADDA nDotProd(a,b) = sum a*conj(b). cublasZdotc conjugates its first
     * operand, hence dotc(b,a) gives exactly conj(b)*a. */
    if (failCublas(cublasZdotc(g.cublas,static_cast<int>(g.cfg.nrows),b,1,a,1,&out),
                   "cublasZdotc(iterative)")) return -1;
    *out_re=out.x; *out_im=out.y;
    return 0;
}

extern "C" int adda_cuda_iter_dotu_self_norm2(const void *a_id,double *out_re,double *out_im,double *norm2)
{
    g_error[0]='\0';
    if (out_re == nullptr || out_im == nullptr || norm2 == nullptr) {
        setError("null iterative dotu/norm result pointer"); return -1;
    }
    cuDoubleComplex *a=iterDeviceVector(a_id); if(!a)return -1;
    cuDoubleComplex out{};
    if (failCublas(cublasZdotu(g.cublas,static_cast<int>(g.cfg.nrows),a,1,a,1,&out),
                   "cublasZdotu(iterative self)")) return -1;
    if (iterNorm2(a,norm2,"cublasDznrm2(iterative self)")) return -1;
    *out_re=out.x; *out_im=out.y;
    return 0;
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
