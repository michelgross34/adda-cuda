/* CUDA-only kernels for the split ADDA CUDA backend.
 * Host state, CUDA allocations, cuFFT/cuBLAS orchestration and the public ABI
 * live in wrappermatvec_backend.cpp.  This file is compiled only by nvcc.
 */
#include "cudamatvec_backend.h"
#include "kernel.h"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <climits>
#include <stdint.h>

#ifdef ADDA_CUDA_SINGLE_BACKEND
#  define cuDoubleComplex cuFloatComplex
#  define make_cuDoubleComplex make_cuFloatComplex
typedef float AddaCudaReal;
#else
typedef double AddaCudaReal;
typedef cuDoubleComplex AddaCudaDoubleComplex;
#endif

namespace {
__host__ __device__ inline cuDoubleComplex cmul(const cuDoubleComplex a, const cuDoubleComplex b)
{
    return make_cuDoubleComplex(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

__host__ __device__ inline cuDoubleComplex cadd_hd(const cuDoubleComplex a, const cuDoubleComplex b)
{
    return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
}

__host__ __device__ inline cuDoubleComplex csub_hd(const cuDoubleComplex a, const cuDoubleComplex b)
{
    return make_cuDoubleComplex(a.x - b.x, a.y - b.y);
}

__host__ __device__ inline cuDoubleComplex cscale_hd(const cuDoubleComplex a, const AddaCudaReal s)
{
    return make_cuDoubleComplex(a.x*s,a.y*s);
}

__host__ __device__ inline cuDoubleComplex cdiv_hd(const cuDoubleComplex a, const cuDoubleComplex b)
{
    const AddaCudaReal den=b.x*b.x+b.y*b.y;
    return make_cuDoubleComplex((a.x*b.x+a.y*b.y)/den,
                                (a.y*b.x-a.x*b.y)/den);
}

__host__ __device__ inline AddaCudaReal cabs2_hd(const cuDoubleComplex a)
{
    return a.x*a.x+a.y*a.y;
}

__host__ __device__ inline cuDoubleComplex cneg_hd(const cuDoubleComplex a)
{
    return make_cuDoubleComplex(-a.x, -a.y);
}

__host__ __device__ inline cuDoubleComplex cconj_hd(const cuDoubleComplex a)
{
    return make_cuDoubleComplex(a.x, -a.y);
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

__global__ void lanierBoxScatterKernel(const cuDoubleComplex *arg,
                                       cuDoubleComplex *grid,
                                       const unsigned short *position,
                                       size_t ndip,size_t nx,size_t ny,size_t n)
{
    const size_t i=blockIdx.x*static_cast<size_t>(blockDim.x)+threadIdx.x;
    if(i>=ndip) return;
    const size_t p=3*i;
    const size_t index=(static_cast<size_t>(position[p+2])*ny+position[p+1])*nx+position[p];
#pragma unroll
    for(int c=0;c<3;c++) grid[static_cast<size_t>(c)*n+index]=arg[p+static_cast<size_t>(c)];
}

__global__ void lanierBoxGatherKernel(const cuDoubleComplex *grid,
                                      cuDoubleComplex *out,
                                      const unsigned short *position,
                                      size_t ndip,size_t nx,size_t ny,size_t n)
{
    const size_t i=blockIdx.x*static_cast<size_t>(blockDim.x)+threadIdx.x;
    if(i>=ndip) return;
    const size_t p=3*i;
    const size_t index=(static_cast<size_t>(position[p+2])*ny+position[p+1])*nx+position[p];
#pragma unroll
    for(int c=0;c<3;c++) out[p+static_cast<size_t>(c)]=grid[static_cast<size_t>(c)*n+index];
}

__global__ void lanierFullBoxScatterKernel(const cuDoubleComplex *arg,
                                           cuDoubleComplex *grid,
                                           const unsigned short *position,
                                           const unsigned char *material,
                                           const cuDoubleComplex *cc,
                                           size_t ndip,size_t nx,size_t ny,size_t n,AddaCudaReal sqrt_dipvol,
                                           int active_material,size_t x0,size_t y0,size_t z0)
{
    const size_t i=blockIdx.x*static_cast<size_t>(blockDim.x)+threadIdx.x;
    if(i>=ndip) return;
    const size_t mat=material[i];
    if(active_material>=0 && mat!=static_cast<size_t>(active_material)) return;
    const size_t p=3*i;
    const size_t gx=position[p],gy=position[p+1],gz=position[p+2];
    if(gx<x0 || gy<y0 || gz<z0) return;
    const size_t lx=gx-x0,ly=gy-y0,lz=gz-z0;
    if(lx>=nx || ly>=ny) return;
    const size_t index=(lz*ny+ly)*nx+lx;
    if(index>=n) return;
#pragma unroll
    for(int c=0;c<3;c++) {
        cuDoubleComplex v=cdiv_hd(arg[p+static_cast<size_t>(c)],cc[3*mat+static_cast<size_t>(c)]);
        grid[static_cast<size_t>(c)*n+index]=make_cuDoubleComplex(v.x*sqrt_dipvol,v.y*sqrt_dipvol);
    }
}

__global__ void lanierFullBoxGatherKernel(const cuDoubleComplex *grid,
                                          cuDoubleComplex *out,
                                          const unsigned short *position,
                                          const unsigned char *material,
                                          const cuDoubleComplex *cc,
                                          size_t ndip,size_t nx,size_t ny,size_t n,AddaCudaReal sqrt_dipvol,
                                          int active_material,size_t x0,size_t y0,size_t z0)
{
    const size_t i=blockIdx.x*static_cast<size_t>(blockDim.x)+threadIdx.x;
    if(i>=ndip) return;
    const size_t p=3*i;
    const size_t mat=material[i];
    if(active_material>=0 && mat!=static_cast<size_t>(active_material)) {
#pragma unroll
        for(int c=0;c<3;c++) out[p+static_cast<size_t>(c)]=make_cuDoubleComplex(0,0);
        return;
    }
    const size_t gx=position[p],gy=position[p+1],gz=position[p+2];
    if(gx<x0 || gy<y0 || gz<z0) {
#pragma unroll
        for(int c=0;c<3;c++) out[p+static_cast<size_t>(c)]=make_cuDoubleComplex(0,0);
        return;
    }
    const size_t lx=gx-x0,ly=gy-y0,lz=gz-z0;
    const size_t index=(lz*ny+ly)*nx+lx;
    if(lx>=nx || ly>=ny || index>=n) {
#pragma unroll
        for(int c=0;c<3;c++) out[p+static_cast<size_t>(c)]=make_cuDoubleComplex(0,0);
        return;
    }
#pragma unroll
    for(int c=0;c<3;c++) {
        cuDoubleComplex v=cdiv_hd(grid[static_cast<size_t>(c)*n+index],cc[3*mat+static_cast<size_t>(c)]);
        out[p+static_cast<size_t>(c)]=make_cuDoubleComplex(v.x*sqrt_dipvol,v.y*sqrt_dipvol);
    }
}

__global__ void materialProjectionKernel(cuDoubleComplex *v,const unsigned char *material,
                                         size_t ndip,int active_material)
{
    const size_t i=blockIdx.x*static_cast<size_t>(blockDim.x)+threadIdx.x;
    if(i>=ndip || material[i]==static_cast<unsigned char>(active_material)) return;
    const size_t p=3*i;
    v[p]=v[p+1]=v[p+2]=make_cuDoubleComplex(0,0);
}

__global__ void lanierReducedMultiplyKernel(cuDoubleComplex *grid,
                                            const cuDoubleComplex *coef,
                                            size_t nx,size_t ny,size_t nz,size_t n,
                                            size_t rx,size_t ry,size_t rz,size_t nred)
{
    const size_t index=blockIdx.x*static_cast<size_t>(blockDim.x)+threadIdx.x;
    if(index>=n) return;
    const size_t x=index%nx, yz=index/nx, y=yz%ny, z=yz/ny;
    const bool xr=x>nx/2, yr=y>ny/2, zr=z>nz/2;
    const size_t mx=xr ? nx-x : x;
    const size_t my=yr ? ny-y : y;
    const size_t mz=zr ? nz-z : z;
    const size_t ridx=(mx*ry+my)*rz+mz;
    (void)rx;
    const AddaCudaReal sxy=(xr^yr) ? static_cast<AddaCudaReal>(-1) : static_cast<AddaCudaReal>(1);
    const AddaCudaReal sxz=(xr^zr) ? static_cast<AddaCudaReal>(-1) : static_cast<AddaCudaReal>(1);
    const AddaCudaReal syz=(yr^zr) ? static_cast<AddaCudaReal>(-1) : static_cast<AddaCudaReal>(1);
    cuDoubleComplex m[6],v[3],w[3];
    m[0]=coef[0*nred+ridx];
    m[1]=cscale_hd(coef[1*nred+ridx],sxy);
    m[2]=cscale_hd(coef[2*nred+ridx],sxz);
    m[3]=coef[3*nred+ridx];
    m[4]=cscale_hd(coef[4*nred+ridx],syz);
    m[5]=coef[5*nred+ridx];
#pragma unroll
    for(int c=0;c<3;c++) v[c]=grid[static_cast<size_t>(c)*n+index];
    symMatVec(m,v,w);
#pragma unroll
    for(int c=0;c<3;c++) grid[static_cast<size_t>(c)*n+index]=w[c];
}

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

__global__ void convertComplexF32ToF64Kernel(const cuFloatComplex *src,
                                             AddaCudaKernelDoubleComplex *dst,
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
                                                 AddaCudaKernelDoubleComplex *dst_a,
                                                 AddaCudaKernelDoubleComplex *dst_b,
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
}

extern "C" {
void adda_kernel_scatterKernel(unsigned int dipBlocks, unsigned int threads, const AddaCudaKernelComplex *arg, AddaCudaKernelComplex *grid, const unsigned char *material, const unsigned short *position, const AddaCudaKernelComplex *cc, size_t ndip, size_t gridX, size_t gridY, size_t gridN, int her)
{
    scatterKernel<<<dipBlocks,threads>>>(arg, grid, material, position, cc, ndip, gridX, gridY, gridN, her);
}
void adda_kernel_spectralMultiplyKernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *grid, const AddaCudaKernelComplex *gridR, const AddaCudaKernelComplex *D, const AddaCudaKernelComplex *R, size_t gridX, size_t gridY, size_t gridZ, size_t gridN, size_t DsizeY, size_t DsizeZ, size_t RsizeY, int reduced, int transposed, int surface)
{
    spectralMultiplyKernel<<<dipBlocks,threads>>>(grid, gridR, D, R, gridX, gridY, gridZ, gridN, DsizeY, DsizeZ, RsizeY, reduced, transposed, surface);
}
void adda_kernel_gatherKernel(unsigned int dipBlocks, unsigned int threads, const AddaCudaKernelComplex *arg, AddaCudaKernelComplex *result, const AddaCudaKernelComplex *grid, const unsigned char *material, const unsigned short *position, const AddaCudaKernelComplex *cc, size_t ndip, size_t gridX, size_t gridY, size_t gridN, int her)
{
    gatherKernel<<<dipBlocks,threads>>>(arg, result, grid, material, position, cc, ndip, gridX, gridY, gridN, her);
}
void adda_kernel_lanierBoxScatterKernel(unsigned int dipBlocks, unsigned int threads, const AddaCudaKernelComplex *arg, AddaCudaKernelComplex *grid, const unsigned short *position, size_t ndip,size_t nx,size_t ny,size_t n)
{
    lanierBoxScatterKernel<<<dipBlocks,threads>>>(arg, grid, position, ndip, nx, ny, n);
}
void adda_kernel_lanierBoxGatherKernel(unsigned int dipBlocks, unsigned int threads, const AddaCudaKernelComplex *grid, AddaCudaKernelComplex *out, const unsigned short *position, size_t ndip,size_t nx,size_t ny,size_t n)
{
    lanierBoxGatherKernel<<<dipBlocks,threads>>>(grid, out, position, ndip, nx, ny, n);
}
void adda_kernel_lanierFullBoxScatterKernel(unsigned int dipBlocks, unsigned int threads, const AddaCudaKernelComplex *arg, AddaCudaKernelComplex *grid, const unsigned short *position, const unsigned char *material, const AddaCudaKernelComplex *cc, size_t ndip,size_t nx,size_t ny,size_t n,AddaCudaKernelReal sqrt_dipvol, int active_material,size_t x0,size_t y0,size_t z0)
{
    lanierFullBoxScatterKernel<<<dipBlocks,threads>>>(arg, grid, position, material, cc, ndip, nx, ny, n, sqrt_dipvol, active_material, x0, y0, z0);
}
void adda_kernel_lanierFullBoxGatherKernel(unsigned int dipBlocks, unsigned int threads, const AddaCudaKernelComplex *grid, AddaCudaKernelComplex *out, const unsigned short *position, const unsigned char *material, const AddaCudaKernelComplex *cc, size_t ndip,size_t nx,size_t ny,size_t n,AddaCudaKernelReal sqrt_dipvol, int active_material,size_t x0,size_t y0,size_t z0)
{
    lanierFullBoxGatherKernel<<<dipBlocks,threads>>>(grid, out, position, material, cc, ndip, nx, ny, n, sqrt_dipvol, active_material, x0, y0, z0);
}
void adda_kernel_materialProjectionKernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *v,const unsigned char *material, size_t ndip,int active_material)
{
    materialProjectionKernel<<<dipBlocks,threads>>>(v, material, ndip, active_material);
}
void adda_kernel_lanierReducedMultiplyKernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *grid, const AddaCudaKernelComplex *coef, size_t nx,size_t ny,size_t nz,size_t n, size_t rx,size_t ry,size_t rz,size_t nred)
{
    lanierReducedMultiplyKernel<<<dipBlocks,threads>>>(grid, coef, nx, ny, nz, n, rx, ry, rz, nred);
}
void adda_kernel_scatterSliceKernel(unsigned int dipBlocks, unsigned int threads, const AddaCudaKernelComplex *arg, AddaCudaKernelComplex *slice, const unsigned char *material, const unsigned short *position, const AddaCudaKernelComplex *cc, size_t ndip, size_t boxX, size_t boxY, size_t sliceN, int her)
{
    scatterSliceKernel<<<dipBlocks,threads>>>(arg, slice, material, position, cc, ndip, boxX, boxY, sliceN, her);
}
void adda_kernel_sliceToPlaneBatchKernel(unsigned int dipBlocks, unsigned int threads, const AddaCudaKernelComplex *slice, AddaCudaKernelComplex *plane, size_t kz0, size_t activeSlices, size_t boxX, size_t boxY, size_t gridX, size_t boxXY, size_t sliceN, size_t planeN)
{
    sliceToPlaneBatchKernel<<<dipBlocks,threads>>>(slice, plane, kz0, activeSlices, boxX, boxY, gridX, boxXY, sliceN, planeN);
}
void adda_kernel_spectralMultiplySliceBatchKernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *plane, const AddaCudaKernelComplex *D, size_t kz0, size_t activeSlices, size_t gridX, size_t gridY, size_t gridZ, size_t planeN, size_t DsizeX, size_t DsizeY, size_t DsizeZ, int reduced, int transposed, int lowMemGreen)
{
    spectralMultiplySliceBatchKernel<<<dipBlocks,threads>>>(plane, D, kz0, activeSlices, gridX, gridY, gridZ, planeN, DsizeX, DsizeY, DsizeZ, reduced, transposed, lowMemGreen);
}
void adda_kernel_planeToSliceBatchKernel(unsigned int dipBlocks, unsigned int threads, const AddaCudaKernelComplex *plane, AddaCudaKernelComplex *slice, size_t kz0, size_t activeSlices, size_t boxX, size_t gridX, size_t boxXY, size_t sliceN, size_t planeN)
{
    planeToSliceBatchKernel<<<dipBlocks,threads>>>(plane, slice, kz0, activeSlices, boxX, gridX, boxXY, sliceN, planeN);
}
void adda_kernel_gatherSliceKernel(unsigned int dipBlocks, unsigned int threads, const AddaCudaKernelComplex *arg, AddaCudaKernelComplex *result, const AddaCudaKernelComplex *slice, const unsigned char *material, const unsigned short *position, const AddaCudaKernelComplex *cc, size_t ndip, size_t boxX, size_t boxY, size_t sliceN, int her)
{
    gatherSliceKernel<<<dipBlocks,threads>>>(arg, result, slice, material, position, cc, ndip, boxX, boxY, sliceN, her);
}
void adda_kernel_qmrMultKernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *a, const AddaCudaKernelComplex *b, AddaCudaKernelComplex c, size_t n)
{
    qmrMultKernel<<<dipBlocks,threads>>>(a, b, c, n);
}
void adda_kernel_qmrMultSelfKernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *a, AddaCudaKernelComplex c, size_t n)
{
    qmrMultSelfKernel<<<dipBlocks,threads>>>(a, c, n);
}
void adda_kernel_qmrLinComb1Kernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *a, const AddaCudaKernelComplex *b, const AddaCudaKernelComplex *c, AddaCudaKernelComplex c1, size_t n)
{
    qmrLinComb1Kernel<<<dipBlocks,threads>>>(a, b, c, c1, n);
}
void adda_kernel_qmrLinCombKernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *a, const AddaCudaKernelComplex *b, const AddaCudaKernelComplex *c, AddaCudaKernelComplex c1, AddaCudaKernelComplex c2, size_t n)
{
    qmrLinCombKernel<<<dipBlocks,threads>>>(a, b, c, c1, c2, n);
}
void adda_kernel_qmrIncrem110Kernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *a, const AddaCudaKernelComplex *b, const AddaCudaKernelComplex *c, AddaCudaKernelComplex c1, AddaCudaKernelComplex c2, size_t n)
{
    qmrIncrem110Kernel<<<dipBlocks,threads>>>(a, b, c, c1, c2, n);
}
void adda_kernel_qmrIncrem111Kernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *a, const AddaCudaKernelComplex *b, const AddaCudaKernelComplex *c, AddaCudaKernelComplex c1, AddaCudaKernelComplex c2, AddaCudaKernelComplex c3, size_t n)
{
    qmrIncrem111Kernel<<<dipBlocks,threads>>>(a, b, c, c1, c2, c3, n);
}
void adda_kernel_qmrIncrem01Kernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *a, const AddaCudaKernelComplex *b, AddaCudaKernelComplex c, size_t n)
{
    qmrIncrem01Kernel<<<dipBlocks,threads>>>(a, b, c, n);
}
void adda_kernel_qmrIncrem11DCKernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *a, const AddaCudaKernelComplex *b, AddaCudaKernelReal c1, AddaCudaKernelComplex c2, size_t n)
{
    qmrIncrem11DCKernel<<<dipBlocks,threads>>>(a, b, c1, c2, n);
}
void adda_kernel_iterCopyKernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *a, const AddaCudaKernelComplex *b, size_t n)
{
    iterCopyKernel<<<dipBlocks,threads>>>(a, b, n);
}
void adda_kernel_iterIncrem10Kernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *a, const AddaCudaKernelComplex *b, AddaCudaKernelComplex c, size_t n)
{
    iterIncrem10Kernel<<<dipBlocks,threads>>>(a, b, c, n);
}
void adda_kernel_iterIncrem011Kernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *a, const AddaCudaKernelComplex *b, const AddaCudaKernelComplex *c, AddaCudaKernelComplex c1, AddaCudaKernelComplex c2, size_t n)
{
    iterIncrem011Kernel<<<dipBlocks,threads>>>(a, b, c, c1, c2, n);
}
void adda_kernel_iterLinComb1ConjKernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *a, const AddaCudaKernelComplex *b, const AddaCudaKernelComplex *c, AddaCudaKernelComplex c1, size_t n)
{
    iterLinComb1ConjKernel<<<dipBlocks,threads>>>(a, b, c, c1, n);
}
void adda_kernel_iterIncrem110DCConjKernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *a, const AddaCudaKernelComplex *b, const AddaCudaKernelComplex *c, AddaCudaKernelReal c1, AddaCudaKernelComplex c2, size_t n)
{
    iterIncrem110DCConjKernel<<<dipBlocks,threads>>>(a, b, c, c1, c2, n);
}
void adda_kernel_iterMultSelfConjKernel(unsigned int dipBlocks, unsigned int threads, AddaCudaKernelComplex *a, AddaCudaKernelReal c, size_t n)
{
    iterMultSelfConjKernel<<<dipBlocks,threads>>>(a, c, n);
}
void adda_kernel_convertComplexF32ToF64Kernel(unsigned int dipBlocks, unsigned int threads, const cuFloatComplex *src, AddaCudaKernelDoubleComplex *dst, size_t n)
{
    convertComplexF32ToF64Kernel<<<dipBlocks,threads>>>(src, dst, n);
}
void adda_kernel_convertComplexPairF32ToF64Kernel(unsigned int dipBlocks, unsigned int threads, const cuFloatComplex *src_a, const cuFloatComplex *src_b, AddaCudaKernelDoubleComplex *dst_a, AddaCudaKernelDoubleComplex *dst_b, size_t n)
{
    convertComplexPairF32ToF64Kernel<<<dipBlocks,threads>>>(src_a, src_b, dst_a, dst_b, n);
}
}
