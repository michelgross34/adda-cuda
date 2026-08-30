#include "cudamatvec_backend.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using cd = std::complex<double>;

struct CaseData {
    AddaCudaMatVecConfig cfg{};
    std::vector<cd> D;
    std::vector<cd> R;
    std::vector<unsigned char> material;
    std::vector<unsigned short> position;
    std::vector<cd> cc;
    std::vector<cd> arg;
};

static size_t idx3(size_t x,size_t y,size_t z,size_t gx,size_t gy)
{
    return (z*gy+y)*gx+x;
}

static std::vector<cd> dft3(const std::vector<cd>& in,size_t gx,size_t gy,size_t gz,
                            int sx,int sy,int sz)
{
    const double twopi=2.0*std::acos(-1.0);
    const size_t n=gx*gy*gz;
    std::vector<cd> out(n,cd(0,0));
    for(size_t kz=0;kz<gz;kz++) for(size_t ky=0;ky<gy;ky++) for(size_t kx=0;kx<gx;kx++) {
        cd sum=0;
        for(size_t z=0;z<gz;z++) for(size_t y=0;y<gy;y++) for(size_t x=0;x<gx;x++) {
            const double phase=twopi*(sx*(double)(kx*x)/gx + sy*(double)(ky*y)/gy + sz*(double)(kz*z)/gz);
            sum += in[idx3(x,y,z,gx,gy)]*std::exp(cd(0,phase));
        }
        out[idx3(kx,ky,kz,gx,gy)]=sum;
    }
    return out;
}

static void symmv(const cd f[6],const cd x[3],cd y[3])
{
    y[0]=f[0]*x[0]+f[1]*x[1]+f[2]*x[2];
    y[1]=f[1]*x[0]+f[3]*x[1]+f[4]*x[2];
    y[2]=f[2]*x[0]+f[4]*x[1]+f[5]*x[2];
}

static void reflmv(const cd f[6],const cd x[3],cd y[3])
{
    y[0]=f[0]*x[0]+f[1]*x[1]+f[2]*x[2];
    y[1]=f[1]*x[0]+f[3]*x[1]+f[4]*x[2];
    y[2]=-f[2]*x[0]-f[4]*x[1]+f[5]*x[2];
}

static std::vector<cd> reference(const CaseData& tc,bool her,double *norm)
{
    const auto& c=tc.cfg;
    const size_t n=c.gridX*c.gridY*c.gridZ;
    std::vector<cd> spatial(3*n,0.0);
    for(size_t i=0;i<c.ndip;i++) {
        const size_t p=3*i;
        const size_t gi=idx3(tc.position[p],tc.position[p+1],tc.position[p+2],c.gridX,c.gridY);
        for(size_t comp=0;comp<3;comp++) {
            cd a=tc.arg[p+comp];
            if(her) a=std::conj(a);
            spatial[comp*n+gi]=tc.cc[3*tc.material[i]+comp]*a;
        }
    }

    std::vector<cd> freq(3*n),freqR;
    for(size_t comp=0;comp<3;comp++) {
        std::vector<cd> one(spatial.begin()+comp*n,spatial.begin()+(comp+1)*n);
        auto f=dft3(one,c.gridX,c.gridY,c.gridZ,-1,-1,-1);
        std::copy(f.begin(),f.end(),freq.begin()+comp*n);
    }
    if(c.surface) {
        freqR.resize(3*n);
        for(size_t comp=0;comp<3;comp++) {
            std::vector<cd> one(spatial.begin()+comp*n,spatial.begin()+(comp+1)*n);
            auto f=dft3(one,c.gridX,c.gridY,c.gridZ,-1,-1,+1);
            std::copy(f.begin(),f.end(),freqR.begin()+comp*n);
        }
    }

    const bool transposed=(!c.reduced_fft)&&her;
    for(size_t z0=0;z0<c.gridZ;z0++) for(size_t y0=0;y0<c.gridY;y0++) for(size_t x0=0;x0<c.gridX;x0++) {
        const size_t ii=idx3(x0,y0,z0,c.gridX,c.gridY);
        cd xv[3],yv[3],f[6];
        for(int q=0;q<3;q++) xv[q]=freq[(size_t)q*n+ii];

        size_t x=x0,y=y0,z=z0;
        if(transposed) {
            if(x>0) x=c.gridX-x;
            if(y>0) y=c.gridY-y;
            if(z>0) z=c.gridZ-z;
        } else {
            if(y>=c.DsizeY) y=c.gridY-y;
            if(z>=c.DsizeZ) z=c.gridZ-z;
        }
        const size_t db=6*((x*c.DsizeZ+z)*c.DsizeY+y);
        for(int q=0;q<6;q++) f[q]=tc.D[db+(size_t)q];
        if(c.reduced_fft) {
            if(y0>=c.DsizeY) {
                f[1]=-f[1];
                if(z0>=c.DsizeZ) f[2]=-f[2];
                else f[4]=-f[4];
            } else if(z0>=c.DsizeZ) {
                f[2]=-f[2];
                f[4]=-f[4];
            }
        }
        symmv(f,xv,yv);

        if(c.surface) {
            cd xr[3],yr[3];
            for(int q=0;q<3;q++) xr[q]=freqR[(size_t)q*n+ii];
            x=x0;y=y0;z=z0;
            if(transposed) {
                if(x>0) x=c.gridX-x;
                if(y>0) y=c.gridY-y;
            } else if(y>=c.RsizeY) y=c.gridY-y;
            const size_t rb=6*((x*c.gridZ+z)*c.RsizeY+y);
            for(int q=0;q<6;q++) f[q]=tc.R[rb+(size_t)q];
            if(c.reduced_fft && y0>=c.RsizeY) {
                f[1]=-f[1];
                f[4]=-f[4];
            }
            if(transposed) {
                f[2]=-f[2];
                f[4]=-f[4];
            }
            reflmv(f,xr,yr);
            for(int q=0;q<3;q++) yv[q]+=yr[q];
        }
        for(int q=0;q<3;q++) freq[(size_t)q*n+ii]=yv[q];
    }

    std::vector<cd> outGrid(3*n);
    for(size_t comp=0;comp<3;comp++) {
        std::vector<cd> one(freq.begin()+comp*n,freq.begin()+(comp+1)*n);
        auto f=dft3(one,c.gridX,c.gridY,c.gridZ,+1,+1,+1);
        std::copy(f.begin(),f.end(),outGrid.begin()+comp*n);
    }

    std::vector<cd> out(c.nrows);
    double sum=0;
    for(size_t i=0;i<c.ndip;i++) {
        const size_t p=3*i;
        const size_t gi=idx3(tc.position[p],tc.position[p+1],tc.position[p+2],c.gridX,c.gridY);
        for(size_t comp=0;comp<3;comp++) {
            cd a=tc.arg[p+comp];
            if(her) a=std::conj(a);
            cd r=a+tc.cc[3*tc.material[i]+comp]*outGrid[comp*n+gi];
            if(her) r=std::conj(r);
            out[p+comp]=r;
            sum+=std::norm(r);
        }
    }
    if(norm) *norm=sum;
    return out;
}

static CaseData makeCase(size_t gx,size_t gy,size_t gz,bool reduced,bool surface)
{
    CaseData t;
    t.cfg.gridX=gx;t.cfg.gridY=gy;t.cfg.gridZ=gz;
    t.cfg.DsizeY=reduced?gy/2+1:gy;
    t.cfg.DsizeZ=reduced?gz/2+1:gz;
    t.cfg.RsizeY=surface?(reduced?gy/2+1:gy):0;
    t.cfg.surface=surface?1:0;
    t.cfg.reduced_fft=reduced?1:0;
    t.cfg.device=-1;

    const size_t maxDip=std::min<size_t>(6,gx*gy*gz);
    t.cfg.ndip=maxDip;
    t.cfg.nrows=3*maxDip;
    t.material.resize(maxDip);
    t.position.resize(3*maxDip);
    for(size_t i=0;i<maxDip;i++) {
        const size_t flat=(i*7+1)%(gx*gy*gz);
        const size_t x=flat%gx;
        const size_t yz=flat/gx;
        const size_t y=yz%gy;
        const size_t z=yz/gy;
        t.position[3*i]=(unsigned short)x;
        t.position[3*i+1]=(unsigned short)y;
        t.position[3*i+2]=(unsigned short)z;
        t.material[i]=(unsigned char)(i%2);
    }

    t.cc.resize(6);
    for(size_t i=0;i<t.cc.size();i++) t.cc[i]=cd(0.08+0.01*i,-0.015+0.004*i);
    t.arg.resize(t.cfg.nrows);
    for(size_t i=0;i<t.arg.size();i++) t.arg[i]=cd(0.11+0.013*i,-0.07+0.009*i);

    t.D.resize(6*gx*t.cfg.DsizeZ*t.cfg.DsizeY);
    for(size_t i=0;i<t.D.size();i++) t.D[i]=cd(0.0007*(1+(i%19)),-0.00031*(1+(i%13)));
    if(surface) {
        t.R.resize(6*gx*gz*t.cfg.RsizeY);
        for(size_t i=0;i<t.R.size();i++) t.R[i]=cd(-0.00041*(1+(i%17)),0.00023*(1+(i%11)));
    }
    return t;
}

static bool closeEnough(cd a,cd b,double atol,double rtol)
{
    const double err=std::abs(a-b);
    const double scale=std::max(std::abs(a),std::abs(b));
    return err<=atol+rtol*scale;
}

static int runCase(const char *name,CaseData& tc,bool her)
{
    if(adda_cuda_matvec_init(&tc.cfg,tc.D.data(),tc.cfg.surface?tc.R.data():nullptr,
                             tc.material.data(),tc.position.data())!=0) {
        std::fprintf(stderr,"%s: init failed: %s\n",name,adda_cuda_matvec_last_error());
        return 1;
    }
    if(adda_cuda_matvec_update_cc(tc.cc.data(),tc.cc.size())!=0) {
        std::fprintf(stderr,"%s: cc update failed: %s\n",name,adda_cuda_matvec_last_error());
        adda_cuda_matvec_free();
        return 1;
    }

    std::vector<cd> before=tc.arg;
    std::vector<cd> gpu(tc.cfg.nrows);
    double gpuNorm=0,refNorm=0;
    if(adda_cuda_matvec_execute(tc.arg.data(),gpu.data(),her?1:0,&gpuNorm)!=0) {
        std::fprintf(stderr,"%s: execute failed: %s\n",name,adda_cuda_matvec_last_error());
        adda_cuda_matvec_free();
        return 1;
    }
    auto ref=reference(tc,her,&refNorm);
    adda_cuda_matvec_free();

    if(std::memcmp(before.data(),tc.arg.data(),before.size()*sizeof(cd))!=0) {
        std::fprintf(stderr,"%s: input vector was modified\n",name);
        return 1;
    }

    const double atol=2e-9,rtol=2e-10;
    double maxErr=0;
    for(size_t i=0;i<gpu.size();i++) {
        maxErr=std::max(maxErr,std::abs(gpu[i]-ref[i]));
        if(!closeEnough(gpu[i],ref[i],atol,rtol)) {
            std::fprintf(stderr,"%s: mismatch at %zu gpu=(%.17g,%.17g) ref=(%.17g,%.17g) err=%.3e\n",
                         name,i,gpu[i].real(),gpu[i].imag(),ref[i].real(),ref[i].imag(),std::abs(gpu[i]-ref[i]));
            return 1;
        }
    }
    const double normErr=std::abs(gpuNorm-refNorm);
    if(normErr>atol+rtol*std::max(std::abs(gpuNorm),std::abs(refNorm))) {
        std::fprintf(stderr,"%s: norm mismatch gpu=%.17g ref=%.17g err=%.3e\n",name,gpuNorm,refNorm,normErr);
        return 1;
    }
    std::printf("PASS %-28s max|delta|=%.3e norm_delta=%.3e\n",name,maxErr,normErr);
    return 0;
}

int main()
{
    int count=0;
    cudaError_t ce=cudaGetDeviceCount(&count);
    if(ce!=cudaSuccess || count==0) {
        std::fprintf(stderr,"SKIP: no CUDA device available\n");
        return 77;
    }

    int failures=0;
    auto a=makeCase(2,2,2,true,false);
    failures+=runCase("2x2x2 reduced",a,false);
    auto b=makeCase(4,4,4,true,true);
    failures+=runCase("4x4x4 reduced surface her",b,true);
    auto c=makeCase(4,4,2,false,true);
    failures+=runCase("4x4x2 full surface her",c,true);
    auto d=makeCase(4,6,4,false,false);
    failures+=runCase("4x6x4 full",d,false);

    if(failures) {
        std::fprintf(stderr,"CUDA MatVec equivalence: %d case(s) failed\n",failures);
        return 1;
    }
    std::printf("CUDA MatVec equivalence: all cases passed\n");
    return 0;
}
