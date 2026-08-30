/* ADDA integration wrapper for CUDA MatVec and CUDA-resident iterative solvers.
 *
 * The CUDA implementation is intentionally sequential-only.  CPU/MPI/OpenCL
 * configurations are untouched.  The traditional MatVec() uploads/downloads
 * its Krylov vectors, whereas MatVec_GPU() addresses CUDA-resident iterative-solver vectors
 * and performs no vector transfer across PCIe.
 */
#include "const.h" /* keep this first */
#include "cudamatvec.h"

#ifdef ADDA_CUDA

#include "comm.h"
#include "cudamatvec_backend.h"
#include "io.h"
#include "prec_time.h"
#include "vars.h"

#include <complex.h>
#include <stdlib.h>

#ifdef PARALLEL
# error "ADDA_CUDA MatVec currently supports the sequential backend only; use the existing CPU/MPI backend for PARALLEL."
#endif
#ifdef SPARSE
# error "ADDA_CUDA MatVec requires FFT mode and is incompatible with SPARSE."
#endif
#ifdef OPENCL
# error "ADDA_CUDA and OPENCL are mutually exclusive backends."
#endif

extern doublecomplex * restrict Dmatrix,* restrict Rmatrix;
extern const size_t DsizeY,DsizeZ,RsizeY;
extern size_t TotalMatVec;

/* Iterative-solver vectors are defined in calculator.c. */
extern doublecomplex *rvec;
extern doublecomplex * restrict vec1,* restrict vec2,* restrict vec3,* restrict vec4,* restrict Avecbuffer;

#ifdef PRECISE_TIMING
void FreeEverything(void);
#endif

static void CudaCheck(const int status,const char *where)
{
    if (status != 0) LogError(ONE_POS,"CUDA error in %s: %s",where,adda_cuda_matvec_last_error());
}

static doublecomplex MakeComplex(const double re,const double im)
{
    return re + I*im;
}

void CudaMatVecInit(void)
{
    if (prognosis) return;
    AddaCudaMatVecConfig cfg;
    cfg.gridX=gridX;
    cfg.gridY=gridY;
    cfg.gridZ=gridZ;
    cfg.DsizeY=DsizeY;
    cfg.DsizeZ=DsizeZ;
    cfg.RsizeY=surface ? RsizeY : 0;
    cfg.ndip=local_nvoid_Ndip;
    cfg.nrows=local_nRows;
    cfg.surface=surface ? 1 : 0;
    cfg.reduced_fft=reduced_FFT ? 1 : 0;
    cfg.device=-1;
    CudaCheck(adda_cuda_matvec_init(&cfg,Dmatrix,surface ? Rmatrix : NULL,material,position),"CUDA backend initialization");
}

void CudaMatVecUpdateCC(void)
{
    if (prognosis) return;
    CudaCheck(adda_cuda_matvec_update_cc(cc_sqrt,(size_t)MAX_NMAT*3),"cc_sqrt update");
}

void CudaMatVecFree(void)
{
    if (!prognosis) adda_cuda_matvec_free();
}

static void MatVecCommon(doublecomplex * restrict argvec,
                         doublecomplex * restrict resultvec,
                         double *inprod,
                         const bool her,
                         TIME_TYPE *timing,
                         TIME_TYPE *comm_timing,
                         const bool resident)
{
    const bool ipr=(inprod!=NULL);
    TIME_TYPE tstart=GET_TIME();
    if (ipr && !ipr_required) LogError(ONE_POS,"Incompatibility error in CUDA MatVec");

#ifdef PRECISE_TIMING
    SYSTEM_TIME tvp[2];
    GET_SYSTEM_TIME(tvp);
#endif

    double cuda_time_ms=0.0;
    if (resident)
        CudaCheck(adda_cuda_matvec_execute_gpu(argvec,resultvec,her ? 1 : 0,inprod,&cuda_time_ms),"MatVec_GPU execution");
    else
        CudaCheck(adda_cuda_matvec_execute(argvec,resultvec,her ? 1 : 0,inprod,&cuda_time_ms),"MatVec execution");

    PrintBoth(logfile,"CUDA MatVec GPU time%s: %.6f ms\n",resident ? " (resident vectors)" : " (without CPU-GPU copies)",cuda_time_ms);
    if (ipr) MyInnerProduct(inprod,double_type,1,comm_timing);

#ifdef PRECISE_TIMING
    GET_SYSTEM_TIME(tvp+1);
    if (IFROOT) {
        PrintBoth(logfile,
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
            "             CUDA MatVec timing           \n"
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
            "Total = "FFORMPT"\n\n"
            "%s\n",
            DiffSystemTime(tvp,tvp+1),
            resident ? "MatVec_GPU uses resident CUDA vectors (no H2D/D2H vector copies)."
                     : "MatVec includes H2D/D2H vector transfers; reported GPU time excludes those copies.");
        PRINTFB("\nPrecise timing is complete. Finishing execution.\n");
    }
    FreeEverything();
    Stop(EXIT_SUCCESS);
#endif

    (*timing)+=GET_TIME()-tstart;
    TotalMatVec++;
}

void MatVec(doublecomplex * restrict argvec,
            doublecomplex * restrict resultvec,
            double *inprod,
            const bool her,
            TIME_TYPE *timing,
            TIME_TYPE *comm_timing)
{
    MatVecCommon(argvec,resultvec,inprod,her,timing,comm_timing,false);
}

void MatVec_GPU(doublecomplex * restrict argvec,
                doublecomplex * restrict resultvec,
                double *inprod,
                const bool her,
                TIME_TYPE *timing,
                TIME_TYPE *comm_timing)
{
    MatVecCommon(argvec,resultvec,inprod,her,timing,comm_timing,true);
}

void CudaIterInit(const int method)
{
    const void *v1=NULL,*v2=NULL,*v3=NULL,*v4=NULL;
    switch ((enum iter)method) {
        case IT_BCGS2:    v1=vec1; v2=vec2; v3=vec3; v4=vec4; break;
        case IT_BICG_CS:
        case IT_CGNR:     break;
        case IT_BICGSTAB:
        case IT_QMR_CS:   v1=vec1; v2=vec2; v3=vec3; break;
        case IT_CSYM:
        case IT_QMR_CS_2: v1=vec1; v2=vec2; break;
        default: LogError(ONE_POS,"Unknown iterative method (%d) in CUDA initialization",method);
    }
    CudaCheck(adda_cuda_iter_init(xvec,rvec,pvec,v1,v2,v3,v4,Avecbuffer),
              "iterative resident-vector initialization");
}

void CudaIterSyncToHost(void)
{
    CudaCheck(adda_cuda_iter_download_all(),"iterative device-to-host synchronization");
}

static void IgnoreComm(TIME_TYPE *comm_timing)
{
    (void)comm_timing; /* ADDA_CUDA is sequential; no MPI reduction is needed. */
}

double CudaIterNorm2(const doublecomplex * restrict a,TIME_TYPE *comm_timing)
{
    double res=0.0;
    IgnoreComm(comm_timing);
    CudaCheck(adda_cuda_iter_norm2(a,&res),"iterative cublasDznrm2");
    return res;
}

doublecomplex CudaIterDotProd(const doublecomplex * restrict a,
                              const doublecomplex * restrict b,
                              TIME_TYPE *comm_timing)
{
    double re=0.0,im=0.0;
    IgnoreComm(comm_timing);
    CudaCheck(adda_cuda_iter_dotc(a,b,&re,&im),"iterative cublasZdotc");
    return MakeComplex(re,im);
}

doublecomplex CudaIterDotProd_conj(const doublecomplex * restrict a,
                                   const doublecomplex * restrict b,
                                   TIME_TYPE *comm_timing)
{
    double re=0.0,im=0.0;
    IgnoreComm(comm_timing);
    CudaCheck(adda_cuda_iter_dotu(a,b,&re,&im),"iterative cublasZdotu");
    return MakeComplex(re,im);
}

doublecomplex CudaIterDotProdSelf_conj(const doublecomplex * restrict a,TIME_TYPE *comm_timing)
{
    return CudaIterDotProd_conj(a,a,comm_timing);
}

doublecomplex CudaIterDotProdSelf_conj_Norm2(const doublecomplex * restrict a,
                                             double * restrict norm,
                                             TIME_TYPE *comm_timing)
{
    double re=0.0,im=0.0;
    IgnoreComm(comm_timing);
    CudaCheck(adda_cuda_iter_dotu_self_norm2(a,&re,&im,norm),
              "iterative cublasZdotu+cublasDznrm2");
    return MakeComplex(re,im);
}

void CudaIterCopy(doublecomplex * restrict a,const doublecomplex * restrict b)
{
    CudaCheck(adda_cuda_iter_copy(a,b),"iterative copy kernel");
}

void CudaIterMult(doublecomplex * restrict a,const doublecomplex * restrict b,const double c)
{
    CudaCheck(adda_cuda_iter_mult(a,b,c,0.0),"iterative real multiply kernel");
}

void CudaIterMult_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,const doublecomplex c)
{
    CudaCheck(adda_cuda_iter_mult(a,b,creal(c),cimag(c)),"iterative complex multiply kernel");
}

void CudaIterMultSelf(doublecomplex * restrict a,const double c)
{
    CudaCheck(adda_cuda_iter_mult_self(a,c,0.0),"iterative real self-multiply kernel");
}

void CudaIterMultSelf_conj(doublecomplex * restrict a,const double c)
{
    CudaCheck(adda_cuda_iter_mult_self_conj(a,c),"iterative conjugate self-multiply kernel");
}

void CudaIterMultSelf_cmplx(doublecomplex * restrict a,const doublecomplex c)
{
    CudaCheck(adda_cuda_iter_mult_self(a,creal(c),cimag(c)),"iterative complex self-multiply kernel");
}

void CudaIterIncrem(doublecomplex * restrict a,const doublecomplex * restrict b,
                    double * restrict inprod,TIME_TYPE *comm_timing)
{
    IgnoreComm(comm_timing);
    CudaCheck(adda_cuda_iter_increm01(a,b,1.0,0.0,inprod),"iterative a+=b kernel");
}

void CudaIterIncrem01(doublecomplex * restrict a,const doublecomplex * restrict b,const double c,
                      double * restrict inprod,TIME_TYPE *comm_timing)
{
    IgnoreComm(comm_timing);
    CudaCheck(adda_cuda_iter_increm01(a,b,c,0.0,inprod),"iterative Increm01 real kernel");
}

void CudaIterIncrem10(doublecomplex * restrict a,const doublecomplex * restrict b,const double c,
                      double * restrict inprod,TIME_TYPE *comm_timing)
{
    IgnoreComm(comm_timing);
    CudaCheck(adda_cuda_iter_increm10(a,b,c,0.0,inprod),"iterative Increm10 real kernel");
}

void CudaIterIncrem01_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,const doublecomplex c,
                            double * restrict inprod,TIME_TYPE *comm_timing)
{
    IgnoreComm(comm_timing);
    CudaCheck(adda_cuda_iter_increm01(a,b,creal(c),cimag(c),inprod),"iterative Increm01 complex kernel");
}

void CudaIterIncrem10_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,const doublecomplex c,
                            double * restrict inprod,TIME_TYPE *comm_timing)
{
    IgnoreComm(comm_timing);
    CudaCheck(adda_cuda_iter_increm10(a,b,creal(c),cimag(c),inprod),"iterative Increm10 complex kernel");
}

void CudaIterIncrem011_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,
                             const doublecomplex * restrict c,const doublecomplex c1,const doublecomplex c2)
{
    CudaCheck(adda_cuda_iter_increm011(a,b,c,creal(c1),cimag(c1),creal(c2),cimag(c2),NULL),
              "iterative Increm011 kernel");
}

void CudaIterIncrem110_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,
                             const doublecomplex * restrict c,const doublecomplex c1,const doublecomplex c2)
{
    CudaCheck(adda_cuda_iter_increm110(a,b,c,creal(c1),cimag(c1),creal(c2),cimag(c2)),
              "iterative Increm110 kernel");
}

void CudaIterIncrem111_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,
                             const doublecomplex * restrict c,const doublecomplex c1,const doublecomplex c2,
                             const doublecomplex c3)
{
    CudaCheck(adda_cuda_iter_increm111(a,b,c,creal(c1),cimag(c1),creal(c2),cimag(c2),creal(c3),cimag(c3)),
              "iterative Increm111 kernel");
}

void CudaIterIncrem11_d_c(doublecomplex * restrict a,const doublecomplex * restrict b,const double c1,
                          const doublecomplex c2,double * restrict inprod,TIME_TYPE *comm_timing)
{
    IgnoreComm(comm_timing);
    CudaCheck(adda_cuda_iter_increm11_d_c(a,b,c1,creal(c2),cimag(c2),inprod),
              "iterative Increm11_d_c kernel");
}

void CudaIterIncrem110_d_c_conj(doublecomplex * restrict a,const doublecomplex * restrict b,
                                const doublecomplex * restrict c,const double c1,const doublecomplex c2,
                                double * restrict inprod,TIME_TYPE *comm_timing)
{
    IgnoreComm(comm_timing);
    CudaCheck(adda_cuda_iter_increm110_d_c_conj(a,b,c,c1,creal(c2),cimag(c2),inprod),
              "iterative Increm110_d_c_conj kernel");
}

void CudaIterLinComb_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,
                           const doublecomplex * restrict c,const doublecomplex c1,const doublecomplex c2,
                           double * restrict inprod,TIME_TYPE *comm_timing)
{
    IgnoreComm(comm_timing);
    CudaCheck(adda_cuda_iter_lincomb(a,b,c,creal(c1),cimag(c1),creal(c2),cimag(c2),inprod),
              "iterative LinComb kernel");
}

void CudaIterLinComb1_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,
                            const doublecomplex * restrict c,const doublecomplex c1,
                            double * restrict inprod,TIME_TYPE *comm_timing)
{
    IgnoreComm(comm_timing);
    CudaCheck(adda_cuda_iter_lincomb1(a,b,c,creal(c1),cimag(c1),inprod),"iterative LinComb1 kernel");
}

void CudaIterLinComb1_cmplx_conj(doublecomplex * restrict a,const doublecomplex * restrict b,
                                 const doublecomplex * restrict c,const doublecomplex c1,
                                 double * restrict inprod,TIME_TYPE *comm_timing)
{
    IgnoreComm(comm_timing);
    CudaCheck(adda_cuda_iter_lincomb1_conj(a,b,c,creal(c1),cimag(c1),inprod),
              "iterative LinComb1 conjugate kernel");
}

#endif /* ADDA_CUDA */
