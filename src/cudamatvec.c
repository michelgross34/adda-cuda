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
extern doublecomplex * restrict vec1,* restrict vec2,* restrict vec3,* restrict vec4,* restrict vec5,* restrict vec6,* restrict vec7,* restrict Avecbuffer;

#ifdef ADDA_CUDA_INFO
/* Extra diagnostic state.  Disabled completely when CMake INFO=OFF. */
static SYSTEM_TIME previous_matvec_exit;
static bool have_previous_matvec_exit = false;
static bool have_printed_cc_memory = false;
#endif

#ifdef PRECISE_TIMING
void FreeEverything(void);
#endif

static void CudaCheck(const int status,const char *where)
{
    if (status != 0) LogError(ONE_POS,"CUDA error in %s: %s",where,adda_cuda_matvec_last_error());
}

static double complex MakeComplex(const double re,const double im)
{
    return re + I*im;
}

#ifdef ADDA_CUDA_INFO
static void PrintCudaMemory(const char *stage)
{
    AddaCudaMemoryInfo info;
    CudaCheck(adda_cuda_memory_info(&info),"CUDA memory query");
    const double mib=1024.0*1024.0;
    const size_t library_bytes = info.free_after_context_bytes > info.free_after_libraries_bytes
                               ? info.free_after_context_bytes-info.free_after_libraries_bytes : 0;
    const size_t backend_from_context = info.free_after_context_bytes > info.current_free_bytes
                                      ? info.free_after_context_bytes-info.current_free_bytes : 0;
    const size_t device_used = info.device_total_bytes > info.current_free_bytes
                             ? info.device_total_bytes-info.current_free_bytes : 0;
    const size_t fft_explicit = info.fft_grid_bytes + info.surface_fft_grid_bytes +
                                info.slice_z_bytes + info.slice_xy_bytes + info.fft_workspace_bytes;
    const size_t geometry = info.material_bytes + info.position_bytes;
    const size_t known_nonfft = info.green_tensor_bytes + info.surface_tensor_bytes +
                                info.reduction_scratch_bytes +
                                info.matvec_vector_bytes + info.iterative_vector_bytes +
                                info.cc_bytes + geometry;
    const size_t category_sum = known_nonfft + fft_explicit;
    const size_t nonexplicit_delta = backend_from_context > info.explicit_current_bytes
                                   ? backend_from_context-info.explicit_current_bytes : 0;

    /* PrintBoth() is limited to MAX_PARAGRAPH (600 bytes), so keep each
     * block comfortably below that limit instead of building one long paragraph. */
    PrintBoth(logfile,"CUDA GPU memory breakdown [%s]:\n",stage);
    PrintBoth(logfile,
              "  Green tensor Dmatrix             : %10.3f MiB\n"
              "  Surface tensor Rmatrix           : %10.3f MiB\n"
              "  Full FFT grid d_grid             : %10.3f MiB\n"
              "  Surface FFT grid d_gridR         : %10.3f MiB\n"
              "  Slice compact Z buffer           : %10.3f MiB\n"
              "  Slice XY batch buffer            : %10.3f MiB\n",
              (double)info.green_tensor_bytes/mib,
              (double)info.surface_tensor_bytes/mib,
              (double)info.fft_grid_bytes/mib,
              (double)info.surface_fft_grid_bytes/mib,
              (double)info.slice_z_bytes/mib,
              (double)info.slice_xy_bytes/mib);
    PrintBoth(logfile,
              "  Explicit cuFFT workspace         : %10.3f MiB\n"
              "  FP64 reduction scratch           : %10.3f MiB\n"
              "  MatVec vectors d_arg+d_result    : %10.3f MiB\n"
              "  Iterative solver extra vectors   : %10.3f MiB\n"
              "  cc_sqrt                          : %10.3f MiB\n"
              "  material                         : %10.3f MiB\n"
              "  position                         : %10.3f MiB\n",
              (double)info.fft_workspace_bytes/mib,
              (double)info.reduction_scratch_bytes/mib,
              (double)info.matvec_vector_bytes/mib,
              (double)info.iterative_vector_bytes/mib,
              (double)info.cc_bytes/mib,
              (double)info.material_bytes/mib,
              (double)info.position_bytes/mib);
    PrintBoth(logfile,
              "  ------------------------------------------------\n"
              "  Explicit FFT-related subtotal    : %10.3f MiB\n"
              "  Explicit category sum            : %10.3f MiB\n"
              "  Explicit ADDA total              : %10.3f MiB\n"
              "  Peak explicit ADDA allocation    : %10.3f MiB\n",
              (double)fft_explicit/mib,
              (double)category_sum/mib,
              (double)info.explicit_current_bytes/mib,
              (double)info.explicit_peak_bytes/mib);
    PrintBoth(logfile,
              "  CUDA/cuBLAS/cuFFT init delta     : %10.3f MiB\n"
              "  Non-explicit delta since context : %10.3f MiB\n"
              "  Backend delta since context      : %10.3f MiB\n"
              "  Device used / total              : %10.3f / %.3f MiB\n"
              "  Device free                      : %10.3f MiB\n",
              (double)library_bytes/mib,
              (double)nonexplicit_delta/mib,
              (double)backend_from_context/mib,
              (double)device_used/mib,
              (double)info.device_total_bytes/mib,
              (double)info.current_free_bytes/mib);
}
#endif /* ADDA_CUDA_INFO */

void CudaMatVecInit(void)
{
    if (prognosis) return;
    {
        const int expected_real_bytes=(int)(sizeof(doublecomplex)/2);
        const int backend_real_bytes=adda_cuda_backend_real_bytes();
        if (backend_real_bytes!=expected_real_bytes)
            LogError(ONE_POS,
                "CUDA backend precision mismatch: executable expects %d-byte real/complex data but loaded DLL uses %d-byte reals. "
                "Rebuild/copy the matching CUDA backend DLL.",
                expected_real_bytes,backend_real_bytes);
    }
#ifdef ADDA_CUDA_INFO
    have_printed_cc_memory=false;
#endif
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
#ifdef ADDA_CUDA_LOW_MEM
    CudaCheck(adda_cuda_matvec_init_low_mem(&cfg,(size_t)boxX,(size_t)boxY,(size_t)boxZ,
                                            Dmatrix,surface ? Rmatrix : NULL,material,position),
              "CUDA low-memory backend initialization");
#elif defined(ADDA_CUDA_SLICE)
    CudaCheck(adda_cuda_matvec_init_slice(&cfg,(size_t)boxX,(size_t)boxY,(size_t)boxZ,
                                          Dmatrix,surface ? Rmatrix : NULL,material,position),
              "CUDA slice backend initialization");
#else
    CudaCheck(adda_cuda_matvec_init(&cfg,Dmatrix,surface ? Rmatrix : NULL,material,position),
              "CUDA backend initialization");
#endif
#ifdef ADDA_CUDA_INFO
#ifdef ADDA_SINGLE
    PrintBoth(logfile,"CUDA numerical data path: float32 complex vectors/FFT; dot/norm reductions use chunked cuBLAS FP64 with CPU double accumulation; solver recurrence coefficients remain double/double-complex.\n");
#else
    PrintBoth(logfile,"CUDA numerical data path: float64 complex.\n");
#endif
#ifdef ADDA_CUDA_SLICE
    {
        const size_t compact_complex=(size_t)3*gridZ*(size_t)boxX*(size_t)boxY;
        const size_t plane_components=(size_t)3*ADDA_CUDA_SLICE_BATCH;
        const size_t plane_complex=plane_components*gridX*gridY;
        const size_t full_complex=(size_t)3*gridX*gridY*gridZ;
        const double mib=1024.0*1024.0;
        PrintBoth(logfile,
                  "CUDA slice FFT enabled (batch=%d): Z workspace %.3f MiB + XY batch workspace %.3f MiB = %.3f MiB "
                  "(full 3-D vector grid would be %.3f MiB)\n",
                  ADDA_CUDA_SLICE_BATCH,
                  (double)(compact_complex*sizeof(doublecomplex))/mib,
                  (double)(plane_complex*sizeof(doublecomplex))/mib,
                  (double)((compact_complex+plane_complex)*sizeof(doublecomplex))/mib,
                  (double)(full_complex*sizeof(doublecomplex))/mib);
    }
#endif
#ifdef ADDA_CUDA_LOW_MEM
    {
        const size_t dx=gridX/2+1;
        const size_t compact=6*dx*DsizeY*DsizeZ;
        const size_t legacy=6*gridX*DsizeY*DsizeZ;
        const double mib=1024.0*1024.0;
        PrintBoth(logfile,
                  "CUDA low-memory Green tensor enabled: GPU Dmatrix %zux%zux%zu = %.3f MiB "
                  "(slice reference stores %zux%zux%zu = %.3f MiB)\n",
                  dx,DsizeY,DsizeZ,(double)(compact*sizeof(doublecomplex))/mib,
                  gridX,DsizeY,DsizeZ,(double)(legacy*sizeof(doublecomplex))/mib);
    }
#endif
    PrintCudaMemory("after MatVec initialization");
#endif /* ADDA_CUDA_INFO */
}

void CudaMatVecUpdateCC(void)
{
    if (prognosis) return;
    CudaCheck(adda_cuda_matvec_update_cc(cc_sqrt,(size_t)MAX_NMAT*3),"cc_sqrt update");
#ifdef ADDA_CUDA_INFO
    if (!have_printed_cc_memory) {
        PrintCudaMemory("after cc_sqrt allocation/update");
        have_printed_cc_memory=true;
    }
#endif
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

#if defined(PRECISE_TIMING) && defined(ADDA_CUDA_INFO)
    SYSTEM_TIME tvp[2];
    GET_SYSTEM_TIME(tvp);
#endif

    double cuda_time_ms=0.0;
    if (resident)
        CudaCheck(adda_cuda_matvec_execute_gpu(argvec,resultvec,her ? 1 : 0,inprod,&cuda_time_ms),"MatVec_GPU execution");
    else
        CudaCheck(adda_cuda_matvec_execute(argvec,resultvec,her ? 1 : 0,inprod,&cuda_time_ms),"MatVec execution");

    if (ipr) MyInnerProduct(inprod,double_type,1,comm_timing);

#ifdef ADDA_CUDA_INFO
    SYSTEM_TIME current_matvec_exit;
    GET_SYSTEM_TIME(&current_matvec_exit);
    if (have_previous_matvec_exit) {
        const double total_interval_ms =
            1000.0*DiffSystemTime(&previous_matvec_exit,&current_matvec_exit);
        PrintBoth(logfile,
#ifdef ADDA_CUDA_SLICE
                  "CUDA Slice MatVec GPU time%s: %.6f ms; time since previous MatVec end: %.6f ms\n",
#else
                  "CUDA MatVec GPU time%s: %.6f ms; time since previous MatVec end: %.6f ms\n",
#endif
                  resident ? " (resident vectors)" : " (without CPU-GPU copies)",
                  cuda_time_ms,total_interval_ms);
    } else {
        PrintBoth(logfile,
#ifdef ADDA_CUDA_SLICE
                  "CUDA Slice MatVec GPU time%s: %.6f ms; time since previous MatVec end: N/A (first MatVec)\n",
#else
                  "CUDA MatVec GPU time%s: %.6f ms; time since previous MatVec end: N/A (first MatVec)\n",
#endif
                  resident ? " (resident vectors)" : " (without CPU-GPU copies)",
                  cuda_time_ms);
    }
    previous_matvec_exit = current_matvec_exit;
    have_previous_matvec_exit = true;
#endif /* ADDA_CUDA_INFO */

#if defined(PRECISE_TIMING) && defined(ADDA_CUDA_INFO)
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
    const void *v1=NULL,*v2=NULL,*v3=NULL,*v4=NULL,*v5=NULL,*v6=NULL,*v7=NULL;
    const char *method_name=NULL;
    int resident_vectors=0;
    switch ((enum iter)method) {
        case IT_BCGS2:
            v1=vec1; v2=vec2; v3=vec3; v4=vec4;
            method_name="BiCGStab(2)/BCGS2"; resident_vectors=8;
            break;
        case IT_BICG_CS:
            method_name="BiCG_CS"; resident_vectors=4;
            break;
        case IT_CGNR:
            method_name="CGNR"; resident_vectors=4;
            break;
        case IT_BICGSTAB:
            v1=vec1; v2=vec2; v3=vec3;
            method_name="BiCGStab"; resident_vectors=7;
            break;
        case IT_GPBICGSTAB2:
            v1=vec1; v2=vec2; v3=vec3; v4=vec4; v5=vec5; v6=vec6; v7=vec7;
            method_name="GPBiCGStab(2)"; resident_vectors=11;
            break;
        case IT_QMR_CS:
            v1=vec1; v2=vec2; v3=vec3;
            method_name="QMR_CS"; resident_vectors=7;
            break;
        case IT_CSYM:
            v1=vec1; v2=vec2;
            method_name="CSYM"; resident_vectors=6;
            break;
        case IT_QMR_CS_2:
            v1=vec1; v2=vec2;
            method_name="QMR_CS_2"; resident_vectors=6;
            break;
        default:
            LogError(ONE_POS,"Unknown iterative method (%d) in CUDA initialization",method);
    }
    CudaCheck(adda_cuda_iter_init(xvec,rvec,pvec,v1,v2,v3,v4,v5,v6,v7,Avecbuffer),
              "iterative resident-vector initialization");

#ifdef ADDA_CUDA_INFO
    /* d_arg and d_result already exist for MatVec and are reused as xvec/rvec.
     * Therefore only resident_vectors-2 new nrows-sized CUDA vectors are allocated. */
    const size_t bytes_per_vector=local_nRows*sizeof(doublecomplex);
    const size_t extra_bytes=(size_t)(resident_vectors-2)*bytes_per_vector;
    const size_t total_bytes=(size_t)resident_vectors*bytes_per_vector;
    PrintBoth(logfile,
              "CUDA iterative vectors for %s: %d resident vectors, %d extra allocations = %.3f MiB; "
              "total resident-vector footprint = %.3f MiB\n",
              method_name,resident_vectors,resident_vectors-2,
              (double)extra_bytes/(1024.0*1024.0),(double)total_bytes/(1024.0*1024.0));
    PrintCudaMemory("after iterative-solver allocation");
#else
    (void)method_name;
    (void)resident_vectors;
#endif /* ADDA_CUDA_INFO */
}

void CudaIterInitList(const void * const *host_ids,size_t count,const char *method_name)
{
    if (host_ids==NULL || count<2) LogError(ONE_POS,"Invalid CUDA iterative vector list");
    CudaCheck(adda_cuda_iter_init_list(host_ids,count),"iterative resident-vector list initialization");
#ifdef ADDA_CUDA_INFO
    const size_t bytes_per_vector=local_nRows*sizeof(doublecomplex);
    const size_t extra_count=count-2; /* x/r reuse d_arg/d_result */
    const size_t extra_bytes=extra_count*bytes_per_vector;
    const size_t total_bytes=count*bytes_per_vector;
    PrintBoth(logfile,
              "CUDA iterative vectors for %s: %zu resident vectors, %zu extra allocations = %.3f MiB; "
              "total resident-vector footprint = %.3f MiB\n",
              method_name,count,extra_count,
              (double)extra_bytes/(1024.0*1024.0),(double)total_bytes/(1024.0*1024.0));
    PrintCudaMemory("after iterative-solver allocation");
#else
    (void)method_name;
#endif /* ADDA_CUDA_INFO */
}

void CudaIterPrintMemoryBeforeLoop(void)
{
    /* Called after PHASE_INIT and immediately before the first solver iteration.
     * At this point all resident solver buffers are allocated and initialized,
     * so this is the most representative pre-iteration GPU-memory snapshot. */
#ifdef ADDA_CUDA_INFO
    PrintCudaMemory("immediately before iterative loop");
#endif
}

void CudaIterSyncToHost(void)
{
    CudaCheck(adda_cuda_iter_download_all(),"iterative device-to-host synchronization");
}

void CudaIterUploadOne(const doublecomplex *host_id)
{
    CudaCheck(adda_cuda_iter_upload_one(host_id),"single iterative host-to-device upload");
}

void CudaIterDownloadOne(doublecomplex *host_id)
{
    CudaCheck(adda_cuda_iter_download_one(host_id),"single iterative device-to-host download");
}

void CudaIterRelease(void)
{
    CudaCheck(adda_cuda_iter_release(),"iterative resident-vector release");
#ifdef ADDA_CUDA_INFO
    PrintCudaMemory("after iterative-solver release");
#endif
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

double complex CudaIterDotProd(const doublecomplex * restrict a,
                              const doublecomplex * restrict b,
                              TIME_TYPE *comm_timing)
{
    double re=0.0,im=0.0;
    IgnoreComm(comm_timing);
    CudaCheck(adda_cuda_iter_dotc(a,b,&re,&im),"iterative cublasZdotc");
    return MakeComplex(re,im);
}

double complex CudaIterDotProd64(const doublecomplex * restrict a,
                                 const doublecomplex * restrict b,
                                 TIME_TYPE *comm_timing)
{
    double re=0.0,im=0.0;
    IgnoreComm(comm_timing);
    CudaCheck(adda_cuda_iter_dotc(a,b,&re,&im),"iterative FP64 cublasZdotc");
    return re + I*im;
}

double complex CudaIterDotProd_conj(const doublecomplex * restrict a,
                                   const doublecomplex * restrict b,
                                   TIME_TYPE *comm_timing)
{
    double re=0.0,im=0.0;
    IgnoreComm(comm_timing);
    CudaCheck(adda_cuda_iter_dotu(a,b,&re,&im),"iterative cublasZdotu");
    return MakeComplex(re,im);
}

double complex CudaIterDotProdSelf_conj(const doublecomplex * restrict a,TIME_TYPE *comm_timing)
{
    return CudaIterDotProd_conj(a,a,comm_timing);
}

double complex CudaIterDotProdSelf_conj_Norm2(const doublecomplex * restrict a,
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

void CudaIterMult_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,const double complex c)
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

void CudaIterMultSelf_cmplx(doublecomplex * restrict a,const double complex c)
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

void CudaIterIncrem01_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,const double complex c,
                            double * restrict inprod,TIME_TYPE *comm_timing)
{
    IgnoreComm(comm_timing);
    CudaCheck(adda_cuda_iter_increm01(a,b,creal(c),cimag(c),inprod),"iterative Increm01 complex kernel");
}

void CudaIterIncrem10_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,const double complex c,
                            double * restrict inprod,TIME_TYPE *comm_timing)
{
    IgnoreComm(comm_timing);
    CudaCheck(adda_cuda_iter_increm10(a,b,creal(c),cimag(c),inprod),"iterative Increm10 complex kernel");
}

void CudaIterIncrem011_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,
                             const doublecomplex * restrict c,const double complex c1,const double complex c2)
{
    CudaCheck(adda_cuda_iter_increm011(a,b,c,creal(c1),cimag(c1),creal(c2),cimag(c2),NULL),
              "iterative Increm011 kernel");
}

void CudaIterIncrem110_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,
                             const doublecomplex * restrict c,const double complex c1,const double complex c2)
{
    CudaCheck(adda_cuda_iter_increm110(a,b,c,creal(c1),cimag(c1),creal(c2),cimag(c2)),
              "iterative Increm110 kernel");
}

void CudaIterIncrem111_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,
                             const doublecomplex * restrict c,const double complex c1,const double complex c2,
                             const double complex c3)
{
    CudaCheck(adda_cuda_iter_increm111(a,b,c,creal(c1),cimag(c1),creal(c2),cimag(c2),creal(c3),cimag(c3)),
              "iterative Increm111 kernel");
}

void CudaIterIncrem11_d_c(doublecomplex * restrict a,const doublecomplex * restrict b,const double c1,
                          const double complex c2,double * restrict inprod,TIME_TYPE *comm_timing)
{
    IgnoreComm(comm_timing);
    CudaCheck(adda_cuda_iter_increm11_d_c(a,b,c1,creal(c2),cimag(c2),inprod),
              "iterative Increm11_d_c kernel");
}

void CudaIterIncrem110_d_c_conj(doublecomplex * restrict a,const doublecomplex * restrict b,
                                const doublecomplex * restrict c,const double c1,const double complex c2,
                                double * restrict inprod,TIME_TYPE *comm_timing)
{
    IgnoreComm(comm_timing);
    CudaCheck(adda_cuda_iter_increm110_d_c_conj(a,b,c,c1,creal(c2),cimag(c2),inprod),
              "iterative Increm110_d_c_conj kernel");
}

void CudaIterLinComb_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,
                           const doublecomplex * restrict c,const double complex c1,const double complex c2,
                           double * restrict inprod,TIME_TYPE *comm_timing)
{
    IgnoreComm(comm_timing);
    CudaCheck(adda_cuda_iter_lincomb(a,b,c,creal(c1),cimag(c1),creal(c2),cimag(c2),inprod),
              "iterative LinComb kernel");
}

void CudaIterLinComb1_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,
                            const doublecomplex * restrict c,const double complex c1,
                            double * restrict inprod,TIME_TYPE *comm_timing)
{
    IgnoreComm(comm_timing);
    CudaCheck(adda_cuda_iter_lincomb1(a,b,c,creal(c1),cimag(c1),inprod),"iterative LinComb1 kernel");
}

void CudaIterLinComb1_cmplx_conj(doublecomplex * restrict a,const doublecomplex * restrict b,
                                 const doublecomplex * restrict c,const double complex c1,
                                 double * restrict inprod,TIME_TYPE *comm_timing)
{
    IgnoreComm(comm_timing);
    CudaCheck(adda_cuda_iter_lincomb1_conj(a,b,c,creal(c1),cimag(c1),inprod),
              "iterative LinComb1 conjugate kernel");
}

#endif /* ADDA_CUDA */
