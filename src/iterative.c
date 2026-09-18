/* A few iterative techniques to solve DDA equations
 *
 * The linear system is composed so that diagonal terms are equal to 1, therefore use of Jacobi preconditioners does not
 * have any effect.
 *
 * CS methods still converge to the right result even when matrix is slightly non-symmetric (e.g. -int so), however they
 * do it much slowly than usually. It is recommended then to use BiCGStab or BCGS2.
 *
 * Copyright (C) ADDA contributors
 * This file is part of ADDA.
 *
 * ADDA is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 *
 * ADDA is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with ADDA. If not, see
 * <http://www.gnu.org/licenses/>.
 */
#include "const.h" // keep this first
// project headers
#include "cmplx.h"
#include "comm.h"
#ifdef ADDA_CUDA
#	include "cudamatvec.h"
#endif
#include "debug.h"
#include "io.h"
#include "linalg.h"
#include "memory.h"
#include "timing.h"
#include "vars.h"
// system headers
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h> // for time_t & time

/* Precision of iterative recurrence scalars. In the CUDA float32 executable
 * the large vectors/FFT stay single precision, but dot products are reduced in
 * FP64 by the CUDA backend. Keep those FP64 results, and all recurrence
 * coefficients derived from them, in double complex instead of narrowing them
 * back to ADDA_SINGLE's float complex. CPU-single behavior is unchanged. */
#if defined(ADDA_CUDA) && defined(ADDA_SINGLE)
typedef double complex itercomplex;
#else
typedef doublecomplex itercomplex;
#endif
#if defined(ADDA_CUDA) && defined(ADDA_SINGLE)
_Static_assert(sizeof(itercomplex)==2*sizeof(double),
               "CUDA single recurrence scalars must remain double complex");
#endif

static inline double IterAbs2(const itercomplex a)
{
	const double ar=creal(a),ai=cimag(a);
	return ar*ar+ai*ai;
}

#ifdef OCL_BLAS
#	include "oclcore.h"
#	include <clBLAS.h> //external library
#	include <clBLAS.version.h>
#endif

// SEMI-GLOBAL VARIABLES

// defined and initialized in calculator.c
extern doublecomplex *rvec; // can't be declared restrict due to SwapPointers
extern doublecomplex * restrict vec1,* restrict vec2,* restrict vec3,* restrict vec4,* restrict vec5,* restrict vec6,* restrict vec7,* restrict Avecbuffer;
// defined and initialized in fft.c
#if !defined(OPENCL) && !defined(SPARSE)
extern doublecomplex * restrict Xmatrix; // used as storage for arrays in WKB init field
#endif
// defined and initialized in param.c
extern const double iter_eps;
extern const enum init_field InitField;
extern const char *infi_fnameY,*infi_fnameX;
extern const bool recalc_resid;
extern const bool reliable_resid;
extern const bool reliable_resid_force_restart;
extern const enum chpoint chp_type;
extern const time_t chp_time;
extern const char *chp_dir;
// defined and initialized in timing.c
extern time_t last_chp_wt;
extern TIME_TYPE Timing_OneIter,Timing_OneIterComm,Timing_InitIter,Timing_InitIterComm,Timing_IntFieldOneComm,
	Timing_MVP,Timing_MVPComm,Timing_OneIterMVP,Timing_OneIterMVPComm;
extern size_t TotalIter;

// LOCAL VARIABLES

#define RESID_STRING "RE_%03d = "EFORM // string containing residual value
#define FFORM_PROG "% .6f"  // format for progress value

static double inprodR;     // used as |r_0|^2 and best squared norm of residual up to some iteration
static double inprodRp1;   // used as |r_k+1|^2 and squared norm of current residual
static double epsB;        // stopping criterion
static double resid_scale; // scale to get square of relative error
static double prev_err;    // previous relative error; used in ProgressReport, initialized in IterativeSolver
static int ind_m;          // index of iterative method
static int niter;          // iteration count
static int counter;        // number of successive iterations without residual decrease
static bool chp_exit;      // checkpoint occurred - exit
static bool complete;      // complete iteration was performed (not stopped in the middle)
	// whether matrix-vector product computed during initialization can be reused at first iteration
static bool matvec_ready;
#ifdef ADDA_CUDA
/* LANIER_PARTITION_V2 asks the ordinary solver core to keep the transformed
 * xvec supplied by the previous regional/full solve instead of rebuilding an
 * initial field from InitField. */
static bool lanier_partition_use_existing_x=false;
static bool lanier_partition_internal_call=false;
#endif
static bool gp2_restart_request; // request PHASE_INIT to rebuild GPBiCGStab(2) after a true-residual check
static bool bcgs2_restart_request; // request PHASE_INIT to rebuild BiCGStab(2)/BCGS2 after a true-residual check
/* This flag is referenced by solver PHASE_INIT code in all builds, but it is
 * only asserted by the CUDA LANIER_FULL reliable-residual controller. */
static bool lanier_hard_restart_request=false;
#ifdef ADDA_CUDA
/* LANIER_FULL uses the DDSCAT hard-reset/reliable-residual policy for every
 * CUDA solver and every CUDA executable (full, slice, low-memory; single/double).
 * A hard restart keeps the current physical solution x, replaces the recurrence
 * residual by an independently recomputed one, and rebuilds only Krylov history. */
static bool lanier_true_converged=false;
static double lanier_physical_rhs_norm2=0.0;
static double lanier_reset_baseline=1.0;
static int lanier_reset_mandatory_count=0;
static int lanier_reset_ratio_count=0;
static int lanier_reset_reliable_count=0;
static doublecomplex *lanier_recursive_residual_vec=NULL;
#endif
typedef struct // data for checkpoints
{
	void *ptr; // pointer to the data
	int size;  // size of one element
} chp_data;
chp_data * restrict scalars,* restrict vectors;
enum phase {
	PHASE_VARS, // Initialization of variables, and linking them to scalars and vectors
	PHASE_INIT, // Initialization of iterations
	PHASE_ITER  // Each iteration
};
struct iter_params_struct {
	enum iter meth;   // identifier
	int mc;           // maximum allowed number of iterations without residual decrease
	int sc_N;         // number of additional scalars to describe the state
	int vec_N;        // number of additional vectors to describe the state
	void (*func)(const enum phase); // pointer to implementation of the iterative solver
};

#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 1100) && (__INTEL_COMPILER < 1200)
#	define WORKAROUND146 // workaround for issue 146 (should not be relevant to modern compilers)
static doublecomplex dumb;
#endif

#define ITER_FUNC(name) static void name(const enum phase ph)

ITER_FUNC(BCGS2);
ITER_FUNC(BiCG_CS);
ITER_FUNC(BiCGStab);
ITER_FUNC(BiCGStab4);
ITER_FUNC(BiCGStab8);
ITER_FUNC(BiCGStab12);
ITER_FUNC(CGNR);
ITER_FUNC(CSYM);
ITER_FUNC(GPBiCGStab2);
ITER_FUNC(GPBiCGStab4);
ITER_FUNC(QMR_CS);
ITER_FUNC(QMR_CS_2);
/* TO ADD NEW ITERATIVE SOLVER
 * Add the line to this list in the alphabetical order, analogous to the ones already present. The variable part is the
 * name of the function, implementing the method. The macros expands to a function prototype.
 */

static const struct iter_params_struct params[]={
	{IT_BCGS2,15000,3,1,BCGS2},
	{IT_BICG_CS,50000,1,0,BiCG_CS},
	{IT_BICGSTAB,30000,3,3,BiCGStab},
	{IT_BICGSTAB4,30000,3,9,BiCGStab4},
	{IT_BICGSTAB8,30000,3,17,BiCGStab8},
	{IT_BICGSTAB12,30000,3,25,BiCGStab12},
	{IT_CGNR,10,1,0,CGNR},
	{IT_CSYM,10,6,2,CSYM},
	{IT_GPBICGSTAB2,30000,1,7,GPBiCGStab2},
	{IT_GPBICGSTAB4,30000,0,11,GPBiCGStab4},
	{IT_QMR_CS,50000,8,3,QMR_CS},
	{IT_QMR_CS_2,50000,5,2,QMR_CS_2}
	/* TO ADD NEW ITERATIVE SOLVER
	 * Add its parameters to this list in the alphabetical order. The parameters, in order of appearance, are identifier
	 * (specified in const.h), maximum allowed number of iterations without the residual decrease, numbers of additional
	 * scalars and vectors to describe the state of the iterative solver (see comment before function SaveCheckpoint),
	 * and name of a function, implementing the method.
	 */
};

// EXTERNAL FUNCTIONS

// matvec.c
void MatVec(doublecomplex * restrict in,doublecomplex * restrict out,double * inprod,bool her,TIME_TYPE *timing,
	TIME_TYPE *comm_timing);

#ifdef OCL_BLAS
// Test clBLAS version (specific numbers is because we never considered earlier versions)
#	define CLBLAS_VER_REQ 2
#	define CLBLAS_SUBVER_REQ 12
#	if !GREATER_EQ2(clblasVersionMajor,clblasVersionMinor,CLBLAS_VER_REQ,CLBLAS_SUBVER_REQ)
#		error "clBLAS version is too old"
#	endif

// Error-checking functionality for clBLAS
#	define CLBLAS_CH_ERR(a) Check_clBLAS_Err(a,ALL_POS)

//======================================================================================================================

static const char *Print_clBLAS_Errstring(clblasStatus err)
// produces meaningful error message from the clBLAS-specific error code, based on clBLAS.h v.2.12.0 (NULL if not found)
{
	switch (err) {
		case clblasNotImplemented:      return "Functionality not implemented";
		case clblasNotInitialized:      return "Library not initialized";
		case clblasInvalidMatA:         return "Invalid matrix A";
		case clblasInvalidMatB:         return "Invalid matrix B";
		case clblasInvalidMatC:         return "Invalid matrix C";
		case clblasInvalidVecX:         return "Invalid vector X";
		case clblasInvalidVecY:         return "Invalid vector Y";
		case clblasInvalidDim:          return "Invalid input dimensions";
		case clblasInvalidLeadDimA:     return "Leading dimension of matrix A smaller than the first size";
		case clblasInvalidLeadDimB:     return "Leading dimension of matrix B smaller than the second size";
		case clblasInvalidLeadDimC:     return "Leading dimension of matrix C smaller than the third size";
		case clblasInvalidIncX:         return "Null increment for vector X";
		case clblasInvalidIncY:         return "Null increment for vector Y";
		case clblasInsufficientMemMatA: return "Insufficient memory for matrix A";
		case clblasInsufficientMemMatB: return "Insufficient memory for matrix B";
		case clblasInsufficientMemMatC: return "Insufficient memory for matrix C";
		case clblasInsufficientMemVecX: return "Insufficient memory for vector X";
		case clblasInsufficientMemVecY: return "Insufficient memory for vector Y";
		default:                        return NULL;
	}
}

//======================================================================================================================

static void Check_clBLAS_Err(const clblasStatus err,ERR_LOC_DECL)
/* Checks error code for clBLAS calls and prints error if necessary. First searches among clBLAS specific errors. If not
 * found, uses general error processing for CL calls (since clBLAS error codes can take standard CL values as well).
 */
{
	if (err != clblasSuccess) {
		const char *str=Print_clBLAS_Errstring(err);
		if (str!=NULL) LogError(ERR_LOC_CALL,"clBLAS error code %d: %s\n",err,str);
		else PrintCLErr((cl_int)err,ERR_LOC_CALL,NULL);
	}
}

#endif

//======================================================================================================================

static void MatVec_wrapper(doublecomplex * restrict in,doublecomplex * restrict out,double * inprod,bool her,
	TIME_TYPE *timing,TIME_TYPE *comm_timing)
/* function wrapper for MatVec to be called within the iterative solver if the solver is able to use clBLAS, i.e.
 * the host and GPU memory does not have to be synchronized. Currently it is only used in the BiCG solver.
 */
{
#ifdef OCL_BLAS
	bufupload=false;
#endif
	MatVec(in,out,inprod,her,timing,comm_timing);
#ifdef OCL_BLAS
	bufupload=true;
#endif
}

//======================================================================================================================

static inline void SwapPointers(doublecomplex **a,doublecomplex **b)
/* swap two pointers of (doublecomplex *) type; should work for others but will give "Suspicious pointer conversion"
 * warning. While this is a convenient function that can save some copying between memory blocks, it doesn't allow
 * consistent usage of 'restrict' keyword for affected pointers. This may hamper some optimizations. Hopefully, the most
 * important optimizations are those in the linalg.c, which can be improved by using 'restrict' keyword in the functions
 * themselves.
 */
{
	doublecomplex *tmp;

	tmp=*a;
	*a=*b;
	*b=tmp;
}

//======================================================================================================================

/* Checkpoint systems saves the current state of the iterative solver to the file. By default (for every iterative
 * solver) a number of scalars and vectors are saved. The scalars include, among others, inprodR. There are 3 default
 * vectors: xvec, rvec, pvec (Avecbuffer is _not_ saved). If the iterative solver requires any other scalars or vectors
 * to describe its state, this information should be specified in structure arrays 'scalars' and 'vectors'.
 */

static void SaveIterChpoint(void)
/* save a binary checkpoint; only limitedly foolproof - user should take care to load checkpoints on the same machine
 * (number of processors) and with the same command line.
 */
{
	int i;
	char fname[MAX_FNAME];
	FILE * restrict chp_file;
	TIME_TYPE tstart;

	tstart=GET_TIME();
#ifdef ADDA_CUDA
	/* During CUDA iterative solves the device copies are authoritative.
	 * Checkpoints use host file I/O, so synchronize the registered vectors here. */
	CudaIterSyncToHost();
#endif
	if (IFROOT) {
		// create directory "chp_dir" if needed and open info file
		SnprintfErr(ONE_POS,fname,MAX_FNAME,"%s/"F_CHP_LOG,chp_dir);
		if ((chp_file=fopen(fname,"w"))==NULL) {
			MkDirErr(chp_dir,ONE_POS);
			chp_file=FOpenErr(fname,"w",ONE_POS);
		}
		// write info and close file
		fprintf(chp_file,"Info about the run, which produced the checkpoint, can be found in ../%s",directory);
		FCloseErr(chp_file,fname,ONE_POS);
	}
	// wait to ensure that directory exists
	Synchronize();
	// open output file; writing errors are checked only for vectors
	SnprintfErr(ALL_POS,fname,MAX_FNAME,"%s/"F_CHP,chp_dir,ringid);
	chp_file=FOpenErr(fname,"wb",ALL_POS);
	// write common scalars
	fwrite(&ind_m,sizeof(int),1,chp_file);
	fwrite(&local_nRows,sizeof(size_t),1,chp_file);
	fwrite(&niter,sizeof(int),1,chp_file);
	fwrite(&counter,sizeof(int),1,chp_file);
	fwrite(&inprodR,sizeof(double),1,chp_file);
	fwrite(&prev_err,sizeof(double),1,chp_file); // written on ALL processors but used only on root
	fwrite(&resid_scale,sizeof(double),1,chp_file);
	// write specific scalars
	for (i=0;i<params[ind_m].sc_N;i++) fwrite(scalars[i].ptr,scalars[i].size,1,chp_file);
	// write common vectors
	if (fwrite(xvec,sizeof(doublecomplex),local_nRows,chp_file)!=local_nRows)
		LogError(ALL_POS,"Failed writing to file '%s'",fname);
	if (fwrite(rvec,sizeof(doublecomplex),local_nRows,chp_file)!=local_nRows)
		LogError(ALL_POS,"Failed writing to file '%s'",fname);
	if (fwrite(pvec,sizeof(doublecomplex),local_nRows,chp_file)!=local_nRows)
		LogError(ALL_POS,"Failed writing to file '%s'",fname);
	// write specific vectors
	for (i=0;i<params[ind_m].vec_N;i++) if (fwrite(vectors[i].ptr,vectors[i].size,local_nRows,chp_file)!=local_nRows)
		LogError(ALL_POS,"Failed writing to file '%s'",fname);
	// close file
	FCloseErr(chp_file,fname,ALL_POS);
	// write info to logfile after everyone is finished
	Synchronize();
	if (IFROOT) PrintBoth(logfile,"Checkpoint (iteration) saved\n");
	Timing_FileIO+=GET_TIME()-tstart;
	Synchronize(); // this is to ensure that message above appears if and only if OK
}

//======================================================================================================================

static void LoadIterChpoint(void)
/* load a binary checkpoint; only limitedly foolproof - user should take care to load checkpoints on the same machine
 * (number of processors) and with the same command line.
 */
{
	int i;
	int ind_m_new;
	size_t local_nRows_new;
	char fname[MAX_FNAME],ch;
	FILE * restrict chp_file;
	TIME_TYPE tstart;

	tstart=GET_TIME();
	// open input file
	SnprintfErr(ALL_POS,fname,MAX_FNAME,"%s/"F_CHP,chp_dir,ringid);
	chp_file=FOpenErr(fname,"rb",ALL_POS);
	/* check for consistency. This implies that the same index corresponds to the same iterative solver in list params.
	 * So if the ADDA executable was changed, e.g. by adding a new iterative solver, between writing and reading
	 * checkpoint, this test may fail.
	 */
	if (fread(&ind_m_new,sizeof(int),1,chp_file)!=1)
		LogError(ALL_POS,"Failed reading from file '%s'",fname);
	if (ind_m_new!=ind_m) LogError(ALL_POS,"File '%s' is for different iterative method",fname);
	if (fread(&local_nRows_new,sizeof(size_t),1,chp_file)!=1)
		LogError(ALL_POS,"Failed reading from file '%s'",fname);
	if (local_nRows_new!=local_nRows) LogError(ALL_POS,"File '%s' is for different vector size",fname);
	// read common scalars
	if (fread(&niter,sizeof(int),1,chp_file)!=1) LogError(ALL_POS,"Failed reading from file '%s'",fname);
	if (fread(&counter,sizeof(int),1,chp_file)!=1) LogError(ALL_POS,"Failed reading from file '%s'",fname);
	if (fread(&inprodR,sizeof(double),1,chp_file)!=1) LogError(ALL_POS,"Failed reading from file '%s'",fname);
	// read on ALL processors but used only on root
	if (fread(&prev_err,sizeof(double),1,chp_file)!=1) LogError(ALL_POS,"Failed reading from file '%s'",fname);
	if (fread(&resid_scale,sizeof(double),1,chp_file)!=1) LogError(ALL_POS,"Failed reading from file '%s'",fname);
	// read specific scalars
	for (i=0;i<params[ind_m].sc_N;i++) if (fread(scalars[i].ptr,scalars[i].size,1,chp_file)!=1)
		LogError(ALL_POS,"Failed reading from file '%s'",fname);
	// read common vectors
	if (fread(xvec,sizeof(doublecomplex),local_nRows,chp_file)!=local_nRows)
		LogError(ALL_POS,"Failed reading from file '%s'",fname);
	if (fread(rvec,sizeof(doublecomplex),local_nRows,chp_file)!=local_nRows)
		LogError(ALL_POS,"Failed reading from file '%s'",fname);
	if (fread(pvec,sizeof(doublecomplex),local_nRows,chp_file)!=local_nRows)
		LogError(ALL_POS,"Failed reading from file '%s'",fname);
	// read specific vectors
	for (i=0;i<params[ind_m].vec_N;i++) if (fread(vectors[i].ptr,vectors[i].size,local_nRows,chp_file)!=local_nRows)
		LogError(ALL_POS,"Failed reading from file '%s'",fname);
	// check if EOF reached and close file
	if (fread(&ch,1,1,chp_file)!=0) LogError(ALL_POS,"File '%s' is too long",fname);
	FCloseErr(chp_file,fname,ALL_POS);
	// initialize auxiliary variables
	epsB=iter_eps*iter_eps/resid_scale;
	// print info
	if (IFROOT) {
		PrintBoth(logfile,"Checkpoint (iteration) loaded\n");
		// if residual is stagnating print info about last minimum
		if (counter!=0) fprintf(logfile,"Residual has been stagnating already for %d iterations since:\n"
			RESID_STRING"\n...\n",counter,niter-counter-1,sqrt(resid_scale*inprodR));
	}
	Timing_FileIO+=GET_TIME()-tstart;
}

//======================================================================================================================

static void ProgressReport(void)
// Do common procedures; show progress in logfile and stdout; also check for checkpoint condition
{
	double err,progr,elapsed;
	char progr_string[MAX_LINE];
	const char *temp;
	time_t wt;

	if (inprodRp1<=inprodR) {
		inprodR=inprodRp1;
		counter=0;
	}
	else counter++;
	if (IFROOT) {
		err=sqrt(resid_scale*inprodRp1);
		progr=1-err/prev_err;
		if (counter==0) temp="+ ";
		else if (progr>0) temp="-+";
		else temp="- ";
		SnprintfErr(ONE_POS,progr_string,MAX_LINE,RESID_STRING"  %s",niter,err,temp);
		if (!orient_avg) fprintf(logfile,"%s  progress ="FFORM_PROG"\n",progr_string,progr);
		PRINTFB("%s\n",progr_string);
		prev_err=err;
	}
	niter++;
	TotalIter++;
	// check condition for checkpoint; checkpoint is saved at first time
	if (chp_type!=CHP_NONE && chp_time!=UNDEF && complete) {
		time(&wt);
		elapsed=difftime(wt,last_chp_wt);
		if (chp_time<elapsed) {
			SaveIterChpoint();
			time(&last_chp_wt);
			if (chp_type!=CHP_REGULAR) chp_exit=true;
		}
	}
}

//======================================================================================================================

static double ResidualNorm2(doublecomplex * restrict x,doublecomplex * restrict r,doublecomplex * restrict buffer,
	TIME_TYPE *mvp_timing,TIME_TYPE *mvp_comm_timing,TIME_TYPE *comm_timing)
/* Computes ||Ax-b||^2, where b=sqrt(C).Einc; buffer is used for Ax, r contains Ax-b at the end; comm_timing is
 * incremented with communication time. If only the norm is required, the calculation can be done without using vector
 * r, but this does not make a lot of sense, since memory is allocated anyway.
 */
{
	double res;

	TIME_TYPE mc_time=0;
	MatVec(x,buffer,NULL,false,mvp_timing,&mc_time);
	(*mvp_comm_timing) += mc_time;
	(*comm_timing) += mc_time;
	nMult_mat(r,Einc,cc_sqrt);
	nDecrem(r,buffer,&res,comm_timing);
	return res;
}

//======================================================================================================================

/* Large-vector operations inside all iterative solvers stay in C source. In
 * the CUDA executable they dispatch to resident GPU vectors; the normal build
 * continues to call the original linalg.c/MatVec routines.
 *
 * LANIER_FULL integration:
 *  - general Krylov methods use the exact right-preconditioned operator A*P;
 *  - CGNR additionally uses (A*P)^H=P^H*A^H;
 *  - complex-symmetric methods use the congruence P*A*P so their required
 *    transpose symmetry is preserved. Their residual is converted to P*r0
 *    once before PHASE_INIT.
 * In every case xvec remains the physical ADDA solution because every solver
 * update x += alpha*d is mapped to x += alpha*P*d. */
#ifdef ADDA_CUDA
static bool lanier_cs_congruence=false;

static bool LanierComplexSymmetricMethod(const enum iter method)
{
    return method==IT_BICG_CS || method==IT_CSYM ||
           method==IT_QMR_CS || method==IT_QMR_CS_2;
}

static bool LanierSchwarzSupportedMethod(const enum iter method)
{
    /* V1 intentionally supports only methods that need A*P, not P^H*A^H or
     * complex-symmetric congruence. */
    return method==IT_BCGS2 || method==IT_BICGSTAB || method==IT_BICGSTAB4 ||
           method==IT_GPBICGSTAB2 || method==IT_GPBICGSTAB4;
}

static void KrylovMatVec(doublecomplex * restrict in,doublecomplex * restrict out,double *inprod,
                         const bool her,TIME_TYPE *timing,TIME_TYPE *comm_timing)
{
    if (!lanier_precon) MatVec_GPU(in,out,inprod,her,timing,comm_timing);
    else if (lanier_cs_congruence) {
        if (her) LogError(ONE_POS,"Internal error: Hermitian MatVec requested for Lanier complex-symmetric congruence mode");
        CudaLanierMatVecCongruence(in,out,inprod,timing,comm_timing);
    }
    else CudaLanierMatVec(in,out,inprod,her,timing,comm_timing);
}

static void KrylovXIncrem(const doublecomplex * restrict direction)
{
    if (lanier_precon) CudaLanierAxpy(xvec,direction,1.0);
    else CudaIterIncrem(xvec,direction,NULL,NULL);
}

static void KrylovXIncrem01(const doublecomplex * restrict direction,const double alpha)
{
    if (lanier_precon) CudaLanierAxpy(xvec,direction,alpha);
    else CudaIterIncrem01(xvec,direction,alpha,NULL,NULL);
}

static void KrylovXIncrem01Cmplx(const doublecomplex * restrict direction,const itercomplex alpha)
{
    if (lanier_precon) CudaLanierAxpy(xvec,direction,(double complex)alpha);
    else CudaIterIncrem01_cmplx(xvec,direction,(double complex)alpha,NULL,NULL);
}

static void KrylovXIncrem011Cmplx(const doublecomplex * restrict d1,const doublecomplex * restrict d2,
                                  const itercomplex a1,const itercomplex a2)
{
    if (lanier_precon) {
        CudaLanierAxpy(xvec,d1,(double complex)a1);
        CudaLanierAxpy(xvec,d2,(double complex)a2);
    }
    else CudaIterIncrem011_cmplx(xvec,d1,d2,(double complex)a1,(double complex)a2);
}

# define IT_MATVEC                 KrylovMatVec
# define BICGCS_MATVEC             KrylovMatVec
# define IT_COPY                   CudaIterCopy
# define IT_NORM2                  CudaIterNorm2
# define IT_DOT                    CudaIterDotProd
# define IT_DOTU                   CudaIterDotProd_conj
# define IT_DOT_SELF               CudaIterDotProdSelf_conj
# define IT_DOT_SELF_NORM2         CudaIterDotProdSelf_conj_Norm2
# define IT_MULT                   CudaIterMult
# define IT_MULT_CMPLX             CudaIterMult_cmplx
# define IT_MULT_SELF              CudaIterMultSelf
# define IT_MULT_SELF_CONJ         CudaIterMultSelf_conj
# define IT_MULT_SELF_CMPLX        CudaIterMultSelf_cmplx
# define IT_INCREM                 CudaIterIncrem
# define IT_INCREM01               CudaIterIncrem01
# define IT_INCREM10               CudaIterIncrem10
# define IT_INCREM01_CMPLX         CudaIterIncrem01_cmplx
# define IT_INCREM10_CMPLX         CudaIterIncrem10_cmplx
# define IT_INCREM011_CMPLX        CudaIterIncrem011_cmplx
# define IT_INCREM110_CMPLX        CudaIterIncrem110_cmplx
# define IT_INCREM111_CMPLX        CudaIterIncrem111_cmplx
# define IT_INCREM11_D_C           CudaIterIncrem11_d_c
# define IT_INCREM110_D_C_CONJ     CudaIterIncrem110_d_c_conj
# define IT_LINCOMB_CMPLX          CudaIterLinComb_cmplx
# define IT_LINCOMB1_CMPLX         CudaIterLinComb1_cmplx
# define IT_LINCOMB1_CMPLX_CONJ    CudaIterLinComb1_cmplx_conj
# define IT_X_INCREM(d)            KrylovXIncrem((d))
# define IT_X_INCREM01(d,a)        KrylovXIncrem01((d),(a))
# define IT_X_INCREM01_CMPLX(d,a)  KrylovXIncrem01Cmplx((d),(a))
# define IT_X_INCREM011_CMPLX(d1,d2,a1,a2) KrylovXIncrem011Cmplx((d1),(d2),(a1),(a2))
#else
# define IT_MATVEC                 MatVec
# define BICGCS_MATVEC             MatVec_wrapper
# define IT_COPY                   nCopy
# define IT_NORM2                  nNorm2
# define IT_DOT                    nDotProd
# define IT_DOTU                   nDotProd_conj
# define IT_DOT_SELF               nDotProdSelf_conj
# define IT_DOT_SELF_NORM2         nDotProdSelf_conj_Norm2
# define IT_MULT                   nMult
# define IT_MULT_CMPLX             nMult_cmplx
# define IT_MULT_SELF              nMultSelf
# define IT_MULT_SELF_CONJ         nMultSelf_conj
# define IT_MULT_SELF_CMPLX        nMultSelf_cmplx
# define IT_INCREM                 nIncrem
# define IT_INCREM01               nIncrem01
# define IT_INCREM10               nIncrem10
# define IT_INCREM01_CMPLX         nIncrem01_cmplx
# define IT_INCREM10_CMPLX         nIncrem10_cmplx
# define IT_INCREM011_CMPLX        nIncrem011_cmplx
# define IT_INCREM110_CMPLX        nIncrem110_cmplx
# define IT_INCREM111_CMPLX        nIncrem111_cmplx
# define IT_INCREM11_D_C           nIncrem11_d_c
# define IT_INCREM110_D_C_CONJ     nIncrem110_d_c_conj
# define IT_LINCOMB_CMPLX          nLinComb_cmplx
# define IT_LINCOMB1_CMPLX         nLinComb1_cmplx
# define IT_LINCOMB1_CMPLX_CONJ    nLinComb1_cmplx_conj
# define IT_X_INCREM(d)            nIncrem(xvec,(d),NULL,NULL)
# define IT_X_INCREM01(d,a)        nIncrem01(xvec,(d),(a),NULL,NULL)
# define IT_X_INCREM01_CMPLX(d,a)  nIncrem01_cmplx(xvec,(d),(a),NULL,NULL)
# define IT_X_INCREM011_CMPLX(d1,d2,a1,a2) nIncrem011_cmplx(xvec,(d1),(d2),(a1),(a2))
#endif

/* Right-preconditioned BCGS2 helpers.  r and all recurrence vectors remain in
 * the physical residual space.  Only operator inputs and solution increments
 * are mapped through M^-1.  Therefore xvec remains the physical ADDA solution,
 * so checkpoints and true-residual recomputation keep their original meaning. */
static void BCGS2Operator(doublecomplex * restrict in,doublecomplex * restrict out,double *inprod,
                          TIME_TYPE *timing,TIME_TYPE *comm_timing)
{
    IT_MATVEC(in,out,inprod,false,timing,comm_timing);
}

static void BCGS2UpdateX(const doublecomplex * restrict direction,const itercomplex alpha)
{
    IT_X_INCREM01_CMPLX(direction,alpha);
}

#ifdef ADDA_CUDA
#define RELIABLE_GAP_TOL 1E-3
#define RECALC_RELIABLE_GAP_TOL 1E-2
#define RECALC_RELIABLE_PERIOD 20
#define LANIER_RELIABLE_GAP_TOL 1E-2
#define LANIER_RELIABLE_PERIOD 20
#define LANIER_RESET_RATIO 100.0

typedef struct
{
	double physical_true_norm2;
	double recurrence_true_norm2;
	double gap_norm2;
	double gap;
	double physical_rel;
	bool restart;
	bool forced_restart;
	bool true_converged;
	bool false_convergence;
} reliable_residual_result;

static const char *LanierMethodName(const enum iter method)
{
	switch (method) {
		case IT_BCGS2: return "BCGS2";
		case IT_BICG_CS: return "BiCG-CS";
		case IT_BICGSTAB: return "BiCGStab";
		case IT_BICGSTAB4: return "BiCGStab(4)";
		case IT_BICGSTAB8: return "BiCGStab(8)";
		case IT_BICGSTAB12: return "BiCGStab(12)";
		case IT_CGNR: return "CGNR";
		case IT_CSYM: return "CSYM";
		case IT_GPBICGSTAB2: return "GPBiCGStab(2)";
		case IT_GPBICGSTAB4: return "GPBiCGStab(4)";
		case IT_QMR_CS: return "QMR-CS";
		case IT_QMR_CS_2: return "QMR2-CS";
		default: return "unknown";
	}
}

static double PhysicalRHSNorm2(void)
/* ||sqrt(C) Einc||^2 evaluated in double accumulation. This remains the
 * physical normalization even when a complex-symmetric solver works on P*A*P. */
{
	TIME_TYPE host_comm=0;
	register size_t i,k;
	double sum=0;
	LARGE_LOOP;
	for (i=0,k=0;i<local_nvoid_Ndip;i++,k+=3) {
		const doublecomplex * restrict val=cc_sqrt[material[i]];
		int j;
		for (j=0;j<3;j++) {
			const double complex b=(double complex)val[j]*(double complex)Einc[k+j];
			sum+=creal(b)*creal(b)+cimag(b)*cimag(b);
		}
	}
	MyInnerProduct(&sum,double_type,1,&host_comm);
	Timing_InitIterComm+=host_comm;
	return sum;
}

static reliable_residual_result ReliableResidualCheck(const enum iter method,const double recursive_norm2,
	const double gap_threshold,const bool force_restart)
/* Independently recompute the physical residual r_true=b-A*x.
 *
 * For normal right-preconditioned solvers, the Krylov residual and physical
 * residual live in the same space. For complex-symmetric congruence solvers,
 * the Krylov recurrence lives in P*r, so the independently recomputed physical
 * residual is additionally mapped through P before evaluating the vector gap.
 * Convergence itself is always checked with the physical residual.
 *
 * lanier_recursive_residual_vec normally points to rvec. CSYM, which does not
 * explicitly retain r_k, reconstructs r_k=tau*conj(q_{k+1}) in Avecbuffer at
 * the end of each iteration and publishes that vector through this pointer. */
{
	reliable_residual_result result={0,0,0,0,0,false,false,false,false};
	TIME_TYPE host_comm=0;
	register size_t i,k;
	double true_sum=0,gap_sum=0,rec_true_sum=0;
	doublecomplex *rec_host;
	doublecomplex *rec_vec=(lanier_recursive_residual_vec ? lanier_recursive_residual_vec : rvec);

	rec_host=(doublecomplex *)malloc(local_nRows*sizeof(doublecomplex));
	if (rec_host==NULL) LogError(ALL_POS,"Failed allocating temporary reliable-residual host vector");
	CudaIterDownloadOne(rec_vec);
	memcpy(rec_host,rec_vec,local_nRows*sizeof(doublecomplex));

	/* Physical A*x, never the preconditioned operator. xvec is kept physical by
	 * the LANIER_FULL integration. */
	MatVec_GPU(xvec,Avecbuffer,NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm);
	CudaIterDownloadOne(Avecbuffer);

	/* Form the physical true residual in the host backing of Avecbuffer. */
	LARGE_LOOP;
	for (i=0,k=0;i<local_nvoid_Ndip;i++,k+=3) {
		const doublecomplex * restrict val=cc_sqrt[material[i]];
		int j;
		for (j=0;j<3;j++) {
			const double complex b=(double complex)val[j]*(double complex)Einc[k+j];
			const double complex ax=(double complex)Avecbuffer[k+j];
			const double complex rt=b-ax;
			Avecbuffer[k+j]=(doublecomplex)rt;
			true_sum+=creal(rt)*creal(rt)+cimag(rt)*cimag(rt);
		}
	}
	MyInnerProduct(&true_sum,double_type,1,&host_comm);
	Timing_OneIterComm+=host_comm;
	result.physical_true_norm2=true_sum;
	result.physical_rel=(lanier_physical_rhs_norm2>0 ? sqrt(MAX(0.0,true_sum/lanier_physical_rhs_norm2)) : HUGE_VAL);
	result.true_converged=(lanier_physical_rhs_norm2>0 && true_sum<=iter_eps*iter_eps*lanier_physical_rhs_norm2);

	/* Convert r_true to the residual space of the recurrence when necessary. */
	if (lanier_cs_congruence) {
		CudaIterUploadOne(Avecbuffer);
		CudaLanierApply(Avecbuffer,Avecbuffer);
		CudaIterDownloadOne(Avecbuffer);
		LARGE_LOOP;
		for (i=0;i<local_nRows;i++) {
			const double complex z=(double complex)Avecbuffer[i];
			rec_true_sum+=creal(z)*creal(z)+cimag(z)*cimag(z);
		}
		host_comm=0;
		MyInnerProduct(&rec_true_sum,double_type,1,&host_comm);
		Timing_OneIterComm+=host_comm;
		result.recurrence_true_norm2=rec_true_sum;
	}
	else result.recurrence_true_norm2=true_sum;

	/* Vector residual gap in the actual recurrence space. */
	LARGE_LOOP;
	for (i=0;i<local_nRows;i++) {
		const double complex rt=(double complex)Avecbuffer[i];
		const double complex rr=(double complex)rec_host[i];
		const double complex dg=rt-rr;
		gap_sum+=creal(dg)*creal(dg)+cimag(dg)*cimag(dg);
	}
	free(rec_host);
	host_comm=0;
	MyInnerProduct(&gap_sum,double_type,1,&host_comm);
	Timing_OneIterComm+=host_comm;
	result.gap_norm2=gap_sum;
	result.gap=(result.recurrence_true_norm2>0 ?
		sqrt(MAX(0.0,gap_sum/result.recurrence_true_norm2)) : (gap_sum>0 ? HUGE_VAL : 0));
	result.false_convergence=(recursive_norm2<=epsB && !result.true_converged);
	result.forced_restart=(force_restart && !result.true_converged);
	result.restart=(!result.true_converged &&
		(result.forced_restart || result.gap>=gap_threshold || result.false_convergence));

	if (result.restart) {
		/* Avecbuffer contains r_true in recurrence space (physical r_true for A*P,
		 * P*r_true for P*A*P). Materialize it as the new recursive residual. */
		memcpy(rvec,Avecbuffer,local_nRows*sizeof(doublecomplex));
		CudaIterUploadOne(rvec);
		matvec_ready=false;
	}
	return result;
}

static void RestartKrylovFromTrueResidual(const enum iter method,const double recurrence_norm2)
/* Rebuild only recurrence/history state. Keep xvec (the physical solution).
 * The independently recomputed residual has already been uploaded to rvec. */
{
	inprodR=recurrence_norm2;
	inprodRp1=recurrence_norm2;
	counter=0;
	complete=true;
	matvec_ready=false;
	lanier_hard_restart_request=true;
	if (method==IT_BCGS2) bcgs2_restart_request=true;
	if (method==IT_GPBICGSTAB2) gp2_restart_request=true;
	(*params[ind_m].func)(PHASE_INIT);
	bcgs2_restart_request=false;
	gp2_restart_request=false;
	lanier_hard_restart_request=false;
}
#endif

ITER_FUNC(BCGS2)
/* Enhanced Bi-CGStab(2) method.
 * Based on the code by M.A. Botchev and D.R. Fokkema - http://www.math.uu.nl/people/vorst/zbcg2.f90 and
 * D. R. Fokkema, "Enhanced implementation of BiCGstab(l) for solving linear systems of equations," Preprint 976,
 * Department of Mathematics, Utrecht University (1996).
 *
 * The original ADDA port removed the Botchev/Fokkema "reliable update part", since double-precision tests using
 * '-recalc_resid' showed it was almost never needed. For CUDA float32, ADDA now provides an external periodic
 * true-residual monitor/restart through '-reliable_resid'; it is deliberately kept outside this recurrence.
 *
 * For l=1, the method is equivalent to BiCGStab, rewritten through 2-term recurrences (as QMR2 is equivalent to QMR),
 * so we use l=2 here. In many cases one iteration of this method is similar to two iterations of BiCGStab, but overall
 * convergence is slightly better.
 * Breakdown tests were made to coincide with that for BiCGStab for l=1.
 *
 * !!! This iterative solver produces segmentation fault when compiled with icc 11.1. Probably that is related to
 * issue 146. But we leave it be (assume that this is a compiler bug). Even if someone uses this compiler, he can
 * live fine without this iterative solver.
 */
{
#define LL 2 // potentially the method will also work for l=1 (but memory allocation and freeing need to be adjusted)
#define EPS1 1E-10 // for 1/|beta|
#define EPS2 1E-10 // for |u_j+1.r~|/|r_j.r~|
	static doublecomplex * restrict r[LL+1],* restrict u[LL+1];
	static itercomplex matrix_z[LL+1][LL+1],y0[LL+1],yl[LL+1],zy0[LL+1],zyl[LL+1];
	static int i,j;
	static itercomplex alpha,beta,omega,rho0,rho1,sigma,varrho,hatgamma,temp1;
	static double kappa0,kappal,dtmp;
	static bool fresh_start;

	switch (ph) {
		case PHASE_VARS:
			/* rename some vectors; this doesn't contradict with 'restrict' keyword, since new names are not used
			 * together with old names
			 */
			r[0]=rvec;
			r[1]=vec1;
			u[0]=vec2;
			u[1]=Avecbuffer;
			if (LL==2) {
				r[2]=vec3;
				u[2]=vec4;
			}
			// initialize data structure for checkpoints
			scalars[0].ptr=&rho0;
			scalars[1].ptr=&alpha;
			scalars[2].ptr=&fresh_start;
			scalars[0].size=scalars[1].size=sizeof(itercomplex);
			scalars[2].size=sizeof(bool);
			vectors[0].ptr=vec2; // u[0]
			vectors[0].size=sizeof(doublecomplex);
#ifdef __INTEL_COMPILER // workaround for issue 286
			/* otherwise, the Intel compiler Classic (last tested for v. 2021.2) produces broken code with -O1 and
			 * higher, by ignoring some of the pointer assignments above r[1],r[2],u[1],u[2]. We were not able to solve
			 * the issue by dumb variable assignments
			 */
			fflush(stdout);
#endif
			return;
		case PHASE_INIT:
			if (!load_chpoint || bcgs2_restart_request) {
				IT_COPY(pvec,rvec); // (pvec = r~0) = r0
				rho0=-1;
				fresh_start=true;
			}
			return;
		case PHASE_ITER:
			// --- The BiCG part ---
			for (j=0;j<LL;j++) {
				rho1=IT_DOT(r[j],pvec,&Timing_OneIterComm); // rho1 = r_j.r~0
				// u_i = r_i - beta*u_i
				if (fresh_start && j==0) IT_COPY(u[0],r[0]);
				else {
					// test for zero rho0 (1/beta)
					dtmp=cabs(rho0)/(cabs(rho1)*cabs(alpha)); // assume that rho1 is not exactly zero
					Dz("1/|beta|="GFORM_DEBUG,dtmp);
					if (dtmp<EPS1) LogError(ONE_POS,"BCGS2 fails: 1/|beta| is too small ("GFORM_DEBUG").",dtmp);
					beta=alpha*rho1/rho0;
					// u_i = r_i - beta*u_i
					temp1=-beta;
					for (i=0;i<=j;i++) IT_INCREM10_CMPLX(u[i],r[i],temp1,NULL,NULL);
				}
				rho0=rho1;
				// u_j+1 = (A*M^-1).u_j when Lanier right preconditioning is active
				if (niter==1 && j==0 && matvec_ready && !lanier_precon) {} // cached vector contains physical A*r0 only
				else BCGS2Operator(u[j],u[j+1],NULL,&Timing_OneIterMVP,&Timing_OneIterMVPComm);
				sigma=IT_DOT(u[j+1],pvec,&Timing_OneIterComm); // sigma = u_j+1.r~0
				// test for zero sigma (1/alpha)
				dtmp=cabs(sigma)/cabs(rho1); // assume that rho1 is not exactly zero
				Dz("|u_%d.r~|/|r_%d.r~|="GFORM_DEBUG,j+1,j,dtmp);
				if (dtmp<EPS1)
					LogError(ONE_POS,"BCGS2 fails: |u_%d.r~|/|r_%d.r~| is too small ("GFORM_DEBUG").",j+1,j,dtmp);
				alpha = rho1/sigma;
				BCGS2UpdateX(u[0],alpha); // x = x + alpha*M^-1*u_0 for right preconditioning
				// r_i = r_i - alpha*u_i+1
				temp1=-alpha;
				for (i=0;i<=j;i++) IT_INCREM01_CMPLX(r[i],u[i+1],temp1,NULL,NULL);
				BCGS2Operator(r[j],r[j+1],NULL,&Timing_OneIterMVP,&Timing_OneIterMVPComm);
			}
			// --- The convex polynomial part ---
			// Z = R'R
			for(i=0;i<=LL;i++) for (j=0;j<=i;j++) {
				matrix_z[i][j]=IT_DOT(r[j],r[i],&Timing_OneIterComm);
				if (i!=j) matrix_z[j][i]=conj(matrix_z[i][j]);
			}
			// small vectors y0 and yl
			y0[0]=-1;
			if (LL==2) y0[1]=matrix_z[1][0]/matrix_z[1][1]; // works only for l<=2
			y0[LL]=0;
			yl[0]=0;
			if (LL==2) yl[1]=matrix_z[1][2]/matrix_z[1][1]; // works only for l<=2
			yl[LL]=-1;
			//  Convex combination
			// compute Zy0 and Zyl
			for (i=0;i<=LL;i++) {
				zy0[i]=zyl[i]=0;
				for (j=0;j<=LL;j++) {
					zy0[i]+=matrix_z[i][j]*y0[j];
					zyl[i]+=matrix_z[i][j]*yl[j];
				}
			}
			// kappa0 = sqrt(y0.Zy0); kappal = sqrt(yl.Zyl); employs that dot products are always real
			dtmp=0;
			for (i=0;i<=LL;i++) dtmp+=creal(zy0[i]*conj(y0[i]));
			kappa0=sqrt(dtmp);
			dtmp=0;
			for (i=0;i<=LL;i++) dtmp+=creal(zyl[i]*conj(yl[i]));
			kappal=sqrt(dtmp);
			// varrho = Zy0.yl/(kappa0*kappal)
			varrho=0;
			for (i=0;i<=LL;i++) varrho+=zy0[i]*conj(yl[i]);
			varrho/=kappa0*kappal;
			// hatgamma = varrho/abs(varrho) * max( abs(varrho),0.7)
			dtmp=cabs(varrho);
			hatgamma=varrho*MAX(dtmp,0.7)/dtmp;
			// y0 = y0 - (hatgamma*kappa0/kappal)*yl
			temp1=-hatgamma*kappa0/kappal;
			for (i=0;i<=LL;i++) y0[i]+=temp1*yl[i];
			// Update
			omega = y0[LL];
			for (i=1;i<=LL;i++) {
				temp1=-y0[i];
				IT_INCREM01_CMPLX(u[0],u[i],temp1,NULL,NULL);   // u_0 = u_0 - y0[i]*u_i
				BCGS2UpdateX(r[i-1],y0[i]); // x = x + y0[i]*M^-1*r_i-1
				IT_INCREM01_CMPLX(r[0],r[i],temp1,NULL,NULL);   // r_0 = r_0 - y0[i]*r_i
			}
			// y0 has changed; compute Zy0 once more
			for (i=0;i<=LL;i++) {
				zy0[i]=0;
				for (j=0;j<=LL;j++) zy0[i]+=matrix_z[i][j]*y0[j];
			}
			// |r|^2 = y0.Zy0
			inprodRp1=0;
			for (i=0;i<=LL;i++) inprodRp1+=creal(zy0[i]*conj(y0[i]));
			// rho0 = -omega*rho0; moved from the beginning of the iteration
			rho0*=-omega;
			fresh_start=false;
			return; // end of PHASE_ITER
	}
	LogError(ONE_POS,"Unknown phase (%d) of the iterative solver",(int)ph);
}
#undef LL
#undef EPS1
#undef EPS2

//======================================================================================================================

static bool GPSolveSmallN(double complex *a,double complex *b,const int n,const int ld)
/* Pivoted Gaussian elimination for the small dense residual-minimization
 * systems used by GPBiCGStab(L).  ld is the physical row stride of a. */
{
	int i,j,k,piv;
	double best;
	double complex tmp,factor;
#define GP_A(ii,jj) a[(ii)*ld+(jj)]
	for (k=0;k<n;k++) {
		piv=k;
		best=cabs(GP_A(k,k));
		for (i=k+1;i<n;i++) if (cabs(GP_A(i,k))>best) { best=cabs(GP_A(i,k)); piv=i; }
		if (best<1E-30) return false;
		if (piv!=k) {
			for (j=k;j<n;j++) { tmp=GP_A(k,j); GP_A(k,j)=GP_A(piv,j); GP_A(piv,j)=tmp; }
			tmp=b[k]; b[k]=b[piv]; b[piv]=tmp;
		}
		for (i=k+1;i<n;i++) {
			factor=GP_A(i,k)/GP_A(k,k);
			GP_A(i,k)=0;
			for (j=k+1;j<n;j++) GP_A(i,j)-=factor*GP_A(k,j);
			b[i]-=factor*b[k];
		}
	}
	for (i=n-1;i>=0;i--) {
		tmp=b[i];
		for (j=i+1;j<n;j++) tmp-=GP_A(i,j)*b[j];
		if (cabs(GP_A(i,i))<1E-30) return false;
		b[i]=tmp/GP_A(i,i);
	}
#undef GP_A
	return true;
}

static int GPSolveSmallNAdaptive(const double complex *a_src,const double complex *b_src,
	double complex *a_work,double complex *b_work,const int n,const int ld)
/* Rank-adaptive wrapper for the tiny MR normal equations used by L=4 methods.
 * A Krylov basis may become exactly/nearly rank deficient on an easy regional
 * LANIER_PARTITION subproblem. That is a happy/near breakdown, not a reason
 * to abort the whole run. Try the full degree first, then truncate the MR
 * polynomial to the largest nonsingular leading subspace. */
{
	int i,j,k;
	for (k=n;k>=1;k--) {
		for (i=0;i<n;i++) {
			b_work[i]=0;
			for (j=0;j<ld;j++) a_work[i*ld+j]=0;
		}
		for (i=0;i<k;i++) {
			b_work[i]=b_src[i];
			for (j=0;j<k;j++) a_work[i*ld+j]=a_src[i*ld+j];
		}
		if (GPSolveSmallN(a_work,b_work,k,ld)) return k;
	}
	return 0;
}

static double complex GPDot(const doublecomplex * restrict a,const doublecomplex * restrict b,TIME_TYPE *comm)
/* IFDDA convention <a,b> = sum(conj(a)*b). ADDA nDotProd(a,b) is
 * sum(a*conj(b)), hence the reversed operands.
 *
 * Precision policy:
 * - CUDA float32: CudaIterDotProd64() performs the chunked FP64 cuBLAS reduction.
 * - CPU float32: explicitly promote each complex value to double complex before
 *   multiplication and accumulation. This keeps the GPBiCGStab(2) recurrence
 *   scalars and its 2x2/3x3 minimizations in FP64 without changing the large
 *   CPU vectors or MatVec, which remain float32.
 * - CPU double: use the native ADDA dot product.
 * This routine is also used by the CPU-only L=4 solvers. */
{
#ifdef ADDA_CUDA
	return CudaIterDotProd64(b,a,comm);
#elif defined(ADDA_SINGLE)
	register size_t i;
	register const size_t n=local_nRows;
	double complex sum=0;
	LARGE_LOOP;
	for (i=0;i<n;i++) {
		const double complex ad=(double)crealf(a[i]) + I*(double)cimagf(a[i]);
		const double complex bd=(double)crealf(b[i]) + I*(double)cimagf(b[i]);
		sum+=conj(ad)*bd;
	}
	/* In sequential CPU builds this is a no-op. In MPI builds cmplx_type is
	 * represented by the historical double-complex reduction datatype. */
	MyInnerProduct(&sum,cmplx_type,1,comm);
	return sum;
#else
	return (double complex)IT_DOT(b,a,comm);
#endif
}

static doublecomplex *cpu_l4_workspace=NULL;
static size_t cpu_l4_workspace_rows=0,cpu_l4_workspace_vecs=0;

static doublecomplex *EnsureCPUL4Workspace(const size_t nvec,const char *name)
/* One contiguous CPU-only workspace is shared by the L=4 solvers. Only one
 * iterative method is active at a time, and it is released at the end of
 * IterativeSolver(), so no persistent CUDA ABI/global-vector changes are needed. */
{
	const size_t total=MultOverflow(nvec,local_nRows,ALL_POS,name);
	if (cpu_l4_workspace!=NULL && (cpu_l4_workspace_rows!=local_nRows || cpu_l4_workspace_vecs<nvec)) {
		Free_cVector(cpu_l4_workspace);
		cpu_l4_workspace=NULL;
		cpu_l4_workspace_rows=cpu_l4_workspace_vecs=0;
	}
	if (cpu_l4_workspace==NULL) {
		cpu_l4_workspace=complexVector(total,ALL_POS,name);
		cpu_l4_workspace_rows=local_nRows;
		cpu_l4_workspace_vecs=nvec;
	}
	return cpu_l4_workspace;
}

static void FreeCPUL4Workspace(void)
{
	if (cpu_l4_workspace!=NULL) Free_cVector(cpu_l4_workspace);
	cpu_l4_workspace=NULL;
	cpu_l4_workspace_rows=cpu_l4_workspace_vecs=0;
}

//======================================================================================================================

ITER_FUNC(BiCGStab4)
/* BiCGStab(L), L=4, CPU implementation.  The BiCG part follows the
 * Sleijpen/Fokkema BiCGStab(L) recurrence and the MR polynomial is formed by
 * modified Gram-Schmidt.  The large vectors stay in the native ADDA precision;
 * GPDot() promotes/accumulates scalar products in FP64 in adda_single.
 *
 * Workspace: the common rvec/u0=pvec plus 9 extra vectors:
 *   r~0, r1..r4, u1..u4.  Thus the complete method uses the theoretical
 *   2L+3=11 Krylov vectors including r0 and u0, without any double MatVec. */
{
#define BSL4_L 4
#define BSL4_EPS 1E-30
	static doublecomplex *r[BSL4_L+1],*u[BSL4_L+1],*rtilda;
	static double complex rho,alpha,omega;
	doublecomplex *base;
	double complex rho1,beta,den,temp;
	double complex tau[BSL4_L][BSL4_L],gamma_p[BSL4_L+1],gamma[BSL4_L+1],gamma_pp[BSL4_L+1];
	double sigma[BSL4_L+1];
	int i,j,mr_rank;

	switch (ph) {
		case PHASE_VARS:
			base=EnsureCPUL4Workspace(9,"BiCGStab(4) L=4 workspace");
			r[0]=rvec; u[0]=pvec; rtilda=base;
			for (i=1;i<=BSL4_L;i++) r[i]=base+(size_t)i*local_nRows;
			for (i=1;i<=BSL4_L;i++) u[i]=base+(size_t)(BSL4_L+i)*local_nRows;
			for (i=0;i<9;i++) { vectors[i].ptr=base+(size_t)i*local_nRows; vectors[i].size=sizeof(doublecomplex); }
			scalars[0].ptr=&rho; scalars[1].ptr=&alpha; scalars[2].ptr=&omega;
			scalars[0].size=scalars[1].size=scalars[2].size=sizeof(double complex);
			return;

		case PHASE_INIT:
			if (!load_chpoint || lanier_hard_restart_request) {
				double nrm;
				IT_COPY(rtilda,rvec);
				nrm=sqrt(MAX(0.0,creal(GPDot(rtilda,rtilda,&Timing_InitIterComm))));
				if (nrm<BSL4_EPS) LogError(ONE_POS,"BiCGStab(4) cannot start from a zero residual");
				IT_MULT_SELF(rtilda,1.0/nrm);
				IT_MULT_CMPLX(u[0],rvec,0);
				rho=1; alpha=0; omega=1;
			}
			return;

		case PHASE_ITER:
			rho=-omega*rho;
			for (j=0;j<BSL4_L;j++) {
				if (cabs(rho)<BSL4_EPS) LogError(ONE_POS,"BiCGStab(4) breakdown: rho is zero");
				rho1=GPDot(rtilda,r[j],&Timing_OneIterComm);
				beta=alpha*rho1/rho;
				rho=rho1;
				temp=-beta;
				for (i=0;i<=j;i++) IT_INCREM10_CMPLX(u[i],r[i],temp,NULL,NULL); /* u_i=r_i-beta*u_i */
#ifdef ADDA_CUDA
				/* Save one full CUDA vector: do not register Avecbuffer for this solver. */
				IT_MATVEC(u[j],u[j+1],NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm);
#else
				if (niter==1 && j==0 && matvec_ready) IT_COPY(u[1],Avecbuffer);
				else IT_MATVEC(u[j],u[j+1],NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm);
#endif
				den=GPDot(rtilda,u[j+1],&Timing_OneIterComm);
				if (cabs(den)<BSL4_EPS) LogError(ONE_POS,"BiCGStab(4) breakdown: <r~,u_%d> is zero",j+1);
				alpha=rho/den;
				temp=-alpha;
				for (i=0;i<=j;i++) IT_INCREM01_CMPLX(r[i],u[i+1],temp,NULL,NULL);
				IT_MATVEC(r[j],r[j+1],NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm);
				IT_X_INCREM01_CMPLX(u[0],alpha);
			}

			memset(tau,0,sizeof(tau));
			memset(gamma_p,0,sizeof(gamma_p));
			memset(gamma,0,sizeof(gamma));
			memset(gamma_pp,0,sizeof(gamma_pp));
			mr_rank=BSL4_L;
			for (j=1;j<=BSL4_L;j++) {
				for (i=1;i<j;i++) {
					tau[j-1][i-1]=GPDot(r[i],r[j],&Timing_OneIterComm)/sigma[i];
					temp=-tau[j-1][i-1];
					IT_INCREM01_CMPLX(r[j],r[i],temp,NULL,NULL);
				}
				sigma[j]=creal(GPDot(r[j],r[j],&Timing_OneIterComm));
				if (!(sigma[j]>BSL4_EPS)) { mr_rank=j-1; break; }
				gamma_p[j]=GPDot(r[j],r[0],&Timing_OneIterComm)/sigma[j];
			}
			if (mr_rank<BSL4_L && IFROOT)
				PrintBoth(logfile,"BiCGStab(4) MR happy/near breakdown: using degree %d instead of 4 at iteration %d.\n",mr_rank,niter);
			if (mr_rank==0) {
				/* No usable MR direction remains. Keep the valid BiCG update already
				 * accumulated in x/r0 and restart the L=4 recurrence next iteration. */
				inprodRp1=MAX(0.0,creal(GPDot(r[0],r[0],&Timing_OneIterComm)));
				if (inprodRp1>BSL4_EPS) {
					double nrm=sqrt(inprodRp1);
					IT_COPY(rtilda,r[0]); IT_MULT_SELF(rtilda,1.0/nrm);
					IT_MULT_CMPLX(u[0],r[0],0); rho=1; alpha=0; omega=1;
				}
				return;
			}
			omega=gamma[mr_rank]=gamma_p[mr_rank];
			for (j=mr_rank-1;j>=1;j--) {
				gamma[j]=gamma_p[j];
				for (i=j+1;i<=mr_rank;i++) gamma[j]-=tau[i-1][j-1]*gamma[i];
			}
			for (j=1;j<mr_rank;j++) {
				gamma_pp[j]=gamma[j+1];
				for (i=j+1;i<mr_rank;i++) gamma_pp[j]+=tau[i-1][j-1]*gamma[i+1];
			}
			IT_X_INCREM01_CMPLX(r[0],gamma[1]);
			temp=-gamma_p[mr_rank]; IT_INCREM01_CMPLX(r[0],r[mr_rank],temp,NULL,NULL);
			temp=-gamma[mr_rank]; IT_INCREM01_CMPLX(u[0],u[mr_rank],temp,NULL,NULL);
			for (j=1;j<mr_rank;j++) {
				IT_X_INCREM01_CMPLX(r[j],gamma_pp[j]);
				temp=-gamma_p[j]; IT_INCREM01_CMPLX(r[0],r[j],temp,NULL,NULL);
				temp=-gamma[j]; IT_INCREM01_CMPLX(u[0],u[j],temp,NULL,NULL);
			}
			inprodRp1=MAX(0.0,creal(GPDot(r[0],r[0],&Timing_OneIterComm)));
			return;
	}
	LogError(ONE_POS,"Unknown phase (%d) of BiCGStab(4)",(int)ph);
#undef BSL4_L
#undef BSL4_EPS
}

ITER_FUNC(BiCGStab8)
/* BiCGStab(L), L=8, CPU implementation.  The BiCG part follows the
 * Sleijpen/Fokkema BiCGStab(L) recurrence and the MR polynomial is formed by
 * modified Gram-Schmidt.  The large vectors stay in the native ADDA precision;
 * GPDot() promotes/accumulates scalar products in FP64 in adda_single.
 *
 * Workspace: the common rvec/u0=pvec plus 17 extra vectors:
 *   r~0, r1..r8, u1..u8.  Thus the complete method uses the theoretical
 *   2L+3=19 Krylov vectors including r0 and u0, without any double MatVec. */
{
#define BSL8_L 8
#define BSL8_EPS 1E-30
	static doublecomplex *r[BSL8_L+1],*u[BSL8_L+1],*rtilda;
	static double complex rho,alpha,omega;
	doublecomplex *base;
	double complex rho1,beta,den,temp;
	double complex tau[BSL8_L][BSL8_L],gamma_p[BSL8_L+1],gamma[BSL8_L+1],gamma_pp[BSL8_L+1];
	double sigma[BSL8_L+1];
	int i,j,mr_rank;

	switch (ph) {
		case PHASE_VARS:
			base=EnsureCPUL4Workspace(17,"BiCGStab(8) L=8 workspace");
			r[0]=rvec; u[0]=pvec; rtilda=base;
			for (i=1;i<=BSL8_L;i++) r[i]=base+(size_t)i*local_nRows;
			for (i=1;i<=BSL8_L;i++) u[i]=base+(size_t)(BSL8_L+i)*local_nRows;
			for (i=0;i<17;i++) { vectors[i].ptr=base+(size_t)i*local_nRows; vectors[i].size=sizeof(doublecomplex); }
			scalars[0].ptr=&rho; scalars[1].ptr=&alpha; scalars[2].ptr=&omega;
			scalars[0].size=scalars[1].size=scalars[2].size=sizeof(double complex);
			return;

		case PHASE_INIT:
			if (!load_chpoint || lanier_hard_restart_request) {
				double nrm;
				IT_COPY(rtilda,rvec);
				nrm=sqrt(MAX(0.0,creal(GPDot(rtilda,rtilda,&Timing_InitIterComm))));
				if (nrm<BSL8_EPS) LogError(ONE_POS,"BiCGStab(8) cannot start from a zero residual");
				IT_MULT_SELF(rtilda,1.0/nrm);
				IT_MULT_CMPLX(u[0],rvec,0);
				rho=1; alpha=0; omega=1;
			}
			return;

		case PHASE_ITER:
			rho=-omega*rho;
			for (j=0;j<BSL8_L;j++) {
				if (cabs(rho)<BSL8_EPS) LogError(ONE_POS,"BiCGStab(8) breakdown: rho is zero");
				rho1=GPDot(rtilda,r[j],&Timing_OneIterComm);
				beta=alpha*rho1/rho;
				rho=rho1;
				temp=-beta;
				for (i=0;i<=j;i++) IT_INCREM10_CMPLX(u[i],r[i],temp,NULL,NULL); /* u_i=r_i-beta*u_i */
#ifdef ADDA_CUDA
				/* Save one full CUDA vector: do not register Avecbuffer for this solver. */
				IT_MATVEC(u[j],u[j+1],NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm);
#else
				if (niter==1 && j==0 && matvec_ready) IT_COPY(u[1],Avecbuffer);
				else IT_MATVEC(u[j],u[j+1],NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm);
#endif
				den=GPDot(rtilda,u[j+1],&Timing_OneIterComm);
				if (cabs(den)<BSL8_EPS) LogError(ONE_POS,"BiCGStab(8) breakdown: <r~,u_%d> is zero",j+1);
				alpha=rho/den;
				temp=-alpha;
				for (i=0;i<=j;i++) IT_INCREM01_CMPLX(r[i],u[i+1],temp,NULL,NULL);
				IT_MATVEC(r[j],r[j+1],NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm);
				IT_X_INCREM01_CMPLX(u[0],alpha);
			}

			memset(tau,0,sizeof(tau));
			memset(gamma_p,0,sizeof(gamma_p));
			memset(gamma,0,sizeof(gamma));
			memset(gamma_pp,0,sizeof(gamma_pp));
			mr_rank=BSL8_L;
			for (j=1;j<=BSL8_L;j++) {
				for (i=1;i<j;i++) {
					tau[j-1][i-1]=GPDot(r[i],r[j],&Timing_OneIterComm)/sigma[i];
					temp=-tau[j-1][i-1];
					IT_INCREM01_CMPLX(r[j],r[i],temp,NULL,NULL);
				}
				sigma[j]=creal(GPDot(r[j],r[j],&Timing_OneIterComm));
				if (!(sigma[j]>BSL8_EPS)) { mr_rank=j-1; break; }
				gamma_p[j]=GPDot(r[j],r[0],&Timing_OneIterComm)/sigma[j];
			}
			if (mr_rank<BSL8_L && IFROOT)
				PrintBoth(logfile,"BiCGStab(8) MR happy/near breakdown: using degree %d instead of 8 at iteration %d.\n",mr_rank,niter);
			if (mr_rank==0) {
				/* No usable MR direction remains. Keep the valid BiCG update already
				 * accumulated in x/r0 and restart the L=8 recurrence next iteration. */
				inprodRp1=MAX(0.0,creal(GPDot(r[0],r[0],&Timing_OneIterComm)));
				if (inprodRp1>BSL8_EPS) {
					double nrm=sqrt(inprodRp1);
					IT_COPY(rtilda,r[0]); IT_MULT_SELF(rtilda,1.0/nrm);
					IT_MULT_CMPLX(u[0],r[0],0); rho=1; alpha=0; omega=1;
				}
				return;
			}
			omega=gamma[mr_rank]=gamma_p[mr_rank];
			for (j=mr_rank-1;j>=1;j--) {
				gamma[j]=gamma_p[j];
				for (i=j+1;i<=mr_rank;i++) gamma[j]-=tau[i-1][j-1]*gamma[i];
			}
			for (j=1;j<mr_rank;j++) {
				gamma_pp[j]=gamma[j+1];
				for (i=j+1;i<mr_rank;i++) gamma_pp[j]+=tau[i-1][j-1]*gamma[i+1];
			}
			IT_X_INCREM01_CMPLX(r[0],gamma[1]);
			temp=-gamma_p[mr_rank]; IT_INCREM01_CMPLX(r[0],r[mr_rank],temp,NULL,NULL);
			temp=-gamma[mr_rank]; IT_INCREM01_CMPLX(u[0],u[mr_rank],temp,NULL,NULL);
			for (j=1;j<mr_rank;j++) {
				IT_X_INCREM01_CMPLX(r[j],gamma_pp[j]);
				temp=-gamma_p[j]; IT_INCREM01_CMPLX(r[0],r[j],temp,NULL,NULL);
				temp=-gamma[j]; IT_INCREM01_CMPLX(u[0],u[j],temp,NULL,NULL);
			}
			inprodRp1=MAX(0.0,creal(GPDot(r[0],r[0],&Timing_OneIterComm)));
			return;
	}
	LogError(ONE_POS,"Unknown phase (%d) of BiCGStab(8)",(int)ph);
#undef BSL8_L
#undef BSL8_EPS
}

ITER_FUNC(BiCGStab12)
/* BiCGStab(L), L=12, CPU implementation.  The BiCG part follows the
 * Sleijpen/Fokkema BiCGStab(L) recurrence and the MR polynomial is formed by
 * modified Gram-Schmidt.  The large vectors stay in the native ADDA precision;
 * GPDot() promotes/accumulates scalar products in FP64 in adda_single.
 *
 * Workspace: the common rvec/u0=pvec plus 25 extra vectors:
 *   r~0, r1..r12, u1..u12.  Thus the complete method uses the theoretical
 *   2L+3=27 Krylov vectors including r0 and u0, without any double MatVec. */
{
#define BSL12_L 12
#define BSL12_EPS 1E-30
	static doublecomplex *r[BSL12_L+1],*u[BSL12_L+1],*rtilda;
	static double complex rho,alpha,omega;
	doublecomplex *base;
	double complex rho1,beta,den,temp;
	double complex tau[BSL12_L][BSL12_L],gamma_p[BSL12_L+1],gamma[BSL12_L+1],gamma_pp[BSL12_L+1];
	double sigma[BSL12_L+1];
	int i,j,mr_rank;

	switch (ph) {
		case PHASE_VARS:
			base=EnsureCPUL4Workspace(25,"BiCGStab(12) L=12 workspace");
			r[0]=rvec; u[0]=pvec; rtilda=base;
			for (i=1;i<=BSL12_L;i++) r[i]=base+(size_t)i*local_nRows;
			for (i=1;i<=BSL12_L;i++) u[i]=base+(size_t)(BSL12_L+i)*local_nRows;
			for (i=0;i<25;i++) { vectors[i].ptr=base+(size_t)i*local_nRows; vectors[i].size=sizeof(doublecomplex); }
			scalars[0].ptr=&rho; scalars[1].ptr=&alpha; scalars[2].ptr=&omega;
			scalars[0].size=scalars[1].size=scalars[2].size=sizeof(double complex);
			return;

		case PHASE_INIT:
			if (!load_chpoint || lanier_hard_restart_request) {
				double nrm;
				IT_COPY(rtilda,rvec);
				nrm=sqrt(MAX(0.0,creal(GPDot(rtilda,rtilda,&Timing_InitIterComm))));
				if (nrm<BSL12_EPS) LogError(ONE_POS,"BiCGStab(12) cannot start from a zero residual");
				IT_MULT_SELF(rtilda,1.0/nrm);
				IT_MULT_CMPLX(u[0],rvec,0);
				rho=1; alpha=0; omega=1;
			}
			return;

		case PHASE_ITER:
			rho=-omega*rho;
			for (j=0;j<BSL12_L;j++) {
				if (cabs(rho)<BSL12_EPS) LogError(ONE_POS,"BiCGStab(12) breakdown: rho is zero");
				rho1=GPDot(rtilda,r[j],&Timing_OneIterComm);
				beta=alpha*rho1/rho;
				rho=rho1;
				temp=-beta;
				for (i=0;i<=j;i++) IT_INCREM10_CMPLX(u[i],r[i],temp,NULL,NULL); /* u_i=r_i-beta*u_i */
#ifdef ADDA_CUDA
				/* Save one full CUDA vector: do not register Avecbuffer for this solver. */
				IT_MATVEC(u[j],u[j+1],NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm);
#else
				if (niter==1 && j==0 && matvec_ready) IT_COPY(u[1],Avecbuffer);
				else IT_MATVEC(u[j],u[j+1],NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm);
#endif
				den=GPDot(rtilda,u[j+1],&Timing_OneIterComm);
				if (cabs(den)<BSL12_EPS) LogError(ONE_POS,"BiCGStab(12) breakdown: <r~,u_%d> is zero",j+1);
				alpha=rho/den;
				temp=-alpha;
				for (i=0;i<=j;i++) IT_INCREM01_CMPLX(r[i],u[i+1],temp,NULL,NULL);
				IT_MATVEC(r[j],r[j+1],NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm);
				IT_X_INCREM01_CMPLX(u[0],alpha);
			}

			memset(tau,0,sizeof(tau));
			memset(gamma_p,0,sizeof(gamma_p));
			memset(gamma,0,sizeof(gamma));
			memset(gamma_pp,0,sizeof(gamma_pp));
			mr_rank=BSL12_L;
			for (j=1;j<=BSL12_L;j++) {
				for (i=1;i<j;i++) {
					tau[j-1][i-1]=GPDot(r[i],r[j],&Timing_OneIterComm)/sigma[i];
					temp=-tau[j-1][i-1];
					IT_INCREM01_CMPLX(r[j],r[i],temp,NULL,NULL);
				}
				sigma[j]=creal(GPDot(r[j],r[j],&Timing_OneIterComm));
				if (!(sigma[j]>BSL12_EPS)) { mr_rank=j-1; break; }
				gamma_p[j]=GPDot(r[j],r[0],&Timing_OneIterComm)/sigma[j];
			}
			if (mr_rank<BSL12_L && IFROOT)
				PrintBoth(logfile,"BiCGStab(12) MR happy/near breakdown: using degree %d instead of 12 at iteration %d.\n",mr_rank,niter);
			if (mr_rank==0) {
				/* No usable MR direction remains. Keep the valid BiCG update already
				 * accumulated in x/r0 and restart the L=12 recurrence next iteration. */
				inprodRp1=MAX(0.0,creal(GPDot(r[0],r[0],&Timing_OneIterComm)));
				if (inprodRp1>BSL12_EPS) {
					double nrm=sqrt(inprodRp1);
					IT_COPY(rtilda,r[0]); IT_MULT_SELF(rtilda,1.0/nrm);
					IT_MULT_CMPLX(u[0],r[0],0); rho=1; alpha=0; omega=1;
				}
				return;
			}
			omega=gamma[mr_rank]=gamma_p[mr_rank];
			for (j=mr_rank-1;j>=1;j--) {
				gamma[j]=gamma_p[j];
				for (i=j+1;i<=mr_rank;i++) gamma[j]-=tau[i-1][j-1]*gamma[i];
			}
			for (j=1;j<mr_rank;j++) {
				gamma_pp[j]=gamma[j+1];
				for (i=j+1;i<mr_rank;i++) gamma_pp[j]+=tau[i-1][j-1]*gamma[i+1];
			}
			IT_X_INCREM01_CMPLX(r[0],gamma[1]);
			temp=-gamma_p[mr_rank]; IT_INCREM01_CMPLX(r[0],r[mr_rank],temp,NULL,NULL);
			temp=-gamma[mr_rank]; IT_INCREM01_CMPLX(u[0],u[mr_rank],temp,NULL,NULL);
			for (j=1;j<mr_rank;j++) {
				IT_X_INCREM01_CMPLX(r[j],gamma_pp[j]);
				temp=-gamma_p[j]; IT_INCREM01_CMPLX(r[0],r[j],temp,NULL,NULL);
				temp=-gamma[j]; IT_INCREM01_CMPLX(u[0],u[j],temp,NULL,NULL);
			}
			inprodRp1=MAX(0.0,creal(GPDot(r[0],r[0],&Timing_OneIterComm)));
			return;
	}
	LogError(ONE_POS,"Unknown phase (%d) of BiCGStab(12)",(int)ph);
#undef BSL12_L
#undef BSL12_EPS
}

//======================================================================================================================

ITER_FUNC(GPBiCGStab2)
/* GPBiCGStab(2), specialized to L=2 and adapted from the IFDDA
 * GPBICGSTABL recurrence.  This implementation deliberately avoids the raw
 * 4*L+8 workspace layout.  Eleven resident vectors are sufficient by reusing
 * the old q/s history after each BiCG substep.  MatVec remains unchanged.
 *
 * Fixed persistent state between outer iterations:
 *   rvec=r0, pvec=p0, vec1=r~0, vec2=z, vec3=y, vec4=u,
 *   vec5=s0(previous r1), vec6=q0(previous p1), vec7=q1(previous p2).
 * Avecbuffer is the only free work vector at the outer-iteration boundary.
 */
{
#define GP_EPS 1E-30
	static doublecomplex * restrict rtilda,* restrict z,* restrict y,* restrict u;
	static doublecomplex * restrict hs,* restrict hq0,* restrict hq1,* restrict work;
	static double complex rho,sigma,alpha,beta,zeta1,zeta2,eta;
	static double complex mat[3][3],rhs[3];
	static double complex temp;
	static bool fresh_start;

	switch (ph) {
		case PHASE_VARS:
			rtilda=vec1; z=vec2; y=vec3; u=vec4;
			hs=vec5; hq0=vec6; hq1=vec7; work=Avecbuffer;
			scalars[0].ptr=&fresh_start;
			scalars[0].size=sizeof(bool);
			/* The seven semantic state vectors have fixed physical identities at
			 * every iteration boundary, so checkpoints need no permutation metadata. */
			vectors[0].ptr=vec1; vectors[1].ptr=vec2; vectors[2].ptr=vec3; vectors[3].ptr=vec4;
			vectors[4].ptr=vec5; vectors[5].ptr=vec6; vectors[6].ptr=vec7;
			vectors[0].size=vectors[1].size=vectors[2].size=vectors[3].size=
				vectors[4].size=vectors[5].size=vectors[6].size=sizeof(doublecomplex);
			return;

		case PHASE_INIT:
			if (!load_chpoint || gp2_restart_request) {
				IT_COPY(rtilda,rvec); /* r~0 = r0 */
				IT_COPY(pvec,rvec);   /* p0  = r0 */
				IT_MULT_CMPLX(z,rvec,0); /* z=0 */
				fresh_start=true;
			}
			return;

		case PHASE_ITER:
			if (fresh_start) {
				/* -------- Initial BiCG polynomial, j=1 -------- */
				rho=GPDot(rtilda,rvec,&Timing_OneIterComm);
				if (niter==1 && matvec_ready) { /* work=Avecbuffer already equals A*r0=A*p0 */ }
				else IT_MATVEC(pvec,work,NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm); /* p1 */
				sigma=GPDot(rtilda,work,&Timing_OneIterComm);
				if (cabs(sigma)<GP_EPS) LogError(ONE_POS,"GPBiCGStab(2) breakdown: sigma1 is zero");
				alpha=rho/sigma;
				IT_X_INCREM01_CMPLX(pvec,alpha);
				temp=-alpha; IT_INCREM01_CMPLX(rvec,work,temp,NULL,NULL);
				IT_MATVEC(rvec,hs,NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm); /* r1 */
				rho=GPDot(rtilda,hs,&Timing_OneIterComm);
				beta=rho/sigma;
				temp=-beta; IT_INCREM10_CMPLX(pvec,rvec,temp,NULL,NULL); /* p0=r0-beta*p0 */
				IT_INCREM10_CMPLX(work,hs,temp,NULL,NULL);             /* p1=r1-beta*p1 */

				/* -------- Initial BiCG polynomial, j=2 -------- */
				IT_MATVEC(work,hq1,NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm); /* p2 */
				sigma=GPDot(rtilda,hq1,&Timing_OneIterComm);
				if (cabs(sigma)<GP_EPS) LogError(ONE_POS,"GPBiCGStab(2) breakdown: sigma2 is zero");
				alpha=rho/sigma;
				IT_X_INCREM01_CMPLX(pvec,alpha);
				temp=-alpha;
				IT_INCREM01_CMPLX(rvec,work,temp,NULL,NULL);
				IT_INCREM01_CMPLX(hs,hq1,temp,NULL,NULL); /* r1 -= alpha*p2 */
				IT_MATVEC(hs,hq0,NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm); /* r2 */
				rho=GPDot(rtilda,hq0,&Timing_OneIterComm);
				beta=rho/sigma;
				temp=-beta;
				IT_INCREM10_CMPLX(pvec,rvec,temp,NULL,NULL);
				IT_INCREM10_CMPLX(work,hs,temp,NULL,NULL);
				IT_INCREM10_CMPLX(hq1,hq0,temp,NULL,NULL); /* p2=r2-beta*p2 */

				/* L=2 residual minimization: M=[r1,r2]. */
				mat[0][0]=GPDot(hs,hs,&Timing_OneIterComm);
				mat[0][1]=GPDot(hs,hq0,&Timing_OneIterComm);
				mat[1][0]=conj(mat[0][1]);
				mat[1][1]=GPDot(hq0,hq0,&Timing_OneIterComm);
				rhs[0]=GPDot(hs,rvec,&Timing_OneIterComm);
				rhs[1]=GPDot(hq0,rvec,&Timing_OneIterComm);
				if (!GPSolveSmallN(&mat[0][0],rhs,2,3)) LogError(ONE_POS,"GPBiCGStab(2) breakdown in initial 2x2 minimization");
				zeta1=rhs[0]; zeta2=rhs[1];

				/* z=zeta1*r0+zeta2*r1; x+=z.  Carry y/u directly as the
				 * correction removed from r0/p0, eliminating r' and p' vectors. */
				IT_LINCOMB_CMPLX(z,rvec,hs,zeta1,zeta2,NULL,NULL);
				IT_X_INCREM(z);
				IT_LINCOMB_CMPLX(y,hs,hq0,zeta1,zeta2,NULL,NULL);
				IT_LINCOMB_CMPLX(u,work,hq1,zeta1,zeta2,NULL,NULL);
				IT_INCREM01_CMPLX(rvec,y,-1,NULL,NULL);
				IT_INCREM01_CMPLX(pvec,u,-1,NULL,NULL);
				/* Preserve histories for the first full GP cycle: hs=r1, q0=p1, q1=p2.
				 * hq0 currently holds dead r2; work holds p1. */
				IT_COPY(hq0,work);
				inprodRp1=IT_NORM2(rvec,&Timing_OneIterComm);
				fresh_start=false;
				return;
			}

			/* ================= Full GPBiCGStab(2) cycle ================= */
			rho=GPDot(rtilda,rvec,&Timing_OneIterComm);

			/* j=1: p1=A*p0. */
			IT_MATVEC(pvec,work,NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm);
			sigma=GPDot(rtilda,work,&Timing_OneIterComm);
			if (cabs(sigma)<GP_EPS) LogError(ONE_POS,"GPBiCGStab(2) breakdown: sigma1 is zero");
			alpha=rho/sigma;
			IT_X_INCREM01_CMPLX(pvec,alpha);
			temp=-alpha; IT_INCREM01_CMPLX(z,u,temp,NULL,NULL); /* z-=alpha*u */
			IT_INCREM011_CMPLX(y,hq0,work,-alpha,alpha);       /* y-=alpha*(q0-p1) */
			IT_INCREM01_CMPLX(rvec,work,temp,NULL,NULL);
			/* s0 <- s0-alpha*q1; q1 is then free and becomes r1. */
			IT_INCREM01_CMPLX(hs,hq1,temp,NULL,NULL);
			IT_MATVEC(rvec,hq1,NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm); /* r1 */
			rho=GPDot(rtilda,hq1,&Timing_OneIterComm);
			beta=rho/sigma;
			temp=-beta;
			IT_INCREM10_CMPLX(pvec,rvec,temp,NULL,NULL); /* p0 */
			IT_INCREM10_CMPLX(work,hq1,temp,NULL,NULL); /* p1 */
			IT_INCREM10_CMPLX(hq0,hs,temp,NULL,NULL);   /* q0=s0-beta*q0; hs now free */
			IT_INCREM10_CMPLX(u,y,temp,NULL,NULL);       /* u=y-beta*u */

			/* j=2: hs is free and becomes p2. */
			IT_MATVEC(work,hs,NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm);
			sigma=GPDot(rtilda,hs,&Timing_OneIterComm);
			if (cabs(sigma)<GP_EPS) LogError(ONE_POS,"GPBiCGStab(2) breakdown: sigma2 is zero");
			alpha=rho/sigma;
			IT_X_INCREM01_CMPLX(pvec,alpha);
			temp=-alpha; IT_INCREM01_CMPLX(z,u,temp,NULL,NULL);
			IT_INCREM011_CMPLX(y,hq0,work,-alpha,alpha); /* y-=alpha*(q0-p1) */
			/* q0 history is dead after the previous line and can become r2. */
			IT_INCREM01_CMPLX(rvec,work,temp,NULL,NULL);
			IT_INCREM01_CMPLX(hq1,hs,temp,NULL,NULL); /* r1 -= alpha*p2 */
			IT_MATVEC(hq1,hq0,NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm); /* r2 */
			rho=GPDot(rtilda,hq0,&Timing_OneIterComm);
			beta=rho/sigma;
			temp=-beta;
			IT_INCREM10_CMPLX(pvec,rvec,temp,NULL,NULL);
			IT_INCREM10_CMPLX(work,hq1,temp,NULL,NULL); /* p1 */
			IT_INCREM10_CMPLX(hs,hq0,temp,NULL,NULL);   /* p2 */
			IT_INCREM10_CMPLX(u,y,temp,NULL,NULL);       /* u=y-beta*u */

			/* 3x3 local minimization with basis [r1,r2,y]. */
			mat[0][0]=GPDot(hq1,hq1,&Timing_OneIterComm);
			mat[0][1]=GPDot(hq1,hq0,&Timing_OneIterComm);
			mat[0][2]=GPDot(hq1,y,&Timing_OneIterComm);
			mat[1][0]=conj(mat[0][1]);
			mat[1][1]=GPDot(hq0,hq0,&Timing_OneIterComm);
			mat[1][2]=GPDot(hq0,y,&Timing_OneIterComm);
			mat[2][0]=conj(mat[0][2]);
			mat[2][1]=conj(mat[1][2]);
			mat[2][2]=GPDot(y,y,&Timing_OneIterComm);
			rhs[0]=GPDot(hq1,rvec,&Timing_OneIterComm);
			rhs[1]=GPDot(hq0,rvec,&Timing_OneIterComm);
			rhs[2]=GPDot(y,rvec,&Timing_OneIterComm);
			if (!GPSolveSmallN(&mat[0][0],rhs,3,3)) LogError(ONE_POS,"GPBiCGStab(2) breakdown in 3x3 minimization");
			zeta1=rhs[0]; zeta2=rhs[1]; eta=rhs[2];

			/* z=eta*z+zeta1*r0+zeta2*r1; x+=z. */
			IT_INCREM111_CMPLX(z,rvec,hq1,eta,zeta1,zeta2);
			IT_X_INCREM(z);
			/* Carry the exact correction vectors into the next outer cycle:
			 * y_next=eta*y+zeta1*r1+zeta2*r2,
			 * u_next=eta*u+zeta1*p1+zeta2*p2. */
			IT_INCREM111_CMPLX(y,hq1,hq0,eta,zeta1,zeta2);
			IT_INCREM111_CMPLX(u,work,hs,eta,zeta1,zeta2);
			IT_INCREM01_CMPLX(rvec,y,-1,NULL,NULL);
			IT_INCREM01_CMPLX(pvec,u,-1,NULL,NULL);

			/* Restore fixed history identities with D2D copies.  r2 is dead, so
			 * hq0 is a temporary: hq0<-p2; hs<-r1; hq1<-p2; hq0<-p1. */
			IT_COPY(hq0,hs);
			IT_COPY(hs,hq1);
			IT_COPY(hq1,hq0);
			IT_COPY(hq0,work);

			inprodRp1=IT_NORM2(rvec,&Timing_OneIterComm);
			return;
	}
	LogError(ONE_POS,"Unknown phase (%d) of GPBiCGStab(2)",(int)ph);
#undef GP_EPS
}

//======================================================================================================================

ITER_FUNC(GPBiCGStab4)
/* GPBiCGStab(4), memory-reduced L=4 specialization.
 *
 * Algebraically follows the IFDDA GPBiCGStab(L) recurrence for L=4, but folds
 * large-vector lifetimes exactly as in the existing GPBiCGStab(2) CUDA port.
 * Persistent boundary state: r0,p0,r~0,z,y,u,s0..s2,q0..q3 plus one transient
 * work vector.  This needs 15 CUDA-resident vectors including x/r/p/work,
 * instead of the raw IFDDA 25-vector layout. MatVec stays in native precision;
 * GPDot() preserves the chunked FP64 reductions in CUDA single precision.
 */
{
#define GP4_L 4
#define GP4_EPS 1E-30
	static doublecomplex *rtilda,*z,*y,*u,*hs[GP4_L-1],*hq[GP4_L],*work;
	static double complex rho,sigma_c,alpha,beta,temp,zeta[GP4_L],eta;
	static bool fresh_start;
	doublecomplex *base;
	doublecomplex *rr[GP4_L+1],*pp[GP4_L+1];
	double complex mat[GP4_L+1][GP4_L+1],rhs[GP4_L+1];
	double complex mat_work[GP4_L+1][GP4_L+1],rhs_work[GP4_L+1];
	int i,j,mr_rank;

#define GP4_BIND_POLY() do { \
	rr[0]=rvec; rr[1]=hq[3]; rr[2]=hq[2]; rr[3]=hq[1]; rr[4]=hq[0]; \
	pp[0]=pvec; pp[1]=work; pp[2]=hs[2]; pp[3]=hs[1]; pp[4]=hs[0]; \
} while(0)
#define GP4_RESTORE_HISTORY() do { \
	IT_COPY(hq[0],hs[0]); IT_COPY(hs[0],hq[3]); IT_COPY(hq[3],hq[0]); \
	IT_COPY(hq[0],hs[1]); IT_COPY(hs[1],hq[2]); IT_COPY(hq[2],hq[0]); \
	IT_COPY(hq[0],hs[2]); IT_COPY(hs[2],hq[1]); IT_COPY(hq[1],hq[0]); \
	IT_COPY(hq[0],work); \
} while(0)

	switch (ph) {
		case PHASE_VARS:
			base=EnsureCPUL4Workspace(11,"GPBiCGStab(4) L=4 workspace");
			rtilda=base; z=base+local_nRows; y=base+2*local_nRows; u=base+3*local_nRows;
			for (i=0;i<GP4_L-1;i++) hs[i]=base+(size_t)(4+i)*local_nRows;
			for (i=0;i<GP4_L;i++) hq[i]=base+(size_t)(7+i)*local_nRows;
			work=Avecbuffer;
			for (i=0;i<11;i++) { vectors[i].ptr=base+(size_t)i*local_nRows; vectors[i].size=sizeof(doublecomplex); }
			return;
		case PHASE_INIT:
			if (!load_chpoint || lanier_hard_restart_request) {
				IT_COPY(rtilda,rvec); IT_COPY(pvec,rvec); IT_MULT_CMPLX(z,rvec,0); fresh_start=true;
			}
			else fresh_start=false;
			return;
		case PHASE_ITER:
			GP4_BIND_POLY();
			if (fresh_start) {
				rho=GPDot(rtilda,rr[0],&Timing_OneIterComm);
				for (j=1;j<=GP4_L;j++) {
					if (j==1 && matvec_ready) { /* work==Avecbuffer is already ready */ }
					else IT_MATVEC(pp[j-1],pp[j],NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm);
					sigma_c=GPDot(rtilda,pp[j],&Timing_OneIterComm);
					if (cabs(sigma_c)<GP4_EPS) LogError(ONE_POS,"GPBiCGStab(4) breakdown: initial sigma%d is zero",j);
					alpha=rho/sigma_c; IT_X_INCREM01_CMPLX(pp[0],alpha); temp=-alpha;
					for (i=0;i<j;i++) IT_INCREM01_CMPLX(rr[i],pp[i+1],temp,NULL,NULL);
					IT_MATVEC(rr[j-1],rr[j],NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm);
					rho=GPDot(rtilda,rr[j],&Timing_OneIterComm); beta=rho/sigma_c; temp=-beta;
					for (i=0;i<=j;i++) IT_INCREM10_CMPLX(pp[i],rr[i],temp,NULL,NULL);
				}
				memset(mat,0,sizeof(mat)); memset(rhs,0,sizeof(rhs));
				for (i=0;i<GP4_L;i++) { for (j=0;j<GP4_L;j++) mat[i][j]=GPDot(rr[i+1],rr[j+1],&Timing_OneIterComm); rhs[i]=GPDot(rr[i+1],rr[0],&Timing_OneIterComm); }
				mr_rank=GPSolveSmallNAdaptive(&mat[0][0],rhs,&mat_work[0][0],rhs_work,GP4_L,GP4_L+1);
				if (mr_rank==0) LogError(ONE_POS,"GPBiCGStab(4) breakdown in initial MR minimization (rank 0)");
				if (mr_rank<GP4_L && IFROOT)
					PrintBoth(logfile,"GPBiCGStab(4) initial MR happy/near breakdown: using rank %d instead of 4 at iteration %d.\n",mr_rank,niter);
				for (i=0;i<GP4_L;i++) zeta[i]=(i<mr_rank ? rhs_work[i] : 0);
				IT_MULT_CMPLX(z,rr[0],zeta[0]); for (i=1;i<GP4_L;i++) IT_INCREM01_CMPLX(z,rr[i],zeta[i],NULL,NULL); IT_X_INCREM(z);
				IT_MULT_CMPLX(y,rr[1],zeta[0]); IT_MULT_CMPLX(u,pp[1],zeta[0]);
				for (i=1;i<GP4_L;i++) { IT_INCREM01_CMPLX(y,rr[i+1],zeta[i],NULL,NULL); IT_INCREM01_CMPLX(u,pp[i+1],zeta[i],NULL,NULL); }
				IT_INCREM01_CMPLX(rr[0],y,-1,NULL,NULL); IT_INCREM01_CMPLX(pp[0],u,-1,NULL,NULL);
				GP4_RESTORE_HISTORY(); inprodRp1=IT_NORM2(rvec,&Timing_OneIterComm); fresh_start=false; return;
			}

			rho=GPDot(rtilda,rvec,&Timing_OneIterComm);
			/* j=1 */
			IT_MATVEC(pvec,work,NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm); sigma_c=GPDot(rtilda,work,&Timing_OneIterComm);
			if (cabs(sigma_c)<GP4_EPS) LogError(ONE_POS,"GPBiCGStab(4) breakdown: sigma1 is zero");
			alpha=rho/sigma_c; IT_X_INCREM01_CMPLX(pvec,alpha); temp=-alpha; IT_INCREM01_CMPLX(z,u,temp,NULL,NULL); IT_INCREM011_CMPLX(y,hq[0],work,-alpha,alpha); IT_INCREM01_CMPLX(rvec,work,temp,NULL,NULL);
			for (i=0;i<3;i++) IT_INCREM01_CMPLX(hs[i],hq[i+1],temp,NULL,NULL);
			IT_MATVEC(rvec,hq[3],NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm); rho=GPDot(rtilda,hq[3],&Timing_OneIterComm); beta=rho/sigma_c; temp=-beta;
			IT_INCREM10_CMPLX(pvec,rvec,temp,NULL,NULL); IT_INCREM10_CMPLX(work,hq[3],temp,NULL,NULL); for(i=0;i<3;i++) IT_INCREM10_CMPLX(hq[i],hs[i],temp,NULL,NULL); IT_INCREM10_CMPLX(u,y,temp,NULL,NULL);
			/* j=2 */
			IT_MATVEC(work,hs[2],NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm); sigma_c=GPDot(rtilda,hs[2],&Timing_OneIterComm); if(cabs(sigma_c)<GP4_EPS) LogError(ONE_POS,"GPBiCGStab(4) breakdown: sigma2 is zero");
			alpha=rho/sigma_c; IT_X_INCREM01_CMPLX(pvec,alpha); temp=-alpha; IT_INCREM01_CMPLX(z,u,temp,NULL,NULL); IT_INCREM011_CMPLX(y,hq[0],work,-alpha,alpha); IT_INCREM01_CMPLX(rvec,work,temp,NULL,NULL); IT_INCREM01_CMPLX(hq[3],hs[2],temp,NULL,NULL);
			for (i=0;i<2;i++) IT_INCREM01_CMPLX(hs[i],hq[i+1],temp,NULL,NULL);
			IT_MATVEC(hq[3],hq[2],NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm);
			rho=GPDot(rtilda,hq[2],&Timing_OneIterComm); beta=rho/sigma_c; temp=-beta;
			IT_INCREM10_CMPLX(pvec,rvec,temp,NULL,NULL); IT_INCREM10_CMPLX(work,hq[3],temp,NULL,NULL); IT_INCREM10_CMPLX(hs[2],hq[2],temp,NULL,NULL); for(i=0;i<2;i++) IT_INCREM10_CMPLX(hq[i],hs[i],temp,NULL,NULL); IT_INCREM10_CMPLX(u,y,temp,NULL,NULL);
			/* j=3 */
			IT_MATVEC(hs[2],hs[1],NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm); sigma_c=GPDot(rtilda,hs[1],&Timing_OneIterComm); if(cabs(sigma_c)<GP4_EPS) LogError(ONE_POS,"GPBiCGStab(4) breakdown: sigma3 is zero");
			alpha=rho/sigma_c; IT_X_INCREM01_CMPLX(pvec,alpha); temp=-alpha; IT_INCREM01_CMPLX(z,u,temp,NULL,NULL); IT_INCREM011_CMPLX(y,hq[0],work,-alpha,alpha); IT_INCREM01_CMPLX(rvec,work,temp,NULL,NULL); IT_INCREM01_CMPLX(hq[3],hs[2],temp,NULL,NULL); IT_INCREM01_CMPLX(hq[2],hs[1],temp,NULL,NULL); IT_INCREM01_CMPLX(hs[0],hq[1],temp,NULL,NULL);
			IT_MATVEC(hq[2],hq[1],NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm); rho=GPDot(rtilda,hq[1],&Timing_OneIterComm); beta=rho/sigma_c; temp=-beta; IT_INCREM10_CMPLX(pvec,rvec,temp,NULL,NULL); IT_INCREM10_CMPLX(work,hq[3],temp,NULL,NULL); IT_INCREM10_CMPLX(hs[2],hq[2],temp,NULL,NULL); IT_INCREM10_CMPLX(hs[1],hq[1],temp,NULL,NULL); IT_INCREM10_CMPLX(hq[0],hs[0],temp,NULL,NULL); IT_INCREM10_CMPLX(u,y,temp,NULL,NULL);
			/* j=4 */
			IT_MATVEC(hs[1],hs[0],NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm); sigma_c=GPDot(rtilda,hs[0],&Timing_OneIterComm); if(cabs(sigma_c)<GP4_EPS) LogError(ONE_POS,"GPBiCGStab(4) breakdown: sigma4 is zero");
			alpha=rho/sigma_c; IT_X_INCREM01_CMPLX(pvec,alpha); temp=-alpha; IT_INCREM01_CMPLX(z,u,temp,NULL,NULL); IT_INCREM011_CMPLX(y,hq[0],work,-alpha,alpha); IT_INCREM01_CMPLX(rvec,work,temp,NULL,NULL); IT_INCREM01_CMPLX(hq[3],hs[2],temp,NULL,NULL); IT_INCREM01_CMPLX(hq[2],hs[1],temp,NULL,NULL); IT_INCREM01_CMPLX(hq[1],hs[0],temp,NULL,NULL);
			IT_MATVEC(hq[1],hq[0],NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm); rho=GPDot(rtilda,hq[0],&Timing_OneIterComm); beta=rho/sigma_c; temp=-beta; IT_INCREM10_CMPLX(pvec,rvec,temp,NULL,NULL); IT_INCREM10_CMPLX(work,hq[3],temp,NULL,NULL); IT_INCREM10_CMPLX(hs[2],hq[2],temp,NULL,NULL); IT_INCREM10_CMPLX(hs[1],hq[1],temp,NULL,NULL); IT_INCREM10_CMPLX(hs[0],hq[0],temp,NULL,NULL); IT_INCREM10_CMPLX(u,y,temp,NULL,NULL);

			GP4_BIND_POLY(); memset(mat,0,sizeof(mat)); memset(rhs,0,sizeof(rhs));
			for(i=0;i<GP4_L;i++){ for(j=0;j<GP4_L;j++) mat[i][j]=GPDot(rr[i+1],rr[j+1],&Timing_OneIterComm); mat[i][GP4_L]=GPDot(rr[i+1],y,&Timing_OneIterComm); mat[GP4_L][i]=conj(mat[i][GP4_L]); rhs[i]=GPDot(rr[i+1],rr[0],&Timing_OneIterComm); }
			mat[GP4_L][GP4_L]=GPDot(y,y,&Timing_OneIterComm); rhs[GP4_L]=GPDot(y,rr[0],&Timing_OneIterComm);
			mr_rank=GPSolveSmallNAdaptive(&mat[0][0],rhs,&mat_work[0][0],rhs_work,GP4_L+1,GP4_L+1);
			if (mr_rank==0) LogError(ONE_POS,"GPBiCGStab(4) breakdown in MR minimization (rank 0)");
			if (mr_rank<GP4_L+1 && IFROOT)
				PrintBoth(logfile,"GPBiCGStab(4) MR happy/near breakdown: using rank %d instead of 5 at iteration %d.\n",mr_rank,niter);
			for (i=0;i<GP4_L;i++) zeta[i]=(i<mr_rank ? rhs_work[i] : 0);
			eta=(mr_rank>GP4_L ? rhs_work[GP4_L] : 0);
			IT_MULT_SELF_CMPLX(z,eta); for(i=0;i<GP4_L;i++) IT_INCREM01_CMPLX(z,rr[i],zeta[i],NULL,NULL); IT_X_INCREM(z);
			IT_MULT_SELF_CMPLX(y,eta); IT_MULT_SELF_CMPLX(u,eta); for(i=0;i<GP4_L;i++){ IT_INCREM01_CMPLX(y,rr[i+1],zeta[i],NULL,NULL); IT_INCREM01_CMPLX(u,pp[i+1],zeta[i],NULL,NULL); }
			IT_INCREM01_CMPLX(rr[0],y,-1,NULL,NULL); IT_INCREM01_CMPLX(pp[0],u,-1,NULL,NULL); GP4_RESTORE_HISTORY(); inprodRp1=IT_NORM2(rvec,&Timing_OneIterComm); return;
	}
	LogError(ONE_POS,"Unknown phase (%d) of GPBiCGStab(4)",(int)ph);
#undef GP4_RESTORE_HISTORY
#undef GP4_BIND_POLY
#undef GP4_L
#undef GP4_EPS
}

//======================================================================================================================

ITER_FUNC(BiCG_CS)
/* Bi-Conjugate Gradient for Complex Symmetric systems, based on:
 * Freund R.W. "Conjugate gradient-type methods for linear systems with complex symmetric coefficient matrices",
 * SIAM Journal of Scientific Statistics and Computation, 13(1):425-448,1992.
 *
 * it is also identical to COCG, described in:
 * van der Vorst H.A., Melissen J.B.M. "A Petrov-Galerkin type method for solving Ax=b, where A is symmetric complex",
 * IEEE Transactions on Magnetics, 26(2):706-708, 1990.
 *
 * Notation that is used here actually corresponds to Figure 2.7 of Barrett et al. "Templates for the Solution of Linear
 * Systems: Building Blocks for Iterative Methods", 2nd ed., SIAM, 1994. http://www.netlib.org/templates/templates.pdf
 * after removing the second path with transposed matrix (the description in the book is for general matrix).
 * TODO: change the notation to that of Freund (similar to QMR).
 */
{
#define EPS1 1E-10 // for (rT.r)/(r.r)
#define EPS2 1E-10 // for (pT.A.p)/(rT.r)
	static itercomplex alpha, mu;
	static itercomplex beta,ro_new,ro_old,temp;
	static double dtmp,abs_ro_new;
	static bool fresh_start;
#ifdef OCL_BLAS
	cl_mem bufro_new;
	cl_mem bufmu;
	cl_mem bufinprodRp1;
	cl_mem bufpvec=bufargvec;
	cl_mem bufAvecbuffer=bufresultvec;
#endif

	switch (ph) {
		case PHASE_VARS:
			scalars[0].ptr=&ro_old;
			scalars[0].size=sizeof(itercomplex);
			return;
		case PHASE_INIT: {
#ifdef OCL_BLAS
			/* This initialization part need to be moved somewhere during further adoption of clBLAS
			 * For now, we use braces around this case to allow internal variable declaration
			 */
			cl_uint major,minor,patch;
			CLBLAS_CH_ERR(clblasGetVersion(&major,&minor,&patch));
			if (!GREATER_EQ2(major,minor,CLBLAS_VER_REQ,CLBLAS_SUBVER_REQ)) LogError(ONE_POS,
				"clBLAS library version (%u.%u) is too old. Version %d.%d or newer is required",
				major,minor,CLBLAS_VER_REQ,CLBLAS_SUBVER_REQ);
			D("clBLAS library version - %u.%u.%u",major,minor,patch);
			D("clblasSetup started");
			CLBLAS_CH_ERR(clblasSetup());
			CL_CH_ERR(clEnqueueWriteBuffer(command_queue,bufpvec,CL_FALSE,0,sizeof(doublecomplex)*local_nRows,pvec,0,
				NULL,NULL));
			CL_CH_ERR(clEnqueueWriteBuffer(command_queue,bufrvec,CL_FALSE,0,sizeof(doublecomplex)*local_nRows,rvec,0,
				NULL,NULL));
			CL_CH_ERR(clEnqueueWriteBuffer(command_queue,bufxvec,CL_FALSE,0,sizeof(doublecomplex)*local_nRows,xvec,0,
				NULL,NULL));
#endif
			if (!load_chpoint || lanier_hard_restart_request) fresh_start=true;
			else fresh_start=false;
			return; // no specific initialization required (if not OCL_BLAS)
		}
		case PHASE_ITER:
#ifdef OCL_BLAS
			/* TODO: Initialization of this two and one other scalar buffers (and then their release) happens at each
			 * iteration. This is not a bottleneck, but still seems redundant. But to solve this problem we probably
			 * need to add additional phase of the iterative solver, like PHASE_RELEASE, where all such buffers can be
			 * released.
			 */
			CREATE_CL_BUFFER(bufro_new,CL_MEM_READ_WRITE,sizeof(doublecomplex),NULL);
			CREATE_CL_BUFFER(bufmu,CL_MEM_READ_WRITE,sizeof(doublecomplex),NULL);
			CLBLAS_CH_ERR(clblasZdotu(local_nRows,bufro_new,0,bufrvec,0,1,bufrvec,0,1,buftmp,1,&command_queue,0,NULL,
				NULL));
			CL_CH_ERR(clEnqueueReadBuffer(command_queue,bufro_new,CL_TRUE,0,sizeof(doublecomplex),&ro_new,0,NULL,NULL));
#else
			ro_new=IT_DOT_SELF(rvec,&Timing_OneIterComm);
#endif
			// ro_k-1=r_k-1(*).r_k-1; check for ro_k-1!=0
			abs_ro_new=cabs(ro_new);
			dtmp=abs_ro_new/inprodR;
			Dz("|rT.r|/(r.r)="GFORM_DEBUG,dtmp);
			if (dtmp<EPS1) LogError(ONE_POS,"BiCG_CS fails: |rT.r|/(r.r) is too small ("GFORM_DEBUG").",dtmp);
			if (fresh_start) {
#ifdef OCL_BLAS
				clEnqueueCopyBuffer(command_queue,bufrvec,bufpvec,0,0,sizeof(doublecomplex)*local_nRows,0,NULL,NULL);
#else
				IT_COPY(pvec,rvec); // p_1=r_0
#endif
			}
			else {
				// beta_k-1=ro_k-1/ro_k-2
				beta=ro_new/ro_old;
				// p_k=beta_k-1*p_k-1+r_k-1
#ifdef OCL_BLAS
				cl_double2 clbeta = {.s={creal(beta),cimag(beta)}};
				CLBLAS_CH_ERR(clblasZscal(local_nRows,clbeta,bufpvec,0,1,1,&command_queue,0,NULL,NULL));
				cl_double2 clunit = {.s={1,0}};
				CLBLAS_CH_ERR(clblasZaxpy(local_nRows,clunit,bufrvec,0,1,bufpvec,0,1,1,&command_queue,0,NULL,NULL));
#else
				IT_INCREM10_CMPLX(pvec,rvec,beta,NULL,NULL);
#endif
			}
			// q_k=Avecbuffer=A.p_k
			if (fresh_start && matvec_ready) {} // do nothing, Avecbuffer is ready to use
			else BICGCS_MATVEC(pvec,Avecbuffer,NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm);
			// mu_k=p_k.q_k; check for mu_k!=0
#ifdef OCL_BLAS
			CLBLAS_CH_ERR(clblasZdotu(local_nRows,bufmu,0,bufpvec,0,1,bufAvecbuffer,0,1,buftmp,1,&command_queue,0,NULL,
				NULL));
			CL_CH_ERR(clEnqueueReadBuffer(command_queue,bufmu,CL_TRUE,0,sizeof(doublecomplex),&mu,0,NULL,NULL));
#else
			mu=IT_DOTU(pvec,Avecbuffer,&Timing_OneIterComm);
#endif
			dtmp=cabs(mu)/abs_ro_new;
			Dz("|pT.A.p|/(rT.r)="GFORM_DEBUG,dtmp);
			if (dtmp<EPS2) LogError(ONE_POS,"BiCG_CS fails: |pT.A.p|/(rT.r) is too small ("GFORM_DEBUG").",dtmp);
			// alpha_k=ro_k/mu_k
			alpha=ro_new/mu;
			// x_k=x_k-1+alpha_k*p_k
#ifdef OCL_BLAS
			cl_double2 clalpha = {.s={creal(alpha),cimag(alpha)}};
			CLBLAS_CH_ERR(clblasZaxpy(local_nRows,clalpha,bufpvec,0,1,bufxvec,0,1,1,&command_queue,0,NULL,NULL));
#else
			IT_X_INCREM01_CMPLX(pvec,alpha);
#endif
			// r_k=r_k-1-alpha_k*A.p_k and |r_k|^2
			temp=-alpha;
#ifdef OCL_BLAS
			cl_double2 cltemp = {.s={creal(temp),cimag(temp)}};
			CREATE_CL_BUFFER(bufinprodRp1,CL_MEM_READ_WRITE,2*sizeof(double),NULL); // 2 due to workaround below
			CLBLAS_CH_ERR(clblasZaxpy(local_nRows,cltemp,bufAvecbuffer,0,1,bufrvec,0,1,1,&command_queue,0,NULL,NULL));
			/* kernel for function clblasDznrm2 fails to compile (during ADDA execution) with modern OpenCL
			 * implementations, since the latter strictly impose conformance to the standard. Since, the compilation
			 * options for these kernels are not accessible, here we use a workaround through the complex dot-product
			 * function. This workaround requires twice larger memory for result, but twice smaller for buftmp, and
			 * returns the square of the norm.
			 * TODO: Since the clBlas is no more developed, this will stay here until we switch to some other library.
			 * The commented out parts will then facilitate reverting to calling a norm function.
			 */
			//CLBLAS_CH_ERR(clblasDznrm2(local_nRows,bufinprodRp1,0,bufrvec,0,1,buftmp,1,&command_queue,0,NULL,NULL));
			CLBLAS_CH_ERR(clblasZdotc(local_nRows,bufinprodRp1,0,bufrvec,0,1,bufrvec,0,1,buftmp,1,&command_queue,0,NULL,NULL));
			CL_CH_ERR(clEnqueueReadBuffer(command_queue,bufinprodRp1,CL_TRUE,0,sizeof(double),&inprodRp1,0,NULL,NULL));
			//inprodRp1=inprodRp1*inprodRp1; // dot product returns already squared norm
#else
			IT_INCREM01_CMPLX(rvec,Avecbuffer,temp,&inprodRp1,&Timing_OneIterComm);
#endif
#ifdef OCL_BLAS
			CL_CH_ERR(clFinish(command_queue)); // finish queue before freeing resources
			my_clReleaseBuffer(bufinprodRp1);
			my_clReleaseBuffer(bufro_new);
			my_clReleaseBuffer(bufmu);
			if (inprodRp1<epsB){
				CL_CH_ERR(clEnqueueReadBuffer(command_queue,bufpvec,CL_FALSE,0,sizeof(doublecomplex)*local_nRows,pvec,0,
					NULL,NULL));
				CL_CH_ERR(clEnqueueReadBuffer(command_queue,bufrvec,CL_FALSE,0,sizeof(doublecomplex)*local_nRows,rvec,0,
					NULL,NULL));
				CL_CH_ERR(clEnqueueReadBuffer(command_queue,bufxvec,CL_TRUE,0,sizeof(doublecomplex)*local_nRows,xvec,0,
					NULL,NULL));
			}
#endif
			// initialize ro_old -> ro_k-2 for next iteration
			ro_old=ro_new;
			fresh_start=false;
			return; // end of PHASE_ITER
	}
	LogError(ONE_POS,"Unknown phase (%d) of the iterative solver",(int)ph);
}
#undef EPS1
#undef EPS2

//======================================================================================================================

ITER_FUNC(BiCGStab)
/* Bi-Conjugate Gradient Stabilized, based on
 * Barrett et al. "Templates for the Solution of Linear Systems: Building Blocks for Iterative Methods", 2nd ed.,
 * SIAM, 1994. http://www.netlib.org/templates/templates.pdf
 */
{
#define EPS1 1E-10 // for 1/|beta|
#define EPS2 1E-10 // for |v.r~|/|r.r~|
	static double denumOmega,dtmp;
	static itercomplex beta,ro_new,ro_old,omega,alpha,temp1,temp2;
	static doublecomplex * restrict v,* restrict s,* restrict rtilda;
	static bool fresh_start;

	switch (ph) {
		case PHASE_VARS:
			/* rename some vectors; this doesn't contradict with 'restrict' keyword, since new names are not used
			 * together with old names
			 */
			v=vec1;
			s=vec2;
			rtilda=vec3;
			// initialize data structure for checkpoints
			scalars[0].ptr=&ro_old;
			scalars[1].ptr=&omega;
			scalars[2].ptr=&alpha;
			scalars[0].size=scalars[1].size=scalars[2].size=sizeof(itercomplex);
			vectors[0].ptr=vec1; // v
			vectors[1].ptr=vec2; // s
			vectors[2].ptr=vec3; // rtilda
			vectors[0].size=vectors[1].size=vectors[2].size=sizeof(doublecomplex);
			return;
		case PHASE_INIT:
			if (!load_chpoint || lanier_hard_restart_request) { IT_COPY(rtilda,rvec); fresh_start=true; } // r~=r_0
			else fresh_start=false;
			return;
		case PHASE_ITER:
			// ro_k-1=r_k-1.r~ ; check for ro_k-1!=0
			ro_new=IT_DOT(rvec,rtilda,&Timing_OneIterComm);
			if (fresh_start) IT_COPY(pvec,rvec); // p_1=r_0
			else {
				// beta_k-1=(ro_k-1/ro_k-2)*(alpha_k-1/omega_k-1)
				temp1=ro_new*alpha;
				temp2=ro_old*omega;
				// check that omega_k-1!=0; assume that ro_new is not exactly zero
				dtmp=cabs(temp2)/cabs(temp1);
				Dz("1/|beta|="GFORM_DEBUG,dtmp);
				if (dtmp<EPS1) LogError(ONE_POS,"BiCGStab fails: 1/|beta| is too small ("GFORM_DEBUG").",dtmp);
				beta=temp1/temp2;
				// p_k=beta_k-1*(p_k-1-omega_k-1*v_k-1)+r_k-1
				temp1=-beta*omega;
				IT_INCREM110_CMPLX(pvec,v,rvec,beta,temp1);
			}
			// calculate v_k=A.p_k
			if (fresh_start && matvec_ready) IT_COPY(v,Avecbuffer);
			else IT_MATVEC(pvec,v,NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm);
			// alpha_k=ro_new/(v_k.r~)
			temp1=IT_DOT(v,rtilda,&Timing_OneIterComm);
			dtmp=cabs(temp1)/cabs(ro_new); // assume that ro_new is not exactly zero
			Dz("|v.r~|/|r.r~|="GFORM_DEBUG,dtmp);
			if (dtmp<EPS2) LogError(ONE_POS,"BiCGStab fails: |v.r~|/|r.r~| is too small ("GFORM_DEBUG").",dtmp);
			alpha=ro_new/temp1;
			// s=r_k-1-alpha*v_k-1
			temp1=-alpha;
			IT_LINCOMB1_CMPLX(s,v,rvec,temp1,&inprodRp1,&Timing_OneIterComm);
			// check convergence at this step; if yes, checkpoint should not be saved afterwards
			if (inprodRp1<epsB && chp_type!=CHP_ALWAYS) {
				// x_k=x_k-1+alpha_k*p_k
				IT_X_INCREM01_CMPLX(pvec,alpha);
				complete=false;
			}
			else {
				// t=Avecbuffer=A.s
				IT_MATVEC(s,Avecbuffer,&denumOmega,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm);
				// omega_k=s.t/|t|^2
				omega=IT_DOT(s,Avecbuffer,&Timing_OneIterComm)/denumOmega;
				// x_k=x_k-1+alpha_k*p_k+omega_k*s
				IT_X_INCREM011_CMPLX(pvec,s,alpha,omega);
				// r_k=s-omega_k*t and |r_k|^2
				temp1=-omega;
				IT_LINCOMB1_CMPLX(rvec,Avecbuffer,s,temp1,&inprodRp1,&Timing_OneIterComm);
				// initialize ro_old -> ro_k-2 for next iteration
				ro_old=ro_new;
			}
			fresh_start=false;
			return; // end of PHASE_ITER
	}
	LogError(ONE_POS,"Unknown phase (%d) of the iterative solver",(int)ph);
}
#undef EPS1
#undef EPS2

//======================================================================================================================

ITER_FUNC(CGNR)
/* Conjugate Gradient applied to Normalized Equations with minimization of Residual Norm, based on
 * Barrett et al. "Templates for the Solution of Linear Systems: Building Blocks for Iterative Methods", 2nd ed.,
 * SIAM, 1994. http://www.netlib.org/templates/templates.pdf
 */
{
	static double alpha, denumeratorAlpha;
	static double beta,ro_new,ro_old;
	static bool fresh_start;

	switch (ph) {
		case PHASE_VARS:
			scalars[0].ptr=&ro_old;
			scalars[0].size=sizeof(double);
			return;
		case PHASE_INIT:
			fresh_start=(!load_chpoint || lanier_hard_restart_request);
			return; // no other specific initialization required
		case PHASE_ITER:
			// p_1=Ah.r_0 and ro_new=ro_0=|Ah.r_0|^2
			// since first product is with Ah , matvec_ready can't be employed
			if (fresh_start) IT_MATVEC(rvec,pvec,&ro_new,true,&Timing_OneIterMVP,&Timing_OneIterMVPComm);
			else {
				// Avecbuffer=AH.r_k-1, ro_new=ro_k-1=|AH.r_k-1|^2
				IT_MATVEC(rvec,Avecbuffer,&ro_new,true,&Timing_OneIterMVP,&Timing_OneIterMVPComm);
				// beta_k-1=ro_k-1/ro_k-2
				beta=ro_new/ro_old;
				// p_k=beta_k-1*p_k-1+AH.r_k-1
				IT_INCREM10(pvec,Avecbuffer,beta,NULL,NULL);
			}
			// alpha_k=ro_k-1/|A.p_k|^2
			// Avecbuffer=A.p_k
			IT_MATVEC(pvec,Avecbuffer,&denumeratorAlpha,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm);
			alpha=ro_new/denumeratorAlpha;
			// x_k=x_k-1+alpha_k*p_k
			IT_X_INCREM01(pvec,alpha);
			// r_k=r_k-1-alpha_k*A.p_k and |r_k|^2
			IT_INCREM01(rvec,Avecbuffer,-alpha,&inprodRp1,&Timing_OneIterComm);
			// initialize ro_old -> ro_k-2 for next iteration
			ro_old=ro_new;
			fresh_start=false;
			return; // end of PHASE_ITER
	}
	LogError(ONE_POS,"Unknown phase (%d) of the iterative solver",(int)ph);
}

//======================================================================================================================

ITER_FUNC(CSYM)
/* Bi-Conjugate Gradient for Complex Symmetric systems, based on
 * A. Bunse-Gerstner and R. Stover, "On a conjugate gradient-type method for solving complex symmetric linear systems,"
 * Lin. Alg. Appl. 287, 105-123 (1999). with rearrangement of operations (IT_MATVEC is now calculated in the beginning of
 * the iteration)
 *
 * Consumes one less vector than QMR-CS, because rvec does not need to be explicitly computed. The residual should
 * always decrease and always be smaller than that of CGNR (for the same number of matrix-vector products).
 */
{
	static itercomplex alpha,gamma,invksi,theta,eta,tau,temp1,temp2,s_old,s_new;
	static double dtmp,beta,c_old,c_new;
	static doublecomplex *q_new,*q_old,*p_new,*p_old; // can't be declared restrict due to SwapPointers
	static int cycle_iter; // iteration number since the latest reliable/hard restart

	switch (ph) {
		case PHASE_VARS:
			// rename some vectors
			q_new=rvec;  // q_k
			q_old=vec1;  // q_k-1
			p_new=pvec;  // p_k-1
			p_old=vec2;  // p_k-2
			// initialize data structure for checkpoints
			scalars[0].ptr=&beta;
			scalars[1].ptr=&c_old;
			scalars[2].ptr=&c_new;
			scalars[3].ptr=&tau;
			scalars[4].ptr=&s_old;
			scalars[5].ptr=&s_new;
			scalars[0].size=scalars[1].size=scalars[2].size=sizeof(double);
			scalars[3].size=scalars[4].size=scalars[5].size=sizeof(itercomplex);
			vectors[0].ptr=vec1; // now it is q_old, but can be changed further by swapping
			vectors[1].ptr=vec2; // now it is p_old, but can be changed further by swapping
			vectors[0].size=vectors[1].size=sizeof(doublecomplex);
			return;
		case PHASE_INIT:
			if (load_chpoint && !lanier_hard_restart_request) { // change pointers names according to count parity
				/* change pointers names according to count parity. Based on the fact that first two are swapped at each
				 * iteration (and niter>=1), while second two - at each iteration starting from niter=2.
				 */
				if (IS_EVEN(niter)) SwapPointers(&q_old,&q_new);
				else if (niter>1) SwapPointers(&p_old,&p_new);
				cycle_iter=niter;
			}
			else {
				/* Restore canonical aliases after a hard restart; repeated swaps from the
				 * previous Krylov cycle must not leak into the new cycle. */
				q_new=rvec; q_old=vec1; p_new=pvec; p_old=vec2;
				// tau_1 = ||r_0||; q_1 = r_0(*)/||r_0||
				tau=sqrt(inprodR);
				IT_MULT_SELF_CONJ(q_new,1/creal(tau));
				// c_0=1; c_-1=0; s_0=s_-1=0
				c_new=1;
				c_old=0;
				s_new=s_old=0;
				cycle_iter=1;
			}
			return;
		case PHASE_ITER:
			/* Avecbuffer = A.q_k. Since q_1 is r_0(*), mat-vec product for niter==1 is equivalent to Ah.r_0 (as in
			 * CGNR). Thus, matvec_ready can't be employed.
			 */
			IT_MATVEC(q_new,Avecbuffer,NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm);
			// alpha_k = q_k(T).A.q_k
			alpha=IT_DOTU(q_new,Avecbuffer,&Timing_OneIterComm);
			// eta_k = c_k-2*c_k-1*beta_k + s_k-1(*)*alpha_k
			eta = c_old*c_new*beta + alpha*conj(s_new);
			// gamma_k = c_k-1*alpha_k - c_k-2*s_k-1*beta_k
			gamma = c_new*alpha - c_old*s_new*beta;
			// theta_k = s_k-2(*)*beta_k
			theta=beta*conj(s_old);
			// w = Aq_k - alpha_k*q_k(*) - beta_k*q_k-1(*); w is stored in q_old
			temp1=-alpha; // temp1 = -alpha_k
			// use explicitly that q_0=0
			if (cycle_iter==1) IT_LINCOMB1_CMPLX_CONJ(q_old,q_new,Avecbuffer,temp1,&dtmp,&Timing_OneIterComm);
			else IT_INCREM110_D_C_CONJ(q_old,q_new,Avecbuffer,-beta,temp1,&dtmp,&Timing_OneIterComm);
			// beta_k+1 = ||w|| (after that beta is beta_k+1)
			// if beta=0 this is the last iteration, following formulae work fine in this case
			beta=sqrt(dtmp);
			// (k-1)-th values are moved to "old", while new (k-th) values are calculated next
			c_old=c_new;
			s_old=s_new;
			dtmp=cabs(gamma);
			if (dtmp==0) {
				// the following condition should never occur
				if (beta==0) LogError(ONE_POS,"Fatal error in CSYM iterative solver. Interaction matrix is singular");
				c_new=0;
				s_new=1;
				invksi=1/beta;
			}
			else {
				// c_k = |gamma_k| / sqrt(|gamma_k|^2 + beta_k+1^2), computed to avoid overflows
				if (dtmp<beta) {
					dtmp=dtmp/beta;
					c_new=dtmp/sqrt(1+dtmp*dtmp);
				}
				else {
					dtmp=beta/dtmp;
					c_new=1/sqrt(1+dtmp*dtmp);
				}
				// 1/ksi_k = c_k/gamma_k
				invksi=c_new/gamma;
				// s_k = beta_k+1/ksi_k = beta_k+1*c_k/gamma_k
				s_new=beta*invksi;
			}
			// p_k=(-theta_k*p_k-2-eta_k*p_k-1+q_k)/ksi_k
			if (cycle_iter==1) IT_MULT_CMPLX(p_new,q_new,invksi); // use implicitly that p_0=p_-1=0
			else {
				temp1=-eta*invksi;
				if (cycle_iter==2) IT_LINCOMB_CMPLX(p_old,p_new,q_new,temp1,invksi,NULL,NULL); // use explicitly that p_0=0
				else {
					temp2=-theta*invksi;
					IT_INCREM111_CMPLX(p_old,p_new,q_new,temp2,temp1,invksi);
				}
				SwapPointers(&p_old,&p_new);
			}
			// x_k=x_k-1+tau_k*c_k*p_k
			temp1=c_new*tau;
			IT_X_INCREM01_CMPLX(p_new,temp1);
			// q_k+1 = w(*)/beta_k+1; it is first stored into q_old and then swapped
			IT_MULT_SELF_CONJ(q_old,1/beta);
			SwapPointers(&q_old,&q_new);
			// tau_k+1 = -s_k*tau_k; ||r_k|| = |tau_k+1|
			tau*=-s_new;
			inprodRp1=IterAbs2(tau);
#ifdef ADDA_CUDA
			if (lanier_cs_congruence) {
				/* CSYM stores only ||r_k|| recursively. Reconstruct the actual
				 * recurrence residual r_k=tau*conj(q_{k+1}) in scratch so the
				 * DDSCAT-style vector-gap test remains exact. */
				IT_COPY(Avecbuffer,q_new);
				IT_MULT_SELF_CONJ(Avecbuffer,1.0);
				IT_MULT_SELF_CMPLX(Avecbuffer,tau);
				lanier_recursive_residual_vec=Avecbuffer;
			}
#endif
			cycle_iter++;
#ifdef WORKAROUND146
			dumb=tau;
#endif
			return; // end of PHASE_ITER
	}
	LogError(ONE_POS,"Unknown phase (%d) of the iterative solver",(int)ph);
}

//======================================================================================================================

/* QMR_CS remains one C algorithm; the generic iterative dispatch above selects CPU or CUDA primitives. */
ITER_FUNC(QMR_CS)
/* Quasi Minimum Residual for Complex Symmetric systems, based on:
 * Freund R.W. "Conjugate gradient-type methods for linear systems with complex symmetric coefficient matrices",
 * SIAM Journal of Scientific Statistics and Computation, 13(1):425-448,1992.
 */
{
#define EPS1 1E-10 // for (vT.v)/(v.v)
#define EPS2 1E-40 // for overflow of exponent number
	static double c_old,c_new,omega_old,omega_new,zetaabs,dtmp1,dtmp2;
	static itercomplex alpha,beta,theta,eta,zeta,zetatilda,tau,tautilda;
	static itercomplex s_new,s_old,temp1,temp2,temp4;
	static doublecomplex *v,*vtilda,*p_new,*p_old; // can't be declared restrict due to SwapPointers
	static int cycle_iter; // iteration number since the latest reliable/hard restart

	switch (ph) {
		case PHASE_VARS:
			// rename some vectors
			v=vec1;      // v_k
			vtilda=vec2; // also v_k-1
			p_new=pvec;  // p_k
			p_old=vec3;  // p_k-1
			// initialize data structure for checkpoints
			scalars[0].ptr=&omega_old;
			scalars[1].ptr=&omega_new;
			scalars[2].ptr=&c_old;
			scalars[3].ptr=&c_new;
			scalars[4].ptr=&beta;
			scalars[5].ptr=&tautilda;
			scalars[6].ptr=&s_old;
			scalars[7].ptr=&s_new;
			scalars[0].size=scalars[1].size=scalars[2].size=scalars[3].size=sizeof(double);
			scalars[4].size=scalars[5].size=scalars[6].size=scalars[7].size=sizeof(itercomplex);
			vectors[0].ptr=vec1; // now it is v, but can be changed further by swapping
			vectors[1].ptr=vec2; // now it is vtilda, but can be changed further by swapping
			vectors[2].ptr=vec3; // now it is p_old, but can be changed further by swapping
			vectors[0].size=vectors[1].size=vectors[2].size=sizeof(doublecomplex);
			return;
		case PHASE_INIT:
			if (load_chpoint && !lanier_hard_restart_request) {
				/* change pointers names according to count parity. Based on the fact that first two are swapped at each
				 * iteration (and niter>=1), while second two - at each iteration starting from niter=2.
				 */
				if (IS_EVEN(niter)) SwapPointers(&v,&vtilda);
				else if (niter>1) SwapPointers(&p_old,&p_new);
				cycle_iter=niter;
			}
			else {
				/* Restore canonical aliases after a reliable/hard restart. */
				v=vec1; vtilda=vec2; p_new=pvec; p_old=vec3;
				// omega_0=||v_0||=0
				omega_old=0.0;
				// beta_1=sqrt(v~_1(*).v~_1); omega_1=||v~_1||/|beta_1|; (v~_1=r_0)
				beta=csqrt(IT_DOT_SELF(rvec,&Timing_InitIterComm));
				omega_new=sqrt(inprodR)/cabs(beta); // inprodR=IT_NORM2(r_0)
				// v_1=v~_1/beta_1
				temp1=1/beta;
				IT_MULT_CMPLX(v,rvec,temp1);
				// tau~_1=omega_1*beta_1
				tautilda=omega_new*beta;
				// c_0=c_-1=1; s_0=s_-1=0
				c_new=c_old=1.0;
				s_new=s_old=0.0;
				cycle_iter=1;
#ifdef WORKAROUND146
				dumb=beta;
#endif
			}
			return;
		case PHASE_ITER:
			// check for very high omega (very small beta/||v||)
			dtmp1=1/(omega_new*omega_new);
			Dz("|vT.v|/(v.v)="GFORM_DEBUG,dtmp1);
			if (dtmp1<EPS1) LogError(ONE_POS,"QMR_CS fails: |vT.v|/(v.v) is too small ("GFORM_DEBUG").",dtmp1);
			// A.v_k; alpha_k=v_k(*).(A.v_k)
			if (cycle_iter==1 && matvec_ready) { // uses that v_1=r_0/beta
				temp1=1/beta;
				IT_MULT_SELF_CMPLX(Avecbuffer,temp1);
			}
			else IT_MATVEC(v,Avecbuffer,NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm);
			alpha=IT_DOTU(v,Avecbuffer,&Timing_OneIterComm);
			// v~_k+1=-beta_k*v_k-1-alpha_k*v_k+A.v_k
			temp2=-alpha;
			if (cycle_iter==1) IT_LINCOMB1_CMPLX(vtilda,v,Avecbuffer,temp2,NULL,NULL); // use explicitly that v_0=0
			else {
				temp1=-beta;
				IT_INCREM110_CMPLX(vtilda,v,Avecbuffer,temp1,temp2);
			}
			// theta_k=s_k-2(*)*omega_k-1*beta_k
			theta=conj(s_old)*omega_old*beta;
			// eta_k=c_k-1*c_k-2*omega_k-1*beta_k+s_k-1(*)*omega_k*alpha_k
			eta = c_old*c_new*omega_old*beta;
			eta += alpha*conj(s_new)*omega_new;
			// zeta~_k=c_k-1*omega_k*alpha_k-s_k-1*c_k-2*omega_k-1*beta_k
			zetatilda = c_new*omega_new*alpha - s_new*c_old*omega_old*beta;
			// beta_k+1=sqrt(v~_k+1(*).v~_k+1); omega_k+1=||v~_k+1||/|beta_k+1|
			omega_old=omega_new;
			temp1=IT_DOT_SELF_NORM2(vtilda,&dtmp1,&Timing_OneIterComm); // dtmp1=||v~||^2
			beta=csqrt(temp1);
			/* Here we do not check for zero beta, since exact zero is very improbable and the following code (until the
			 * end of iteration) employs only the product omega_k+1*beta_k+1. So the (almost) breakdown is instead
			 * checked at the beginning of the iteration.
			 */
			omega_new=sqrt(dtmp1)/cabs(beta);
			// |zeta_k|=sqrt(|zeta~_k|^2+omega_k+1^2*|beta_k+1|^2)
			dtmp2=IterAbs2(zetatilda); // dtmp2=|zeta~_k|^2
			zetaabs=sqrt(dtmp2+dtmp1);
			dtmp1=sqrt(dtmp2); // dtmp1=|zeta~_k|
			// if (|zeta~_k|==0) zeta_k=|zeta_k|; else zeta=|zeta_k|*zeta~_k/|zeta~_k|
			if (dtmp1<EPS2) zeta=zetaabs;
			else zeta=(zetaabs/dtmp1)*zetatilda;
			// c_k=zeta~_k/zeta_k = |zeta~_k|/|zeta_k|
			c_old=c_new;
			c_new=dtmp1/zetaabs;
			// s_k+1=omega_k+1*beta_k+1/zeta_k
			s_old=s_new;
			s_new=omega_new*beta/zeta;
			// p_k=(-theta_k*p_k-2-eta_k*p_k-1+v_k)/zeta_k
			temp4=1/zeta; // temp4=1/zeta_k;
			if (cycle_iter==1) IT_MULT_CMPLX(p_new,v,temp4); // use implicitly that p_0=p_-1=0
			else {
				temp2=-eta*temp4; // temp2=-eta_k/zeta_k
				if (cycle_iter==2) IT_LINCOMB_CMPLX(p_old,p_new,v,temp2,temp4,NULL,NULL);
				else {
					temp1=-theta*temp4; // temp1=-theta_k/zeta_k
					IT_INCREM111_CMPLX(p_old,p_new,v,temp1,temp2,temp4);
				}
				SwapPointers(&p_old,&p_new);
			}
			// tau_k=c_k*tau~_k
			tau=c_new*tautilda;
			// tau~_k+1=-s_k*tau~_k
			tautilda=-s_new*tautilda;
			// x_k=x_k-1+tau_k*p_k
			IT_X_INCREM01_CMPLX(p_new,tau);
			// v_k+1=v~_k+1/beta_k+1
			temp1=1/beta;
			IT_MULT_SELF_CMPLX(vtilda,temp1);
			SwapPointers(&v,&vtilda); // v~ is as v_k-1 at next iteration
			// r_k = |s_k|^2*r_k-1 + (c_k*tau~_k+1/omega_k+1)*v_k+1
			temp1=(c_new/omega_new)*tautilda;
			IT_INCREM11_D_C(rvec,v,IterAbs2(s_new),temp1,&inprodRp1,&Timing_OneIterComm);
			cycle_iter++;
			return; // end of PHASE_ITER
	}
	LogError(ONE_POS,"Unknown phase (%d) of the iterative solver",(int)ph);
}
#undef EPS1
#undef EPS2

//======================================================================================================================

ITER_FUNC(QMR_CS_2)
/* Quasi Minimum Residual for Complex Symmetric systems based on:
 * R.W. Freund and N.M. Nachtigal, "An implementation of the qmr method based on coupled 2-term recurrences,"
 * SIAM J. Sci. Comp. 15, 313-337 (1994).
 *
 * We use recommended values of omega_k=1, which correspond to omega_k=||v_k|| used in QMR_CS above
 */
{
// breakdown tests are similar to that of BiCG_CS
#define EPS1  1E-10 // for vT.v
#define EPS2  1E-10 // for pT.A.p
	static double c_old,c_new,theta_old,theta_new,ro_old,ro_new,sabs2,dtmp1;
	static itercomplex eps,beta,delta,eta,temp1;
	static doublecomplex * restrict v,* restrict d;
	static int cycle_iter; // iteration number since the latest reliable/hard restart

	switch (ph) {
		case PHASE_VARS:
			// rename some vectors
			v=vec1;      // v_k and v~_k+1
			d=vec2;      // d_k
			// initialize data structure for checkpoints
			scalars[0].ptr=&c_old;
			scalars[1].ptr=&theta_old;
			scalars[2].ptr=&ro_old;
			scalars[3].ptr=&eps;
			scalars[4].ptr=&eta;
			scalars[0].size=scalars[1].size=scalars[2].size=sizeof(double);
			scalars[3].size=scalars[4].size=sizeof(itercomplex);
			vectors[0].ptr=vec1; // v
			vectors[1].ptr=vec2; // d
			vectors[0].size=vectors[1].size=sizeof(doublecomplex);
			return;
		case PHASE_INIT:
			if (!load_chpoint || lanier_hard_restart_request) {
				// ro_1=||r_0||; v~_1=r_0
				ro_old=sqrt(inprodR);
				IT_COPY(v,rvec);
				// c_0=eps_0=1; theta_0=0; eta_0=-1
				c_old=1;
				eps=1;
				theta_old=0;
				eta=-1;
				cycle_iter=1;
#ifdef WORKAROUND146
				dumb=eps;
#endif
			}
			else cycle_iter=niter;
			return;
		case PHASE_ITER:
			// v_k = v~_k/ro_k; this is rearranged as compared to the original algorithm
			// if ro_k=0 then c_k=1,v~_k=0, hence r_k-1=0 and iteration should have stopped by now
			IT_MULT_SELF(v,1/ro_old);
			// delta_k = v(*).v ; test it to be non zero
			delta=IT_DOT_SELF(v,&Timing_OneIterComm);
			dtmp1=cabs(delta);
			Dz("|vT.v|="GFORM_DEBUG,dtmp1);
			if (dtmp1<EPS1) LogError(ONE_POS,"QMR_CS_2 fails: |vT.v| is too small ("GFORM_DEBUG").",dtmp1);
			// p_k = v_k - p_k-1*ro_k*delta_k/eps_k-1
			if (cycle_iter==1) IT_COPY(pvec,v); // use explicitly that p_0=0
			else {
				temp1=-ro_old*delta/eps;
				IT_INCREM10_CMPLX(pvec,v,temp1,NULL,NULL);
			}
			// A.p_k
			if (cycle_iter==1 && matvec_ready) { // uses that p_1=v_1=r_0/ro_1
				IT_MULT_SELF(Avecbuffer,1/ro_old);
			}
			else IT_MATVEC(pvec,Avecbuffer,NULL,false,&Timing_OneIterMVP,&Timing_OneIterMVPComm);
			// eps_k = p_k(*).(A.p_k); beta_k = eps_k/delta_k
			eps=IT_DOTU(pvec,Avecbuffer,&Timing_OneIterComm);
			beta=eps/delta;
			dtmp1=cabs(eps);
			Dz("|pT.A.p|="GFORM_DEBUG,dtmp1);
			if (dtmp1<EPS1) LogError(ONE_POS,"QMR_CS_2 fails: |pT.A.p| is too small ("GFORM_DEBUG").",dtmp1);
			// v~_k+1 = A.p_k - beta_k*v_k; stored in the same vector v
			temp1=-beta;
			IT_INCREM10_CMPLX(v,Avecbuffer,temp1,&dtmp1,&Timing_OneIterComm);
			ro_new=sqrt(dtmp1); // ro_k+1 = ||v~_k+1||
			// theta_k = ro_k+1/(c_k-1*|beta_k|);
			theta_new=ro_new/(c_old*cabs(beta));
			// c_k = 1/sqrt(1+theta_k^2), |s_k|^2 = 1-c_k^2
			dtmp1=theta_new*theta_new;
			c_new=1/sqrt(1+dtmp1);
			sabs2=dtmp1/(1+dtmp1);
			// eta_k = -eta_k-1*ro_k*c_k^2/(beta_k*c_k-1^2)
			dtmp1=c_new/c_old;
			eta=-ro_old*dtmp1*dtmp1*eta/beta;
			// d_k = p_k*eta_k + d_k-1*(theta_k-1*c_k)^2
			if (cycle_iter==1) IT_MULT_CMPLX(d,pvec,eta); // use explicitly that d_0=0
			else {
				dtmp1=theta_old*c_new;
				IT_INCREM11_D_C(d,pvec,dtmp1*dtmp1,eta,NULL,NULL);
			}
			// x_k = x_k-1 + d_k
			IT_X_INCREM(d);
			/* The following formula to update residual was not given in the original publication, we derived it
			 * ourselves; r_k = (1-c_k^2)*r_k-1 - eta_k*v~_k+1
			 */
			temp1=-eta;
			IT_INCREM11_D_C(rvec,v,sabs2,temp1,&inprodRp1,&Timing_OneIterComm);
			// update variables for next iteration
			ro_old=ro_new;
			theta_old=theta_new;
			c_old=c_new;
			cycle_iter++;
			return; // end of PHASE_ITER
	}
	LogError(ONE_POS,"Unknown phase (%d) of the iterative solver",(int)ph);
}
#undef EPS1
#undef EPS2

//======================================================================================================================

#undef IT_MATVEC
#undef BICGCS_MATVEC
#undef IT_COPY
#undef IT_NORM2
#undef IT_DOT
#undef IT_DOTU
#undef IT_DOT_SELF
#undef IT_DOT_SELF_NORM2
#undef IT_MULT
#undef IT_MULT_CMPLX
#undef IT_MULT_SELF
#undef IT_MULT_SELF_CONJ
#undef IT_MULT_SELF_CMPLX
#undef IT_INCREM
#undef IT_INCREM01
#undef IT_INCREM10
#undef IT_INCREM01_CMPLX
#undef IT_INCREM10_CMPLX
#undef IT_INCREM011_CMPLX
#undef IT_INCREM110_CMPLX
#undef IT_INCREM111_CMPLX
#undef IT_INCREM11_D_C
#undef IT_INCREM110_D_C_CONJ
#undef IT_LINCOMB_CMPLX
#undef IT_LINCOMB1_CMPLX
#undef IT_LINCOMB1_CMPLX_CONJ
#undef IT_X_INCREM
#undef IT_X_INCREM01
#undef IT_X_INCREM01_CMPLX
#undef IT_X_INCREM011_CMPLX

/* TO ADD NEW ITERATIVE SOLVER
 * Add the function implementing the iterative method to the list above in the alphabetical order. The template for the
 * function is provided below together with additional comments. Please also look at the iterative solvers, already
 * present, for examples. For operations on complex numbers you are advised to use functions from cmplx.h, for switching
 * vectors - SwapPointers (above), for linear algebra - functions from linalg.c, for multiplication of vector with
 * matrix of the linear system - MatVec function from matvec.c. Some of these functions take account of the time spent
 * on communication between different processors (in parallel mode), and increment their last argument by the
 * corresponding amount. You may also use values of variables, defined in the beginning of this source file, especially
 * niter, resid_scale, and epsB.
 */
#if 0
ITER_FUNC(_name_) // only '_name_' should be changed, the macro expansion will do the rest
// Short comment, providing full name of the iterative solver
{
/* It is recommended to define all nontrivial constants here. Do not forget to undef them at the end of this function to
 * avoid conflicts with other iterative solvers.
 */
#define EPS1 1E-30
	/* all internal variables should be defined here as static, since the function will be called many times (once per
	 * iteration).
	 */
	static double xxx;

	/* The function accepts a single argument 'ph' describing a phase, which it should perform at a particular run. This
	 * is done to move all common parts to the function IterativeSolver. Possible phases are defined and briefly
	 * explained in the definition of 'enum phase' in the beginning of this source file.
	 */
	switch (ph) {
		case PHASE_VARS:
			/* Here variables are linked to structure arrays 'scalars' and 'vectors' to initialize checkpoint system
			 * (see comment before function SaveCheckpoint). For example:
			 */
			scalars[0].ptr=&xxx;
			scalars[0].size=sizeof(double);
			/* Also, if auxiliary vectors vec1,... are used, their names may be changed to a more meaningful ones (using
			 * pointer assignments)
			 */
			return;
		case PHASE_INIT:
			/* Initialization of the iterative solver. You may use 'load_chpoint' to distinguish between the plain run
			 * and the one restarted from a checkpoint. Actual loading of checkpoint happens just before this phase. For
			 * gathering communication time use variable Timing_InitIterComm.
			 */
			return;
		case PHASE_ITER:
			/* Performs a general iteration. As a result, inprodRp1 (current residual) should be calculated. For
			 * gathering communication time use variable Timing_OneIterComm.
			 */

			// an example for checking of convergence failure (optional)
			if (xxx<EPS1) LogError(ONE_POS,"_name_ fails: xxx is too small ("GFORM_DEBUG").",xxx);

			/* _Some_ iterative solvers contain extra checks for convergence in the _middle_ of an iteration, designed
			 * to save time of, e.g., one matrix-vector product in some cases. They should be performed as follows. In
			 * particular, the intermediate test will be skipped if final checkpoint is required (checkpoint of type
			 * 'always').
			 */
			if (inprodRp1<epsB && chp_type!=CHP_ALWAYS) {
				// Additional code, e.g. to set xvec to the final value
				complete=false; // this is required to skip saving checkpoint and some timing
			}
			else {
				// Continue the iteration normally
			}
			/* Common check for convergence at the end of an iteration should not be done here, because it is performed in
			 * the function IterativeSolver.
			 */
			return;
	}
	LogError(ONE_POS,"Unknown phase (%d) of the iterative solver",(int)ph);
#undef EPS1
}
#endif

#undef ITER_FUNC

//======================================================================================================================

static void CalcFieldWKB(doublecomplex * restrict Efield)
// Calculate internal electric field in the WKB approximation; uses Einc and stores the result in Efield
{
#ifndef SPARSE	//currently no support for WKB in sparse mode
	if (prop[2]!=1) LogError(ONE_POS,"WKB initial field currently works only with default incident direction "
		"of the incoming wave (along z-axis)");
	doublecomplex vals[Nmat+1],tmpc;
	int i,k; // for traversing single-axis dimensions
	size_t dip,ind,dip_sl; // for traversing slices or up to local_nRows
	size_t boxX_l=(size_t)boxX; // to remove type conversion in indexing
#define INDEX_GRID(i) (position[(i)+2]*boxXY+position[(i)+1]*boxX_l+position[i])
	/* can be optimized by reusing material_tmp from make_particle.c or keeping the values between the calls. But
	 * this will require usage of extra memory. So the current option can be considered as corresponding to
	 * '-opt mem'
	 */
	unsigned char *mat; // same as material, but defined on whole grid (local_Ndip)
	doublecomplex *arg; // argument of exponent for corrections of incident field
#ifdef PARALLEL
	doublecomplex *bottom; // value of arg at bottom of current processor
#endif
	doublecomplex *top; // propagating value of arg at planes between the voxels

#if defined(OPENCL) || defined(ADDA_CUDA) // Xmatrix is not allocated for GPU MatVec backends
	bool a_arg=false;
	bool a_mat=false;
	bool a_top=false;
	MAXIMIZE(memPeak,memory);
	if (local_Ndip<=local_nRows) arg=Avecbuffer;
	else {
		MALLOC_VECTOR(arg,complex,local_Ndip,ALL);
		memPeak+=local_Ndip*sizeof(doublecomplex);
		a_arg=true;
	}
	if (local_Ndip*sizeof(char)<=sizeof(doublecomplex)*local_nRows) mat=(unsigned char *)Efield;
	else {
		MALLOC_VECTOR(mat,uchar,local_Ndip,ALL);
		memPeak+=local_Ndip*sizeof(char);
		a_mat=true;
	}
	if (boxXY<local_nRows) top=rvec;
	else {
		MALLOC_VECTOR(top,complex,boxXY,ALL);
		memPeak+=boxXY*sizeof(doublecomplex);
		a_top=true;
	}
#else // define all vectors using memory assigned to Xmatrix; kind of weird but should be OK
	arg=Xmatrix;
#	ifdef PARALLEL
	bottom=Xmatrix+local_Ndip;
	top=bottom+boxXY;
#	else
	top=Xmatrix+local_Ndip;
#	endif
	mat=(unsigned char *)(top + boxXY);
#endif
	// calculate function of refractive index
	for (i=0;i<Nmat;i++) vals[i]=I*(ref_index[i]-1)*kdZ/2;
	vals[Nmat]=0;
	// calculate values of mat (the same algorithm as in matvec), for void voxels mat=Nmat
	for (dip=0;dip<local_Ndip;dip++) mat[dip]=(unsigned char)Nmat;
	for (dip=0,ind=0;dip<local_nvoid_Ndip;dip++,ind+=3) mat[INDEX_GRID(ind)]=material[dip];
	/* main part responsible for calculation of arg; arg[i,j,k+1]=arg[i,j,k]+vals[i,j,k]+vals[i,j,k+1]
	 * but that is done with temporary variables (not to index both k and k+1 simultaneously
	 * 'ind' traverses one slice, and 'dip' - all dipoles (voxels)
	 */
	// First, calculate shifts relative to the bottom of current processor
	for(ind=0;ind<boxXY;ind++) top[ind]=0;
	for(k=local_z0,dip_sl=0;k<local_z1_coer;k++,dip_sl+=boxXY) for(ind=0,dip=dip_sl;ind<boxXY;ind++,dip++) {
		arg[dip]=top[ind]+vals[mat[dip]];
		top[ind]=arg[dip]+vals[mat[dip]];
	}
#ifdef PARALLEL
	// Second, fulfill boundary by exchanging shift values at top and bottom
	if (ExchangePhaseShifts(bottom,top,&Timing_InitIterComm))
		// Third (if required) update shift from the obtained values on the bottom
		for(k=local_z0,dip_sl=0;k<local_z1_coer;k++,dip_sl+=boxXY) for(ind=0,dip=dip_sl;ind<boxXY;ind++,dip++)
			arg[dip]+=bottom[ind];
#endif
	// E=Einc*Exp(arg), but arg is defined on a set of all (including void) voxels
	for (ind=0;ind<local_nRows;ind+=3) {
		tmpc=cexp(arg[INDEX_GRID(ind)]);
		cvMultScal_cmplx(tmpc,Einc+ind,Efield+ind);
	}
#if defined(OPENCL) || defined(ADDA_CUDA) // free those buffers that were allocated
	if (a_arg) Free_cVector(arg);
	if (a_mat) Free_general(mat);
	if (a_top) Free_cVector(top);
#endif
// end of !SPARSE
#else
	// should never come to this
	LogError(ONE_POS,"WKB initial field is not supported in sparse mode");
	Efield[0]=0; // redundant, to eliminate unused parameter warning
#endif
}

//======================================================================================================================

static void InitFieldfromE(void)
/* sets starting vector for linear system x_0, as well as A.x_0, r_0=b-A.x_0, and |r_0|^2 from given electric field;
 * assumes that xvec contains initial electric field, it is then replaced by x_0
 */
{
	// calculate x = (1/cc_sqrt)*V*chi*E (both x and E are stored in xvec)
	doublecomplex mult[MAX_NMAT][3];
	int i,j;
	for (i=0;i<Nmat;i++) for (j=0;j<3;j++) mult[i][j]=1/(cc_sqrt[i][j]*chi_inv[i][j]);
	nMultSelf_mat(xvec,mult);
	// calculate A.x_0, r_0=b-A.x_0, and |r_0|^2
	MatVec(xvec,Avecbuffer,NULL,false,&Timing_MVP,&Timing_MVPComm);
	nSubtr(rvec,pvec,Avecbuffer,&inprodR,&Timing_InitIterComm);
}

//======================================================================================================================

static const char *CalcInitField(double zero_resid,const enum incpol which)
/* Initializes the field as the starting point of the iterative solver. Assumes that pvec contains the right-hand side
 * of equations (b). At the end of this function xvec should contain initial vector for the iterative solver (x_0), rvec
 * - corresponding residual r_0, and inprodR - the norm of the latter residual. Returns string containing description of
 * the initial field used.
 */
{
	switch (InitField) {
		case IF_AUTO:
			/* This code is somewhat inelegant, but there seem to be no easy way to completely reuse code for other
			 * cases. Moreover, this option will probably be changed afterwards.
			 */
			// calculate A.(x_0=b), r_0=b-A.(x_0=b) and |r_0|^2
			MatVec(pvec,Avecbuffer,NULL,false,&Timing_MVP,&Timing_MVPComm);
			nSubtr(rvec,pvec,Avecbuffer,&inprodR,&Timing_InitIterComm);
			// check which x_0 is better
			if (zero_resid<inprodR) { // use x_0=0
				nInit(xvec);
				nCopy(rvec,pvec);
				inprodR=zero_resid;
				matvec_ready=true; // here Avecbuffer = A.r_0
				return "x_0 = 0";
			}
			else { // use x_0=Einc
				nCopy(xvec,pvec);
				return "x_0 = E_inc";
			}
		case IF_ZERO:
			nInit(xvec); // x_0=0
			nCopy(rvec,pvec); // r_0=b
			inprodR=zero_resid;
			return "x_0 = 0";
		case IF_INC:
			nCopy(xvec,pvec); // x_0=b, i.e. E_exc=E_inc
			// calculate A.(x_0=b), r_0=b-A.(x_0=b) and |r_0|^2
			MatVec(xvec,Avecbuffer,NULL,false,&Timing_MVP,&Timing_MVPComm);
			nSubtr(rvec,pvec,Avecbuffer,&inprodR,&Timing_InitIterComm);
			return "x_0 = E_inc";
		case IF_WKB:
			CalcFieldWKB(xvec); // calculate WKB electric field
			InitFieldfromE(); // transform it into starting vector
			return "x_0 = result of WKB";
		case IF_READ: {
			const char *fname;
			if (which==INCPOL_Y) fname=infi_fnameY;
			else fname=infi_fnameX; // which==INCPOL_X
			ReadField(fname,xvec); // read electric field
			InitFieldfromE(); // transform it into starting vector
			return dyn_sprintf("x_0 = from file %s",fname);
		}
	}
	LogError(ONE_POS,"Unknown method to calculate initial field (%d)",(int)InitField);
}

//======================================================================================================================

static int IterativeSolverCore(const enum iter method_in,const enum incpol which)
/* choose required iterative method; do common initialization part;
 * 'which' is used only if the initial field is read from file
 */
{
	double temp;
	char tmp_str[MAX_LINE];
	const char *init_descr=NULL;
	TIME_TYPE tstart,time_tmp,time_tmp2,time_tmp3;

	// redundant initialization to remove warnings
	time_tmp=time_tmp2=time_tmp3=0;

	/* Instead of solving system (I+D.C).x=b , C - diagonal matrix with couple constants
	 *                                         D - symmetric interaction matrix of Green's tensor
	 * we solve system (I+S.D.S).(S.x)=(S.b), S=sqrt(C), then total interaction matrix is symmetric and
	 * Jacobi-preconditioned for any distribution of refractive index.
	 *
	 * Relative residual is defined by dividing by the norm of the righ-hand-side, i.e. |S.Einc|. This is more robust,
	 * when various non-zero initial guesses are used.
	 */
	/* p=b=(S.Einc) is right part of the linear system; used only here. In iteration methods themselves p is completely
	 * different vector. To avoid confusion this is done before any other initializations, specific to iterative solvers
	 */
	Timing_InitIterComm=Timing_MVP=Timing_MVPComm=0;
	tstart=GET_TIME();
	matvec_ready=false; // can be set to true only in CalcInitField (if !load_chpoint)
#ifdef ADDA_CUDA
	lanier_true_converged=false;
	lanier_physical_rhs_norm2=0.0;
	lanier_reset_baseline=1.0;
	lanier_reset_mandatory_count=lanier_reset_ratio_count=lanier_reset_reliable_count=0;
	lanier_recursive_residual_vec=rvec;
#endif
	if (!load_chpoint) {
		nMult_mat(pvec,Einc,cc_sqrt);
		temp=nNorm2(pvec,&Timing_InitIterComm); // |S.Einc|^2, but also equal to |r_0|^2 when x_0=0
#ifdef ADDA_CUDA
		lanier_physical_rhs_norm2=temp;
#endif
		resid_scale=1/temp;
		epsB=iter_eps*iter_eps*temp;
		// Calculate initial field. Printing is delayed until after an optional
		// Lanier complex-symmetric residual-space transformation.
#ifdef ADDA_CUDA
		if (lanier_partition_use_existing_x) {
			/* xvec is already in ADDA's transformed unknown y=S^-1 p. */
			MatVec(xvec,Avecbuffer,NULL,false,&Timing_MVP,&Timing_MVPComm);
			nSubtr(rvec,pvec,Avecbuffer,&inprodR,&Timing_InitIterComm);
			init_descr="x_0 = LANIER_PARTITION warm start";
		}
		else
#endif
			init_descr=CalcInitField(temp,which);
		// initialize counters
		niter=1;
		counter=0;
	}
	/* determine index of the iterative solver, which is further used to get its parameters from list 'params'. This way
	 * it should be resistant to inconsistencies in orders of iterative solvers inside the list of identifiers in
	 * const.h and in the list 'params' above.
	 */
	ind_m=0;
	while (params[ind_m].meth!=method_in) {
		ind_m++;
		if (ind_m>=LENGTH(params))
			LogError(ONE_POS,"Parameters for the given iterative solver are not found in list 'params'");
	}
#if defined(ADDA_CUDA) && defined(ADDA_SINGLE)
	if (reliable_resid && method_in!=IT_GPBICGSTAB2 && method_in!=IT_BCGS2)
		LogWarning(EC_WARN,ONE_POS,"-reliable_resid currently applies only to CUDA float32 BiCGStab(2)/BCGS2 and GPBiCGStab(2); option ignored");
#else
	if (reliable_resid)
		LogWarning(EC_WARN,ONE_POS,"-reliable_resid currently applies only to CUDA float32 BiCGStab(2)/BCGS2 and GPBiCGStab(2); option ignored");
#endif
	if (lanier_precon) {
#ifndef ADDA_CUDA
		LogError(ONE_POS,"Lanier preconditioners require an ADDA CUDA executable");
#else
		/* The validated reference 1x/1.5x path remains deliberately restricted
		 * to BCGS2. LANIER_FULL is integrated with every CUDA iterative solver. */
		if (!lanier_full_precon && method_in!=IT_BCGS2)
			LogError(ONE_POS,"Lanier reference 1x/1.5x currently supports only -iter bcgs2; use -precon lanier_full for the all-solver path");
		if (!lanier_full_precon && Nmat!=1)
			LogError(ONE_POS,"Lanier reference preconditioner currently requires one homogeneous material");
		if (surface)
			LogError(ONE_POS,"Lanier preconditioners currently do not support -surf");
		if (lanier_full_precon && rectDip)
			LogError(ONE_POS,"Lanier full TQC-v1 currently requires isotropic cubic voxels (no -rect_dip)");
		if ((lanier_multizone_schwarz_precon || lanier_multizone_schwarz_sym_precon || lanier_multizone_schwarz_reverse_precon || lanier_multizone_schur_precon) && !LanierSchwarzSupportedMethod(method_in))
			LogError(ONE_POS,"LANIER_MULTIZONE nonsymmetric Schwarz modes and Schur V2.1 support -iter bcgs2, bicgstab, bicgstab4, gpbicgstab2, or gpbicgstab4. BiCG/CGNR and complex-symmetric solver integration remain intentionally disabled.");
		lanier_cs_congruence=lanier_full_precon && !lanier_multizone_schwarz_precon && !lanier_multizone_schwarz_sym_precon && !lanier_multizone_schwarz_reverse_precon && !lanier_multizone_schur_precon && LanierComplexSymmetricMethod(method_in);
#endif
	}
#ifdef ADDA_CUDA
	else lanier_cs_congruence=false;
#endif
	// initialize data required for checkpoints and specific variables
	chp_exit=false;
	complete=true;
	if (params[ind_m].sc_N>0)
		scalars=(chp_data *)voidVector(params[ind_m].sc_N*sizeof(chp_data),ALL_POS,"list of scalars");
	else scalars=NULL;
	if (params[ind_m].vec_N>0)
		vectors=(chp_data *)voidVector(params[ind_m].vec_N*sizeof(chp_data),ALL_POS,"list of scalars");
	else vectors=NULL;
	(*params[ind_m].func)(PHASE_VARS);
	// load checkpoint, if needed, and finish initialization of the iterative solver
	if (load_chpoint) LoadIterChpoint();
#ifdef ADDA_CUDA
	/* Initial-field/checkpoint construction above is still host-side. Upload the
	 * complete iterative-solver state once, immediately before the solver starts using GPU vectors. */
	if (method_in==IT_BICGSTAB4 || method_in==IT_BICGSTAB8 || method_in==IT_BICGSTAB12 || method_in==IT_GPBICGSTAB4) {
		const size_t extra=(size_t)params[ind_m].vec_N;
		/* Reliable true-residual control needs Avecbuffer as a resident scratch
		 * vector for both L=4 solvers. GPBiCGStab(4) already required it;
		 * BiCGStab(4) now registers the same scratch explicitly. */
		const size_t work_extra=1u;
		const size_t count=3u+extra+work_extra;
		const void *ids[32];
		size_t ci=0,k;
		ids[ci++]=xvec; ids[ci++]=rvec; ids[ci++]=pvec;
		for (k=0;k<extra;k++) ids[ci++]=vectors[k].ptr;
		ids[ci++]=Avecbuffer;
		CudaIterInitList(ids,count,method_in==IT_BICGSTAB4 ? "BiCGStab(4)" : (method_in==IT_BICGSTAB8 ? "BiCGStab(8)" : (method_in==IT_BICGSTAB12 ? "BiCGStab(12)" : "GPBiCGStab(4)")));
	}
	else CudaIterInit((int)method_in);
	if (lanier_precon) {
		CudaLanierInit();
		if (lanier_full_precon && !(lanier_physical_rhs_norm2>0.0))
			lanier_physical_rhs_norm2=PhysicalRHSNorm2();
		/* Any cached CalcInitField MatVec is the unpreconditioned physical A
		 * product and must not be reused by a preconditioned recurrence. */
		matvec_ready=false;
		if (lanier_cs_congruence && !load_chpoint) {
			/* Complex-symmetric solvers run on P*A*P. Keep xvec physical, but
			 * transform r0 -> P*r0 and scale convergence by ||P*b||. */
			CudaLanierApply(rvec,rvec);
			CudaLanierApply(pvec,Avecbuffer);
			inprodR=CudaIterNorm2(rvec,&Timing_InitIterComm);
			temp=CudaIterNorm2(Avecbuffer,&Timing_InitIterComm);
			if (temp==0) LogError(ONE_POS,"Lanier complex-symmetric transformed right-hand side has zero norm");
			resid_scale=1/temp;
			epsB=iter_eps*iter_eps*temp;
			if (IFROOT) PrintBoth(logfile,
				lanier_multizone_schwarz_sym_precon ? "LANIER_MULTIZONE_SCHWARZ_SYM solver integration: symmetric multiplicative Schwarz right preconditioning A*P_SMS.\n" : (lanier_multizone_schwarz_precon ? "LANIER_MULTIZONE_SCHWARZ solver integration: fixed-order multiplicative Schwarz right preconditioning A*P_MS.\n" : (lanier_multizone_precon ? "LANIER_MULTIZONE solver integration: N-zone block-Jacobi complex-symmetric congruence P*A*P with residual P*(b-Ax).\n" :
				(lanier_nested_precon ? "LANIER_NESTED solver integration: block-Jacobi complex-symmetric congruence P*A*P with residual P*(b-Ax).\n" :
				"LANIER_FULL solver integration: complex-symmetric congruence P*A*P with residual P*(b-Ax).\n"))));
		}
	}
#endif
	if (!load_chpoint && IFROOT) {
		prev_err=sqrt(resid_scale*inprodR);
		SnprintfErr(ONE_POS,tmp_str,MAX_LINE,RESID_STRING"\n",0,prev_err);
		if (!orient_avg) fprintf(logfile,"%s\n%s",init_descr,tmp_str);
		PRINTFB("%s\n%s",init_descr,tmp_str);
	}
	(*params[ind_m].func)(PHASE_INIT);
#ifdef ADDA_CUDA
	if (lanier_full_precon) {
		lanier_reset_baseline=sqrt(MAX(0.0,resid_scale*inprodR));
		if (IFROOT) PrintBoth(logfile,
			lanier_multizone_schur_precon ? "LANIER_MULTIZONE_SCHUR DDSCAT reset policy: mandatory reset after iteration 3; ratio R=100; true physical residual every 20 iterations; vector-gap restart >= 0.01.\n" :
			(lanier_multizone_schwarz_reverse_precon ? "LANIER_MULTIZONE_SCHWARZ_REVERSE DDSCAT reset policy: mandatory reset after iteration 3; ratio R=100; true physical residual every 20 iterations; vector-gap restart >= 0.01.\n" :
			(lanier_multizone_schwarz_sym_precon ? "LANIER_MULTIZONE_SCHWARZ_SYM DDSCAT reset policy: mandatory reset after iteration 3; ratio R=100; true physical residual every 20 iterations; vector-gap restart >= 0.01.\n" :
			(lanier_multizone_schwarz_precon ? "LANIER_MULTIZONE_SCHWARZ DDSCAT reset policy: mandatory reset after iteration 3; ratio R=100; true physical residual every 20 iterations; vector-gap restart >= 0.01.\n" :
			(lanier_multizone_precon ? "LANIER_MULTIZONE DDSCAT reset policy: mandatory reset after iteration 3; ratio R=100; true physical residual every 20 iterations; vector-gap restart >= 0.01.\n" :
			(lanier_nested_precon ? "LANIER_NESTED DDSCAT reset policy: mandatory reset after iteration 3; ratio R=100; true physical residual every 20 iterations; vector-gap restart >= 0.01.\n" :
			"LANIER_FULL DDSCAT reset policy: mandatory reset after iteration 3; ratio R=100; true physical residual every 20 iterations; vector-gap restart >= 0.01.\n"))))));
	}
#endif
	// Initialization time includes generating the incident beam
	Timing_InitIter = GET_TIME() - tstart;
	Timing_InitIterComm += Timing_MVPComm; // Timing_MVPComm should (by here) include only iteration initialization
	Timing_IntFieldOneComm=Timing_InitIterComm;
#ifdef ADDA_CUDA
	/* Snapshot the actual GPU footprint after solver PHASE_INIT and immediately
	 * before entering the iterative loop. */
	CudaIterPrintMemoryBeforeLoop();
#endif
	// main iteration cycle
	/* For Lanier complex-symmetric congruence solvers the recurrence residual
	 * lives in the transformed space P*(b-Ax).  That norm may fall below its
	 * transformed tolerance while the physical residual ||b-Ax||/||b|| is still
	 * above iter_eps.  Therefore transformed inprodR must never terminate a
	 * congruence solve.  ReliableResidualCheck() is the sole convergence authority
	 * for that path and sets lanier_true_converged only after a physical check. */
#ifdef ADDA_CUDA
	while ((lanier_cs_congruence ? !lanier_true_converged :
		(inprodR>epsB && !lanier_true_converged)) &&
		niter<=maxiter && counter<=params[ind_m].mc && !chp_exit) {
#else
	while (inprodR>epsB && niter<=maxiter && counter<=params[ind_m].mc && !chp_exit) {
#endif
		// initialize time
		Timing_OneIterComm=Timing_OneIterMVP=Timing_OneIterMVPComm=0;
		tstart=GET_TIME();
		// main execution
#ifdef ADDA_CUDA
		lanier_recursive_residual_vec=rvec; // CSYM overrides this with its reconstructed r_k
#endif
		(*params[ind_m].func)(PHASE_ITER);
#ifdef ADDA_CUDA
		if (lanier_full_precon) {
			const double recursive_norm2=inprodRp1;
			const double recursive_err=sqrt(MAX(0.0,resid_scale*recursive_norm2));
			const bool mandatory=(niter==3 && niter<maxiter);
			const bool ratio_reset=(niter>3 && niter<maxiter && lanier_reset_baseline>0.0 &&
				recursive_err < lanier_reset_baseline/LANIER_RESET_RATIO);
			const bool periodic=((niter%LANIER_RELIABLE_PERIOD)==0);
			const bool recursive_claim=(recursive_norm2<=epsB);
			if (mandatory || ratio_reset || periodic || recursive_claim) {
				const bool force_restart=(mandatory || ratio_reset);
				const reliable_residual_result rr=ReliableResidualCheck(method_in,recursive_norm2,
					LANIER_RELIABLE_GAP_TOL,force_restart);
				const char *reason=mandatory ? "mandatory-after-3" :
					(ratio_reset ? "ratio-R100" : (periodic ? "periodic-20" : "recursive-convergence"));
				if (IFROOT) PrintBoth(logfile,
					"%s %s reliable residual at iteration %d: reason=%s recursive="EFORM
					", physical_true="EFORM", gap="GFORM"%s\n",
					LanierMethodName(method_in),lanier_multizone_schur_precon ? "LANIER_MULTIZONE_SCHUR" : (lanier_multizone_schwarz_reverse_precon ? "LANIER_MULTIZONE_SCHWARZ_REVERSE" : (lanier_multizone_schwarz_sym_precon ? "LANIER_MULTIZONE_SCHWARZ_SYM" : (lanier_multizone_schwarz_precon ? "LANIER_MULTIZONE_SCHWARZ" : (lanier_multizone_precon ? "LANIER_MULTIZONE" : (lanier_nested_precon ? "LANIER_NESTED" : "LANIER_FULL"))))),niter,reason,recursive_err,rr.physical_rel,rr.gap,
					rr.true_converged ? "; TRUE CONVERGENCE" : (rr.restart ? "; RESET" : "; continue"));
				if (rr.true_converged) {
					lanier_true_converged=true;
				}
				else if (rr.restart) {
					RestartKrylovFromTrueResidual(method_in,rr.recurrence_true_norm2);
					lanier_reset_baseline=sqrt(MAX(0.0,resid_scale*rr.recurrence_true_norm2));
					if (mandatory) lanier_reset_mandatory_count++;
					else if (ratio_reset) lanier_reset_ratio_count++;
					else lanier_reset_reliable_count++;
				}
			}
		}
		/* -recalc_resid is also an in-iteration reliability policy in ADDA-CUDA.
		 * For every CUDA solver, independently recompute the physical residual every
		 * RECALC_RELIABLE_PERIOD iterations and whenever recursive convergence is
		 * claimed. If the vector residual gap is too large, restart the Krylov
		 * recurrence from the independently recomputed true residual.
		 *
		 * LANIER_FULL/NESTED/MULTIZONE use the stronger policy above (mandatory
		 * iteration-3 and ratio-R100 resets in addition to the same periodic idea),
		 * so this generic branch is reached only for the remaining configurations. */
		else if (recalc_resid &&
			((niter%RECALC_RELIABLE_PERIOD)==0 || inprodRp1<=epsB)) {
			const double recursive_norm2=inprodRp1;
			const reliable_residual_result rr=ReliableResidualCheck(method_in,recursive_norm2,
				RECALC_RELIABLE_GAP_TOL,false);
			if (IFROOT) {
				const double recursive_err=sqrt(MAX(0.0,resid_scale*recursive_norm2));
				const char *reason=((niter%RECALC_RELIABLE_PERIOD)==0 ? "periodic-20" : "recursive-convergence");
				PrintBoth(logfile,"%s -recalc_resid check at iteration %d: reason=%s recursive="EFORM
					", physical_true="EFORM", gap="GFORM"%s\n",LanierMethodName(method_in),niter,reason,
					recursive_err,rr.physical_rel,rr.gap,
					rr.true_converged ? "; TRUE CONVERGENCE" : (rr.restart ? "; RESET" : "; continue"));
			}
			if (rr.restart) RestartKrylovFromTrueResidual(method_in,rr.recurrence_true_norm2);
			else if (rr.true_converged) {
				inprodRp1=rr.recurrence_true_norm2;
				inprodR=rr.recurrence_true_norm2;
			}
		}
#if defined(ADDA_SINGLE)
		else if (reliable_resid && (method_in==IT_GPBICGSTAB2 || method_in==IT_BCGS2) &&
			((niter%20)==0 || inprodRp1<=epsB)) {
			const double recursive_norm2=inprodRp1;
			const reliable_residual_result rr=ReliableResidualCheck(method_in,recursive_norm2,
				RELIABLE_GAP_TOL,reliable_resid_force_restart);
			if (IFROOT) {
				const char *method_name=(method_in==IT_GPBICGSTAB2 ? "GPBiCGStab(2)" : "BiCGStab(2)/BCGS2");
				const double recursive_err=sqrt(MAX(0.0,resid_scale*recursive_norm2));
				PrintBoth(logfile,"%s reliable residual at iteration %d: recursive="EFORM
					", true="EFORM", gap="GFORM"%s\n",method_name,niter,recursive_err,rr.physical_rel,rr.gap,
					rr.true_converged ? "; true residual confirms convergence" : (rr.restart ? "; restarted" : "; continuing"));
			}
			if (rr.restart) RestartKrylovFromTrueResidual(method_in,rr.recurrence_true_norm2);
			else if (rr.true_converged) { inprodRp1=rr.recurrence_true_norm2; inprodR=rr.recurrence_true_norm2; }
		}
#endif
#endif
		// finalize time; time for incomplete iteration may be inadequate
		Timing_OneIterComm+=Timing_OneIterMVPComm;
		Timing_IntFieldOneComm+=Timing_OneIterComm;
		Timing_MVP+=Timing_OneIterMVP;
		Timing_MVPComm+=Timing_OneIterMVPComm;
		if (complete) {
			Timing_OneIter=GET_TIME()-tstart;
			time_tmp=Timing_OneIterComm;
			time_tmp2=Timing_OneIterMVP;
			time_tmp3=Timing_OneIterMVPComm;
		}
		// use result from the previous iteration (assumed to be available by this time)
		else {
			Timing_OneIterComm=time_tmp;
			Timing_OneIterMVP=time_tmp2;
			Timing_OneIterMVPComm=time_tmp3;
		}
		/* check progress; it takes negligible time by itself (O(1) operations), but may lead to saving checkpoint.
		 * Since the latter is not relevant to the iteration itself, the ProgressReport is called after finalizing the
		 * time of a single iteration.
		 */
		ProgressReport();
	}
	// Save checkpoint of type always
	if (chp_type==CHP_ALWAYS && !chp_exit) SaveIterChpoint();
#ifdef ADDA_CUDA
	if (lanier_full_precon && IFROOT) PrintBoth(logfile,
		lanier_multizone_schur_precon ? "LANIER_MULTIZONE_SCHUR DDSCAT reset summary: mandatory=%d ratio-R100=%d reliable-gap=%d; period=%d gap_threshold="GFORM".\n" :
		(lanier_multizone_schwarz_reverse_precon ? "LANIER_MULTIZONE_SCHWARZ_REVERSE DDSCAT reset summary: mandatory=%d ratio-R100=%d reliable-gap=%d; period=%d gap_threshold="GFORM".\n" :
		(lanier_multizone_schwarz_sym_precon ? "LANIER_MULTIZONE_SCHWARZ_SYM DDSCAT reset summary: mandatory=%d ratio-R100=%d reliable-gap=%d; period=%d gap_threshold="GFORM".\n" :
		(lanier_multizone_schwarz_precon ? "LANIER_MULTIZONE_SCHWARZ DDSCAT reset summary: mandatory=%d ratio-R100=%d reliable-gap=%d; period=%d gap_threshold="GFORM".\n" :
		(lanier_multizone_precon ? "LANIER_MULTIZONE DDSCAT reset summary: mandatory=%d ratio-R100=%d reliable-gap=%d; period=%d gap_threshold="GFORM".\n" :
		(lanier_nested_precon ? "LANIER_NESTED DDSCAT reset summary: mandatory=%d ratio-R100=%d reliable-gap=%d; period=%d gap_threshold="GFORM".\n" :
		"LANIER_FULL DDSCAT reset summary: mandatory=%d ratio-R100=%d reliable-gap=%d; period=%d gap_threshold="GFORM".\n"))))),
		lanier_reset_mandatory_count,lanier_reset_ratio_count,lanier_reset_reliable_count,
		LANIER_RELIABLE_PERIOD,(double)LANIER_RELIABLE_GAP_TOL);
	/* The rest of IterativeSolver (optional residual recalculation and field
	 * post-processing) consumes host arrays. One final D2H synchronization is
	 * therefore required when an iterative solver is CUDA-resident. */
	CudaIterSyncToHost();
	/* Solver-only vectors are no longer needed after the final D2H sync. Free
	 * them now; d_arg/d_result remain available for CPU-facing MatVec(), e.g.
	 * the optional residual recalculation below. */
	if (lanier_precon) CudaLanierRelease();
	CudaIterRelease();
#endif
	/* process incomplete convergence
	 * Since maxiter can be used in several reasonable ways, e.g. to control execution time, we allow calculation of
	 * (potentially inaccurate) scattering quantities, when it is reached. We leave the warning although it may be
	 * redundant in e.g. scattering-order-formulation controlled by maxiter.
	 * By contrast, stagnation (especially, e.g. for CGNR) is rarely something that should be tolerated. In principle,
	 * this may happen in the end of a long run at already low residual. However, to account for such cases one should
	 * better use maxiter.
	 */
#ifdef ADDA_CUDA
	if (lanier_cs_congruence ? !lanier_true_converged : inprodR>epsB) {
#else
	if (inprodR>epsB) {
#endif
		if (niter>maxiter) LogWarning(EC_WARN,ONE_POS,"Iterations haven't converged in %d iterations. Further "
			"calculated scattering quantities may be less accurate.",maxiter);
		else if (counter>params[ind_m].mc) LogError(ONE_POS,"Residual norm haven't decreased for maximum allowed "
			"number of iterations (%d)",params[ind_m].mc);
	}
	/* Final physical-residual validation is mandatory in every single-precision
	 * build. In double precision it remains enabled by -recalc_resid. */
#if defined(ADDA_SINGLE)
	if (1) {
#else
	if (recalc_resid) {
#endif
		double report_resid_scale=resid_scale;
#ifdef ADDA_CUDA
		/* Complex-symmetric Lanier solvers converge in the congruence residual
		 * norm ||P(b-Ax)||/||Pb||.  -recalc_resid must nevertheless retain
		 * ADDA's normal physical definition ||b-Ax||/||b|| so results remain
		 * directly comparable with NONE and with the right-preconditioned solvers. */
		if (lanier_cs_congruence) {
			nMult_mat(Avecbuffer,Einc,cc_sqrt);
			temp=nNorm2(Avecbuffer,&Timing_IntFieldOneComm);
			if (temp==0) LogError(ONE_POS,"Physical right-hand side has zero norm during residual recalculation");
			report_resid_scale=1/temp;
		}
#endif
		inprodR=ResidualNorm2(xvec,rvec,Avecbuffer,&Timing_MVP,&Timing_MVPComm,&Timing_IntFieldOneComm);
		if (IFROOT) {
			temp=sqrt(report_resid_scale*inprodR);
			SnprintfErr(ONE_POS,tmp_str,MAX_LINE,"Final (recalculated) residual norm: "EFORM"\n",temp);
			if (!orient_avg) fprintf(logfile,"%s",tmp_str);
			PRINTFB("%s",tmp_str);
		}
	}
	if (method_in==IT_BICGSTAB4 || method_in==IT_BICGSTAB8 || method_in==IT_BICGSTAB12 || method_in==IT_GPBICGSTAB4) FreeCPUL4Workspace();
	// post-processing
	if (params[ind_m].sc_N>0) Free_general(scalars);
	if (params[ind_m].vec_N>0) Free_general(vectors);
	/* x is a solution of a modified system, not exactly internal field; should not be used further except for adaptive
	 * technique (as starting vector for next system)
	 */
	nMult_mat(pvec,xvec,cc_sqrt); // p now contains polarizations. Can be used to calculate e.g. scattered field faster.
	if (chp_exit) return CHP_EXIT; // check if exiting after checkpoint
	return (niter-1); // the number of iterations elapsed
}

#ifdef ADDA_CUDA
/* ---------------- LANIER_PARTITION_V2 ----------------
 * Article-style separated-material partition sweep (Fig. 3):
 *   solve object 1 -> scatter into object 2 -> solve object 2 -> reverse,
 *   reusing previous transformed polarizations as warm starts; combine the
 *   regional solutions and finally solve the complete physical problem.
 *
 * V2 keeps one regional FULL6 conditioner resident at a time and uses
 * the user-selected iterative method for every regional and final solve.
 * Multi-plan cuFFT caching remains deferred.
 */
static double PartitionRelChange(const doublecomplex *a,const doublecomplex *b,const size_t n)
{
    size_t i; double d=0,den=0;
    for(i=0;i<n;i++) {
        const double dr=creal(a[i]-b[i]),di=cimag(a[i]-b[i]);
        const double ar=creal(a[i]),ai=cimag(a[i]);
        d+=dr*dr+di*di; den+=ar*ar+ai*ai;
    }
    if(den==0) return d==0 ? 0 : HUGE_VAL;
    return sqrt(d/den);
}

static int PartitionEnvInt(const char *name,const int defv,const int lo,const int hi)
{
    const char *e=getenv(name); char *end=NULL; long v;
    if(e==NULL || e[0]=='\0') return defv;
    v=strtol(e,&end,10);
    if(end==e || *end!='\0' || v<lo || v>hi) {
        LogWarning(EC_WARN,ONE_POS,"Ignoring invalid %s=%s; using %d",name,e,defv);
        return defv;
    }
    return (int)v;
}

static double PartitionEnvDouble(const char *name,const double defv)
{
    const char *e=getenv(name); char *end=NULL; double v;
    if(e==NULL || e[0]=='\0') return defv;
    v=strtod(e,&end);
    if(end==e || *end!='\0' || !(v>0) || !isfinite(v)) {
        LogWarning(EC_WARN,ONE_POS,"Ignoring invalid %s=%s; using "GFORM,name,e,defv);
        return defv;
    }
    return v;
}

static void PartitionRejectTouchingMaterials(void)
{
    size_t nbox,xy,i; unsigned char *occ;
    if(boxX<=0 || boxY<=0 || boxZ<=0) LogError(ONE_POS,"LANIER_PARTITION invalid physical box");
    xy=(size_t)boxX*(size_t)boxY;
    if(xy/(size_t)boxY!=(size_t)boxX || xy>SIZE_MAX/(size_t)boxZ)
        LogError(ONE_POS,"LANIER_PARTITION occupancy-map size overflow");
    nbox=xy*(size_t)boxZ;
    occ=(unsigned char*)malloc(nbox);
    if(occ==NULL) LogError(ONE_POS,"Insufficient host memory validating LANIER_PARTITION separation");
    memset(occ,0xff,nbox);
    for(i=0;i<local_nvoid_Ndip;i++) {
        const size_t p=3*i,x=position[p],y=position[p+1],z=position[p+2];
        occ[(z*(size_t)boxY+y)*(size_t)boxX+x]=material[i];
    }
    /* Reject face/edge/corner contact. The published method is explicitly for
     * separated regions and is known to become unstable at very small gaps. */
    for(i=0;i<local_nvoid_Ndip;i++) {
        const size_t p=3*i,x=position[p],y=position[p+1],z=position[p+2];
        const unsigned char m=material[i];
        int dx,dy,dz;
        for(dz=-1;dz<=1;dz++)for(dy=-1;dy<=1;dy++)for(dx=-1;dx<=1;dx++) {
            size_t qx,qy,qz; unsigned char q;
            if(dx==0&&dy==0&&dz==0) continue;
            if((dx<0&&x==0)||(dy<0&&y==0)||(dz<0&&z==0)) continue;
            qx=(size_t)((long)x+dx); qy=(size_t)((long)y+dy); qz=(size_t)((long)z+dz);
            if(qx>=(size_t)boxX||qy>=(size_t)boxY||qz>=(size_t)boxZ) continue;
            q=occ[(qz*(size_t)boxY+qy)*(size_t)boxX+qx];
            if(q!=0xff && q!=m) {
                free(occ);
                LogError(ONE_POS,
                    "LANIER_PARTITION requires separated material regions; materials %u and %u touch within one lattice cell near (%zu,%zu,%zu). Use lanier_full or a future nested/contact formulation instead.",
                    (unsigned)m+1,(unsigned)q+1,x,y,z);
            }
        }
    }
    free(occ);
}

static void PartitionBuildExcitation(const int active,const doublecomplex *E0,
                                     const doublecomplex *scat,const int *mats,const int nm,
                                     const size_t nrows)
{
    size_t i; int k;
    (void)nrows;
    for(i=0;i<local_nvoid_Ndip;i++) {
        const size_t p=3*i;
        if((int)material[i]!=active) {
            Einc[p]=Einc[p+1]=Einc[p+2]=0;
            continue;
        }
        for(int c=0;c<3;c++) {
            doublecomplex e=E0[p+(size_t)c];
            for(k=0;k<nm;k++) if(mats[k]!=active)
                e += scat[(size_t)k*nrows+p+(size_t)c];
            Einc[p+(size_t)c]=e;
        }
    }
}

static int PartitionMaterialSlot(const int *mats,const int nm,const int mat)
{
    int k; for(k=0;k<nm;k++) if(mats[k]==mat) return k;
    return -1;
}

static int LanierPartitionLocalSolve(const int active,const int slot,const int *mats,const int nm,
                                     const doublecomplex *E0,doublecomplex *combined,doublecomplex *scat,
                                     const size_t nrows,const enum iter method_in,const enum incpol which)
{
    size_t i; int its; const bool save_lp=lanier_precon,save_lf=lanier_full_precon;
    const bool save_local=lanier_partition_local_mode,save_warm=lanier_partition_use_existing_x;
    const int save_active=lanier_partition_active_material;

    PartitionBuildExcitation(active,E0,scat,mats,nm,nrows);
    /* Warm start only with this object's previous transformed solution. */
    for(i=0;i<local_nvoid_Ndip;i++) {
        const size_t p=3*i;
        if((int)material[i]==active) {
            xvec[p]=combined[p];xvec[p+1]=combined[p+1];xvec[p+2]=combined[p+2];
        } else xvec[p]=xvec[p+1]=xvec[p+2]=0;
    }

    lanier_partition_local_mode=true;
    lanier_partition_active_material=active;
    lanier_partition_use_existing_x=true;
    lanier_precon=true;
    lanier_full_precon=true;
    CudaLanierPartitionProjection(active);
    if(IFROOT) PrintBoth(logfile,"LANIER_PARTITION: solve material %d with %s + regional FULL6, warm-start enabled.\n",
                           active+1,LanierMethodName(method_in));
    its=IterativeSolverCore(method_in,which);
    CudaLanierPartitionProjection(-1);
    if(its==CHP_EXIT) LogError(ONE_POS,"Checkpoints are not supported inside LANIER_PARTITION regional sweeps");

    /* xvec is the transformed regional solution. Update the combined vector. */
    for(i=0;i<local_nvoid_Ndip;i++) if((int)material[i]==active) {
        const size_t p=3*i;
        combined[p]=xvec[p];combined[p+1]=xvec[p+1];combined[p+2]=xvec[p+2];
    }

    /* Full, unprojected field from this object. For A~=I+SDS and p=S*y,
     * E_scat=-D*p=S^-1(y-A~y). */
    MatVec(xvec,Avecbuffer,NULL,false,&Timing_MVP,&Timing_MVPComm);
    for(i=0;i<local_nvoid_Ndip;i++) {
        const size_t p=3*i,tm=(size_t)material[i];
        for(int c=0;c<3;c++)
            scat[(size_t)slot*nrows+p+(size_t)c]=(xvec[p+(size_t)c]-Avecbuffer[p+(size_t)c])/
                                                     cc_sqrt[tm][c];
    }

    lanier_precon=save_lp;lanier_full_precon=save_lf;
    lanier_partition_local_mode=save_local;
    lanier_partition_active_material=save_active;
    lanier_partition_use_existing_x=save_warm;
    return its;
}

static double PartitionGlobalResidual(const doublecomplex *E0,doublecomplex *combined,const size_t nrows)
{
    double b2,r2;
    memcpy(Einc,E0,nrows*sizeof(doublecomplex));
    memcpy(xvec,combined,nrows*sizeof(doublecomplex));
    nMult_mat(pvec,Einc,cc_sqrt);
    b2=nNorm2(pvec,&Timing_IntFieldOneComm);
    r2=ResidualNorm2(xvec,rvec,Avecbuffer,&Timing_MVP,&Timing_MVPComm,&Timing_IntFieldOneComm);
    return b2>0 ? sqrt(r2/b2) : HUGE_VAL;
}

static int LanierPartitionSolver(const enum iter method_in,const enum incpol which)
{
    const size_t nrows=3*local_nvoid_Ndip;
    int mats[MAX_NMAT],nm=0,m,k,sweep,final_its;
    int max_sweeps=PartitionEnvInt("ADDA_LANIER_PARTITION_MAX_SWEEPS",50,1,10000);
    double part_tol=PartitionEnvDouble("ADDA_LANIER_PARTITION_TOL",iter_eps);
    doublecomplex *E0,*combined,*prev,*scat;
    size_t start_total=TotalIter;
    double rel_change=HUGE_VAL,global_resid=HUGE_VAL;
    const bool save_lp=lanier_precon,save_lf=lanier_full_precon;
    const bool save_local=lanier_partition_local_mode,save_warm=lanier_partition_use_existing_x;
    const int save_active=lanier_partition_active_material;

    if(Nmat<2) LogError(ONE_POS,"LANIER_PARTITION requires at least two material domains");
    if(surface) LogError(ONE_POS,"LANIER_PARTITION does not support -surf");
    if(rectDip) LogError(ONE_POS,"LANIER_PARTITION currently requires cubic voxels");
    if(load_chpoint || chp_type!=CHP_NONE)
        LogError(ONE_POS,"LANIER_PARTITION V2 does not support loading/saving iterative checkpoints");
    PartitionRejectTouchingMaterials();
    for(m=0;m<Nmat;m++) {
        bool found=false; size_t i;
        for(i=0;i<local_nvoid_Ndip;i++) if((int)material[i]==m){found=true;break;}
        if(found) mats[nm++]=m;
    }
    if(nm<2) LogError(ONE_POS,"LANIER_PARTITION found fewer than two occupied material regions");

    E0=(doublecomplex*)malloc(nrows*sizeof(doublecomplex));
    combined=(doublecomplex*)calloc(nrows,sizeof(doublecomplex));
    prev=(doublecomplex*)malloc(nrows*sizeof(doublecomplex));
    if((size_t)nm>SIZE_MAX/nrows || (size_t)nm*nrows>SIZE_MAX/sizeof(doublecomplex))
        LogError(ONE_POS,"LANIER_PARTITION scattered-field allocation overflow");
    scat=(doublecomplex*)calloc((size_t)nm*nrows,sizeof(doublecomplex));
    if(E0==NULL||combined==NULL||prev==NULL||scat==NULL) {
        free(E0);free(combined);free(prev);free(scat);
        LogError(ONE_POS,"Insufficient host memory for LANIER_PARTITION work vectors");
    }
    memcpy(E0,Einc,nrows*sizeof(doublecomplex));

    if(IFROOT) PrintBoth(logfile,
        "LANIER_PARTITION V2: %d separated material regions; regional solver=%s+FULL6; final solver=%s; max_roundtrips=%d; partition_tol="GFORM".\n"
        "LANIER_PARTITION V2 uses the -iter solver for every regional solve and the final full-system solve. One regional cuFFT conditioner is resident at a time; regional GPU-plan caching is deferred.\n",
        nm,LanierMethodName(method_in),LanierMethodName(method_in),max_sweeps,part_tol);

    lanier_partition_internal_call=true;
    /* First forward pass: incident field -> object 1 -> object 2 -> ... */
    for(k=0;k<nm;k++) {
        const int slot=PartitionMaterialSlot(mats,nm,mats[k]);
        LanierPartitionLocalSolve(mats[k],slot,mats,nm,E0,combined,scat,nrows,method_in,which);
    }
    global_resid=PartitionGlobalResidual(E0,combined,nrows);
    if(IFROOT) PrintBoth(logfile,"LANIER_PARTITION initial forward pass: full physical residual="EFORM".\n",global_resid);

    for(sweep=1;sweep<=max_sweeps && global_resid>iter_eps;sweep++) {
        memcpy(prev,combined,nrows*sizeof(doublecomplex));
        /* Reverse pass excludes the last object already solved at the end of
         * the forward pass; forward pass excludes object 0 solved at reverse end. */
        for(k=nm-2;k>=0;k--) {
            const int slot=PartitionMaterialSlot(mats,nm,mats[k]);
            LanierPartitionLocalSolve(mats[k],slot,mats,nm,E0,combined,scat,nrows,method_in,which);
        }
        for(k=1;k<nm;k++) {
            const int slot=PartitionMaterialSlot(mats,nm,mats[k]);
            LanierPartitionLocalSolve(mats[k],slot,mats,nm,E0,combined,scat,nrows,method_in,which);
        }
        rel_change=PartitionRelChange(combined,prev,nrows);
        global_resid=PartitionGlobalResidual(E0,combined,nrows);
        if(IFROOT) PrintBoth(logfile,
            "LANIER_PARTITION roundtrip %d: relative solution change="EFORM", full physical residual="EFORM".\n",
            sweep,rel_change,global_resid);
        if(rel_change<=part_tol) {
            if(IFROOT) PrintBoth(logfile,"LANIER_PARTITION regional fixed point reached at roundtrip %d.\n",sweep);
            break;
        }
    }

    /* Fig. 3: Combine Solutions -> Solve Full Problem. Use the combined
     * transformed solution as x0 for the user's selected final solver, with
     * the physical full operator and no single global Lanier conditioner. */
    memcpy(Einc,E0,nrows*sizeof(doublecomplex));
    memcpy(xvec,combined,nrows*sizeof(doublecomplex));
    CudaLanierPartitionProjection(-1);
    lanier_partition_local_mode=false;
    lanier_partition_active_material=-1;
    lanier_precon=false;
    lanier_full_precon=false;
    lanier_partition_use_existing_x=true;
    if(IFROOT) PrintBoth(logfile,
        "LANIER_PARTITION: Combine Solutions -> Solve Full Problem with %s warm start; preconditioner=none.\n",
        LanierMethodName(method_in));
    final_its=IterativeSolverCore(method_in,which);

    if(IFROOT) PrintBoth(logfile,
        "LANIER_PARTITION summary: regional+final Krylov iterations=%zu; final full-solve iterations=%d; last partition residual="EFORM".\n",
        TotalIter-start_total,final_its,global_resid);

    lanier_precon=save_lp;lanier_full_precon=save_lf;
    lanier_partition_local_mode=save_local;
    lanier_partition_active_material=save_active;
    lanier_partition_use_existing_x=save_warm;
    lanier_partition_internal_call=false;
    free(E0);free(combined);free(prev);free(scat);
    return final_its;
}
#endif /* ADDA_CUDA */

int IterativeSolver(const enum iter method_in,const enum incpol which)
{
#ifdef ADDA_CUDA
    if(lanier_partition_precon && !lanier_partition_internal_call)
        return LanierPartitionSolver(method_in,which);
#else
    if(lanier_partition_precon)
        LogError(ONE_POS,"LANIER_PARTITION requires an ADDA CUDA executable");
#endif
    return IterativeSolverCore(method_in,which);
}
