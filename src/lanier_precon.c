/* ADDA_LANIER_REFERENCE15X_V4
 * Reference homogeneous three-level circulant preconditioner for ADDA-CUDA.
 *
 * The construction is intentionally done with ADDA's own InterTerm_int()
 * rather than a duplicate CUDA Green formula.  This keeps the preconditioner
 * consistent with the selected ADDA interaction prescription and sign
 * convention.  The one-time CPU build uses one full complex component plus
 * six reduced spectral octants; application remains entirely on the GPU.
 */
#include "const.h" /* keep this first */
#include "lanier_precon.h"

#include "interaction.h"
#include "io.h"
#include "vars.h"

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <fftw3.h>

#ifdef ADDA_SINGLE
# define LPLAN fftwf_plan
# define LPLAN_DFT_3D fftwf_plan_dft_3d
# define LEXECUTE fftwf_execute
# define LDESTROY fftwf_destroy_plan
# define LMALLOC fftwf_malloc
# define LFREE fftwf_free
# define LCOMPLEX fftwf_complex
# define LFORWARD FFTW_FORWARD
#else
# define LPLAN fftw_plan
# define LPLAN_DFT_3D fftw_plan_dft_3d
# define LEXECUTE fftw_execute
# define LDESTROY fftw_destroy_plan
# define LMALLOC fftw_malloc
# define LFREE fftw_free
# define LCOMPLEX fftw_complex
# define LFORWARD FFTW_FORWARD
#endif

static size_t LIndex(const size_t x,const size_t y,const size_t z,
                     const size_t nx,const size_t ny)
{
    return (z*ny+y)*nx+x;
}

static size_t LAuxDim(const size_t n,const double expansion)
{
    size_t v;
    if (expansion<=1.0000001) {
        v=n;
    }
    else if (fabs(expansion-1.5)<1.0e-12) {
        /* Same direct 1.5x convention as the validated DDSCAT Lanier path:
         * ceil(3*n/2), then round upward to an even transform length. */
        if (n > (SIZE_MAX-1)/3)
            LogError(ONE_POS,"Lanier auxiliary dimension overflow");
        v=(3*n+1)/2;
    }
    else {
        const double q=ceil(expansion*(double)n);
        if (!(q>=1.0) || q>(double)SIZE_MAX)
            LogError(ONE_POS,"Invalid Lanier expansion factor %.17g",expansion);
        v=(size_t)q;
    }
    if (v<n) v=n;
    if (v&1u) {
        if (v==SIZE_MAX) LogError(ONE_POS,"Lanier auxiliary dimension overflow");
        v++;
    }
    return v;
}

static double LParity(const int comp,const int axis)
{
    /* component order: xx,xy,xz,yy,yz,zz */
    static const signed char parity[3][6]={
        {+1,-1,-1,+1,+1,+1}, /* x reflection */
        {+1,-1,+1,+1,-1,+1}, /* y reflection */
        {+1,+1,-1,+1,-1,+1}  /* z reflection */
    };
    return (double)parity[axis][comp];
}

static void AverageX(doublecomplex *a,const int comp,
                     const size_t nx,const size_t ny,const size_t nz)
{
    const size_t pts=(nx-1)/2;
    const double s=LParity(comp,0);
    size_t z,y,x;
    for (z=0;z<nz;z++) for (y=0;y<ny;y++) {
        for (x=1;x<=pts;x++) {
            const double w1=(double)x/(double)nx,w0=1.0-w1;
            const size_t ip=LIndex(x,y,z,nx,ny), im=LIndex(nx-x,y,z,nx,ny);
            const doublecomplex v=w0*a[ip]+(s*w1)*a[im];
            a[ip]=v;
            a[im]=s*v;
        }
        if ((nx%2)==0 && s<0) a[LIndex(nx/2,y,z,nx,ny)]=0;
    }
}

static void AverageY(doublecomplex *a,const int comp,
                     const size_t nx,const size_t ny,const size_t nz)
{
    const size_t pts=(ny-1)/2;
    const double s=LParity(comp,1);
    size_t z,x,y;
    for (z=0;z<nz;z++) for (x=0;x<nx;x++) {
        for (y=1;y<=pts;y++) {
            const double w1=(double)y/(double)ny,w0=1.0-w1;
            const size_t ip=LIndex(x,y,z,nx,ny), im=LIndex(x,ny-y,z,nx,ny);
            const doublecomplex v=w0*a[ip]+(s*w1)*a[im];
            a[ip]=v;
            a[im]=s*v;
        }
        if ((ny%2)==0 && s<0) a[LIndex(x,ny/2,z,nx,ny)]=0;
    }
}

static void AverageZ(doublecomplex *a,const int comp,
                     const size_t nx,const size_t ny,const size_t nz)
{
    const size_t pts=(nz-1)/2;
    const double s=LParity(comp,2);
    size_t y,x,z;
    for (y=0;y<ny;y++) for (x=0;x<nx;x++) {
        for (z=1;z<=pts;z++) {
            const double w1=(double)z/(double)nz,w0=1.0-w1;
            const size_t ip=LIndex(x,y,z,nx,ny), im=LIndex(x,y,nz-z,nx,ny);
            const doublecomplex v=w0*a[ip]+(s*w1)*a[im];
            a[ip]=v;
            a[im]=s*v;
        }
        if ((nz%2)==0 && s<0) a[LIndex(x,y,nz/2,nx,ny)]=0;
    }
}

static void BuildSpatialComponent(doublecomplex *a,const int comp,
                                  const size_t nx,const size_t ny,const size_t nz)
{
    static const unsigned char mu[6]={0,0,0,1,1,2};
    static const unsigned char nu[6]={0,1,2,1,2,2};
    size_t x,y,z;
    for (z=0;z<nz;z++) for (y=0;y<ny;y++) for (x=0;x<nx;x++) {
        const size_t idx=LIndex(x,y,z,nx,ny);
        if (x==0 && y==0 && z==0) {
            a[idx]=(comp==0 || comp==3 || comp==5) ? 1.0 : 0.0;
        }
        else {
            doublecomplex g[6];
            (*InterTerm_int)((int)x,(int)y,(int)z,g);
            /* fft.c stores -FFT(InterTerm)/N and MatVec therefore applies
             * D=-InterTerm.  Build the transformed physical block S D S. */
            a[idx]=-cc_sqrt[0][mu[comp]]*g[comp]*cc_sqrt[0][nu[comp]];
        }
    }
    AverageX(a,comp,nx,ny,nz);
    AverageY(a,comp,nx,ny,nz);
    AverageZ(a,comp,nx,ny,nz);
}

static double LMax6(const doublecomplex a,const doublecomplex b,const doublecomplex c,
                    const doublecomplex d,const doublecomplex e,const doublecomplex f)
{
    double m=cabs(a),q;
    q=cabs(b); if(q>m)m=q; q=cabs(c); if(q>m)m=q;
    q=cabs(d); if(q>m)m=q; q=cabs(e); if(q>m)m=q;
    q=cabs(f); if(q>m)m=q;
    return m>1.0 ? m : 1.0;
}

doublecomplex *LanierBuildReference(double expansion,
                                     size_t *nx,size_t *ny,size_t *nz,
                                     size_t *rx,size_t *ry,size_t *rz)
{
    size_t n,nred,tmp,comp,kx,ky,kz,bad=0;
    doublecomplex *work,*coef;
    LPLAN plan;

    if (nx==NULL || ny==NULL || nz==NULL || rx==NULL || ry==NULL || rz==NULL)
        LogError(ONE_POS,"Internal Lanier reference error: null output dimension pointer");
    if (boxX<=0 || boxY<=0 || boxZ<=0)
        LogError(ONE_POS,"Invalid ADDA box dimensions for Lanier reference preconditioner");
    if (Nmat!=1)
        LogError(ONE_POS,"Reference Lanier preconditioner currently requires one homogeneous material");

    if (!(fabs(expansion-1.0)<1.0e-12 || fabs(expansion-1.5)<1.0e-12))
        LogError(ONE_POS,"Reference Lanier expansion must be 1.0 or 1.5 (got %.17g)",expansion);
    *nx=LAuxDim((size_t)boxX,expansion);
    *ny=LAuxDim((size_t)boxY,expansion);
    *nz=LAuxDim((size_t)boxZ,expansion);
    *rx=*nx/2+1; *ry=*ny/2+1; *rz=*nz/2+1;
    if (*ny > SIZE_MAX/(*nx) || (*nx)*(*ny) > SIZE_MAX/(*nz))
        LogError(ONE_POS,"Lanier auxiliary-grid size overflow");
    n=(*nx)*(*ny)*(*nz);
    if (*ry > SIZE_MAX/(*rx) || (*rx)*(*ry) > SIZE_MAX/(*rz))
        LogError(ONE_POS,"Lanier reduced-grid size overflow");
    nred=(*rx)*(*ry)*(*rz);
    if (nred > SIZE_MAX/6 || 6*nred > SIZE_MAX/sizeof(doublecomplex))
        LogError(ONE_POS,"Lanier coefficient allocation overflow");

    work=(doublecomplex *)LMALLOC(n*sizeof(doublecomplex));
    coef=(doublecomplex *)malloc(6*nred*sizeof(doublecomplex));
    if (work==NULL || coef==NULL) {
        if (work!=NULL) LFREE(work);
        free(coef);
        LogError(ONE_POS,"Insufficient host memory while building Lanier coefficients");
    }

    plan=LPLAN_DFT_3D((int)*nz,(int)*ny,(int)*nx,
                      (LCOMPLEX *)work,(LCOMPLEX *)work,
                      LFORWARD,FFTW_ESTIMATE);
    if (plan==NULL) {
        LFREE(work); free(coef);
        LogError(ONE_POS,"FFTW failed to create the Lanier auxiliary 3-D plan");
    }

    for (comp=0;comp<6;comp++) {
        BuildSpatialComponent(work,(int)comp,*nx,*ny,*nz);
        LEXECUTE(plan);
        for (kx=0;kx<*rx;kx++) for (ky=0;ky<*ry;ky++) for (kz=0;kz<*rz;kz++) {
            const size_t ridx=(kx*(*ry)+ky)*(*rz)+kz;
            const size_t findex=LIndex(kx,ky,kz,*nx,*ny);
            coef[comp*nred+ridx]=work[findex];
        }
    }
    LDESTROY(plan);
    LFREE(work);

    /* Invert each complex-symmetric 3x3 Fourier block.  Scale by 1/N here
     * because cuFFT inverse transforms are unnormalised. */
    for (tmp=0;tmp<nred;tmp++) {
        const doublecomplex a=coef[0*nred+tmp], b=coef[1*nred+tmp], c=coef[2*nred+tmp];
        const doublecomplex d=coef[3*nred+tmp], e=coef[4*nred+tmp], f=coef[5*nred+tmp];
        const doublecomplex c0=d*f-e*e;
        const doublecomplex c1=c*e-b*f;
        const doublecomplex c2=b*e-c*d;
        const doublecomplex c3=a*f-c*c;
        const doublecomplex c4=b*c-a*e;
        const doublecomplex c5=a*d-b*b;
        const doublecomplex det=a*c0+b*c1+c*c2;
        const double scale=LMax6(a,b,c,d,e,f);
#ifdef ADDA_SINGLE
        const double reltol=2.0e-5;
#else
        const double reltol=1.0e-12;
#endif
        if (!(cabs(det)>reltol*scale*scale*scale)) {
            bad++;
            continue;
        }
        const doublecomplex q=1.0/((double)n*det);
        coef[0*nred+tmp]=c0*q; coef[1*nred+tmp]=c1*q; coef[2*nred+tmp]=c2*q;
        coef[3*nred+tmp]=c3*q; coef[4*nred+tmp]=c4*q; coef[5*nred+tmp]=c5*q;
    }
    if (bad!=0) {
        free(coef);
        LogError(ONE_POS,"Lanier reference rejected: %zu/%zu reduced Fourier blocks are singular or ill-conditioned",bad,nred);
    }

    return coef;
}

#undef LPLAN
#undef LPLAN_DFT_3D
#undef LEXECUTE
#undef LDESTROY
#undef LMALLOC
#undef LFREE
#undef LCOMPLEX
#undef LFORWARD
