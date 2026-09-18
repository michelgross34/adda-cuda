/* LANIER_FULL_TQC_V1
 * Separate full preconditioner derived from the user's validated DDSCAT port
 * of Steven Lanier's supplied TQC-v1 archive.  The existing homogeneous
 * lanier1x/lanier15x implementation remains in lanier_precon.c unchanged.
 */
#include "const.h" /* keep this first */
#include "lanier_full_precon.h"
#include "io.h"
#include "vars.h"

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <fftw3.h>

#ifdef ADDA_SINGLE
# define LFPLAN fftwf_plan
# define LFPLAN_DFT_3D fftwf_plan_dft_3d
# define LFEXECUTE fftwf_execute
# define LFDESTROY fftwf_destroy_plan
# define LFMALLOC fftwf_malloc
# define LFFREE fftwf_free
# define LFCOMPLEX fftwf_complex
#else
# define LFPLAN fftw_plan
# define LFPLAN_DFT_3D fftw_plan_dft_3d
# define LFEXECUTE fftw_execute
# define LFDESTROY fftw_destroy_plan
# define LFMALLOC fftw_malloc
# define LFFREE fftw_free
# define LFCOMPLEX fftw_complex
#endif

static const double gl9_x[9]={
 -0.9681602395076261,-0.8360311073266358,-0.6133714327005904,
 -0.3242534234038089,0.0,0.3242534234038089,
  0.6133714327005904,0.8360311073266358,0.9681602395076261};
static const double gl9_w[9]={
 0.08127438836157416,0.18064816069485737,0.2606106964029356,
 0.31234707704000292,0.33023935500125984,0.31234707704000292,
 0.2606106964029356,0.18064816069485737,0.08127438836157416};

static size_t LFIndex(size_t x,size_t y,size_t z,size_t nx,size_t ny){return (z*ny+y)*nx+x;}
static size_t LFAuxDim(size_t n){
 if(n>(SIZE_MAX-1)/3)LogError(ONE_POS,"Lanier full auxiliary dimension overflow");
 size_t v=(3*n+1)/2;if(v<n)v=n;if(v&1u){if(v==SIZE_MAX)LogError(ONE_POS,"Lanier full auxiliary dimension overflow");v++;}return v;
}
static double LFParity(int comp,int axis){static const signed char p[3][6]={{+1,-1,-1,+1,+1,+1},{+1,-1,+1,+1,-1,+1},{+1,+1,-1,+1,-1,+1}};return(double)p[axis][comp];}

static void LFAverageX(doublecomplex*a,int c,size_t nx,size_t ny,size_t nz){size_t pts=(nx-1)/2,z,y,x;double s=LFParity(c,0);for(z=0;z<nz;z++)for(y=0;y<ny;y++){for(x=1;x<=pts;x++){double w1=(double)x/nx,w0=1-w1;size_t ip=LFIndex(x,y,z,nx,ny),im=LFIndex(nx-x,y,z,nx,ny);doublecomplex v=w0*a[ip]+s*w1*a[im];a[ip]=v;a[im]=s*v;}if((nx%2)==0&&s<0)a[LFIndex(nx/2,y,z,nx,ny)]=0;}}
static void LFAverageY(doublecomplex*a,int c,size_t nx,size_t ny,size_t nz){size_t pts=(ny-1)/2,z,x,y;double s=LFParity(c,1);for(z=0;z<nz;z++)for(x=0;x<nx;x++){for(y=1;y<=pts;y++){double w1=(double)y/ny,w0=1-w1;size_t ip=LFIndex(x,y,z,nx,ny),im=LFIndex(x,ny-y,z,nx,ny);doublecomplex v=w0*a[ip]+s*w1*a[im];a[ip]=v;a[im]=s*v;}if((ny%2)==0&&s<0)a[LFIndex(x,ny/2,z,nx,ny)]=0;}}
static void LFAverageZ(doublecomplex*a,int c,size_t nx,size_t ny,size_t nz){size_t pts=(nz-1)/2,y,x,z;double s=LFParity(c,2);for(y=0;y<ny;y++)for(x=0;x<nx;x++){for(z=1;z<=pts;z++){double w1=(double)z/nz,w0=1-w1;size_t ip=LFIndex(x,y,z,nx,ny),im=LFIndex(x,y,nz-z,nx,ny);doublecomplex v=w0*a[ip]+s*w1*a[im];a[ip]=v;a[im]=s*v;}if((nz%2)==0&&s<0)a[LFIndex(x,y,nz/2,nx,ny)]=0;}}

/* Algebraic transcription of the supplied archive's greens3_eval.  Its sign
 * is already the physical off-diagonal block used by M = alpha^-1 I - G. */
static void LFGreenEval(double px,double py,double pz,double k,doublecomplex val[6]){
 double r2=px*px+py*py+pz*pz;if(r2<1e-24){for(int c=0;c<6;c++)val[c]=0;return;}double r=sqrt(r2),invr=1/r,q[3]={px*invr,py*invr,pz*invr},k2=k*k,cs=cos(k*r),sn=sin(k*r);static const unsigned char cm[6][2]={{0,0},{0,1},{0,2},{1,1},{1,2},{2,2}};
 for(int c=0;c<6;c++){int mu=cm[c][0],nu=cm[c][1],diag=(mu==nu);double qmn=q[mu]*q[nu],den=r2,t2re=-1/den,t2im=k*r/den,coeff=3*qmn-(diag?1:0),t1=k2*(qmn-(diag?1:0)),re=t1+coeff*t2re,im=coeff*t2im;val[c]=(cs*invr*re-sn*invr*im)+I*(sn*invr*re+cs*invr*im);}
}

static void LFFullLag(int lx,int ly,int lz,size_t nmax,const AddaCudaLanierFullPrediction*p,doublecomplex out[6]){
 double px=lx,py=ly,pz=lz,rvox=sqrt((double)lx*lx+(double)ly*ly+(double)lz*lz),keff=kd*(1.0+(double)p->k_shift);for(int c=0;c<6;c++)out[c]=0;
 if(p->l_shift!=0){double sr=ceil((double)nmax*(double)p->shift_radius);if(rvox<=sr){double r=sqrt(px*px+py*py+pz*pz),sf=(r+(double)p->l_shift)/fmax(r,1e-12);px*=sf;py*=sf;pz*=sf;}}
 if(px*px+py*py+pz*pz<1e-24)return;doublecomplex point[6];LFGreenEval(px,py,pz,keff,point);if(p->blend_weight==0||rvox>5){for(int c=0;c<6;c++)out[c]=point[c];return;}
 static const unsigned char cm[6][2]={{0,0},{0,1},{0,2},{1,1},{1,2},{2,2}};double pos[3]={px,py,pz};int zero[6];for(int c=0;c<6;c++){int mu=cm[c][0],nu=cm[c][1];zero[c]=(mu!=nu)&&(pos[mu]==0||pos[nu]==0);}
 doublecomplex quad[6]={0,0,0,0,0,0};for(int ia=0;ia<9;ia++)for(int ib=0;ib<9;ib++)for(int ic=0;ic<9;ic++){double qx=px+0.5*gl9_x[ia],qy=py+0.5*gl9_x[ib],qz=pz+0.5*gl9_x[ic],w=gl9_w[ia]*gl9_w[ib]*gl9_w[ic];doublecomplex qv[6];LFGreenEval(qx,qy,qz,keff,qv);for(int c=0;c<6;c++)if(!zero[c])quad[c]+=qv[c]*w;}
 double bw=p->blend_weight,pw=1-bw;for(int c=0;c<6;c++){if(zero[c])out[c]=0;else out[c]=pw*point[c]+bw*(quad[c]*0.125);}
}

static void LFBuildSpatial(doublecomplex*a,int comp,size_t nx,size_t ny,size_t nz,const AddaCudaLanierFullPrediction*p){
 doublecomplex alpha=(double)p->alpha_re+I*(double)p->alpha_im,ainv=1.0/alpha;size_t nmax=nx>ny?(nx>nz?nx:nz):(ny>nz?ny:nz);for(size_t z=0;z<nz;z++)for(size_t y=0;y<ny;y++)for(size_t x=0;x<nx;x++){size_t idx=LFIndex(x,y,z,nx,ny);if(x==0&&y==0&&z==0)a[idx]=(comp==0||comp==3||comp==5)?ainv:0;else{doublecomplex g[6];LFFullLag((int)x,(int)y,(int)z,nmax,p,g);a[idx]=g[comp];}}
 LFAverageX(a,comp,nx,ny,nz);LFAverageY(a,comp,nx,ny,nz);LFAverageZ(a,comp,nx,ny,nz);
}
static double LFMax6(doublecomplex a,doublecomplex b,doublecomplex c,doublecomplex d,doublecomplex e,doublecomplex f){double m=cabs(a),q;q=cabs(b);if(q>m)m=q;q=cabs(c);if(q>m)m=q;q=cabs(d);if(q>m)m=q;q=cabs(e);if(q>m)m=q;q=cabs(f);if(q>m)m=q;return m>1?m:1;}

doublecomplex *LanierBuildFullBox(const AddaCudaLanierFullPrediction*p,size_t phys_x,size_t phys_y,size_t phys_z,size_t*nx,size_t*ny,size_t*nz,size_t*rx,size_t*ry,size_t*rz){
 if(!p||!nx||!ny||!nz||!rx||!ry||!rz)LogError(ONE_POS,"Internal Lanier full null argument");if(phys_x==0||phys_y==0||phys_z==0)LogError(ONE_POS,"Invalid physical dimensions for Lanier full");if(rectDip)LogError(ONE_POS,"Lanier full TQC-v1 requires isotropic cubic voxels (no -rect_dip)");double am=hypot((double)p->alpha_re,(double)p->alpha_im);if(!(am>1e-20)||!isfinite(am))LogError(ONE_POS,"Lanier full actor returned invalid alpha_opt");
 *nx=LFAuxDim(phys_x);*ny=LFAuxDim(phys_y);*nz=LFAuxDim(phys_z);*rx=*nx/2+1;*ry=*ny/2+1;*rz=*nz/2+1;if(*ny>SIZE_MAX/(*nx)||(*nx)*(*ny)>SIZE_MAX/(*nz))LogError(ONE_POS,"Lanier full grid overflow");size_t n=(*nx)*(*ny)*(*nz);if(*ry>SIZE_MAX/(*rx)||(*rx)*(*ry)>SIZE_MAX/(*rz))LogError(ONE_POS,"Lanier full reduced-grid overflow");size_t nred=(*rx)*(*ry)*(*rz);if(nred>SIZE_MAX/6||6*nred>SIZE_MAX/sizeof(doublecomplex))LogError(ONE_POS,"Lanier full coefficient allocation overflow");
 doublecomplex*work=(doublecomplex*)LFMALLOC(n*sizeof(doublecomplex));doublecomplex*coef=(doublecomplex*)malloc(6*nred*sizeof(doublecomplex));if(!work||!coef){if(work)LFFREE(work);free(coef);LogError(ONE_POS,"Insufficient host memory building Lanier full coefficients");}
 LFPLAN plan=LFPLAN_DFT_3D((int)*nz,(int)*ny,(int)*nx,(LFCOMPLEX*)work,(LFCOMPLEX*)work,FFTW_FORWARD,FFTW_ESTIMATE);if(!plan){LFFREE(work);free(coef);LogError(ONE_POS,"FFTW failed to create Lanier full 3-D plan");}
 for(size_t comp=0;comp<6;comp++){LFBuildSpatial(work,(int)comp,*nx,*ny,*nz,p);LFEXECUTE(plan);for(size_t kx=0;kx<*rx;kx++)for(size_t ky=0;ky<*ry;ky++)for(size_t kz=0;kz<*rz;kz++){size_t ri=(kx*(*ry)+ky)*(*rz)+kz,fi=LFIndex(kx,ky,kz,*nx,*ny);coef[comp*nred+ri]=work[fi];}}
 LFDESTROY(plan);LFFREE(work);size_t bad=0;for(size_t t=0;t<nred;t++){doublecomplex a=coef[t],b=coef[nred+t],c=coef[2*nred+t],d=coef[3*nred+t],e=coef[4*nred+t],f=coef[5*nred+t],c0=d*f-e*e,c1=c*e-b*f,c2=b*e-c*d,c3=a*f-c*c,c4=b*c-a*e,c5=a*d-b*b,det=a*c0+b*c1+c*c2;double scale=LFMax6(a,b,c,d,e,f);
#ifdef ADDA_SINGLE
  const double tol=2e-5;
#else
  const double tol=1e-12;
#endif
  if(!(cabs(det)>tol*scale*scale*scale)){bad++;continue;}doublecomplex q=1.0/((double)n*det);coef[t]=c0*q;coef[nred+t]=c1*q;coef[2*nred+t]=c2*q;coef[3*nred+t]=c3*q;coef[4*nred+t]=c4*q;coef[5*nred+t]=c5*q;}
 if(bad){free(coef);LogError(ONE_POS,"Lanier full rejected: %zu/%zu spectral blocks singular/ill-conditioned",bad,nred);}return coef;
}

doublecomplex *LanierBuildFull(const AddaCudaLanierFullPrediction*p,size_t*nx,size_t*ny,size_t*nz,size_t*rx,size_t*ry,size_t*rz){
 if(boxX<=0||boxY<=0||boxZ<=0)LogError(ONE_POS,"Invalid ADDA box dimensions for Lanier full");
 return LanierBuildFullBox(p,(size_t)boxX,(size_t)boxY,(size_t)boxZ,nx,ny,nz,rx,ry,rz);
}

#undef LFPLAN
#undef LFPLAN_DFT_3D
#undef LFEXECUTE
#undef LFDESTROY
#undef LFMALLOC
#undef LFFREE
#undef LFCOMPLEX
