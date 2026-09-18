/* LANIER_FULL_TQC_V1
 * Header-only CUDA/C++ helper included by cudamatvec_backend.cu.
 * It reproduces the supplied DDSCAT/Lanier TQC-v1 EMA actor path:
 * 18 geometric/physical features -> 36 observation -> 5 Mish layers -> 6 tanh actions.
 * The binary asset format and feature formulas are intentionally kept identical
 * to the validated DDSCAT adapter from ddscat_stage39_ROOT_READY_FIX1.
 */
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace lanier_full_ai {
using cd=std::complex<double>;
static constexpr double PI=3.141592653589793238462643383279502884;
static std::string err;
static size_t pidx(int x,int y,int z,int ny,int nz){return ((size_t)x*(size_t)ny+(size_t)y)*(size_t)nz+(size_t)z;}
static void seterr(const char*s){err=s?s:"unknown LANIER_FULL TQC-v1 error";}

struct Layer{int out=0,in=0;std::vector<float>w,b;};
struct Actor{Layer l0,l1,l2,l3,l4,mu;std::string path;bool ready=false;};
static Actor actor;
static bool read_exact(std::ifstream&f,void*p,size_t n){f.read(reinterpret_cast<char*>(p),(std::streamsize)n);return f.good()||(f.eof()&&(size_t)f.gcount()==n);}
static bool read_layer(std::ifstream&f,Layer&l,int out,int in){l.out=out;l.in=in;l.w.resize((size_t)out*in);l.b.resize((size_t)out);return read_exact(f,l.w.data(),l.w.size()*sizeof(float))&&read_exact(f,l.b.data(),l.b.size()*sizeof(float));}
static bool load_actor(const char*path){
 if(!path||!*path){seterr("LANIER_FULL actor path is empty");return false;}
 if(actor.ready&&actor.path==path)return true;
 std::ifstream f(path,std::ios::binary);if(!f){err=std::string("cannot open LANIER_FULL actor asset: ")+path;return false;}
 char magic[9]={};uint32_t version=0,count=0;
 if(!read_exact(f,magic,9)||!read_exact(f,&version,4)||!read_exact(f,&count,4)){seterr("LANIER_FULL actor asset has a truncated header");return false;}
 if(std::string(magic,9)!="LANIERAI1"||version!=1||count!=157958u){seterr("LANIER_FULL actor asset header/version/parameter count mismatch");return false;}
 Actor a;
 if(!read_layer(f,a.l0,256,36)||!read_layer(f,a.l1,256,256)||!read_layer(f,a.l2,192,256)||
    !read_layer(f,a.l3,128,192)||!read_layer(f,a.l4,64,128)||!read_layer(f,a.mu,6,64)){
  seterr("LANIER_FULL actor asset payload is truncated");return false;
 }
 char extra;if(f.read(&extra,1)){seterr("LANIER_FULL actor asset has unexpected trailing bytes");return false;}
 a.path=path;a.ready=true;actor=std::move(a);return true;
}
static float softplus(float x){if(x>20.f)return x;if(x<-20.f)return std::exp(x);return std::log1p(std::exp(x));}
static float mish(float x){return x*std::tanh(softplus(x));}
static std::vector<float> linear(const Layer&l,const std::vector<float>&x,bool use_mish){
 std::vector<float>y((size_t)l.out);
 for(int o=0;o<l.out;o++){double a=(double)l.b[(size_t)o];const float*row=l.w.data()+(size_t)o*l.in;for(int i=0;i<l.in;i++)a+=(double)row[i]*(double)x[(size_t)i];float v=(float)a;y[(size_t)o]=use_mish?mish(v):v;}return y;
}
static std::array<float,6> infer(const std::array<float,36>&obs){
 std::vector<float>x(obs.begin(),obs.end());x=linear(actor.l0,x,true);x=linear(actor.l1,x,true);x=linear(actor.l2,x,true);x=linear(actor.l3,x,true);x=linear(actor.l4,x,true);x=linear(actor.mu,x,false);
 std::array<float,6>a{};for(int i=0;i<6;i++)a[(size_t)i]=std::tanh(x[(size_t)i]);return a;
}
static std::array<double,3> sym_eigs(double a00,double a01,double a02,double a11,double a12,double a22){
 double A[3][3]={{a00,a01,a02},{a01,a11,a12},{a02,a12,a22}};
 for(int it=0;it<40;it++){int p=0,q=1;double m=std::fabs(A[0][1]);if(std::fabs(A[0][2])>m){m=std::fabs(A[0][2]);p=0;q=2;}if(std::fabs(A[1][2])>m){m=std::fabs(A[1][2]);p=1;q=2;}if(m<1e-15)break;double app=A[p][p],aqq=A[q][q],apq=A[p][q];double phi=.5*std::atan2(2*apq,aqq-app),c=std::cos(phi),s=std::sin(phi);for(int k=0;k<3;k++)if(k!=p&&k!=q){double akp=A[k][p],akq=A[k][q];A[k][p]=A[p][k]=c*akp-s*akq;A[k][q]=A[q][k]=s*akp+c*akq;}A[p][p]=c*c*app-2*s*c*apq+s*s*aqq;A[q][q]=s*s*app+2*s*c*apq+c*c*aqq;A[p][q]=A[q][p]=0.;}
 std::array<double,3>e{{std::max(A[0][0],1e-12),std::max(A[1][1],1e-12),std::max(A[2][2],1e-12)}};std::sort(e.begin(),e.end());return e;
}
static std::vector<double> block_means(const std::vector<uint8_t>&mask,int nx,int ny,int nz,int groups){
 int bx=std::max(1,nx/groups),by=std::max(1,ny/groups),bz=std::max(1,nz/groups);int tx=(nx/bx)*bx,ty=(ny/by)*by,tz=(nz/bz)*bz;if(tx==0||ty==0||tz==0)return{};int NX=nx/bx,NY=ny/by,NZ=nz/bz;std::vector<double>out;out.reserve((size_t)NX*NY*NZ);
 for(int I=0;I<NX;I++)for(int J=0;J<NY;J++)for(int K=0;K<NZ;K++){double s=0.;int n=0;for(int ii=0;ii<bx;ii++)for(int jj=0;jj<by;jj++)for(int kk=0;kk<bz;kk++){size_t lin=((((((size_t)I*bx+(size_t)ii)*(size_t)NY+(size_t)J)*(size_t)by+(size_t)jj)*(size_t)NZ+(size_t)K)*(size_t)bz+(size_t)kk);int x=(int)(lin/(size_t)(ty*tz));size_t rem=lin%(size_t)(ty*tz);int y=(int)(rem/(size_t)tz),z=(int)(rem%(size_t)tz);s+=mask[pidx(x,y,z,ny,nz)]?1.:0.;n++;}out.push_back(n?s/(double)n:0.);}
 return out;
}
static double popvar(const std::vector<double>&v){if(v.size()<=1)return 0.;double m=std::accumulate(v.begin(),v.end(),0.)/(double)v.size(),s=0.;for(double x:v){double d=x-m;s+=d*d;}return s/(double)v.size();}
static int fft_features(const std::vector<uint8_t>&mask,int nx,int ny,int nz,double out[4]){
 size_t N=(size_t)nx*ny*nz;if(!N||N>(size_t)INT_MAX){seterr("LANIER_FULL FFT grid invalid");return 1;}
 std::vector<cufftComplex>h(N);for(size_t q=0;q<N;q++)h[q]=make_cuFloatComplex(mask[q]?1.f:0.f,0.f);
 cufftComplex*d=nullptr;cufftHandle plan=0;cudaError_t ce=cudaMalloc((void**)&d,N*sizeof(cufftComplex));if(ce!=cudaSuccess){err=std::string("LANIER_FULL cudaMalloc feature FFT failed: ")+cudaGetErrorString(ce);return 2;}
 auto cleanup=[&](){if(plan)cufftDestroy(plan);if(d)cudaFree(d);};
 ce=cudaMemcpy(d,h.data(),N*sizeof(cufftComplex),cudaMemcpyHostToDevice);if(ce!=cudaSuccess){err="LANIER_FULL feature FFT H2D failed";cleanup();return 3;}
 cufftResult cr=cufftPlan3d(&plan,nx,ny,nz,CUFFT_C2C);if(cr!=CUFFT_SUCCESS){seterr("LANIER_FULL cufftPlan3d feature FFT failed");cleanup();return 4;}
 cr=cufftExecC2C(plan,d,d,CUFFT_FORWARD);if(cr!=CUFFT_SUCCESS){seterr("LANIER_FULL forward feature FFT failed");cleanup();return 5;}cudaDeviceSynchronize();
 ce=cudaMemcpy(h.data(),d,N*sizeof(cufftComplex),cudaMemcpyDeviceToHost);if(ce!=cudaSuccess){err="LANIER_FULL feature FFT D2H failed";cleanup();return 6;}
 int cx=nx/2,cy=ny/2,cz=nz/2;double rmax2=(double)(cx*cx+cy*cy+cz*cz),lt2=.33*.33*rmax2,mt2=.66*.66*rmax2,b1=0,b2=0,b3=0,tot=0;auto sh=[](int i,int n){return(i+(n+1)/2)%n;};
 for(int i=0;i<nx;i++)for(int j=0;j<ny;j++)for(int k=0;k<nz;k++){int dx=i-cx,dy=j-cy,dz=k-cz;double r2=(double)(dx*dx+dy*dy+dz*dz);if(r2<=0)continue;const cufftComplex z=h[pidx(sh(i,nx),sh(j,ny),sh(k,nz),ny,nz)];double p=(double)z.x*z.x+(double)z.y*z.y;tot+=p;if(r2<=lt2)b1+=p;else if(r2<=mt2)b2+=p;else b3+=p;}
 tot=std::max(tot,1e-12);out[0]=b1/tot;out[1]=b2/tot;out[2]=b3/tot;
 for(size_t q=0;q<N;q++){double p=(double)h[q].x*h[q].x+(double)h[q].y*h[q].y;h[q]=make_cuFloatComplex((float)p,0.f);}ce=cudaMemcpy(d,h.data(),N*sizeof(cufftComplex),cudaMemcpyHostToDevice);if(ce!=cudaSuccess){cleanup();seterr("LANIER_FULL power FFT H2D failed");return 7;}cr=cufftExecC2C(plan,d,d,CUFFT_INVERSE);if(cr!=CUFFT_SUCCESS){cleanup();seterr("LANIER_FULL inverse feature FFT failed");return 8;}cudaDeviceSynchronize();ce=cudaMemcpy(h.data(),d,N*sizeof(cufftComplex),cudaMemcpyDeviceToHost);cleanup();if(ce!=cudaSuccess){seterr("LANIER_FULL autocorrelation D2H failed");return 9;}
 double invN=1./(double)N,mx=-std::numeric_limits<double>::infinity();for(size_t q=0;q<N;q++)mx=std::max(mx,(double)h[q].x*invN);mx=std::max(mx,1e-12);double best=0.;
 for(int i=0;i<nx;i++)for(int j=0;j<ny;j++)for(int k=0;k<nz;k++){int dx=i-cx,dy=j-cy,dz=k-cz;double r2=(double)(dx*dx+dy*dy+dz*dz);if(r2<=0)continue;double av=(double)h[pidx(i,j,k,ny,nz)].x*invN/mx;av=std::max(0.,std::min(1.,av));if(av<=.5)best=std::max(best,std::sqrt(r2));}
 double diag=std::sqrt((double)nx*nx+(double)ny*ny+(double)nz*nz);out[3]=best/std::max(diag,1e-12);return 0;
}
static double gnorm_one(int ix,int iy,int iz,double k){double x=ix,y=iy,z=iz,r=std::sqrt(x*x+y*y+z*z);if(r<1e-9)return 0.;double rh[3]={x/r,y/r,z/r};cd e=std::exp(cd(0,k*r))/r,c=(cd(0,k*r)-1.)/(r*r);double s=0;for(int a=0;a<3;a++)for(int b=0;b<3;b++){double rr=rh[a]*rh[b],I=(a==b)?1.:0.;cd G=e*(k*k*(rr-I)+c*(3.*rr-I));s+=std::norm(G);}return std::sqrt(s);}
static void gnorm_features(double wavelength,double out[3]){const double R1_MIN=2.449490,R1_MAX=2.664822,R2_MIN=.306186,R2_MAX=.470133,DIFF_MIN=2.143304,DIFF_MAX=2.194689;double k=2*PI/wavelength,k10=2*PI/10.,g1=gnorm_one(1,0,0,k),g2=gnorm_one(2,0,0,k),g110=gnorm_one(1,0,0,k10),g210=gnorm_one(2,0,0,k10),diff=g1-g2,d10=g110-g210;if(wavelength>=10.){out[0]=(g1-R1_MIN)/(R1_MAX-R1_MIN);out[1]=(g2-R2_MIN)/(R2_MAX-R2_MIN);out[2]=(diff-DIFF_MIN)/(DIFF_MAX-DIFF_MIN);}else{out[0]=(g110-R1_MIN)/(R1_MAX-R1_MIN)+std::log1p(g1-g110);out[1]=(g210-R2_MIN)/(R2_MAX-R2_MIN)+std::log1p(g2-g210);out[2]=(d10-DIFF_MIN)/(DIFF_MAX-DIFF_MIN)+std::log1p(diff-d10);}}

/* Input cc_sqrt uses the ABI precision of this backend. Squaring it recovers ADDA's
 * dimensional coupling/polarizability cc. DDSCAT's actor works in lattice units,
 * so the actor receives alpha_hat=cc/dipvol, i.e. polarizability in units of one
 * dipole volume. The scalar voxel alpha is the mean of the three diagonal
 * components, matching DDSCAT's scalar reduction from CXADIA. */
static int predict(const cuDoubleComplex*ccsqrt,size_t cc_count,const unsigned char*material,const unsigned short*position,size_t ndip,int nx,int ny,int nz,double akd,double dipvol,const char*actor_path,AddaCudaLanierFullPrediction*out){
 err.clear();if(!ccsqrt||!material||!position||!out||nx<=0||ny<=0||nz<=0||ndip==0){seterr("LANIER_FULL invalid prediction arguments");return 1;}if(!(akd>0.)||!std::isfinite(akd)){seterr("LANIER_FULL requires positive finite kd");return 2;}if(!(dipvol>0.)||!std::isfinite(dipvol)){seterr("LANIER_FULL requires positive finite dipole volume");return 3;}if(!load_actor(actor_path))return 4;
 size_t N=(size_t)nx*ny*nz;if(!N||N>(size_t)INT_MAX){seterr("LANIER_FULL physical grid too large");return 5;}std::vector<uint8_t>mask(N,0);cd asum(0,0);double maxmag=0.;int V=0;
 for(size_t q=0;q<ndip;q++){size_t p=3*q;int x=(int)position[p],y=(int)position[p+1],z=(int)position[p+2];if(x<0||x>=nx||y<0||y>=ny||z<0||z>=nz){seterr("LANIER_FULL occupied position outside physical box");return 6;}size_t m=(size_t)material[q];if(3*m+2>=cc_count){seterr("LANIER_FULL material index outside cc_sqrt table");return 7;}cd av(0,0);for(int c=0;c<3;c++){const cuDoubleComplex s=ccsqrt[3*m+(size_t)c];cd zc((double)s.x,(double)s.y);av+=(zc*zc)/dipvol;}av/=3.;size_t pi=pidx(x,y,z,ny,nz);if(!mask[pi]){mask[pi]=1;V++;}asum+=av;maxmag=std::max(maxmag,std::abs(av));}
 if(V<=0){seterr("LANIER_FULL has no occupied voxels");return 8;}cd arep=asum/(double)ndip;
 std::memset(out,0,sizeof(*out));out->occupied_count=V;out->max_alpha_mag=(float)maxmag;
 double sx=0,sy=0,sz=0;for(int i=0;i<nx;i++)for(int j=0;j<ny;j++)for(int k=0;k<nz;k++)if(mask[pidx(i,j,k,ny,nz)]){sx+=i;sy+=j;sz+=k;}double mx=sx/V,my=sy/V,mz=sz/V,c00=0,c01=0,c02=0,c11=0,c12=0,c22=0;int B=0;
 for(int i=0;i<nx;i++)for(int j=0;j<ny;j++)for(int k=0;k<nz;k++)if(mask[pidx(i,j,k,ny,nz)]){double x=i-mx,y=j-my,z=k-mz;c00+=x*x;c01+=x*y;c02+=x*z;c11+=y*y;c12+=y*z;c22+=z*z;bool bd=(i==0||j==0||k==0||i==nx-1||j==ny-1||k==nz-1);if(!bd)bd=!mask[pidx(i-1,j,k,ny,nz)]||!mask[pidx(i+1,j,k,ny,nz)]||!mask[pidx(i,j-1,k,ny,nz)]||!mask[pidx(i,j+1,k,ny,nz)]||!mask[pidx(i,j,k-1,ny,nz)]||!mask[pidx(i,j,k+1,ny,nz)];if(bd)B++;}
 double den=std::max(V-1,1);c00/=den;c01/=den;c02/=den;c11/=den;c12/=den;c22/=den;auto eig=sym_eigs(c00,c01,c02,c11,c12,c22);double Rg=std::sqrt((eig[0]+eig[1]+eig[2])/3.);auto co=block_means(mask,nx,ny,nz,4),fi=block_means(mask,nx,ny,nz,2);double cfvar=std::log1p(popvar(co)/std::max(popvar(fi),1e-12));double ff[4];if(fft_features(mask,nx,ny,nz,ff))return 9;double wavelength=2*PI/akd,gn[3];gnorm_features(wavelength,gn);double aeqobj=std::cbrt(3.*V/(4*PI)),aeqgrid=std::cbrt(3.*(double)N/(4*PI)),ar=(double)std::max({nx,ny,nz})/(double)std::max(1,std::min({nx,ny,nz}));double d[18];d[0]=arep.real();d[1]=std::cbrt(std::fabs(arep.imag()));d[2]=std::log1p(akd*aeqobj);d[3]=std::log1p(akd*aeqgrid);d[4]=std::log1p(akd*Rg);d[5]=std::log1p(std::sqrt(eig[2]/eig[0]));d[6]=(double)V/(double)N;d[7]=std::min(1.,std::max(0.,(double)B/(double)V/10.));d[8]=ff[0];d[9]=ff[1];d[10]=ff[2];d[11]=ff[3];d[12]=gn[0];d[13]=gn[1];d[14]=gn[2];d[15]=cfvar;d[16]=std::cbrt((double)N)/64.;d[17]=ar/(ar+5.);
 std::array<float,36>obs{};for(int i=0;i<18;i++){float v=std::isfinite(d[i])?(float)d[i]:0.f;out->features[i]=v;obs[(size_t)i]=v;}auto a=infer(obs);for(int i=0;i<6;i++)out->action[i]=a[(size_t)i];float mm=(float)maxmag;out->alpha_re=a[0]*1.5f*mm;out->alpha_im=(.85f*a[1]+.65f)*mm;out->k_shift=(a[2]+1.f)*1.5f-1.f;out->l_shift=(a[3]+1.f)*1.5f-1.f;out->blend_weight=(a[4]+1.f)*.5f;out->shift_radius=(a[5]+1.f)*.5f;
 if(!std::isfinite((double)out->alpha_re)||!std::isfinite((double)out->alpha_im)||std::hypot((double)out->alpha_re,(double)out->alpha_im)<=1e-20){seterr("LANIER_FULL actor produced zero/non-finite alpha_opt");return 10;}return 0;
}
static const char*last_error(){return err.c_str();}
} /* namespace lanier_full_ai */
