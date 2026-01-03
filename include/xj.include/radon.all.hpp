/************update in 2021.01.07**************/
/*
    
    
***********************************************/

#ifndef RADON_ALL_HPP
#define RADON_ALL_HPP
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <thread>
#include <future>
#include "../xjc.h"
using namespace std;
using namespace arma;

// 3-D Recover data in the frequence - space domain.

/* discription.
This function will do the local slant-stack over local traces .
float
**trace;
the INPUT local seismic-gather, i.e. trace[ntrace][ns].
int ntrace;
the trace-number of the INPUT gather.
int ns;
the trace length.
float dt;
time- sampling interval, (unit of dt should be second) ,
float *coor;
coor[ntrace] stores the offset of each trace .
float x0;
the beam-center coordinate .
float **tauppanel; the OUTPUT tau-p spectrum, i.e. tauppanel[npsr][ns].
int npsr;
the ray- parameter number of tau-p panel.
float psrmin;
the minimum ray-parameter, (unit is s/m) .
float dpsr;
the interval of ray-parameter， (unit is s/m) .
*/
///////////////////////////////////////////////////////////////////
/*
beamforming/liner_radon变换 传递的参数：
nz（处理数据体Z/T方向采样点）、nx（处理数据体X方向采样点）、
ny（3D数据体测线数）、
nf（数据频率域变换的频率采样点）、np（Radon变换倾角采样点）、
dz（数据Z方向采样间隔）、dx（数据X方向采样间隔）、
df（数据变换后频率采样间隔）、dp（数据变换后倾角采样间隔）、
p0（数据变换后的中心倾角值）、data（原始数据）、
realdataTP（beamforming/liner_radon变换得到的数据实部）、
realrebuild（beamforming/liner_radon反变换重建的数据实部）、
realdatafft（beamforming/liner_radon变换得到的中间矩阵实部，用于检查算法）、
dataTP（beamforming/liner_radon变换得到的复数数据体）、
rebuild（beamforming/liner_radon反变换重建的复数数据体）、
dig_n（对角加权值）、

allAreal[99],allAimag[99]
（是文件路径，用于存放一个转换矩阵A；在处理多批数据时需要重复计算A，
且A较大不好直接存在内存里，故将其存入文件以备读取使用）
*/
class LinerRadon3d
{
public:
    int nt,nx,ny,nf,npx,npy,nthread,
        nf1,nf2,rulef1,rulef2,myid;
    float dt,dx,dy,df,dpx,dpy,p0x,p0y,\
        fmin,fmax,rulefmin,rulefmax;
    float planeWaveAccuracy;
    float planeWaveInterval;
    float weightNoiseLevel;
    float weightSparseness;
/*****************************************************/
    fcube dataTX,dataRealTP,rebuildTX;
    cx_fcube dataFX,rebuildFX,dataFP;
    fcube weightTX;
/*****************************************************/
    fmat weightPxy,coordpy,coordpx,coordx,coordy;
    float digNormL1,digNormL2,digNormMix,parNorm;
    bool regularization, isConverge, fastHessian;
    cx_fmat basePlaneWaveFPX;
    cx_dcube hessfftAll;
    cx_fmat* pbasePlaneWaveFPX;
    cx_dcube* phessfftAll;

    LinerRadon3d();
    ~LinerRadon3d();
    LinerRadon3d & operator=(LinerRadon3d & inp);
    void InitializePar(\
        int nx, int ny, int nt, \
        int npx, int npy, \
        float dpx, float dpy, float dt, \
        float p0x, float p0y, int ncpu);
    void ParUpdate();
    void ClearData();
    void GetCoordpx();
    void GetCoordpy();
    void GetCoordx(fmat &coordx);
    void GetCoordy(fmat &coordy);
    fmat GetCoordx(int nx,float dx,float x0);
    fmat GetCoordy(int ny,float dy,float y0);
    void InputOriginalData(fcube &data);

    void GetBasePlaneWaveFPX();
    void GetDataTPbyLRT(bool dotx2fxTransform=true,bool dofp2tpTransform=true);

    void GetHessianAll();
    void GetDataTPbyLSLRT(int iterations_num,float residual_ratio);
    void GetDataTPbyMixftLSLRT(int iterations_num,float residual_ratio,int halfWinWide);
    
    void RecoverDataTXbyInvLRT(bool dotx2fxTransform=true);

};
LinerRadon3d& LinerRadon3d::operator=(LinerRadon3d & inp)
{
    nt=inp.nt; nx=inp.nx; ny=inp.ny;
    nf=inp.nf;npx=inp.npx; npy=inp.npy;
    nthread=inp.nthread;
    nf1=inp.nf1; nf2=inp.nf2;
    rulef1=inp.rulef1; rulef2=inp.rulef2;
    dt=inp.dt; dx=inp.dx; dy=inp.dy;
    df=inp.df; dpx=inp.dpx; dpy=inp.dpy;
    p0x=inp.p0x; p0y=inp.p0y;
    fmin=inp.fmin; fmax=inp.fmax;
    rulefmin=inp.rulefmin; rulefmax=inp.rulefmax;
    planeWaveAccuracy=inp.planeWaveAccuracy;
    planeWaveInterval=inp.planeWaveInterval;
    dataTX=inp.dataTX;
    dataRealTP=inp.dataRealTP;
    rebuildTX=inp.rebuildTX;
    dataFX=inp.dataFX;
    rebuildFX=inp.rebuildFX;
    dataFP=inp.dataFP;
    weightTX=inp.weightTX;
    weightPxy=inp.weightPxy;
    coordpx=inp.coordpx;
    coordpy=inp.coordpy;
    coordx=inp.coordx;
    coordy=inp.coordy;
    digNormL1=inp.digNormL1;
    digNormL2=inp.digNormL2;
    digNormMix=inp.digNormMix;
    parNorm=inp.parNorm;
    regularization=inp.regularization;
    isConverge=inp.isConverge;
    fastHessian=inp.fastHessian;
    pbasePlaneWaveFPX=inp.pbasePlaneWaveFPX;
    phessfftAll=inp.phessfftAll;
    weightSparseness=inp.weightSparseness;
    weightNoiseLevel=inp.weightNoiseLevel;
    return *this;
}

void GetDataTPbyLRT_thread(LinerRadon3d* lrt3d, int kf1, int kf2);
LinerRadon3d::~LinerRadon3d()
{
    phessfftAll=nullptr;
    pbasePlaneWaveFPX=nullptr;
    this->ClearData();
}
LinerRadon3d::LinerRadon3d()
{
    myid=0;
    planeWaveAccuracy=1.0;
    weightNoiseLevel=0.5;
    weightSparseness=0.5;
    nt=1;nx=1;ny=1;
    npx=1;npy=1;
    nf=1;nf1=0;nf2=0;
    rulef1=0;rulef2=0;
    fmax=150;fmin=1;
    rulefmax=25;rulefmin=1;
    nthread=1;
    dt=0.001;dx=1;dy=1;
    df=1.0/nt/dt;
    dpx=0.000005;dpy=0.000005;
    p0x=0;p0y=0;
    digNormL1=0;digNormL2=0.01;
    digNormMix=0;parNorm=0;
    regularization=true;
    isConverge=false;
    fastHessian=true;
    phessfftAll=nullptr;
    pbasePlaneWaveFPX=nullptr;
}
//设置一组常用的初始化参数
void LinerRadon3d::InitializePar(\
    int nx, int ny, int nt, \
    int npx, int npy, \
    float dpx, float dpy, float dt, \
    float p0x, float p0y, int ncpu)
{
    myid=0;
    planeWaveAccuracy=1.0;
    weightNoiseLevel=0.5;
    weightSparseness=0.5;
    this->nx=nx; this->ny=ny;
    this->nt=nt;this->dt=dt;
    this->npx=npx;this->npy=npy;    
    this->dpx=dpx;this->dpy=dpy;
    this->p0x=p0x;this->p0y=p0y;
    this->nf=nt;
    this->df=1.0/nt/dt;
    this->fmax=150;this->fmin=1;
    this->rulefmax=25;this->rulefmin=1;
    this->nf1=1;this->nf2=this->fmax/this->df;
    this->rulef1=1;
    this->rulef2=this->rulefmax/this->df;
    this->nthread=ncpu;
    this->dx=1;this->dy=1;
    this->digNormL1=0;
    this->digNormL2=0.01;
    this->digNormMix=0;
    this->parNorm=0;
    this->regularization=true;
    this->isConverge=false;
    this->fastHessian=true;
    phessfftAll=nullptr;
    pbasePlaneWaveFPX=nullptr;
    this->GetCoordpx();
    this->GetCoordpy();
} 
void LinerRadon3d::ParUpdate()
{
    this->df=1.0/this->dt/this->nt;
    this->nf=this->nt;
    this->nf2=int(this->fmax/this->df);
    this->rulef2=int(this->rulefmax/this->df);
    if(this->nf2>this->nf/2){
        this->nf2=this->nf/2;
    }
    if(this->rulef2>this->nf2/2){
        this->rulef2=this->nf2/2;
    }
    if(this->dpx==0){
        this->dpx=0.00001;
    }
    if(this->dpy==0){
        this->dpy=this->dpx;
    }
    planeWaveInterval=min(dpx,dpy)*planeWaveAccuracy;
    this->GetCoordpx();
    this->GetCoordpy();
}
void LinerRadon3d::ClearData()
{
    this->dataTX.clear();
    this->dataRealTP.clear();
    this->rebuildTX.clear();
    this->dataFX.clear();
    this->rebuildFX.clear();
    this->dataFP.clear();
    this->weightTX.clear();
    this->weightPxy.clear();
    this->coordpx.clear();
    this->coordpy.clear();
    this->coordx.clear();
    this->coordy.clear();
    this->basePlaneWaveFPX.clear();
    this->hessfftAll.clear();
}

void LinerRadon3d::GetCoordpx()
{
    this->coordpx.zeros(this->npx,1);
    for(int i=0;i<this->npx;i++){
        this->coordpx(i,0)=i*this->dpx+this->p0x;
    }
}
void LinerRadon3d::GetCoordpy()
{
    this->coordpy.zeros(this->npy,1);
    for(int i=0;i<this->npy;i++){
        this->coordpy(i,0)=i*this->dpy+this->p0y;   
    }
}
void LinerRadon3d::GetCoordx(fmat &coordx)
{   
    this->coordx=coordx;
    this->nx=coordx.n_rows;
}
void LinerRadon3d::GetCoordy(fmat &coordy)
{
    this->coordy=coordy;
    this->ny=coordy.n_cols;
}
fmat LinerRadon3d::GetCoordx(int nx, float dx,float x0)
{   
    this->nx=nx;
    this->dx=dx;
    this->coordx.zeros(nx,ny);
    for(int i=0;i<nx;i++){
    for(int iy=0;iy<ny;iy++){
        this->coordx(i,iy)=i*dx+x0;
    }}
    return this->coordx;
}
fmat LinerRadon3d::GetCoordy(int ny, float dy,float y0)
{
    this->ny=ny;
    this->dy=dy;
    this->coordy.zeros(nx,ny);
    for(int ix=0;ix<nx;ix++){
    for(int i=0;i<ny;i++){
        this->coordy(ix,i)=i*dy+y0;  
    }}
    return this->coordy;
}
void LinerRadon3d::InputOriginalData(fcube &data)
{
    this->dataTX=data;
    this->nx=data.n_rows;
    this->ny=data.n_cols;
    this->nt=data.n_slices;
    this->ParUpdate();
}
void LinerRadon3d::GetBasePlaneWaveFPX()
{
    int nfn=nf2-nf1;
    float maxX=abs(coordx).max();
    float maxY=abs(coordy).max();
    float maxPx=abs(coordpx).max();
    float maxPy=abs(coordpy).max();
    float maxTime=max(maxX,maxY)*max(maxPx,maxPy);
    int npt=3+maxTime/planeWaveInterval;
    float memoryGB=nfn*npt*2*sizeof(memoryGB)/1000.0/1000.0/1000.0;
    cout<<"Warning: memory GB will be used: "<<memoryGB<<endl;
    this->basePlaneWaveFPX.zeros(npt,nfn);
    float dpt=planeWaveInterval;
omp_set_num_threads(nthread);
#pragma omp parallel for
    for(int k=nf1;k<nf2;k++){
        cx_float a,b;
        float w=2.0*3.1415926*df*k;
        a.real(0.0);a.imag(1.0);
        for(int i=0;i<npt;i++){
            b=a*w*dpt*float(i);
            basePlaneWaveFPX(i,k-nf1)=exp(b);
            //timeBase(i,k-fn1)=conj(exp(b));
        }
    }
    this->pbasePlaneWaveFPX=&basePlaneWaveFPX;
}
void RecoverDataTXbyInvLRT_thread(LinerRadon3d *lrt3d, int kf1, int kf2)
{
    float dpt=lrt3d[0].planeWaveInterval;
    int kfbeg=lrt3d[0].nf1;

    for(int kf=kf1;kf<kf2;kf++)  
    {
        for(int kpy=0;kpy<lrt3d[0].npy;kpy++){
        for(int kpx=0;kpx<lrt3d[0].npx;kpx++){
            cx_float xplane,yplane,dat,datOut;
            dat=lrt3d[0].dataFP(kpx,kpy,kf);
        for(int ky=0;ky<lrt3d[0].ny;ky++){
            float fpty=lrt3d[0].coordy(0,ky)*lrt3d[0].coordpy(kpy,0);
            int kpty=abs(fpty)/dpt;
            //cout<<kpty<<"/"<<lrt3d[0].basePlaneWaveFPX.n_rows<<endl;
            if(fpty<0){yplane=conj(lrt3d[0].pbasePlaneWaveFPX[0](kpty,kf-kfbeg));}
            else{yplane=(lrt3d[0].pbasePlaneWaveFPX[0](kpty,kf-kfbeg));}
            datOut=dat*yplane;
        for(int kx=0;kx<lrt3d[0].nx;kx++){
            float fpty2=lrt3d[0].coordy(kx,ky)*lrt3d[0].coordpy(kpy,0);
            if(abs(fpty2-fpty)>=dpt){
                fpty=fpty2;
                kpty=abs(fpty)/dpt;
                if(fpty<0){yplane=conj(lrt3d[0].pbasePlaneWaveFPX[0](kpty,kf-kfbeg));}
                else{yplane=(lrt3d[0].pbasePlaneWaveFPX[0](kpty,kf-kfbeg));}
                datOut=dat*yplane;
            }
            float fptx=lrt3d[0].coordx(kx,ky)*lrt3d[0].coordpx(kpx,0);
            int kptx=abs(fptx)/dpt;

            if(fptx<0){xplane=conj(lrt3d[0].pbasePlaneWaveFPX[0](kptx,kf-kfbeg));}
            else{xplane=(lrt3d[0].pbasePlaneWaveFPX[0](kptx,kf-kfbeg));}
            
            lrt3d[0].rebuildFX(kx,ky,kf)+=(datOut*xplane);
        }}
        }}
    }
}
void LinerRadon3d::RecoverDataTXbyInvLRT(bool dotx2fxTransform)
{
    if(dotx2fxTransform){
        this->dataFP.set_size(npx,npy,nf);
        tx2fx_3d_thread(dataFP,dataRealTP,nthread);
    }
    rebuildFX.zeros(nx,ny,nf);
    int ncpu(nthread),pnf1,pnf2,k,kcpu,kf,nfn(nf2-nf1);
    float dnf;
    ncpu=min(ncpu,nfn);
    ncpu=max(ncpu,1);
    thread *pcal;
    pcal=new thread[ncpu];
    
    for(int jn=0;jn<ncpu;jn++){
        int zbeg=round(jn*((nfn+0.001)/nthread))+nf1;
        int zend=round((jn+1)*((nfn+0.001)/nthread))+nf1;
        //cout<<zbeg<<" to "<<zend<<"/"<<nf1<<" to "<<nf2<<endl;
        pcal[jn]=thread(RecoverDataTXbyInvLRT_thread,this,zbeg,zend);
    }
    for(k=0;k<ncpu;k++){
        if(pcal[k].joinable()){
            pcal[k].join();}
    }
    //fp_to_tp3d_linerradon3d(par);
    rebuildTX.zeros(nx,ny,nt);
    fx2tx_3d_thread(rebuildTX,rebuildFX,ncpu);
    dataFX.set_size(1,1,1);
    rebuildTX=4.0*rebuildTX;
    delete[] pcal;
    pcal=nullptr;
}
void GetDataTPbyLRT_thread(LinerRadon3d *lrt3d, int kf1, int kf2)
{
    float dpt=lrt3d[0].planeWaveInterval;
    int kfbeg=lrt3d[0].nf1;
    //cout<<"n1="<<lrt3d[0].pbasePlaneWaveFPX[0].n_rows<<endl;
    //cout<<"n2="<<lrt3d[0].pbasePlaneWaveFPX[0].n_cols<<endl;
    for(int kf=kf1;kf<kf2;kf++)  
    {
        for(int ky=0;ky<lrt3d[0].ny;ky++){
        for(int kx=0;kx<lrt3d[0].nx;kx++){
            cx_float xplane,yplane,dat,datOut;
            dat=lrt3d[0].dataFX(kx,ky,kf);
        for(int kpy=0;kpy<lrt3d[0].npy;kpy++){
            float fpty=lrt3d[0].coordy(kx,ky)*lrt3d[0].coordpy(kpy,0);
            int kpty=abs(fpty)/dpt;
            //cout<<kpty<<"/"<<lrt3d[0].basePlaneWaveFPX.n_rows<<endl;
            if(fpty<0){yplane=conj(lrt3d[0].pbasePlaneWaveFPX[0](kpty,kf-kfbeg));}
            else{yplane=(lrt3d[0].pbasePlaneWaveFPX[0](kpty,kf-kfbeg));}
            datOut=dat*yplane;
        for(int kpx=0;kpx<lrt3d[0].npx;kpx++){
            float fptx=lrt3d[0].coordx(kx,ky)*lrt3d[0].coordpx(kpx,0);
            int kptx=abs(fptx)/dpt;
            if(fptx<0){xplane=conj(lrt3d[0].pbasePlaneWaveFPX[0](kptx,kf-kfbeg));}
            else{xplane=(lrt3d[0].pbasePlaneWaveFPX[0](kptx,kf-kfbeg));}
                lrt3d[0].dataFP(kpx,kpy,kf)+=(datOut*xplane);
            }
        }
        }}
    }
}
void LinerRadon3d::GetDataTPbyLRT(bool dotx2fxTransform,bool dofp2tpTransform)
{
    if(dotx2fxTransform){
        dataFX.zeros(nx,ny,nf);
        tx2fx_3d_thread(dataFX,dataTX,nthread);
    }
    dataFP.zeros(npx,npy,nf);
    int ncpu(nthread),pnf1,pnf2,k,kcpu,kf,nfn(nf2-nf1);
    float dnf;
    ncpu=min(ncpu,nfn);
    ncpu=max(ncpu,1);
    thread *pcal;
    pcal=new thread[ncpu];
    
    for(int jn=0;jn<ncpu;jn++){
        int zbeg=round(jn*((nfn+0.001)/nthread))+nf1;
        int zend=round((jn+1)*((nfn+0.001)/nthread))+nf1;
        //cout<<zbeg<<" to "<<zend<<"/"<<nf1<<" to "<<nf2<<endl;
        pcal[jn]=thread(GetDataTPbyLRT_thread,this,zbeg,zend);
    }
    for(k=0;k<ncpu;k++){
        if(pcal[k].joinable()){
            pcal[k].join();}
    }
    //fp_to_tp3d_linerradon3d(par);
    if(dofp2tpTransform){
        dataRealTP.zeros(npx,npy,nt);
        fx2tx_3d_thread(dataRealTP,dataFP,ncpu);
    }
    //dataFP.set_size(1,1,1);
    delete[] pcal;
    pcal=nullptr;
}

inline cx_dmat GetOneHessianElem(float w,\
 int nx, float kx1, float dx, float dpx)
{
    cx_dmat a(1,1,fill::zeros);
    cx_dmat b(1,1,fill::zeros);
    cx_dmat c(1,1,fill::zeros);
    cx_dmat d(1,1,fill::zeros);
    cx_dmat o(1,1,fill::zeros);
    o(0,0).real(1.0);

    a(0,0).imag(w*dpx*kx1);
    b=exp(a);
    a(0,0).imag(w*dpx*nx*dx);
    c=exp(a)-o;
    a(0,0).imag(w*dpx*dx);
    d=exp(a)-o;

    a=b*c/d;
    return a;
}
cx_fmat GetRegularHessianElem(float w,\
 int nx, float kx1, float dx, float dpx,\
 int ny, float ky1, float dy, float dpy)
{
    cx_dmat a(1,1,fill::zeros),b(1,1,fill::zeros);
    float n2(0.000000000000);
    float pdx(abs(dpx)),pdy(abs(dpy));

    if(pdx>0 && pdy>0){
        a=GetOneHessianElem(w, nx, kx1, dx, dpx);
        b=GetOneHessianElem(w, ny, ky1, dy, dpy);
    }
    else if(pdx<1e-9 && pdy>n2){
        a(0,0).real(nx);
        b=GetOneHessianElem(w, ny, ky1, dy, dpy);
    }
    else if(pdx>0 && pdy<1e-9){
        a=GetOneHessianElem(w, nx, kx1, dx, dpx);
        b(0,0).real(ny);
    }
    else if(pdx<1e-9 && pdy<1e-9){
        a(0,0).real(nx);
        b(0,0).real(ny);
    }
    
    a=a*b;
    cx_fmat af(1,1,fill::zeros);
    af(0,0).real(a(0,0).real());
    af(0,0).imag(a(0,0).imag());
    return af;
}
template <typename type1>
type1 GetHessianForLSLRT(LinerRadon3d * par,\
 type1 &hessiancg_cxfmat_p1ncpu_2npx2npy_small, int pnf)
{
    int kf,kpx,kpy,kpx2,kpy2,kx,ky,i,j;//cout<<"ok"<<endl;
    float w,pi(3.1415926);
    float df(par[0].df),dpx(par[0].dpx),dpy(par[0].dpy),p0x(par[0].p0x);
    int nx(par[0].nx),npx(par[0].npx),nf(par[0].nf),\
        ny(par[0].ny),npy(par[0].npy),nfft,k;
    cx_fmat a(1,1),A(nx,ny),B(nx,ny);

    kf=pnf;
    w=2.0*pi*df*(kf); 
    float fpxmin,fpymin;
    fpxmin=par[0].coordpx.min()-par[0].coordpx.max();
    fpymin=par[0].coordpy.min()-par[0].coordpy.max();
    if(par->regularization){
        float kx1=par->coordx.min();
        float dx=abs(par->coordx.max()-par->coordx.min())/(1e-9+nx-1.0); 
        float ky1=par->coordy.min();
        float dy=abs(par->coordy.max()-par->coordy.min())/(1e-9+ny-1.0);
        //cout<<"(dx,dy)="<<dx<<","<<dy<<endl;
        //cout<<"(fpxmin,fpymin)="<<fpxmin<<","<<fpymin<<endl;
        for(kpx2=0;kpx2<=(2*npx-2);kpx2++){
        for(kpy2=0;kpy2<=(2*npy-2);kpy2++){
            //float minDpx=w*abs(fpxmin+kpx2*dpx);
            //float minDpy=w*abs(fpymin+kpy2*dpy);
            //if(minDpx<0.0552233 && minDpy<0.0552233){
            a=GetRegularHessianElem( w,\
                nx, kx1, dx, fpxmin+kpx2*dpx,\
                ny, ky1, dy, fpymin+kpy2*dpy);
            hessiancg_cxfmat_p1ncpu_2npx2npy_small(kpx2,kpy2)\
                =a(0,0);
            //}
        }}
    }else{
        cx_fcube basey3d;
        cx_fmat basex,basey;
        cx_fmat basepy0,basepx0;
        cx_fmat basedpy,basedpx;
        basey.zeros(nx,ny);basex.zeros(nx,ny);
        basey3d.set_size(nx,ny,2*npy-1);
        basepy0.zeros(nx,ny);basepx0.zeros(nx,ny);
        basedpy.zeros(nx,ny);basedpx.zeros(nx,ny);
        basepx0.set_imag(w*par->coordx*(fpxmin-dpx));  
        basepy0.set_imag(w*par->coordy*(fpymin));
        basedpx.set_imag(w*par->coordx*dpx);  
        basedpy.set_imag(w*par->coordy*dpy);
        basedpy=exp(basedpy);basedpx=exp(basedpx);
        basepy0=exp(basepy0);basepx0=exp(basepx0);

        basex=basepx0;
        basey3d.slice(0)=basepy0;
        for(kpy2=0;kpy2<(2*npy-2);kpy2++){
        for(kx=0;kx<nx;kx++){
            for(ky=0;ky<ny;ky++){
            basey3d(kx,ky,kpy2+1)=basey3d(kx,ky,kpy2)*basedpy(kx,ky);
        }}}
        for(kpx2=0;kpx2<=(2*npx-2);kpx2++){
            for(kx=0;kx<nx;kx++){
            for(ky=0;ky<ny;ky++){
                basex(kx,ky)=basex(kx,ky)*basedpx(kx,ky);
            }}
        for(kpy2=0;kpy2<=(2*npy-2);kpy2++){
            a=cx_fmatmul(basex,basey=basey3d.slice(kpy2));
            hessiancg_cxfmat_p1ncpu_2npx2npy_small(kpx2,kpy2)\
                =a(0,0);
        }}
    }
    /*
    if(kf==90){
        fmat hessreal;
        hessreal.copy_size(hessiancg_cxfmat_p1ncpu_2npx2npy_small);
        for(kpx2=0;kpx2<=(2*npx-2);kpx2++){
        for(kpy2=0;kpy2<=(2*npy-2);kpy2++){
            hessreal(kpx2,kpy2)=\
                real(hessiancg_cxfmat_p1ncpu_2npx2npy_small(kpx2,kpy2));
        }}
        datawrite(hessreal,"hess.dat");
    }*/

    nfft=getfftnum(2*npx-1);
    type1 hessfft(nfft, 2*npy-1);
    for(kpy=0;kpy<npy;kpy++){
        float fpy2=(par[0].coordpy(kpy,0));
        for(kpy2=0;kpy2<npy;kpy2++){
        float fpy1=(par[0].coordpy(kpy2,0));
        int nkpy=round((fpy1-fpy2)/dpy)+npy-1;
        type1 a1(nfft,1,fill::zeros);
        kpx=0;
        float fpx1=(par[0].coordpx(kpx,0));
        for(kpx2=0;kpx2<npx;kpx2++){
            float fpx2=(par[0].coordpx(kpx2,0));
            int nkpx=round((fpx1-fpx2)/dpx)+npx-1;
            a1(kpx2,0)=hessiancg_cxfmat_p1ncpu_2npx2npy_small(nkpx,nkpy);
        }
        for(k=1;k<npx;k++){
            a1(nfft-k,0)=a1(k,0);
        }

        kpx2=0;
        float fpx2=(par[0].coordpx(kpx2,0));
        for(kpx=0;kpx<npx;kpx++){
            float fpx1=(par[0].coordpx(kpx,0));
            int nkpx=round((fpx1-fpx2)/dpx)+npx-1;
            a1(kpx,0)=hessiancg_cxfmat_p1ncpu_2npx2npy_small(nkpx,nkpy);
        }
        hessfft.col(nkpy)=fft(a1.col(0));
    }}
    return hessfft;
}

void LinerRadon3d::GetHessianAll()
{
    int nfn(nf2-nf1);
    cx_dmat hessfft;
    hessfft.zeros(2*npx-1,2*npy-1);
    hessfft=GetHessianForLSLRT(this,hessfft,1);
    float memoryGB=nfn*(hessfft.n_rows)*(hessfft.n_cols)*4\
        *sizeof(memoryGB)/1000.0/1000.0/1000.0;
    cout<<"Warning: memory GB will be used: "<<memoryGB<<endl;
    this->hessfftAll.zeros(hessfft.n_rows,hessfft.n_cols,nfn);
    hessfft.clear();
omp_set_num_threads(nthread);
#pragma omp parallel for
    for(int kf=nf1;kf<nf2;kf++){
        //cout<<"kf="<<kf<<endl;
        cx_dmat hessfftIn;
        hessfftIn.zeros(2*npx-1,2*npy-1);
        hessfftIn=GetHessianForLSLRT(this,hessfftIn,kf);
        hessfftAll.slice(kf-nf1)=hessfftIn;
    }
    this->phessfftAll=&hessfftAll;
}
template <typename type1,typename type2>
void cxMatget_Ap_small_fft(type1 & Ap, \
    type2 & hessfft, type1 & p, LinerRadon3d * par)
{
    Ap.fill(0.0);
    int kf,kpx,kpy,kpx2,kpy2,kx,ky,k,nfft;//cout<<"ok"<<endl;
    float df(par[0].df),dpx(par[0].dpx),dpy(par[0].dpy),p0x(par[0].p0x);
    int nx(par[0].nx),npx(par[0].npx),nf(par[0].nf),\
        ny(par[0].ny),npy(par[0].npy);

    nfft=hessfft.n_rows;
    for(kpy=0;kpy<npy;kpy++){
        float fpy2=(par[0].coordpy(kpy,0));
        type1 dfft(nfft,1,fill::zeros),d(nfft,1,fill::zeros);
        for(kpx=0;kpx<npx;kpx++){
            d(kpx,0)=p(kpx,kpy);
        }
        dfft.col(0)=fft(d.col(0));
        for(kpy2=0;kpy2<npy;kpy2++){
            float fpy1=(par[0].coordpy(kpy2,0));
            int nkpy=round((fpy1-fpy2)/dpy)+npy-1;
            type2 a1fft(nfft,1,fill::zeros),a1ifft(nfft,1,fill::zeros);

            a1fft.col(0)=hessfft.col(nkpy);
            for(k=0;k<nfft;k++){
                a1fft(k,0)*=dfft(k,0);
            }
            a1ifft.col(0)=ifft(a1fft.col(0));
            for(kpx=0;kpx<npx;kpx++){
                Ap(kpx,kpy2)+=a1ifft(kpx,0);
            }
        }
    }
    type1 matp;
    matp.copy_size(par->weightPxy);
    matp.fill(0.0);
    matcopy(matp,par->weightPxy);
    for(kpx=0;kpx<npx;kpx++){
    for(kpy=0;kpy<npy;kpy++){
        Ap(kpx,kpy)+=matp(kpx,kpy)*p(kpx,kpy);
    }}
}

template <typename type1>
inline type1 cxMatmul_CG(type1 & mat1, type1 & mat2)
{
    int nz,nx;
    nz=mat1.n_rows;
    nx=mat1.n_cols;

    type1 a(1,1,fill::zeros);
    int i,j;
    for(i=0;i<nz;i++){
        a+=mat1.row(i)*mat2.row(i).t();
    }
    return a;
}
void LSLRT_CG3d_thread(LinerRadon3d * par,\
 bool *convergence,int kf1, int kf2,bool regularization,\
 int iterations_num, float residual_ratio)
{
    for(int kf=kf1;kf<kf2;kf++){
        cx_dmat hessfft=par[0].phessfftAll[0].slice(kf-par[0].nf1);
        int ip,jp,in,jn,k,iter(0);
        int np1(par->dataFP.n_rows),np2(par->dataFP.n_cols);
        cx_dmat gradient_rk,gradient_rk_1,\
            gradient_cg_pk,gradient_cg_pk_1,\
            datatp_k,datatp_k_1,\
            A_gradient_cg_pk,A_datatp_k;
        dmat sum_num(1,1),residual_pow(1,1),residual_k(1,1);
        cx_dmat beta_k(1,1),alpha_k(1,1);
        datatp_k.copy_size(par[0].dataFP.slice(kf));
        datatp_k_1.copy_size(datatp_k);
        gradient_rk.copy_size(datatp_k);
        gradient_rk_1.copy_size(datatp_k);
        gradient_cg_pk.copy_size(datatp_k);
        gradient_cg_pk_1.copy_size(datatp_k); 
        A_gradient_cg_pk.copy_size(datatp_k);
        A_datatp_k.copy_size(datatp_k);

        iter=0;
        //datatp_k=par[0].datafP.slice(kf);
        cxmatcopy(datatp_k,par[0].dataFP.slice(kf));
        datatp_k_1=datatp_k;
        datatp_k.set_real(real(datatp_k)/par[0].nx/par[0].ny);
        datatp_k.set_imag(imag(datatp_k)/par[0].nx/par[0].ny);
        datatp_k=2.0*datatp_k;
        cxMatget_Ap_small_fft(A_datatp_k,hessfft,datatp_k,par);
        gradient_rk=datatp_k_1-A_datatp_k;
        gradient_cg_pk=gradient_rk;
        cxMatget_Ap_small_fft(A_gradient_cg_pk,hessfft,gradient_cg_pk,par);
        
        sum_num=sum(sum(abs(datatp_k)));
        residual_pow=sum_num(0,0);
        residual_pow*=residual_ratio;    
        
        alpha_k=cxMatmul_CG(gradient_cg_pk,A_gradient_cg_pk);
        if(abs(alpha_k(0,0))>=0.000000000001){
            alpha_k=cxMatmul_CG(gradient_rk,gradient_rk)/alpha_k;
        }else{
            iterations_num=-1;
        }
        //get new solution
        datatp_k_1=datatp_k+alpha_k(0,0)*gradient_cg_pk;
        gradient_rk_1=gradient_rk-alpha_k(0,0)*A_gradient_cg_pk;

        beta_k=cxMatmul_CG(gradient_rk,gradient_rk);
        if(abs(beta_k(0,0))>=0.000000000001){
            beta_k=cxMatmul_CG(gradient_rk_1,gradient_rk_1)/beta_k;
        }else{
            iterations_num=-1;
        }
        gradient_cg_pk_1=gradient_rk_1+beta_k(0,0)*gradient_cg_pk;
        //updata
        datatp_k=datatp_k_1;
        gradient_rk=gradient_rk_1;
        gradient_cg_pk=gradient_cg_pk_1;
        //cal residual_pow
        sum_num=sum(sum(abs(gradient_rk)));
        residual_k=sum_num;

        while(iter<iterations_num && residual_k(0,0)>residual_pow(0,0)){
            iter++;
        cxMatget_Ap_small_fft(A_gradient_cg_pk,hessfft,gradient_cg_pk,par);
            alpha_k=cxMatmul_CG(gradient_cg_pk,A_gradient_cg_pk);
            if(abs(alpha_k(0,0))<0.000000000001){break;}
            alpha_k=cxMatmul_CG(gradient_rk,gradient_rk)/alpha_k;
            //get new solution
            datatp_k_1=datatp_k+alpha_k(0,0)*gradient_cg_pk;
            gradient_rk_1=gradient_rk-alpha_k(0,0)*A_gradient_cg_pk;

            beta_k=cxMatmul_CG(gradient_rk,gradient_rk);
            if(abs(beta_k(0,0))<0.000000000001){break;}
            beta_k=cx_fmatmul_CG(gradient_rk_1,gradient_rk_1)/beta_k;
            gradient_cg_pk_1=gradient_rk_1+real(beta_k(0,0))*gradient_cg_pk;
            //updata
            datatp_k=datatp_k_1;
            gradient_rk=gradient_rk_1;
            gradient_cg_pk=gradient_cg_pk_1;
            //cal residual_pow
            sum_num=sum(sum(sum(abs(gradient_rk))));
            residual_k=sum_num;

        }
        //sum_num=sum(sum(sum(abs(datatp_k))));
        //cout<<residual_pow(0,0)<<"||"<<kf<<"||"<<iter<<"||"\
            <<residual_k(0,0)/residual_pow(0,0)<<"||"<<sum_num(0,0)<<endl;

        if((residual_k(0,0)/residual_pow(0,0))>(1.0)\
            || (residual_k(0,0))!=(residual_k(0,0))){
                par[0].dataFP.slice(kf).fill(0.0);
                //cxmatcopy(par[0].datafP.slice(kf),datatp_k);
                convergence[kf]=false;
            }
            else{
                //par[0].datafP.slice(kf)=datatp_k;
                cxmatcopy(par[0].dataFP.slice(kf),datatp_k);
                convergence[kf]=true;
            }
    }
}
void LSLRT_CG3d_redo_thread(struct LinerRadon3d * par,\
 bool *convergence,int kf1, int kf2,bool regularization,\
 int iterations_num, float residual_ratio)
{
    for(int kf=kf1;kf<kf2;kf++){
        if(convergence[kf]){
            continue;
        }
        else{
        cx_dmat hessfft;
        hessfft=par[0].phessfftAll[0].slice(kf-par[0].nf1);

        int ip,jp,in,jn,k,iter(0);
        int np1(par->dataFP.n_rows),np2(par->dataFP.n_cols);
        cx_dmat gradient_rk,gradient_rk_1,\
            gradient_cg_pk,gradient_cg_pk_1,\
            datatp_k,datatp_k_1,\
            A_gradient_cg_pk,A_datatp_k;
        dmat sum_num(1,1),residual_pow(1,1),residual_k(1,1);
        cx_dmat beta_k(1,1),alpha_k(1,1);
        datatp_k.zeros(np1,np2);
        datatp_k_1.copy_size(datatp_k);
        gradient_rk.copy_size(datatp_k);
        gradient_rk_1.copy_size(datatp_k);
        gradient_cg_pk.copy_size(datatp_k);
        gradient_cg_pk_1.copy_size(datatp_k); 
        A_gradient_cg_pk.copy_size(datatp_k);
        A_datatp_k.copy_size(datatp_k);

        iter=0;
        cxmatcopy(datatp_k,par[0].dataFP.slice(kf));
        sum_num=sum(sum(sum(abs(datatp_k))));
        residual_pow=sum_num(0,0);
        residual_pow*=residual_ratio;
        //datatp_k.fill(0.0);

    //    cxfmatget_Ap_small(A_datatp_k,hess_A,datatp_k,kcpu,par);
        cxMatget_Ap_small_fft(A_datatp_k,hessfft,datatp_k,par);
        gradient_rk=datatp_k-A_datatp_k;
        gradient_cg_pk=gradient_rk;

    //    cxfmatget_Ap_small(A_gradient_cg_pk,hess_A,gradient_cg_pk,kcpu,par);
        cxMatget_Ap_small_fft(A_gradient_cg_pk,hessfft,gradient_cg_pk,par);
            
        alpha_k=cxMatmul_CG(gradient_cg_pk,A_gradient_cg_pk);
        if(abs(alpha_k(0,0))>=0.000000000001){
            alpha_k=cxMatmul_CG(gradient_rk,gradient_rk)/alpha_k;
        }else{
            iterations_num=-1;
        }
        //get new solution
        datatp_k_1=datatp_k+alpha_k(0,0)*gradient_cg_pk;
        gradient_rk_1=gradient_rk-alpha_k(0,0)*A_gradient_cg_pk;

        beta_k=cxMatmul_CG(gradient_rk,gradient_rk);
        if(abs(beta_k(0,0))>=0.000000000001){
            beta_k=cxMatmul_CG(gradient_rk_1,gradient_rk_1)/beta_k;
        }else{
            iterations_num=-1;
        }
        gradient_cg_pk_1=gradient_rk_1+beta_k(0,0)*gradient_cg_pk;
        //updata
        datatp_k=datatp_k_1;
        gradient_rk=gradient_rk_1;
        gradient_cg_pk=gradient_cg_pk_1;
        //cal residual_pow
        sum_num=sum(sum(sum(abs(gradient_rk))));
        residual_k=sum_num;

        while(iter<iterations_num && residual_k(0,0)>residual_pow(0,0)){
            //cout<<iter<<"||"<<residual_k/residual_pow<<endl;
            iter++;

            cxMatget_Ap_small_fft(A_gradient_cg_pk,hessfft,gradient_cg_pk,par);
            
            alpha_k=cxMatmul_CG(gradient_cg_pk,A_gradient_cg_pk);
            if(abs(alpha_k(0,0))<0.000000000001){
                break;
            }
            alpha_k=cxMatmul_CG(gradient_rk,gradient_rk)/alpha_k;
            //get new solution
            datatp_k_1=datatp_k+alpha_k(0,0)*gradient_cg_pk;
            gradient_rk_1=gradient_rk-alpha_k(0,0)*A_gradient_cg_pk;

            beta_k=cxMatmul_CG(gradient_rk,gradient_rk);
            if(abs(beta_k(0,0))<0.000000000001){
                break;
            }
            beta_k=cxMatmul_CG(gradient_rk_1,gradient_rk_1)/beta_k;
            gradient_cg_pk_1=gradient_rk_1+real(beta_k(0,0))*gradient_cg_pk;
            //updata
            datatp_k=datatp_k_1;
            gradient_rk=gradient_rk_1;
            gradient_cg_pk=gradient_cg_pk_1;
            //cal residual_pow
            sum_num=sum(sum(sum(abs(gradient_rk))));
            residual_k=sum_num;
        }
        if((residual_k(0,0)/residual_pow(0,0))>(1.0)\
            ||(residual_k(0,0))!=(residual_k(0,0))){
                par[0].dataFP.slice(kf).fill(0.0);
                convergence[kf]=false;
                cout<<"kf="<<kf<<" ||iteration times:"<<iter<<" ||err level:"<<\
                residual_k(0,0)/residual_pow(0,0)<<" ("<<convergence[kf]<<endl;
            }
            else{
                cxmatcopy(par[0].dataFP.slice(kf),datatp_k);
                convergence[kf]=true;
            }
        }
    }
}

fmat smoothdig(fmat dig,int l,int n)
{
    fmat d1(l,1),d2(l,1);
    d1=dig,d2=dig;
    int j,k;
    for(j=0;j<n;j++)
    {
        for(k=1;k<l-1;k++)
        {
            d2(k,0)=(0.5*d1(k-1,0)+0.5*d1(k+1,0)+d1(k,0))/2;
        }
        d1=d2;
    }
    dig=d2;
    return dig;
}
void GetWeightPxy(LinerRadon3d & par,\
 fmat &digw_fmat_npxnpy)
{
    digw_fmat_npxnpy.fill(0.0);
    int k,i;
    float maxpower,minpower(0);
    cx_fmat cxmat(par.npx,par.npy);
    for(k=par.rulef1;k<par.rulef2;k++){
        digw_fmat_npxnpy+=abs(cxmat=par.dataFP.slice(k));
    } 

    maxpower=((digw_fmat_npxnpy.max()));
    minpower=((digw_fmat_npxnpy.min()));
    digw_fmat_npxnpy=(digw_fmat_npxnpy-minpower)/\
        (maxpower-minpower);
    digw_fmat_npxnpy+=sum(sum(digw_fmat_npxnpy))/\
        digw_fmat_npxnpy.n_elem;
    digw_fmat_npxnpy=1.0/digw_fmat_npxnpy;
    maxpower=((digw_fmat_npxnpy.max()));
    minpower=((digw_fmat_npxnpy.min()));
    digw_fmat_npxnpy=(digw_fmat_npxnpy-minpower)/\
        (maxpower-minpower);

    par.weightPxy=digw_fmat_npxnpy;
    if(par.npy>=5){
        par.weightPxy=fmatsmooth(par.weightPxy,par.npx,par.npy,1);
        maxpower=((par.weightPxy.max()));
        minpower=((par.weightPxy.min()));
        par.weightPxy=(par.weightPxy-minpower)/\
            (maxpower-minpower);
        for(k=0;k<par.npy;k++){
            for(i=0;i<par.npx;i++){
                par.weightPxy(i,k)=1.0/(1.0+exp(100*(0.5-par.weightPxy(i,k))));
            }
            //datawrite(par.p_power,par.npx,par.npy,"dig.bin");
        }
    }
    else{
        fmat digline(par.npx,1);
        for(k=0;k<par.npy;k++){
            par.weightPxy.col(k)=smoothdig(par.weightPxy.col(k),par.npx,99);
        }
        maxpower=((par.weightPxy.max()));
        minpower=((par.weightPxy.min()));
        par.weightPxy=(par.weightPxy-minpower)/\
            (maxpower-minpower);
        par.weightPxy+=par.parNorm;
        for(k=0;k<par.npy;k++){
            for(i=0;i<par.npx;i++){
                par.weightPxy(i,k)=1.0/(1.0+exp(100*(0.5-par.weightPxy(i,k))));
            }
            //datawrite(digline=par.p_power.col(k),par.npx,1,"dig.bin");
        }
    }
    //get diagonally weighted matrix W
    digw_fmat_npxnpy=par.digNormL2+par.digNormL1*par.weightPxy;
    par.weightPxy=digw_fmat_npxnpy;
}

fcube getGauss3dWin(int halfw1, int halfw2, int halfw3)
{
    int n1(2*halfw1+1),n2(2*halfw2+1),n3(2*halfw3+1);
    int s1(halfw1),s2(halfw2),s3(halfw3);
    fcube gauss3d(n1,n2,n3);
    gauss3d.fill(0.0);
    float var1,var2,var3;
    var1=(s1/3.0);var1=var1*var1;
    if(var1<0.000001){var1=1.0;}
    var2=(s2/3.0);var2=var2*var2;
    if(var2<0.000001){var2=1.0;}
    var3=(s3/3.0);var3=var3*var3;
    if(var3<0.000001){var3=1.0;}
    float uniNum(0.0);
    for(int i1=0;i1<n1;i1++){
        float l1=-(i1-s1)*(i1-s1)/var1;
    for(int i2=0;i2<n2;i2++){
        float l2=-(i2-s2)*(i2-s2)/var2;
    for(int i3=0;i3<n3;i3++){
        float l3=-(i3-s3)*(i3-s3)/var3;
        gauss3d(i1,i2,i3)=exp(0.5*(l1+l2+l3));
        uniNum+=gauss3d(i1,i2,i3);
    }}}
    //gauss3d/=uniNum;
    return gauss3d;
}
fcube mulGauss3dWin(const fcube & data3d,\
    const fcube & gauss3d,\
    int s1, int s2, int s3,\
    int halfw1, int halfw2, int halfw3)
{
    fcube dataGabor;
    dataGabor.copy_size(data3d);
    dataGabor.fill(0.0);
    int n1(2*halfw1+1),n2(2*halfw2+1),n3(2*halfw3+1);
    for(int i1=0;i1<n1;i1++){
        int k1=s1+i1-halfw1;
    for(int i2=0;i2<n2;i2++){
        int k2=s2+i2-halfw2;
    for(int i3=0;i3<n3;i3++){
        int k3=s3+i3-halfw3;
        dataGabor(k1,k2,k3)=data3d(k1,k2,k3)\
            *gauss3d(i1,i2,i3);
    }}}
    return dataGabor;    
}

void LinerRadon3d::GetDataTPbyLSLRT(int iterations_num,float residual_ratio)
{
    int ncpu(nthread);
    ncpu=min(ncpu,nf2-nf1);
    ncpu=max(ncpu,1);
    bool isRegularData=this->regularization;
    int kcpu,kf,k,i,j;//cout<<"ok"<<endl;

    fmat dig_w_fmat(npx, npy);
    GetWeightPxy(this[0], dig_w_fmat);
    //datawrite(dig_w_fmat,par.npx,par.npy,"dig.bin");
    
    bool *convergence;
    convergence=new bool[nf2];
    int nfn=nf2-nf1;

    if(ncpu==1){
        for(kf=0;kf<nf2;kf++){convergence[kf]=true;}
        for(kf=nf1;kf<nf2;kf++){
            LSLRT_CG3d_thread(this,\
                convergence,kf,kf+1,isRegularData,\
                iterations_num,residual_ratio);
            //cout<<"LRT CG thread: "<<kf<<" Started"<<endl;
        }
    }else{
        ThreadPool pool(ncpu,ncpu+1);
        for(kf=0;kf<nf2;kf++){convergence[kf]=true;}
        for(kf=nf1;kf<nf2;kf++){
            pool.AddNormalTask(LSLRT_CG3d_thread,this,\
                convergence,kf,kf+1,isRegularData,\
                iterations_num,residual_ratio);
            //cout<<"LRT CG thread: "<<kf<<" Started"<<endl;
        }
        pool.WaitForTask();
    }
//cout<<"==========cg has finished=========="<<endl;
    cx_fmat s1,s2;
    bool *convergence2;
    convergence2=new bool[nf2];
    int kk,k1(0),k2(0);
    for(kf=nf1;kf<nf2;kf++){
        convergence2[kf]=convergence[kf];
        if(!convergence[kf]){
            k2++;
        }
    }
    for(kf=nf2-2;kf>=nf1+1;kf--){
        if(!convergence2[kf]&&convergence2[kf-1]&&convergence2[kf+1]){
            s1=dataFP.slice(kf-1);
            s1+=dataFP.slice(kf+1);
            dataFP.slice(kf).set_real(real(s1)/2.0);
            dataFP.slice(kf).set_imag(imag(s1)/2.0);
            convergence2[kf]=true;
        }
        else if(!convergence2[kf]&&convergence2[kf+1]){
            dataFP.slice(kf)=dataFP.slice(kf+1);
            convergence2[kf]=true;
        }
    }
    dataRealTP.zeros(npx,npy,nt);
    fx2tx_3d_thread(dataRealTP,dataFP,ncpu);
    delete[] convergence;
    delete[] convergence2;

}

///////////////////////////////////////////////////////////////////////////////
inline float fcube_inner_product(fcube &data1, fcube &data2, int nthread=1)
{
    int ni(data1.n_rows),nj(data1.n_cols),nk(data1.n_slices);
    double *inner_num, inner_double(0.0);
    inner_num=new double[ni];

omp_set_num_threads(nthread);
#pragma omp parallel for
    for(int i=0;i<ni;i++){
        inner_num[i]=0.0;
    for(int j=0;j<nj;j++){
    for(int k=0;k<nk;k++){
        inner_num[i]+=double(data1(i,j,k)*data2(i,j,k));
    }}}
    for(int i=0;i<ni;i++){
        inner_double+=inner_num[i];
    }
    delete [] inner_num;
    inner_num=nullptr;
    float inner=float(inner_double);
    return inner;
} 
void fcubemul(fcube& outdata, fcube& mat1, fcube& mat2, int nthread=1)
{
    int nx,ny,nz;
    ny=outdata.n_cols;
    nx=outdata.n_rows;
    nz=outdata.n_slices;

omp_set_num_threads(nthread);
#pragma omp parallel for
    for(int i=0;i<nx;i++){
    for(int j=0;j<ny;j++){
    for(int k=0;k<nz;k++){
        outdata(i,j,k)=mat1(i,j,k)*mat2(i,j,k);
    }}}
}
float fcubedot(fcube& mat1, fcube& mat2, int nthread=1)
{
    int nx,ny,nz;
    ny=mat1.n_cols;
    nx=mat1.n_rows;
    nz=mat1.n_slices;
    float *apow,num;
    apow=new float[nx];
omp_set_num_threads(nthread);
#pragma omp parallel for
    for(int i=0;i<nx;i++){
        apow[i]=0.0;
    for(int j=0;j<ny;j++){
    for(int k=0;k<nz;k++){
        apow[i]+=mat1(i,j,k)*mat2(i,j,k);
    }}}
    num=0.0;
    for(int i=0;i<nx;i++){
        num+=apow[i];
    }
    delete [] apow;
    apow=nullptr;
    return num;
}
void RadonMixFTCG3d_getdigfmat(class LinerRadon3d & par,\
 fmat &digw_fmat_npxnpy)
{
    digw_fmat_npxnpy.fill(par.digNormL2);
    par.weightPxy=digw_fmat_npxnpy;
}
void get_anti_aliasing_weighted_RadonMixFTCG3d\
    (fcube &filter, fcube data1, int nfrule, \
    int nthread=1, int wx=1, int wy=1, int wz=20,
    float weightNoiseLevel=0.5, float weightEdgeSharpness=2.0)
{
    int nx,ny,nz;
    int win;
    cx_fcube datafx;
    fcube data2,dataStru;
    data2=data1;
    datafx.copy_size(data1);
    tx2fx_3d_thread(datafx, data1, nthread);
    for(int k=nfrule;k<datafx.n_slices;k++){
        datafx.slice(k).fill(0.0);
    }
    for(int k=nfrule/2;k<nfrule;k++){
        datafx.slice(k).set_real(real(datafx.slice(k))*Blackman(k,nfrule/2));
        datafx.slice(k).set_imag(imag(datafx.slice(k))*Blackman(k,nfrule/2));
    }
    fx2tx_3d_thread(data1, datafx, nthread);
    data1=data1/data1.max();

    ny=data1.n_cols;
    nx=data1.n_rows;
    nz=data1.n_slices;
    //datawrite3d_bycol_transpose(data1,nz,nx,"./swap/datataup.rule.dat");
    //datawrite3d_bycol_transpose(data2,nz,nx,"./swap/datataup.dat");

    win=(nx-1)/2;
    int wx1=min(win,1);
    win=(ny-1)/2;
    int wy1=min(win,1);
    win=(nz-1)/2;
    int wz1=min(win,1);
    dataStru.copy_size(data1);
    dataStru.fill(0.0);
    float amax2=data2.max()*1e-6;
    float amax1=data1.max()*1e-6;
/*
omp_set_num_threads(nthread);
#pragma omp parallel for
    for(int i=wx1;i<nx-wx1;i++){
        float dx1(0.0),dy1(0.0),dz1(0.0);
        float dx2(0.0),dy2(0.0),dz2(0.0);
    for(int j=wy1;j<ny-wy1;j++){
    for(int k=wz1;k<nz-wz1;k++){
        if(ny>2){
            dy1=data1(i,j+1,k)-data1(i,j-1,k);
            dy2=data2(i,j+1,k)-data2(i,j-1,k);
        }
        if(nx>2){
            dx1=data1(i+1,j,k)-data1(i-1,j,k)+amax1;
            dx2=data2(i+1,j,k)-data2(i-1,j,k);
        }
        if(nz>2){
            dz1=data1(i,j,k+1)-data1(i,j,k-1);
            dz2=data2(i,j,k+1)-data2(i,j,k-1)+amax2;
        }
        dataStru(i,j,k)=(dx1*dx2+dy1*dy2+dz1*dz2)\
            /(sqrt(dx1*dx1+dy1*dy1+dz1*dz1+amax1)\
            /sqrt(dx2*dx2+dy2*dy2+dz2*dz2+amax2));
    }}}
    if(ny>3){
        dataStru=fcubesmooth(dataStru,9);
    }else{
        for(int k=0;k<ny;k++){
            dataStru.col(k)=fmatsmooth(dataStru.col(k),nx,nz,9);
        }
    }
    //datawrite3d_bycol_transpose(dataStru,nz,nx,"./swap/datataup.weight.dat");
*/
    win=(nx-1)/2;
    wx=min(win,wx);
    win=(ny-1)/2;
    wy=min(win,wy);
    win=(nz-1)/2;
    wz=min(win,wz);
    filter.copy_size(data1);
    filter.fill(0.0);
omp_set_num_threads(nthread);
#pragma omp parallel for
    for(int i=0;i<nx;i++){
        float num=0;
    for(int j=0;j<ny;j++){
    for(int k=0;k<nz;k++){
        //fcube win1(2*wx+1,2*wy+1,2*wz+1),\
            win2(2*wx+1,2*wy+1,2*wz+1);
        //win1=data1(span(i-wx,i+wx),span(j-wy,j+wy),span(k-wz,k+wz));
        //win2=data2(span(i-wx,i+wx),span(j-wy,j+wy),span(k-wz,k+wz));
        //filter(i,j,k)=fcubedot(win1,win2);
        for(int i1=max(i-wx,0);i1<=min(i+wx,nx-1);i1++){
        for(int j1=max(j-wy,0);j1<=min(j+wy,ny-1);j1++){
        for(int k1=max(k-wz,0);k1<=min(k+wz,nz-1);k1++){
            //filter(i,j,k)+=(data1(i+i1,j+j1,k+k1)*data2(i+i1,j+j1,k+k1)+\
                data1(i-i1,j+j1,k+k1)*data2(i-i1,j+j1,k+k1)+\
                data1(i+i1,j-j1,k+k1)*data2(i+i1,j-j1,k+k1)+\
                data1(i-i1,j-j1,k+k1)*data2(i-i1,j-j1,k+k1)+\
                data1(i+i1,j+j1,k-k1)*data2(i+i1,j+j1,k-k1)+\
                data1(i-i1,j+j1,k-k1)*data2(i-i1,j+j1,k-k1)+\
                data1(i+i1,j-j1,k-k1)*data2(i+i1,j-j1,k-k1)+\
                data1(i-i1,j-j1,k-k1)*data2(i-i1,j-j1,k-k1));
            filter(i,j,k)+=(data1(i1,j1,k1)*data1(i1,j1,k1));
            //filter(i,j,k)+=abs(dataStru(i1,j1,k1));
            //apow2+=(data2(i+i1,j+j1,k+k1)*data2(i+i1,j+j1,k+k1));
        }}}
        //filter(i,j,k)=sqrt(abs(filter(i,j,k)));
    }}}

    if(ny>3){
        filter=fcubesmooth(filter,8);
    }else{
        for(int k=0;k<ny;k++){
            filter.col(k)=fmatsmooth(filter.col(k),nx,nz,8);
        }
    }
    filter-=filter.min();
    filter=(filter)/filter.max();
    for(int i=0;i<nx;i++){
    for(int j=0;j<ny;j++){
    for(int k=0;k<nz;k++){
        filter(i,j,k)=1.0/(1.0+exp(weightEdgeSharpness\
            *(weightNoiseLevel-filter(i,j,k))));
    }}}
    filter=(filter)/filter.max();
    filter+=0.000001;

    filter=1.0/filter;
    filter-=filter.min();
    filter=(filter)/filter.max();
    float aveNum=accu(filter)/filter.n_elem;
    for(int i=0;i<nx;i++){
    for(int j=0;j<ny;j++){
    for(int k=0;k<nz;k++){
        if(filter(i,j,k)>5*(aveNum)){
            filter(i,j,k)=5*aveNum;
        }
    }}}
    filter=(filter)/filter.max();
}
void RadonMixFTCG3d_LTL_operator(fcube & Ap,
    fcube & p, class LinerRadon3d & par)
{
    cx_fcube pfx,Apfx;
    pfx.copy_size(p);
    Apfx.copy_size(Ap);
    Apfx.fill(0.0);
    pfx.fill(0.0);
    tx2fx_3d_thread(pfx,(&p)[0],par.nthread);

omp_set_num_threads(par.nthread);
#pragma omp parallel for
    for(int kf=par.nf1;kf<par.nf2;kf++){
        cx_fmat Apkf,pkf;
        Apkf.copy_size(par.dataFP.slice(0));
        pkf.copy_size(par.dataFP.slice(0));
        cx_dmat hessfft=par.phessfftAll[0].slice(kf-par.nf1);
        //cxmatcopy(hessfft,par.hessianf2npx2npy[kf-par.nf1]);
        pkf=pfx.slice(kf);
        cxMatget_Ap_small_fft(Apkf,hessfft,pkf,&par);
        Apfx.slice(kf)=Apkf;
    }
    fx2tx_3d_thread((&Ap)[0],Apfx,par.nthread);
    Ap*=2.0;
}
/*
*/
void LinerRadon3d::GetDataTPbyMixftLSLRT(int iterations_num,float residual_ratio,int halfWinWide)
{
    int nthread(this->nthread);
    this->weightPxy.zeros(this->npx,this->npy);
    this->weightTX.copy_size(this->dataRealTP);
    this->weightTX.fill(0.0);
    get_anti_aliasing_weighted_RadonMixFTCG3d\
        (this->weightTX,this->dataRealTP,this->rulef2,\
        nthread,1,1,halfWinWide,this->weightNoiseLevel,\
        1.0/this->weightSparseness);
    float dataxs(1.0);
    this->weightTX*=(this->digNormL1);
    this->weightTX+=(this->digNormL2);
    //datawrite3d_bycol_transpose(this->weightTX,this->nt,\
        this->npx,"./swap/weight.bin");

    int ip,jp,in,jn,k,iter(0);
    int n1(this->dataTX.n_rows),n2(this->dataTX.n_cols),n3(this->dataTX.n_slices),\
        np1(this->dataRealTP.n_rows),np2(this->dataRealTP.n_cols);
    fcube gradient_rk,gradient_rk_1,\
        gradient_cg_pk,gradient_cg_pk_1,\
        datatp_k,datatp_k_1,datatp_weighted,\
        A_gradient_cg_pk,A_datatp_k;
    fmat sum_num(1,1);
    float beta_k,alpha_k,residual_pow,residual_k;
    datatp_weighted.copy_size(this->dataRealTP);
    datatp_k.copy_size(datatp_weighted);
    datatp_k_1.copy_size(datatp_weighted);
    gradient_rk.copy_size(datatp_weighted);
    gradient_rk_1.copy_size(datatp_weighted);
    gradient_cg_pk.copy_size(datatp_weighted);
    gradient_cg_pk_1.copy_size(datatp_weighted); 
    A_gradient_cg_pk.copy_size(datatp_weighted);
    A_datatp_k.copy_size(datatp_weighted);
    
    iter=0;
    datatp_k=this->dataRealTP/float(this->nx*this->ny);
    //slantstack3d_recover_L_operator(recoverdatatx_uk,datatp_k,\
        this->ptrace_coord,this->pline_coord,this->ntrace_coordx,this->nline_coordy,\
        this->dt,this->numthread);
    //slantstack3d_stack_LT_operator(A_datatp_k,recoverdatatx_uk,\
        this->ptrace_coord,this->pline_coord,this->ntrace_coordx,this->nline_coordy,\
        this->dt,this->numthread);
    RadonMixFTCG3d_LTL_operator(A_datatp_k,datatp_k,this[0]);

    //regularization
    fcubemul(datatp_weighted,datatp_k,this->weightTX,nthread);
    A_datatp_k=A_datatp_k+datatp_weighted;
    gradient_rk=this->dataRealTP-A_datatp_k;
    gradient_cg_pk=gradient_rk;

    //cal residual_pow
    sum_num(0,0)=fcube_inner_product(gradient_rk,gradient_rk,nthread);
    residual_k=sum_num(0,0);
    residual_pow=residual_k;
    residual_pow*=residual_ratio;
    //residual_pow/=float(this->nt);
    float max_residual_k=1000.0*residual_k;

    //slantstack3d_recover_L_operator(recoverdatatx_uk,gradient_cg_pk,\
        this->ptrace_coord,this->pline_coord,this->ntrace_coordx,this->nline_coordy,\
        this->dt,this->numthread);
    //slantstack3d_stack_LT_operator(A_gradient_cg_pk,recoverdatatx_uk,\
        this->ptrace_coord,this->pline_coord,this->ntrace_coordx,this->nline_coordy,\
        this->dt,this->numthread);
    RadonMixFTCG3d_LTL_operator(A_gradient_cg_pk,gradient_cg_pk,this[0]);
    //regularization
    fcubemul(datatp_weighted,gradient_cg_pk,this->weightTX,nthread);
    A_gradient_cg_pk=A_gradient_cg_pk+datatp_weighted;

    alpha_k=fcube_inner_product(gradient_cg_pk,A_gradient_cg_pk,nthread);
    if(abs(alpha_k)>=0.000000000001){
        alpha_k=sum_num(0,0)/alpha_k;
    }else{
        iterations_num=-1;
    }
    //get new solution
    datatp_k_1=datatp_k+alpha_k*gradient_cg_pk;
    gradient_rk_1=gradient_rk-alpha_k*A_gradient_cg_pk;

    beta_k=sum_num(0,0);
    if(abs(beta_k)>=0.000000000001){
        beta_k=fcube_inner_product(gradient_rk_1,gradient_rk_1,nthread)/beta_k;
    }else{
        iterations_num=-1;
    }
    gradient_cg_pk_1=gradient_rk_1+beta_k*gradient_cg_pk;
    //updata
    datatp_k=datatp_k_1;
    gradient_rk=gradient_rk_1;
    gradient_cg_pk=gradient_cg_pk_1;

    while(iter<iterations_num && residual_k>residual_pow){
        iter++;
        //cal residual_pow
        //sum_num=sum(sum(sum(abs(gradient_rk))));
        sum_num(0,0)=fcube_inner_product(gradient_rk,gradient_rk,nthread);
        residual_k=sum_num(0,0);
        if(residual_k>max_residual_k){
            break;
        }
        //cout<<"iterations times:"<<iter<<" || "<<\
        "err level:"<<residual_k/residual_pow<<endl;

        //slantstack3d_recover_L_operator(recoverdatatx_uk,gradient_cg_pk,\
            par.ptrace_coord,par.pline_coord,par.ntrace_coordx,par.nline_coordy,\
            par.dt,par.numthread);
        //slantstack3d_stack_LT_operator(A_gradient_cg_pk,recoverdatatx_uk,\
            par.ptrace_coord,par.pline_coord,par.ntrace_coordx,par.nline_coordy,\
            par.dt,par.numthread);
        RadonMixFTCG3d_LTL_operator\
            (A_gradient_cg_pk,gradient_cg_pk,this[0]);
        //regularization
        fcubemul(datatp_weighted,gradient_cg_pk,this->weightTX,nthread);
        A_gradient_cg_pk=A_gradient_cg_pk+datatp_weighted;
        
        alpha_k=fcube_inner_product(gradient_cg_pk,A_gradient_cg_pk,nthread);
        if(abs(alpha_k)<0.000000000001){break;}
        alpha_k=sum_num(0,0)/alpha_k;
        //get new solution
        datatp_k_1=datatp_k+alpha_k*gradient_cg_pk;
        gradient_rk_1=gradient_rk-alpha_k*A_gradient_cg_pk;

        beta_k=sum_num(0,0);
        if(abs(beta_k)<0.000000000001){break;}
        beta_k=fcube_inner_product(gradient_rk_1,gradient_rk_1,nthread)/beta_k;
        gradient_cg_pk_1=gradient_rk_1+beta_k*gradient_cg_pk;
        //updata
        datatp_k=datatp_k_1;
        gradient_rk=gradient_rk_1;
        gradient_cg_pk=gradient_cg_pk_1;
    }
    this->dataRealTP=datatp_k;
    cout<<" myid:"<<this->myid<<" || "<<\
        "iterations times:"<<iter<<" || "<<\
        "err level:"<<residual_k/residual_pow<<endl;
    if(iter<(iterations_num-1)&&(residual_k/residual_pow)<1.01)
        {this[0].isConverge=true;}
}

//    std::mutex mtxDataTP;
void LRT2dLocalOneTrace(int ix, LinerRadon3d* lrt3d,\
    fcube* data3dtx,fcube* win3d,fcube* data3dtp,fmat* coordx,fmat* coordy,\
    int halfw1,int halfw2,int iterations_num,float residual_ratio,\
    int weightHalfw3, float weightNoiseLevel, float weightSparseness)
{
    std::mutex mtxDataTP;
    int nt=lrt3d[0].nt;
    LinerRadon3d lrt3dLocal;
    lrt3dLocal=lrt3d[0];
    fmat coordxLocal,coordyLocal;
    coordxLocal=coordx[0](span(ix-halfw1,ix+halfw1),span::all);
    coordyLocal=coordy[0](span(ix-halfw1,ix+halfw1),span::all);
    lrt3dLocal.GetCoordx(coordxLocal);
    lrt3dLocal.GetCoordy(coordyLocal);
    lrt3dLocal.ParUpdate();
    lrt3dLocal.myid=ix;
    float normL2=lrt3dLocal.digNormL2;

    //lrt3dLocal.GetBasePlaneWaveFPX();
    //lrt3dLocal.GetHessianAll();
    fcube dataLocal3d,dataWin3d;
    //int it=halfw3;
    dataWin3d=mulGauss3dWin(data3dtx[0],win3d[0],ix,0,nt/2,halfw1,halfw2,nt/2-1);
    dataLocal3d=dataWin3d(span(ix-halfw1,ix+halfw1),span::all,span::all);

    lrt3dLocal.InputOriginalData(dataLocal3d);
    lrt3dLocal.ParUpdate();

    //lrt3dLocal.GetDataTPbyLSLRT(iterations_num,residual_ratio);
    lrt3dLocal.weightNoiseLevel=weightNoiseLevel;
    lrt3dLocal.weightSparseness=weightSparseness;

while(true){
    lrt3dLocal.GetDataTPbyLRT(true,true);
    lrt3dLocal.GetDataTPbyMixftLSLRT(iterations_num,residual_ratio,weightHalfw3);
    if(lrt3dLocal.isConverge){break;}
    else{
        residual_ratio*=2.0;
        lrt3dLocal.digNormL2+=normL2;
    }
    if(residual_ratio>1e-2){break;}
}
    mtxDataTP.lock();
    if(lrt3dLocal.isConverge)data3dtp[0]+=lrt3dLocal.dataRealTP;
    mtxDataTP.unlock();
/*
    dataLocal3d=dataWin3d(span(ix-halfw1,ix+halfw1),span::all,span(jt-halfw3,jt+halfw3));
    float accuPow=accu(dataLocal3d);
    if(accuPow>1.0){
        string fileOut="./4.swap/local.tp.dat";
        fileOut=fileOut+numtostr(accuPow,8);
        //datawrite3d_bycol_transpose(dataLocal3d,dataLocal3d.n_slices,\
            dataLocal3d.n_rows,fileOut.c_str());
        datawrite3d_bycol_transpose(lrt3dLocal.dataRealTP,lrt3dLocal.dataRealTP.n_slices,\
            lrt3dLocal.dataRealTP.n_rows,fileOut.c_str());
    }
    */
}
////////////beamforminginv3d use c++ thread/////////////////
/*

*/
#endif
