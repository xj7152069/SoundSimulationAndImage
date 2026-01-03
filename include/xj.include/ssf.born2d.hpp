#ifndef SSF_BORN_2D_HPP
#define SSF_BORN_2D_HPP

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <thread>
#include <future>
#include <time.h>
#include <algorithm>
#include <omp.h>
#include "../../include/xjc.h"

using namespace std;
using namespace arma;
/////////////////////////////////////////////////////

class SSF_BORN_2d{ 
private:
    bool baseExpSexit;
public:
    int nz,nx,nt,nf,nthread,nExp;
    int izBeg,izEnd,absorbWide;
    float dx,dz,dt,df,fmax,nfWrite;
    float minVel,absorb,wfPow;
    fmat vp2d,scatterDown2d,scatterUp2d; 
    fmat inpData2d,outData2d;
    cx_fmat inpFx2d,outFx2d;
    cx_fcube downKzKxKf3d; 
    cx_fcube upKzKxKf3d;
    fvec absorbWin,v0Dep;
    fmat deltaS2d;
    cx_fvec baseExp,baseExpS;

    SSF_BORN_2d();
    ~SSF_BORN_2d();

    void clearAll();
    void initializeParameters();
    void getParams(int nz,int nx,int nt,float dz,float dx,float dt,float fmax);
    void getInpMat(fmat inp2d, fmat vp2d, fmat scatter2d);
    void ssf2dUpToDown();
    void ssf2dDownToUp();
    void ffd2dTwoOrderUpToDown(float sxCoord);
    void ffd2dTwoOrderDownToUp(float sxCoord);
    void accuffd2dTwoOrderUpToDown();
    void accuffd2dTwoOrderDownToUp();
    void upFieldWrite(float f, const char* filename, bool isReal=true);
    void downFieldWrite(float f, const char* filename, bool isReal=true);
};
void SSF_BORN_2d::accuffd2dTwoOrderUpToDown()
{
    int nxfft=getfftnum(inpData2d.n_cols);
    this->initializeParameters();
    float pi=3.1415926;
    inpFx2d.zeros(inpData2d.n_rows,nxfft);
    //cout<<"this->nthread: "<<this->nthread<<endl;
omp_set_num_threads(this->nthread);
#pragma omp parallel for
    for(int ix=0;ix<this->nx;ix++){
        inpFx2d.col(ix)=fft(inpData2d.col(ix));
    }
    float dkx=2.0*pi/(this->dx)/(inpFx2d.n_cols);

omp_set_num_threads(this->nthread);
#pragma omp parallel for
    for(int jf=1;jf<this->nf;jf++){
        cout<<"Frequence: "<<jf*df<<endl;
        float wf=2.0*pi*this->df*jf;
        cx_fmat iz0Kx(1,inpFx2d.n_cols);
        cx_fmat iz1Kx(1,inpFx2d.n_cols);
        fmat iz0Dv(1,this->nx);
        iz0Kx=this->inpFx2d.row(jf);
        for(int ix=0;ix<int(this->nx);ix++){
            iz0Kx(0,ix)+=(upKzKxKf3d(izBeg,ix,jf)\
                *this->scatterUp2d(izBeg,ix));
            if(ix<this->absorbWide)iz0Kx(0,ix)=iz0Kx(0,ix)\
                *this->absorbWin(this->absorbWide-ix);
            if(ix>=(this->nx-this->absorbWide))iz0Kx(0,ix)=iz0Kx(0,ix)\
                *this->absorbWin(ix-this->nx+this->absorbWide);
        }
        this->downKzKxKf3d.slice(jf).row(izBeg)=iz0Kx(span::all,span(0,this->nx-1));

        cx_float a;
        a.imag(1.0);a.real(0.0);
        for(int iz=this->izBeg+1;iz<this->izEnd;iz++){
            iz1Kx.fill(0.0);
            float v0=this->v0Dep(iz-1);
            iz0Dv.row(0)=this->deltaS2d.row(iz-1);
            iz0Kx.row(0)=fft(iz0Kx.row(0));
            float izMaxPow=abs(iz0Kx.row(0)).max();
            izMaxPow=0.0;
            float kz=wf/v0;
            iz1Kx(0,0)=iz0Kx(0,0)*exp(a*float(-1.0*this->dz*kz));
            //*this->baseExp(round(this->dz*kz/1e-6));
            cx_fmat izSwapKx(1,inpFx2d.n_cols,fill::zeros);
            iz1Kx.row(0)=ifft(iz1Kx.row(0));
            for(int ix=0;ix<int(this->nx);ix++){
                if(abs(iz0Dv(0,ix))>0.000000001){
                    float alph=0.0;
                    float iz0Dv2Order=0.0;
                    //if(iz0Dv(0,ix)>0)iz1Kx(0,ix)=iz1Kx(0,ix)\
                        *this->baseExpS(round(wf*this->dz*iz0Dv(0,ix)/1e-8));
                    //else iz1Kx(0,ix)=iz1Kx(0,ix)\
                        *conj(this->baseExpS(round(-wf*this->dz*iz0Dv(0,ix)/1e-8)));
                    iz1Kx(0,ix)=iz1Kx(0,ix)*exp(a*float(-wf*this->dz*(iz0Dv(0,ix)\
                        +iz0Dv2Order)));
                }
            }
            izSwapKx+=iz1Kx;
            for(int ix=1;ix<int(iz0Kx.n_cols/2);ix++){
                float kx=dkx*ix;
                float kz2=wf*wf/v0/v0-kx*kx;
                if(kz2>0.0){
                    kz=sqrt(kz2);
                    int index=(round(this->dz*kz/1e-6));
                    //iz1Kx(0,ix)=iz0Kx(0,ix)*this->baseExp(index);
                    iz1Kx(0,ix)=iz0Kx(0,ix)*exp(a*float(-1.0*this->dz*kz));
                    iz1Kx(0,iz0Kx.n_cols-ix)=iz0Kx(0,iz0Kx.n_cols-ix)\
                        *exp(a*float(-1.0*this->dz*kz));
                        //*this->baseExp(index);
                    cx_fmat iz2Kx(1,inpFx2d.n_cols,fill::zeros);
                    if(abs(iz1Kx(0,ix))>1e-5*izMaxPow || \
                        abs(iz1Kx(0,iz0Kx.n_cols-ix))>1e-5*izMaxPow)
                    {
                        iz2Kx(0,ix)=iz1Kx(0,ix);
                        iz2Kx(0,iz0Kx.n_cols-ix)=iz1Kx(0,iz0Kx.n_cols-ix);
                        iz2Kx.row(0)=ifft(iz2Kx.row(0));
                        for(int ix=0;ix<int(this->nx);ix++){
                            if(abs(iz0Dv(0,ix))>0.000000001){
                                float alph=-kx*kx/wf/wf;
                                float iz0Dv2Order=(1.0/this->vp2d(iz-1,ix)\
                                    -v0/this->vp2d(iz-1,ix)/this->vp2d(iz-1,ix))\
                                    *(this->vp2d(iz-1,ix)*this->vp2d(iz-1,ix)*alph\
                                    /(2.0+0.5*this->vp2d(iz-1,ix)*this->vp2d(iz-1,ix)\
                                    *alph*(1.0+v0/this->vp2d(iz-1,ix)\
                                    +v0*v0/this->vp2d(iz-1,ix)/this->vp2d(iz-1,ix))));
                                //if(iz0Dv(0,ix)>0)iz1Kx(0,ix)=iz1Kx(0,ix)\
                                    *this->baseExpS(round(wf*this->dz*iz0Dv(0,ix)/1e-8));
                                //else iz1Kx(0,ix)=iz1Kx(0,ix)\
                                    *conj(this->baseExpS(round(-wf*this->dz*iz0Dv(0,ix)/1e-8)));
                                iz2Kx(0,ix)=iz2Kx(0,ix)*exp(a*float(-wf*this->dz*(iz0Dv(0,ix)\
                                    +iz0Dv2Order)));
                            }
                        }
                    }
                    izSwapKx+=iz2Kx;
                }
            }
            
            iz0Kx.row(0)=izSwapKx.row(0);
            for(int ix=0;ix<int(this->nx);ix++){
                iz0Kx(0,ix)+=(upKzKxKf3d(iz,ix,jf)\
                    *this->scatterUp2d(iz,ix));
                if(ix<this->absorbWide)iz0Kx(0,ix)=iz0Kx(0,ix)\
                    *this->absorbWin(this->absorbWide-ix);
                if(ix>=(this->nx-this->absorbWide))iz0Kx(0,ix)=iz0Kx(0,ix)\
                    *this->absorbWin(ix-this->nx+this->absorbWide);
            }
            this->downKzKxKf3d.slice(jf).row(iz)=iz0Kx(span::all,span(0,this->nx-1));
        }
    }
    //downFieldWrite(this->nfWrite, "./movie/slice.down.dat")
}

void SSF_BORN_2d::accuffd2dTwoOrderDownToUp()
{
    this->initializeParameters();
    float pi=3.1415926;
    float dkx=2.0*pi/(this->dx)/(inpFx2d.n_cols);
    outFx2d.copy_size(inpData2d);
    outFx2d.fill(0.0);
omp_set_num_threads(this->nthread);
#pragma omp parallel for
    for(int jf=1;jf<this->nf;jf++){
        cout<<"Frequence: "<<jf*df<<endl;
        float wf=2.0*pi*this->df*jf;
        float wf2=wf*wf;
        cx_fmat iz0Kx(1,inpFx2d.n_cols,fill::zeros);
        cx_fmat iz1Kx(1,inpFx2d.n_cols);
        fmat iz0Dv(1,this->nx);
        for(int ix=0;ix<int(this->nx);ix++){
            iz0Kx(0,ix)+=(downKzKxKf3d(izEnd-1,ix,jf)\
                *this->scatterDown2d(izEnd-1,ix))*wf2;
                if(ix<this->absorbWide)iz0Kx(0,ix)=iz0Kx(0,ix)\
                    *this->absorbWin(this->absorbWide-ix);
                if(ix>=(this->nx-this->absorbWide))iz0Kx(0,ix)=iz0Kx(0,ix)\
                    *this->absorbWin(ix-this->nx+this->absorbWide);
            }
        this->upKzKxKf3d.slice(jf).row(izEnd-1)=iz0Kx(span::all,span(0,this->nx-1));
        cx_float a;
        a.imag(1.0);a.real(0.0);
        for(int iz=(this->izEnd-2);iz>=this->izBeg;iz--){
            iz1Kx.fill(0.0);
            float v0=this->v0Dep(iz+1);
            iz0Dv.row(0)=this->deltaS2d.row(iz+1);
            iz0Kx.row(0)=fft(iz0Kx.row(0));
            float izMaxPow=abs(iz0Kx.row(0)).max();
            izMaxPow=0.0;
            float kz=wf/v0;
            iz1Kx(0,0)=iz0Kx(0,0)*exp(a*float(-1.0*this->dz*kz));
            float xs=(kz*kz+0.001*wf*wf/v0/v0);
            //if(iz==this->izBeg)iz1Kx(0,0)=iz0Kx(0,0)/xs;
            cx_fmat izSwapKx(1,inpFx2d.n_cols,fill::zeros);
            iz1Kx.row(0)=ifft(iz1Kx.row(0));
            for(int ix=0;ix<int(this->nx);ix++){
                if(abs(iz0Dv(0,ix))>0.000000001){
                    float alph=0.0;
                    float iz0Dv2Order=0.0;
                    //if(iz0Dv(0,ix)>0)iz1Kx(0,ix)=iz1Kx(0,ix)\
                        *this->baseExpS(round(wf*this->dz*iz0Dv(0,ix)/1e-8));
                    //else iz1Kx(0,ix)=iz1Kx(0,ix)\
                        *conj(this->baseExpS(round(-wf*this->dz*iz0Dv(0,ix)/1e-8)));
                    iz1Kx(0,ix)=iz1Kx(0,ix)*exp(a*float(-wf*this->dz*(iz0Dv(0,ix)\
                        +iz0Dv2Order)));
                }
            }
            izSwapKx+=iz1Kx;
            //this->baseExp(round(this->dz*kz/1e-6));
            for(int ix=1;ix<int(iz0Kx.n_cols/2);ix++){
                float kx=dkx*ix;
                float kz2=wf*wf/v0/v0-kx*kx;
                if(kz2>0.0){
                    kz=sqrt(kz2);
                    int index=(round(this->dz*kz/1e-6));
                    iz1Kx(0,ix)=iz0Kx(0,ix)*exp(a*float(-1.0*this->dz*kz));
                    //this->baseExp(index);
                    iz1Kx(0,iz0Kx.n_cols-ix)=iz0Kx(0,iz0Kx.n_cols-ix)\
                        *exp(a*float(-1.0*this->dz*kz));
                    //*this->baseExp(index);
                    if(iz==this->izBeg){
                        xs=(kz2+0.001*wf*wf/v0/v0);
                        //iz1Kx(0,ix)=iz0Kx(0,ix)/xs;
                        //iz1Kx(0,iz0Kx.n_cols-ix)=iz0Kx(0,iz0Kx.n_cols-ix)/xs;
                    }
                    cx_fmat iz2Kx(1,inpFx2d.n_cols,fill::zeros);
                    if(abs(iz1Kx(0,ix))>1e-5*izMaxPow || \
                        abs(iz1Kx(0,iz0Kx.n_cols-ix))>1e-5*izMaxPow)
                    {
                        iz2Kx(0,ix)=iz1Kx(0,ix);
                        iz2Kx(0,iz0Kx.n_cols-ix)=iz1Kx(0,iz0Kx.n_cols-ix);
                        iz2Kx.row(0)=ifft(iz2Kx.row(0));
                        for(int ix=0;ix<int(this->nx);ix++){
                            if(abs(iz0Dv(0,ix))>0.000000001){
                                float alph=-kx*kx/wf/wf;
                                float iz0Dv2Order=(1.0/this->vp2d(iz+1,ix)\
                                    -v0/this->vp2d(iz+1,ix)/this->vp2d(iz+1,ix))\
                                    *(alph*this->vp2d(iz+1,ix)*this->vp2d(iz+1,ix)\
                                    /(2.0+0.5*alph*this->vp2d(iz+1,ix)*this->vp2d(iz+1,ix)\
                                    *(1.0+v0/this->vp2d(iz+1,ix)\
                                    +v0*v0/this->vp2d(iz+1,ix)/this->vp2d(iz+1,ix))));
                                //if(iz0Dv(0,ix)>0)iz1Kx(0,ix)=iz1Kx(0,ix)\
                                    *this->baseExpS(round(wf*this->dz*iz0Dv(0,ix)/1e-8));
                                //else iz1Kx(0,ix)=iz1Kx(0,ix)\
                                    *conj(this->baseExpS(round(-wf*this->dz*iz0Dv(0,ix)/1e-8)));
                                iz2Kx(0,ix)=iz2Kx(0,ix)*exp(a*float(-wf*this->dz*(iz0Dv(0,ix)\
                                    +iz0Dv2Order)));
                            }
                        }
                    }
                    izSwapKx+=iz2Kx;
                }
            }
            
            iz0Kx.row(0)=izSwapKx.row(0);
            for(int ix=0;ix<int(this->nx);ix++){
                iz0Kx(0,ix)+=(downKzKxKf3d(iz,ix,jf)\
                    *this->scatterDown2d(iz,ix))*wf2;
                if(ix<this->absorbWide)iz0Kx(0,ix)=iz0Kx(0,ix)\
                    *this->absorbWin(this->absorbWide-ix);
                if(ix>=(this->nx-this->absorbWide))iz0Kx(0,ix)=iz0Kx(0,ix)\
                    *this->absorbWin(ix-this->nx+this->absorbWide);
            }
            this->upKzKxKf3d.slice(jf).row(iz)=iz0Kx(span::all,span(0,this->nx-1));
        }
        outFx2d.row(jf)=pow(wf,this->wfPow)*this->upKzKxKf3d.slice(jf).row(izBeg);
    }
    outData2d=inpData2d;

omp_set_num_threads(this->nthread);
#pragma omp parallel for
    for(int ix=0;ix<this->nx;ix++){
        outData2d.col(ix)=2.0*real(ifft(outFx2d.col(ix)));
    }
    outData2d=(1.0/pow(this->v0Dep(izBeg),this->wfPow))*outData2d;
    //upFieldWrite(this->nfWrite, "./movie/slice.down.dat")
}
void SSF_BORN_2d::ffd2dTwoOrderUpToDown(float sxCoord)
{
    int nxfft=getfftnum(inpData2d.n_cols);
    this->initializeParameters();
    float pi=3.1415926;
    inpFx2d.zeros(inpData2d.n_rows,nxfft);
    //cout<<"this->nthread: "<<this->nthread<<endl;
omp_set_num_threads(this->nthread);
#pragma omp parallel for
    for(int ix=0;ix<this->nx;ix++){
        inpFx2d.col(ix)=fft(inpData2d.col(ix));
    }
    float dkx=2.0*pi/(this->dx)/(inpFx2d.n_cols);

omp_set_num_threads(this->nthread);
#pragma omp parallel for
    for(int jf=1;jf<this->nf;jf++){
        float wf=2.0*pi*this->df*jf;
        cx_fmat iz0Kx(1,inpFx2d.n_cols);
        cx_fmat iz1Kx(1,inpFx2d.n_cols);
        fmat iz0Dv(1,this->nx);
        iz0Kx=this->inpFx2d.row(jf);
        for(int ix=0;ix<int(this->nx);ix++){
            iz0Kx(0,ix)+=(upKzKxKf3d(izBeg,ix,jf)\
                *this->scatterUp2d(izBeg,ix));
            if(ix<this->absorbWide)iz0Kx(0,ix)=iz0Kx(0,ix)\
                *this->absorbWin(this->absorbWide-ix);
            if(ix>=(this->nx-this->absorbWide))iz0Kx(0,ix)=iz0Kx(0,ix)\
                *this->absorbWin(ix-this->nx+this->absorbWide);
        }
        this->downKzKxKf3d.slice(jf).row(izBeg)=iz0Kx(span::all,span(0,this->nx-1));

        cx_float a;
        a.imag(1.0);a.real(0.0);
        for(int iz=this->izBeg+1;iz<this->izEnd;iz++){
            iz1Kx.fill(0.0);
            float v0=this->v0Dep(iz-1);
            iz0Dv.row(0)=this->deltaS2d.row(iz-1);
            iz0Kx.row(0)=fft(iz0Kx.row(0));
            float kz=wf/v0;
            iz1Kx(0,0)=iz0Kx(0,0)*exp(a*float(-1.0*this->dz*kz));
            //*this->baseExp(round(this->dz*kz/1e-6));
            for(int ix=1;ix<int(iz0Kx.n_cols/2);ix++){
                float kx=dkx*ix;
                float kz2=wf*wf/v0/v0-kx*kx;
                if(kz2>0.0){
                    kz=sqrt(kz2);
                    int index=(round(this->dz*kz/1e-6));
                    //iz1Kx(0,ix)=iz0Kx(0,ix)*this->baseExp(index);
                    iz1Kx(0,ix)=iz0Kx(0,ix)*exp(a*float(-1.0*this->dz*kz));
                    iz1Kx(0,iz0Kx.n_cols-ix)=iz0Kx(0,iz0Kx.n_cols-ix)\
                        *exp(a*float(-1.0*this->dz*kz));
                        //*this->baseExp(index);
                }
            }
            iz1Kx.row(0)=ifft(iz1Kx.row(0));
            for(int ix=0;ix<int(this->nx);ix++){
                if(abs(iz0Dv(0,ix))>0.000000001){
                    float alph=abs(ix*dx-sxCoord)/(dz*(iz-this->izBeg));
                    alph=alph*alph;
                    //float alph=0.25;
                    float iz0Dv2Order=(1.0/this->vp2d(iz-1,ix)\
                        -v0/this->vp2d(iz-1,ix)/this->vp2d(iz-1,ix))\
                        *(alph/(2.0+0.5*alph*(1.0+v0/this->vp2d(iz-1,ix)\
                        +v0*v0/this->vp2d(iz-1,ix)/this->vp2d(iz-1,ix))));
                    //if(iz0Dv(0,ix)>0)iz1Kx(0,ix)=iz1Kx(0,ix)\
                        *this->baseExpS(round(wf*this->dz*iz0Dv(0,ix)/1e-8));
                    //else iz1Kx(0,ix)=iz1Kx(0,ix)\
                        *conj(this->baseExpS(round(-wf*this->dz*iz0Dv(0,ix)/1e-8)));
                    iz1Kx(0,ix)=iz1Kx(0,ix)*exp(a*float(-wf*this->dz*(iz0Dv(0,ix)\
                        -iz0Dv2Order)));
                }
            }
            iz0Kx.row(0)=iz1Kx.row(0);
            for(int ix=0;ix<int(this->nx);ix++){
                iz0Kx(0,ix)+=(upKzKxKf3d(iz,ix,jf)\
                    *this->scatterUp2d(iz,ix));
                if(ix<this->absorbWide)iz0Kx(0,ix)=iz0Kx(0,ix)\
                    *this->absorbWin(this->absorbWide-ix);
                if(ix>=(this->nx-this->absorbWide))iz0Kx(0,ix)=iz0Kx(0,ix)\
                    *this->absorbWin(ix-this->nx+this->absorbWide);
            }
            this->downKzKxKf3d.slice(jf).row(iz)=iz0Kx(span::all,span(0,this->nx-1));
        }
    }
    //downFieldWrite(this->nfWrite, "./movie/slice.down.dat")
}

void SSF_BORN_2d::ffd2dTwoOrderDownToUp(float sxCoord)
{
    this->initializeParameters();
    float pi=3.1415926;
    float dkx=2.0*pi/(this->dx)/(inpFx2d.n_cols);
    outFx2d.copy_size(inpData2d);
    outFx2d.fill(0.0);
omp_set_num_threads(this->nthread);
#pragma omp parallel for
    for(int jf=1;jf<this->nf;jf++){
        float wf=2.0*pi*this->df*jf;
        cx_fmat iz0Kx(1,inpFx2d.n_cols,fill::zeros);
        cx_fmat iz1Kx(1,inpFx2d.n_cols);
        fmat iz0Dv(1,this->nx);
        for(int ix=0;ix<int(this->nx);ix++){
            iz0Kx(0,ix)+=(downKzKxKf3d(izEnd-1,ix,jf)\
                *this->scatterDown2d(izEnd-1,ix));
                if(ix<this->absorbWide)iz0Kx(0,ix)=iz0Kx(0,ix)\
                    *this->absorbWin(this->absorbWide-ix);
                if(ix>=(this->nx-this->absorbWide))iz0Kx(0,ix)=iz0Kx(0,ix)\
                    *this->absorbWin(ix-this->nx+this->absorbWide);
            }
        this->upKzKxKf3d.slice(jf).row(izEnd-1)=iz0Kx(span::all,span(0,this->nx-1));
        cx_float a;
        a.imag(1.0);a.real(0.0);
        for(int iz=(this->izEnd-2);iz>=this->izBeg;iz--){
            iz1Kx.fill(0.0);
            float v0=this->v0Dep(iz+1);
            iz0Dv.row(0)=this->deltaS2d.row(iz+1);
            iz0Kx.row(0)=fft(iz0Kx.row(0));
            float kz=wf/v0;
            iz1Kx(0,0)=iz0Kx(0,0)*exp(a*float(-1.0*this->dz*kz));
            float xs=(kz*kz+0.001*wf*wf/v0/v0); 
            //if(iz==this->izBeg)iz1Kx(0,0)=iz0Kx(0,0)/xs;
            //this->baseExp(round(this->dz*kz/1e-6));
            for(int ix=1;ix<int(iz0Kx.n_cols/2);ix++){
                float kx=dkx*ix;
                float kz2=wf*wf/v0/v0-kx*kx;
                if(kz2>0.0){
                    kz=sqrt(kz2);
                    int index=(round(this->dz*kz/1e-6));
                    iz1Kx(0,ix)=iz0Kx(0,ix)*exp(a*float(-1.0*this->dz*kz));
                    //this->baseExp(index);
                    iz1Kx(0,iz0Kx.n_cols-ix)=iz0Kx(0,iz0Kx.n_cols-ix)\
                        *exp(a*float(-1.0*this->dz*kz));
                    //*this->baseExp(index);
                    if(iz==this->izBeg){
                        xs=(kz2+0.001*wf*wf/v0/v0);
                        //iz1Kx(0,ix)=iz0Kx(0,ix)/xs;
                        //iz1Kx(0,iz0Kx.n_cols-ix)=iz0Kx(0,iz0Kx.n_cols-ix)/xs;
                    }
                }
            }
            iz1Kx.row(0)=ifft(iz1Kx.row(0));
            for(int ix=0;ix<int(this->nx);ix++){
                if(abs(iz0Dv(0,ix))>0.000000001){
                    //float alph=abs(ix*dx-sxCoord)/(dz*(iz-this->izBeg));
                    //alph=alph*alph;
                    float alph=0.5;
                    float iz0Dv2Order=(1.0/this->vp2d(iz+1,ix)\
                        -v0/this->vp2d(iz+1,ix)/this->vp2d(iz+1,ix))\
                        *(alph/(2.0+0.5*alph*(1.0+v0/this->vp2d(iz+1,ix)\
                        +v0*v0/this->vp2d(iz+1,ix)/this->vp2d(iz+1,ix))));
                    //if(iz0Dv(0,ix)>0)iz1Kx(0,ix)=iz1Kx(0,ix)\
                        *this->baseExpS(round(wf*this->dz*iz0Dv(0,ix)/1e-8));
                    //else iz1Kx(0,ix)=iz1Kx(0,ix)\
                        *conj(this->baseExpS(round(-wf*this->dz*iz0Dv(0,ix)/1e-8)));
                    iz1Kx(0,ix)=iz1Kx(0,ix)*exp(a*float(-wf*this->dz*(iz0Dv(0,ix)\
                        -iz0Dv2Order)));
                }
            }
            iz0Kx.row(0)=iz1Kx.row(0);
            for(int ix=0;ix<int(this->nx);ix++){
                iz0Kx(0,ix)+=(downKzKxKf3d(iz,ix,jf)\
                    *this->scatterDown2d(iz,ix));
                if(ix<this->absorbWide)iz0Kx(0,ix)=iz0Kx(0,ix)\
                    *this->absorbWin(this->absorbWide-ix);
                if(ix>=(this->nx-this->absorbWide))iz0Kx(0,ix)=iz0Kx(0,ix)\
                    *this->absorbWin(ix-this->nx+this->absorbWide);
            }
            this->upKzKxKf3d.slice(jf).row(iz)=iz0Kx(span::all,span(0,this->nx-1));
        }
        outFx2d.row(jf)=pow(wf,this->wfPow)*this->upKzKxKf3d.slice(jf).row(izBeg);
    }
    outData2d=inpData2d;

omp_set_num_threads(this->nthread);
#pragma omp parallel for
    for(int ix=0;ix<this->nx;ix++){
        outData2d.col(ix)=2.0*real(ifft(outFx2d.col(ix)));
    }
    //upFieldWrite(this->nfWrite, "./movie/slice.down.dat")
}

void SSF_BORN_2d::ssf2dDownToUp()
{
    this->initializeParameters();
    float pi=3.1415926;
    float dkx=2.0*pi/(this->dx)/(inpFx2d.n_cols);
    outFx2d.copy_size(inpData2d);
    outFx2d.fill(0.0);
omp_set_num_threads(this->nthread);
#pragma omp parallel for
    for(int jf=1;jf<this->nf;jf++){
        float wf=2.0*pi*this->df*jf;
        float wf2=wf*wf;
        cx_fmat iz0Kx(1,inpFx2d.n_cols,fill::zeros);
        cx_fmat iz1Kx(1,inpFx2d.n_cols);
        fmat iz0Dv(1,this->nx);
        fmat iz0Dv2Order(1,this->nx);
        for(int ix=0;ix<int(this->nx);ix++){
            iz0Kx(0,ix)+=(downKzKxKf3d(izEnd-1,ix,jf)\
                *this->scatterDown2d(izEnd-1,ix))*wf2;
                if(ix<this->absorbWide)iz0Kx(0,ix)=iz0Kx(0,ix)\
                    *this->absorbWin(this->absorbWide-ix);
                if(ix>=(this->nx-this->absorbWide))iz0Kx(0,ix)=iz0Kx(0,ix)\
                    *this->absorbWin(ix-this->nx+this->absorbWide);
            }
        this->upKzKxKf3d.slice(jf).row(izEnd-1)=iz0Kx(span::all,span(0,this->nx-1));
        cx_float a;
        a.imag(1.0);a.real(0.0);
        for(int iz=(this->izEnd-2);iz>=this->izBeg;iz--){
            iz1Kx.fill(0.0);
            float v0=this->v0Dep(iz+1);
            iz0Kx.row(0)=fft(iz0Kx.row(0));
            float kz=wf/v0;
            iz1Kx(0,0)=iz0Kx(0,0)*exp(a*float(-1.0*this->dz*kz));
            //this->baseExp(round(this->dz*kz/1e-6));
            for(int ix=1;ix<int(iz0Kx.n_cols/2);ix++){
                float kx=dkx*ix;
                float kz2=wf*wf/v0/v0-kx*kx;
                if(kz2>0.0){
                    kz=sqrt(kz2);
                    int index=(round(this->dz*kz/1e-6));
                    iz1Kx(0,ix)=iz0Kx(0,ix)*exp(a*float(-1.0*this->dz*kz));
                    //this->baseExp(index);
                    iz1Kx(0,iz0Kx.n_cols-ix)=iz0Kx(0,iz0Kx.n_cols-ix)\
                        *exp(a*float(-1.0*this->dz*kz));
                    //*this->baseExp(index);
                }
            }
            iz1Kx.row(0)=ifft(iz1Kx.row(0));
            iz0Dv.row(0)=this->deltaS2d.row(iz+1);
            for(int ix=0;ix<int(this->nx);ix++){
                if(abs(iz0Dv(0,ix))>0.000000001){
                    //if(iz0Dv(0,ix)>0)iz1Kx(0,ix)=iz1Kx(0,ix)\
                        *this->baseExpS(round(wf*this->dz*iz0Dv(0,ix)/1e-8));
                    //else iz1Kx(0,ix)=iz1Kx(0,ix)\
                        *conj(this->baseExpS(round(-wf*this->dz*iz0Dv(0,ix)/1e-8)));
                    iz1Kx(0,ix)=iz1Kx(0,ix)*exp(a*float(-wf*this->dz*iz0Dv(0,ix)));
                }
            }
            iz0Kx.row(0)=iz1Kx.row(0);
            for(int ix=0;ix<int(this->nx);ix++){
                iz0Kx(0,ix)+=(downKzKxKf3d(iz,ix,jf)\
                    *this->scatterDown2d(iz,ix))*wf2;
                if(ix<this->absorbWide)iz0Kx(0,ix)=iz0Kx(0,ix)\
                    *this->absorbWin(this->absorbWide-ix);
                if(ix>=(this->nx-this->absorbWide))iz0Kx(0,ix)=iz0Kx(0,ix)\
                    *this->absorbWin(ix-this->nx+this->absorbWide);
            }
            this->upKzKxKf3d.slice(jf).row(iz)=iz0Kx(span::all,span(0,this->nx-1));
        }
        outFx2d.row(jf)=pow(wf,this->wfPow)*this->upKzKxKf3d.slice(jf).row(izBeg);
    }
    outData2d=inpData2d;

omp_set_num_threads(this->nthread);
#pragma omp parallel for
    for(int ix=0;ix<this->nx;ix++){
        outData2d.col(ix)=2.0*real(ifft(outFx2d.col(ix)));
    }
    //upFieldWrite(this->nfWrite, "./movie/slice.down.dat")
}

void SSF_BORN_2d::ssf2dUpToDown()
{
    int nxfft=getfftnum(inpData2d.n_cols);
    this->initializeParameters();
    float pi=3.1415926;
    inpFx2d.zeros(inpData2d.n_rows,nxfft);
    //cout<<"this->nthread: "<<this->nthread<<endl;
omp_set_num_threads(this->nthread);
#pragma omp parallel for
    for(int ix=0;ix<this->nx;ix++){
        inpFx2d.col(ix)=fft(inpData2d.col(ix));
    }
    float dkx=2.0*pi/(this->dx)/(inpFx2d.n_cols);

omp_set_num_threads(this->nthread);
#pragma omp parallel for
    for(int jf=1;jf<this->nf;jf++){
        float wf=2.0*pi*this->df*jf;
        cx_fmat iz0Kx(1,inpFx2d.n_cols);
        cx_fmat iz1Kx(1,inpFx2d.n_cols);
        fmat iz0Dv(1,this->nx);
        iz0Kx=this->inpFx2d.row(jf);
        for(int ix=0;ix<int(this->nx);ix++){
            iz0Kx(0,ix)+=(upKzKxKf3d(izBeg,ix,jf)\
                *this->scatterUp2d(izBeg,ix));
            if(ix<this->absorbWide)iz0Kx(0,ix)=iz0Kx(0,ix)\
                *this->absorbWin(this->absorbWide-ix);
            if(ix>=(this->nx-this->absorbWide))iz0Kx(0,ix)=iz0Kx(0,ix)\
                *this->absorbWin(ix-this->nx+this->absorbWide);
        }
        this->downKzKxKf3d.slice(jf).row(izBeg)=iz0Kx(span::all,span(0,this->nx-1));

        cx_float a;
        a.imag(1.0);a.real(0.0);
        for(int iz=this->izBeg+1;iz<this->izEnd;iz++){
            iz1Kx.fill(0.0);
            float v0=this->v0Dep(iz-1);
            iz0Dv.row(0)=this->deltaS2d.row(iz-1);
            iz0Kx.row(0)=fft(iz0Kx.row(0));
            float kz=wf/v0;
            iz1Kx(0,0)=iz0Kx(0,0)*exp(a*float(-1.0*this->dz*kz));
            //*this->baseExp(round(this->dz*kz/1e-6));
            for(int ix=1;ix<int(iz0Kx.n_cols/2);ix++){
                float kx=dkx*ix;
                float  kz2=wf*wf/v0/v0-kx*kx;
                if(kz2>0.0){
                    kz=sqrt(kz2);
                    int index=(round(this->dz*kz/1e-6));
                    //iz1Kx(0,ix)=iz0Kx(0,ix)*this->baseExp(index);
                    iz1Kx(0,ix)=iz0Kx(0,ix)*exp(a*float(-1.0*this->dz*kz));
                    iz1Kx(0,iz0Kx.n_cols-ix)=iz0Kx(0,iz0Kx.n_cols-ix)\
                        *exp(a*float(-1.0*this->dz*kz));
                        //*this->baseExp(index);
                }
            }
            iz1Kx.row(0)=ifft(iz1Kx.row(0));
            for(int ix=0;ix<int(this->nx);ix++){
                if(abs(iz0Dv(0,ix))>0.000000001){
                    //if(iz0Dv(0,ix)>0)iz1Kx(0,ix)=iz1Kx(0,ix)\
                        *this->baseExpS(round(wf*this->dz*iz0Dv(0,ix)/1e-8));
                    //else iz1Kx(0,ix)=iz1Kx(0,ix)\
                        *conj(this->baseExpS(round(-wf*this->dz*iz0Dv(0,ix)/1e-8)));
                    iz1Kx(0,ix)=iz1Kx(0,ix)*exp(a*float(-wf*this->dz*iz0Dv(0,ix)));
                }
            }
            iz0Kx.row(0)=iz1Kx.row(0);
            for(int ix=0;ix<int(this->nx);ix++){
                iz0Kx(0,ix)+=(upKzKxKf3d(iz,ix,jf)\
                    *this->scatterUp2d(iz,ix));
                if(ix<this->absorbWide)iz0Kx(0,ix)=iz0Kx(0,ix)\
                    *this->absorbWin(this->absorbWide-ix);
                if(ix>=(this->nx-this->absorbWide))iz0Kx(0,ix)=iz0Kx(0,ix)\
                    *this->absorbWin(ix-this->nx+this->absorbWide);
            }
            this->downKzKxKf3d.slice(jf).row(iz)=iz0Kx(span::all,span(0,this->nx-1));
        }
    }
    //downFieldWrite(this->nfWrite, "./movie/slice.down.dat")
}

void SSF_BORN_2d::upFieldWrite(float f, const char* filename, bool isReal)
{
    fmat sliceNf;
    if(isReal)sliceNf=real(this->upKzKxKf3d.slice(f/this->df));
    else sliceNf=imag(this->upKzKxKf3d.slice(f/this->df));
    datawrite(sliceNf,filename);
}
void SSF_BORN_2d::downFieldWrite(float f, const char* filename, bool isReal)
{
    fmat sliceNf;
    if(isReal)sliceNf=real(this->downKzKxKf3d.slice(f/this->df));
    else sliceNf=imag(this->downKzKxKf3d.slice(f/this->df));
    datawrite(sliceNf,filename);
}

void SSF_BORN_2d::getInpMat(fmat inp2d, fmat vp2d, fmat scatter2d)
{
    this->inpData2d=inp2d;
    this->vp2d=vp2d;
    this->scatterDown2d=scatter2d;
}

void SSF_BORN_2d::getParams(int nz,int nx,int nt,\
    float dz,float dx,float dt,float fmax)
{
    this->nz=nz,this->nx=nx,this->nt=nt;
    this->dx=dx,this->dz=dz,this->dt=dt;
    this->fmax=fmax,this->df=1.0/nt/dt;
    this->nf=int(fmax/df)+1;
    this->vp2d.set_size(nz,nx);
    this->scatterDown2d.set_size(nz,nx);
    this->scatterUp2d.set_size(nz,nx);
    this->inpData2d.set_size(nt,nx);
    this->outData2d.set_size(nt,nx);
    this->inpFx2d.set_size(nt,nx);
    this->outFx2d.set_size(nt,nx);
    this->downKzKxKf3d.set_size(nz,nx,nf);
    this->upKzKxKf3d.set_size(nz,nx,nf);
}

void SSF_BORN_2d::clearAll(){
    this->vp2d.fill(0.0); 
    this->scatterDown2d.fill(0.0),this->scatterUp2d.fill(0.0);
    this->inpData2d.fill(0.0),this->outData2d.fill(0.0);
    this->inpFx2d.fill(0.0),this->outFx2d.fill(0.0);
    this->downKzKxKf3d.fill(0.0); 
    this->upKzKxKf3d.fill(0.0);
}

SSF_BORN_2d::SSF_BORN_2d(){
    this->nthread=1;
    this->nfWrite=30.0;
    this->izBeg=0,this->izEnd=0;
    this->wfPow=2.0;
    this->absorb=2e-3; 
    this->absorbWide=50; 
    this->minVel=1000.0;
    this->baseExpSexit=true;
}

SSF_BORN_2d::~SSF_BORN_2d(){
    this->vp2d.clear();
    this->scatterDown2d.clear(),this->scatterUp2d.clear();
    this->inpData2d.clear(),this->outData2d.clear();
    this->inpFx2d.clear(),this->outFx2d.clear();
    this->downKzKxKf3d.clear(); 
    this->upKzKxKf3d.clear();
}

fmat GetReverRowFmat(fmat data2d)
{
    int n1=data2d.n_rows;
    int n2=data2d.n_cols;
    fmat revData2d(n1,n2);
    for(int i=0;i<n1;i++){
        for(int j=0;j<n2;j++){
            revData2d(i,j)=data2d(n1-i-1,j);
        }
    }
    return revData2d;
}
void SSF_BORN_2d::initializeParameters()
{
    this->absorbWin.zeros(this->absorbWide+1);
    for(int i=0;i<=this->absorbWide;i++){
        this->absorbWin(i)=exp(-float(this->absorb*i));
    }
    this->v0Dep.zeros(this->nz);
    this->deltaS2d.zeros(this->nz,this->nx);
    for(int i=0;i<this->nz;i++){
        this->v0Dep(i)=accu(this->vp2d.row(i))/this->nx;
        //this->v0Dep(i)=this->vp2d.row(i).min();
        this->deltaS2d.row(i)=1.0/vp2d.row(i);
        this->deltaS2d.row(i)=this->deltaS2d.row(i)\
            -(1.0/this->v0Dep(i));
    }

    if(!this->baseExpSexit){
        this->nExp=(2*3.2*fmax*dz/this->minVel)/1e-6;
        this->baseExp.set_size(nExp);
        cx_float a;
        a.imag(1.0);a.real(0.0);
omp_set_num_threads(this->nthread);
#pragma omp parallel for
        for(int i=0;i<nExp;i++){
            this->baseExp(i)=exp(a*float(-1e-6*float(i)));
        }

        int nExpS=(2*3.2*fmax*dz*abs(this->deltaS2d.max()))/1e-8;
        nExpS=max(nExpS,int(2*3.2*fmax*dz*abs(this->deltaS2d.min())/1e-8));
        this->baseExpS.set_size(nExpS);
omp_set_num_threads(this->nthread);
#pragma omp parallel for
        for(int i=0;i<nExpS;i++){
            this->baseExpS(i)=exp(a*float(-1e-8*float(i)));
        }
        this->baseExpSexit=true;
    }
}

#endif

