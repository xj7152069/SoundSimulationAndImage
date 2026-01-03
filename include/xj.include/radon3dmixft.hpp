/************update in 2023.05.20***************
Anti-aliasing Radon3d in Mix-frequence-time domain;   
    
***********************************************/

#ifndef RADON3DMIXFT_HPP
#define RADON3DMIXFT_HPP
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


/*******************************************************/
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
void RadonMixFTCG3d_getdigfmat(struct linerradon3d & par,\
 fmat &digw_fmat_npxnpy)
{
    digw_fmat_npxnpy.fill(par.dig_n2);
    par.p_power=digw_fmat_npxnpy;
}
/*void get_anti_aliasing_weighted_RadonMixFTCG3d\
    (fcube &filter, fcube data1, int nfrule, \
    int nthread=1, int wx=5, int wy=5, int wz=10)
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

    ny=data1.n_cols;
    nx=data1.n_rows;
    nz=data1.n_slices;
    datawrite3d_bycol_transpose(data1,nz,nx,"./swap/datataup.rule.dat");
    datawrite3d_bycol_transpose(data2,nz,nx,"./swap/datataup.dat");

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
    datawrite3d_bycol_transpose(dataStru,nz,nx,"./swap/datataup.weight.dat");

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
            //filter(i,j,k)+=(data1(i+i1,j+j1,k+k1)*data1(i+i1,j+j1,k+k1));
            filter(i,j,k)+=abs(dataStru(i1,j1,k1));
            //apow2+=(data2(i+i1,j+j1,k+k1)*data2(i+i1,j+j1,k+k1));
        }}}
        //filter(i,j,k)=sqrt(abs(filter(i,j,k)));
    }}}

    if(ny>3){
        filter=fcubesmooth(filter,0);
    }else{
        for(int k=0;k<ny;k++){
            filter.col(k)=fmatsmooth(filter.col(k),nx,nz,0);
        }
    }
    filter-=filter.min();
    filter=(filter)/filter.max();
    for(int i=0;i<nx;i++){
    for(int j=0;j<ny;j++){
    for(int k=0;k<nz;k++){
        //filter(i,j,k)=1.0/(1.001+exp(50*(0.25-filter(i,j,k))));
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
}*/
void get_anti_aliasing_weighted_RadonMixFTCG3d\
    (fcube &filter, fcube data1, int nfrule, \
    int nthread=1, int wx=5, int wy=5, int wz=10)
{
    int i,j,k,i1,j1,nx,ny,nz,k1,k2;
    int win;
    filter.copy_size(data1);
    filter.fill(0.0);
    cx_fcube datafx;
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

    ny=data1.n_cols;
    nx=data1.n_rows;
    nz=data1.n_slices;
    win=(nx-1)/2;
    wx=min(win,wx);
    win=(ny-1)/2;
    wy=min(win,wy);
    win=(nz-1)/2;
    wz=min(win,wz);

omp_set_num_threads(nthread);
#pragma omp parallel for
    for(i=wx;i<nx-wx;i++){
    for(j=wy;j<ny-wy;j++){
    for(k=wz;k<nz-wz;k++){
        for(i1=-wx;i1<=wx;i1++){
        for(j1=-wy;j1<=wy;j1++){
        for(k1=-wz;k1<=wz;k1++){
            filter(i,j,k)+=(data1(i+i1,j+j1,k+k1)*data1(i-i1,j+j1,k+k1)*\
                data1(i+i1,j-j1,k+k1)*data1(i-i1,j-j1,k+k1))*\
                (data1(i+i1,j+j1,k-k1)*data1(i-i1,j+j1,k-k1)*\
                data1(i+i1,j-j1,k-k1)*data1(i-i1,j-j1,k-k1));
            //agcpow1+=data1(i+i1,j+j1,k+k1)*data1(i+i1,j+j1,k+k1);
        }}}
        filter(i,j,k)=sqrt(abs(filter(i,j,k)));
    }}}
    if(ny>3){
        filter=fcubesmooth(filter,9);
    }else{
        for(k=0;k<ny;k++){
            filter.col(k)=fmatsmooth(filter.col(k),nx,nz,9);
        }
    }
    filter-=filter.min();
    filter=(filter)/filter.max();
    for(i=0;i<nx;i++){
    for(j=0;j<ny;j++){
    for(k=0;k<nz;k++){
        filter(i,j,k)=1.0/(1.001+exp(100*(0.1-filter(i,j,k))));
    }}}
    filter=(filter)/filter.max();
    filter+=0.00001;
    filter=1.0/filter;
    filter-=filter.min();
    filter=(filter)/filter.max();
}
void RadonMixFTCG3d_LTL_operator(fcube & Ap,
    fcube & p, struct linerradon3d &par)
{
    if(!par.base5dexist){
omp_set_num_threads(par.numthread);
#pragma omp parallel for
    for(int kf=par.nf1;kf<par.nf2;kf++){
        cx_fmat hessfft,hessian;
        hessian.zeros(par.npx*2-1,par.npy*2-1);
        hessfft=beamforminginv3d_CG_hessianget_thread(&par,hessian,kf);
        par.hessianf2npx2npy[kf-par.nf1].copy_size(hessfft);
        par.hessianf2npx2npy[kf-par.nf1]=hessfft;
        //cxmatcopy(par.hessianf2npx2npy[kf-par.nf1],hessfft);
    }
        par.base5dexist=true;
    }
    cx_fcube pfx,Apfx;
    pfx.copy_size(p);
    Apfx.copy_size(Ap);
    Apfx.fill(0.0);
    pfx.fill(0.0);
    tx2fx_3d_thread(pfx,(&p)[0],par.numthread);

omp_set_num_threads(par.numthread);
#pragma omp parallel for
    for(int kf=par.nf1;kf<par.nf2;kf++){
        cx_fmat Apkf,pkf,hessfft;
        Apkf.copy_size(par.datafP.slice(0));
        pkf.copy_size(par.datafP.slice(0));
        hessfft.copy_size(par.hessianf2npx2npy[0]);
        hessfft=par.hessianf2npx2npy[kf-par.nf1];
        //cxmatcopy(hessfft,par.hessianf2npx2npy[kf-par.nf1]);
        pkf=pfx.slice(kf);
        cxfmatget_Ap_small_fft(Apkf,hessfft,pkf,0,&par);
        Apfx.slice(kf)=Apkf;
    }
    fx2tx_3d_thread((&Ap)[0],Apfx,par.numthread);
    Ap*=2.0;
}

void RadonMixFTCG3d(struct linerradon3d &par,\
 int iterations_num=9, float residual_ratio=1)
{
    int nthread(par.numthread);
    fmat dig_w_fmat(par.npx,par.npy);
    RadonMixFTCG3d_getdigfmat((&par)[0], dig_w_fmat);
    par.weighted_fcube.copy_size(par.realdataTP);
    par.weighted_fcube.fill(0.0);
    get_anti_aliasing_weighted_RadonMixFTCG3d\
        (par.weighted_fcube,par.realdataTP,par.rulef2,nthread);
    float dataxs(1.0);
    par.weighted_fcube*=par.dig_n1;
    par.weighted_fcube+=par.dig_n2;
    par.p_power.fill(0.0);
    datawrite3d_bycol_transpose(par.weighted_fcube,par.nz,\
        par.npx,"weight.bin");

    int ip,jp,in,jn,k,iter(0);
    int n1(par.data.n_rows),n2(par.data.n_cols),n3(par.data.n_slices),\
        np1(par.realdataTP.n_rows),np2(par.realdataTP.n_cols);
    fcube gradient_rk,gradient_rk_1,\
        gradient_cg_pk,gradient_cg_pk_1,\
        datatp_k,datatp_k_1,datatp_weighted,\
        A_gradient_cg_pk,A_datatp_k;
    fmat sum_num(1,1);
    float beta_k,alpha_k,residual_pow,residual_k;
    datatp_weighted.copy_size(par.realdataTP);
    datatp_k.copy_size(datatp_weighted);
    datatp_k_1.copy_size(datatp_weighted);
    gradient_rk.copy_size(datatp_weighted);
    gradient_rk_1.copy_size(datatp_weighted);
    gradient_cg_pk.copy_size(datatp_weighted);
    gradient_cg_pk_1.copy_size(datatp_weighted); 
    A_gradient_cg_pk.copy_size(datatp_weighted);
    A_datatp_k.copy_size(datatp_weighted);
    
    iter=0;
    datatp_k=par.realdataTP*(1.0/float(par.nx*par.ny));
    //slantstack3d_recover_L_operator(recoverdatatx_uk,datatp_k,\
        par.ptrace_coord,par.pline_coord,par.ntrace_coordx,par.nline_coordy,\
        par.dt,par.numthread);
    //slantstack3d_stack_LT_operator(A_datatp_k,recoverdatatx_uk,\
        par.ptrace_coord,par.pline_coord,par.ntrace_coordx,par.nline_coordy,\
        par.dt,par.numthread);
    RadonMixFTCG3d_LTL_operator(A_datatp_k,datatp_k,(&par)[0]);

    //regularization
    fcubemul(datatp_weighted,datatp_k,par.weighted_fcube,nthread);
    A_datatp_k=A_datatp_k+datatp_weighted;
    gradient_rk=par.realdataTP-A_datatp_k;
    gradient_cg_pk=gradient_rk;

    //cal residual_pow
    sum_num(0,0)=fcube_inner_product(gradient_rk,gradient_rk,nthread);
    residual_k=sum_num(0,0);
    residual_pow=residual_k;
    residual_pow*=residual_ratio;

    //slantstack3d_recover_L_operator(recoverdatatx_uk,gradient_cg_pk,\
        par.ptrace_coord,par.pline_coord,par.ntrace_coordx,par.nline_coordy,\
        par.dt,par.numthread);
    //slantstack3d_stack_LT_operator(A_gradient_cg_pk,recoverdatatx_uk,\
        par.ptrace_coord,par.pline_coord,par.ntrace_coordx,par.nline_coordy,\
        par.dt,par.numthread);
    RadonMixFTCG3d_LTL_operator(A_gradient_cg_pk,gradient_cg_pk,(&par)[0]);
    //regularization
    fcubemul(datatp_weighted,gradient_cg_pk,par.weighted_fcube,nthread);
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
        cout<<"iterations times:"<<iter<<" || "<<\
        "err level:"<<residual_k/residual_pow<<endl;

        //slantstack3d_recover_L_operator(recoverdatatx_uk,gradient_cg_pk,\
            par.ptrace_coord,par.pline_coord,par.ntrace_coordx,par.nline_coordy,\
            par.dt,par.numthread);
        //slantstack3d_stack_LT_operator(A_gradient_cg_pk,recoverdatatx_uk,\
            par.ptrace_coord,par.pline_coord,par.ntrace_coordx,par.nline_coordy,\
            par.dt,par.numthread);
        RadonMixFTCG3d_LTL_operator\
            (A_gradient_cg_pk,gradient_cg_pk,(&par)[0]);
        //regularization
        fcubemul(datatp_weighted,gradient_cg_pk,par.weighted_fcube,nthread);
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
    par.realdataTP=datatp_k;
    cout<<"iterations times:"<<iter<<" || "<<\
        "err level:"<<residual_k/residual_pow<<endl;

}

int RadonMix_FT_CG_3D(fcube &tauppanel,fcube &recoverdata,fcube &recovererr,\
 fcube& trace, fmat coordx,fmat coordy, int ns, int ntrace,int nline,float dt,\
 int npx,float pxmin, float dpx,int npy,float pymin, float dpy,\
 float fmax=150,float frule=50,int ncpu=1, float factor_L2=0.1,float factor_L1=0.0,\
 int iterations_num=45, float residual_ratio=0.1, bool dorecover=false,\
 bool fastHessian=false)
{
    struct linerradon3d par;
    int i,j,k;
    int nz(ns),nx(ntrace),ny(nline),nf(ns);
//////////////////////////radon par-set////////////////////////////
    beamforming_parset(nx,ny,nz,par);
    par.dpx=dpx;
    par.dpy=dpy;
    par.dz=dt;

//The default px of central channel is zero
    par.npx=npx;
    par.p0x=pxmin;
    par.npy=npy;
    par.p0y=pymin;  

//ncpu
    par.numthread=ncpu;

//regularization parameter
    par.dig_n2=nx*ny*factor_L2;  //L2, Tikhonov 
    par.dig_n1=nx*ny*factor_L1;  //L1,
//Seismic trace coordinates
    for(i=0;i<nx;i++){
        for(j=0;j<ny;j++){
            par.coordx3d(i,j)=coordx(i,j);
            par.coordy3d(i,j)=coordy(i,j);
        }}
    par.regularization=false;
    par.fastHessian=fastHessian;
//Parameters updated
    beamforming_parupdate(par);
//Frequency calculation range (number)
    par.nf2=min(int(fmax/par.df),nz/2-1); 
    par.nf1=1; 
//Low frequency constraint range (number)
    par.rulef2=min(int(frule/par.df),nz/2-1);
    par.rulef1=1;

    par.data=trace;
    //trace.set_size(1,1,1);
    tauppanel.set_size(1,1,1);
//////////////////////////////////////////////////
    linerradon(par,true); 
    par.hessianf2npx2npy=new cx_fmat[par.nf2-par.nf1];
    par.base5dexist=false;
    RadonMixFTCG3d(par,iterations_num,residual_ratio);
    tauppanel=par.realdataTP;
    //trace=par.data;

    if(dorecover){
        //recover data
        rebuildsignal(par);
        recoverdata=par.realrebuildtx;
        recovererr=recoverdata-trace;
    }
    
    return 0;
}


#endif