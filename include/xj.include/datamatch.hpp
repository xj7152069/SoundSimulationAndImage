/*
    Multiple wave prediction and suppression of seismic data.
    Clear up in 2022.11.05, by Xiang Jian, WPI.

*/

#ifndef DATA_MATCH_HPP
#define DATA_MATCH_HPP

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

#include "../xjc.h"
using namespace std;
using namespace arma;

///////////////////// Function declaration ///////////////////////
void dataMatching(fcube& match, fcube data, fcube &dict,\
 fcube& swap3d, int dataLength, int wienerLength, int halfWinx,\
 int slideLength, float wienerTikhonov, float dt, int ncpu, \
 int num_shift=0, int d_shift=1, float weight_shift=0.1);
void AdaptiveRemoveMultiple2d(\
 fmat & dataResult, fmat & dataSwap, fmat & data, fcube * multiModel,\
 ivec filterLen, int nt,int nx, int nwt,int dnwt,\
 float lmd, int half_winx, int num_shift, int d_shift,\
 float weight_shift, int ncpu);
void AdaptiveRemoveMultiple2dPthread(\
 fmat* dataResult, fmat* dataTimeShift, fmat* dataOrig, \
 fcube* dataModel, ivec* filterLen, int nx1,int dnx, int nxMax, \
 int nt, int nwt, int dnwt, float lmd,  int half_winx, \
 int maxTimeShiftNum, int d_shift, float weight_shift);
///////////////////////////////////////////////////////////
void AdaptiveRemoveMultiple2d(\
 fmat & dataResult, fmat & dataSwap, fmat & data, \
 fcube * multiModel,ivec filterLen, int nt, int nx, 
 int nwt,int dnwt, float lmd, int half_winx, \
 int num_shift, int d_shift, float weight_shift, int ncpu)
 {
    fmat dataOrig(nt,nx);
    float xs;
    xs=data.max()-data.min();
    dataOrig=data/(xs);
    //datavz=data;
    dataResult.fill(0.0);
////////////////////////////////////////////////////////////////////
    thread* pcal;
    bool* end_of_thread;
    ncpu=min(ncpu,nx);
    ncpu=max(ncpu,1);
    pcal=new thread[ncpu];
    end_of_thread=new bool[ncpu];
    int percentage;

    for(int k=0;k<ncpu;k++){
        int nx1=round(k*((nx+0.001)/ncpu));
        int nx2=round((k+1)*((nx+0.001)/ncpu));
        pcal[k]=thread(AdaptiveRemoveMultiple2dPthread,\
            &dataResult, &dataSwap, &dataOrig, multiModel,\
            &filterLen, nx1, 1, nx2,nt, nwt, dnwt,lmd, half_winx, \
            num_shift, d_shift, weight_shift);
    }
    for(int k=0;k<ncpu;k++){
        if(pcal[k].joinable())
            pcal[k].join();
    }
    delete [] pcal;
    delete [] end_of_thread;
    cout<<" Finished"<<endl;
    dataResult*=xs;
}

void GetWienerFilter2d(fmat& mat1,fmat& data2d, fcube* multiModel, fmat& w,
    ivec filterLen, int timeOffset, int timeShift, int spaceOffset, \
    int n1Len, int n2Len, float lmd)
{
    int wt1,kwt,nw(w.n_rows),modelNum(filterLen.n_elem);
    int datanum=n1Len*n2Len;
    int alllen=0;
    fmat submat(n1Len,n2Len);
    for(int km=0;km<modelNum;km++){
        int flen=filterLen(km);
    if(filterLen(km)<1){continue;}
    else{
        for(int ki=0;ki<n2Len;ki++){
            int kn1=ki*n1Len;
        for(int i=0;i<n1Len;i++){
        for(int j=0;j<flen;j++){
            mat1(kn1+i,j+alllen)=multiModel[km]\
                (timeOffset+timeShift+i,spaceOffset+ki,j);
        }}}
        alllen+=flen;
    }}
    submat=data2d(span(timeOffset,timeOffset+n1Len-1),\
        span(spaceOffset,spaceOffset+n2Len-1));
    float *psubmat;
    psubmat=&(submat(0,0));

    fmat matd(psubmat, datanum,1); 
    dmat x2(nw,1,fill::zeros);
    //matcopy(x2,w);
    x2=solveCG_real<dmat>\
        (mat1, x2, matd, lmd, 0.000000000001, nw*nw);
    matcopy(w,x2);
}
void AdaptiveRemoveMultiple2dPthread(\
 fmat* dataResult, fmat* dataTimeShift, fmat* dataOrig, \
 fcube* dataModel, ivec* filterLen, int nx1,int dnx, int nxMax, \
 int nt, int nwt, int dnwt, float lmd,  int half_winx, \
 int maxTimeShiftNum, int d_shift, float weight_shift)
{
    //nwp=0;//nwph=0;//nwpd=0;//nwphd=0;
if(weight_shift<0.0){
    d_shift=max(d_shift,1);
    int num_shift=max(maxTimeShiftNum,0);
    num_shift=max(int(dataTimeShift[0].max()),num_shift);
    int i,j,k,kwt,nw(sum(filterLen[0]));
    fmat datal,matq(nw,1,fill::zeros);
    int winbeg,winend,winnum,kwtbeg;
    kwtbeg=filterLen[0].max();
    datal.zeros(nt,1);
    for(k=nx1;k<nxMax;k+=dnx){
        winbeg=max(0,k-half_winx);
        winend=min(int(dataOrig[0].n_cols-1),k+half_winx);
        winnum=winend-winbeg+1;
        int datanum=nwt*winnum;
        fmat mat1(datanum,nw);
    for(kwt=kwtbeg+(num_shift*d_shift)+1;kwt<(nt-nwt-(num_shift*d_shift));kwt+=dnwt){
        //matq.fill(1.0/nw);
        int kshift=round(dataTimeShift[0](kwt,k));
        GetWienerFilter2d(mat1,dataOrig[0],dataModel,matq,\
            filterLen[0], kwt, kshift, winbeg, nwt, winnum, lmd);
        fmat matd=mat1*matq;
        for(i=kwtbeg;i<nwt;i++){
            datal(i+kwt,0)+=1.0;
            float dataNum=dataOrig[0](i+kwt,k)\
                -matd(i+(k-winbeg)*nwt,0);
            if(abs(dataOrig[0](i+kwt,k))<abs(dataNum)\
                ||dataNum!=dataNum){
                dataNum=dataOrig[0](i+kwt,k);
            }
            dataResult[0](i+kwt,k)+=dataNum;
        }
    }
    for(i=0;i<nt;i++){
        if(datal(i,0)>0.5){
            dataResult[0](i,k)/=datal(i,0);
    }}
    datal.fill(0);
    }
}else{
    d_shift=max(d_shift,1);
    int num_shift=max(maxTimeShiftNum,0);
    num_shift=max(int(dataTimeShift[0].max()),num_shift);
    int i,j,k,kwt,nw(sum(filterLen[0]));
    fmat datal,matqbackup(nw,1),matq(nw,1,fill::zeros);
    int winbeg,winend,winnum,datanum,kwtbeg;
    kwtbeg=filterLen[0].max();
    datal.zeros(nt,1);
    for(k=nx1;k<nxMax;k+=dnx){
        winbeg=max(0,k-half_winx);
        winend=min(int(dataOrig[0].n_cols-1),k+half_winx);
        winnum=winend-winbeg+1;
        int datanum=nwt*winnum;
        fmat mat1(datanum,nw);
    for(kwt=kwtbeg+(num_shift*d_shift)+1;kwt<(nt-nwt-(num_shift*d_shift));kwt+=dnwt){
        fmat matq0(nw,1,fill::zeros);
        GetWienerFilter2d(mat1,dataOrig[0],dataModel, matq0,\
            filterLen[0], kwt, 0, winbeg, nwt, winnum, lmd);
        //matcopy(matq,x2);
        fmat mat1backup=mat1;
        float minPow(1e30),minkshift(0);
    for(int kshift=-num_shift;kshift<=num_shift;kshift+=1){
        matq.fill(0.0);
        //multiple data_win for slip fliter
        //matq=inv(matD+digmat)*mat1.t()*matd;
        //conv=solveCG(a, x, b, I, 0.00001, N*N);
        int dkshift=kshift*d_shift;
        if(dkshift!=0){
            GetWienerFilter2d(mat1,dataOrig[0],dataModel, matq,\
                filterLen[0], kwt, dkshift, winbeg, nwt, winnum, lmd);
        }else{matcopy(matq,matq0);}
        //matqAll.col(dkshift+num_shift)=matq;
        fmat matd=mat1*matq;
        float dataPow(0.0),stableNum(1e-8);
        for(i=kwtbeg;i<nwt;i++){
            float dataNum=dataOrig[0](i+kwt,k)\
                -matd(i+(k-winbeg)*nwt,0);
            if(abs(dataOrig[0](i+kwt,k))<abs(dataNum)\
                ||dataNum!=dataNum){
                dataNum=dataOrig[0](i+kwt,k);
            }
                dataPow+=abs(dataNum);
                //stableNum+=1.0;
        }
        //dataPow=dataPow/stableNum;
        dataPow+=((1.0/mat1.max()+0.01)*weight_shift*dkshift*dkshift);
        if(dataPow<=minPow){
            minkshift=dkshift;
            minPow=dataPow;
            mat1backup=mat1;
            matqbackup=matq;
        }
    }
        fmat matd=mat1backup*matqbackup;
        for(i=kwtbeg;i<nwt;i++){
            datal(i+kwt,0)+=1.0;
            float dataNum=dataOrig[0](i+kwt,k)\
                -matd(i+(k-winbeg)*nwt,0);
            if(abs(dataOrig[0](i+kwt,k))<abs(dataNum)\
                ||dataNum!=dataNum){
                dataNum=dataOrig[0](i+kwt,k);
            }
            dataResult[0](i+kwt,k)+=dataNum;
            dataTimeShift[0](i+kwt,k)+=minkshift;
            //datadown[0](i+kwt,k)=dataOrig[0](i+kwt,k)\
                -dataup[0](i+kwt,k);
        }
    }
        for(i=0;i<nt;i++){
        if(datal(i,0)>0.5){
            dataResult[0](i,k)/=datal(i,0);
            dataTimeShift[0](i,k)/=datal(i,0);
        }}
        datal.fill(0);
    }
}
}

fmat AdaptiveRemoveMultipleMultiShift2d(fmat& data, fmat& multiple, \
 fmat& dataSwap,float dt, int wienerLength, float wienerTikhonov,\
 int dataLength, int slideLength, int half_winx, int numPhase, \
 int num_shift, int d_shift, float weight_shift, int ncpu)
{    
    float pi(3.1415926);
    numPhase=max(numPhase,1);
    num_shift=max(num_shift,0);
    int numTimeShift=2*num_shift+1;
    if(weight_shift>=0.0){numTimeShift=1;}
    else if(dataSwap.max()>0.1 || dataSwap.min()<-0.1){numTimeShift=1;}
    ivec filterLen(numPhase*numTimeShift);
    filterLen.fill(wienerLength);
    fcube* multiModel;
    fcube multiModelOrig(multiple.n_rows,multiple.n_cols,numPhase*numTimeShift);
    multiModel=new fcube[numPhase*numTimeShift];
    //multiModel.slice(0)=multiple/multiple.max();
/***Pseudo multi-channel matching***/
omp_set_num_threads(ncpu);
#pragma omp parallel for
    for(int k=0;k<data.n_cols;k++){
        fmat datal(data.n_rows,1);
        datal=multiple.col(k);
        cx_fmat datasf=fft(datal);
    for(int kp=0;kp<numPhase;kp++){
        float phase=(kp)*2.0*pi/numPhase;
    for(int kt=0;kt<numTimeShift;kt++){
        float tlag=(kt-num_shift)*dt*d_shift;
        if(weight_shift>=0.0){tlag=0.0;}
        else if(dataSwap.max()>0.1 || dataSwap.min()<-0.1){tlag=0.0;}
        cx_fmat datasfSwap=datasf;
        if(abs(tlag)>=(dt*0.5)){
            datasfSwap=shiftTime1Dfft(datasfSwap,tlag,dt);
        }
        if(kp==0){
            datasfSwap=datasfSwap;
        }else{
            datasfSwap=shiftPhase1Dfft(datasfSwap,phase);
        }
        multiModelOrig.slice(kp*numTimeShift+kt).col(k)=2.0*real(ifft(datasfSwap)); 

        //char file[399];
        //file[0]='\0';
        //strcat(file,"m.dat");
        //strcat(file,numtostr(kp*numTimeShift+kt,5));
        //datawrite(datad=multiModelOrig.slice(kp*numTimeShift+kt),file);
    }}
    }
    fmat datad;
    datad.copy_size(data);
    for(int kp=0;kp<numPhase;kp++){
    for(int kt=0;kt<numTimeShift;kt++){
        datad=multiModelOrig.slice(kp*numTimeShift+kt);
        multiModelOrig.slice(kp*numTimeShift+kt)=datad/(datad.max()-datad.min());
    }}
    //multiModel=multiModel*(1.0/filterLen.n_elem);
omp_set_num_threads(ncpu);
#pragma omp parallel for
    for(int km=0;km<filterLen.n_elem;km++){
        multiModel[km].zeros(multiple.n_rows,multiple.n_cols,filterLen(km));
        int kwHalf=floor((filterLen(km)+0.001)/2.0);
        for(int k1=kwHalf;k1<multiple.n_rows-kwHalf;k1++){
        for(int k2=0;k2<multiple.n_cols;k2++){
        for(int k3=0;k3<filterLen(km);k3++){
            int k1w=k1-kwHalf+k3;
            multiModel[km](k1,k2,k3)=multiModelOrig(k1w,k2,km);
        }}}
    }
    multiModelOrig.clear();
    fmat data2dResult;
    data2dResult.copy_size(data);
    //dataSwap.copy_size(data);
    AdaptiveRemoveMultiple2d(\
        data2dResult, dataSwap, data, multiModel,\
        filterLen, data.n_rows,data.n_cols,\
        dataLength,slideLength,wienerTikhonov,half_winx,\
        num_shift, d_shift, weight_shift,ncpu);
    //datawrite(dataSwap,"dataTimeShift.dat");
    delete [] multiModel;
    return data2dResult;
} 
void dataMatching(fcube& match, fcube data, fcube &dict,\
 fcube& swap3d, int dataLength, int wienerLength, int halfWinx,\
 int slideLength, float wienerTikhonov, float dt, int ncpu, \
 int num_shift, int d_shift, float weight_shift)
{
    slideLength=min(dataLength-wienerLength*2,slideLength);
    match.fill(0.0);
    if(weight_shift>=0.0){swap3d.fill(0.0);}
    for(int iline=0;iline<data.n_cols;iline++){
        fmat data2d,multiple2d,demultiple2d,data2dSwap;
        data2d=data.col(iline);
        multiple2d=dict.col(iline);
        data2dSwap=swap3d.col(iline);
        data2d=data2d.st();
        multiple2d=multiple2d.st();
        data2dSwap=data2dSwap.st();
        demultiple2d=AdaptiveRemoveMultipleMultiShift2d(\
            data2d, multiple2d, data2dSwap, \
            dt, wienerLength, wienerTikhonov, dataLength, \
            slideLength, halfWinx, 4, num_shift, \
            d_shift, weight_shift, ncpu);
        //demultiple2d=AdaptiveRemoveMultiple2d(data2d, multiple2d, \
            data2dSwap, dt, wienerLength, wienerTikhonov, dataLength, \
            slideLength, halfWinx,num_shift, d_shift, weight_shift, \
            ncpu);
        //data.col(iline)=demultiple2d.st();
        data2d-=demultiple2d;
        match.col(iline)+=data2d.st();
        if(weight_shift>=0){swap3d.col(iline)+=data2dSwap.st();}
    }
}

#endif
