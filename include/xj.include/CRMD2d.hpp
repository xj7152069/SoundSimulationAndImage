
#ifndef CRMD_2D_HPP
#define CRMD_2D_HPP

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
fmat LinearInterpolateMat2dByRow(fmat data2d);
fmat model2dExpandUp(fmat model, int nExpandRow);
void modelConstantVel2d(fmat &vp, fmat& vs,\
    fmat& rho, float vp0=1500.0, float vs0=0.0);
void elastic2dModelModify(elastic3D_ARMA& obj, \
    fmat modelvp, fmat modelvs, fmat modelrho, \
    int nx, int ny, int nz, \
    float dx, float dy, float dz, float dt,\
    int izFreeSurface, int isPMLSurface, int nthread);
fmat Born2dSRME(\
    elastic3D_ARMA& backGround,fmat data2d,\
    elastic3D_ARMA& scatterField, fmat scatter,\
    int izFreeSurface, int nthread, bool sourceModeling=false);
/////////////////////////////////////////////////////
class CRMD2d{ 
private:
public:
    string filevp,filevs,filerho,\
        fileDirectBeg,fileDirectEnd,\
        fileCSorig,fileCSmul,\
        fileCSdemul,fileCSmatch,\
        fileCOorig,fileCOmul,\
        fileCOdemul,fileCOmatch,\
        filePath,fileOutSwap;
    fmat dataOrig2d,mul2d;
    fmat directBeg2d,directEnd2d;
    fmat coordx,coordy,layerDepth;
    fmat vp2d,vs2d,rho2d,vp2dReverse;
    int nt,nx,ny,nz,ncpu,nSmooth;
    int sxBeg,sxEnd,sxGap,sxNow;
    int offsetBeg,offsetEnd,offsetGap;
    float dx,dy,dz,dt,sx,sy;
    float fmin,fmax,df,velWater;
    int izFreeSurface;
    int nIndxNum;

    CRMD2d();
    ~CRMD2d();
    void ClearData();
    void NewDataMatInfo(fmat& csCoordx,fmat& csCoordy,int ksx);
    void GetOnePar(const char *name, const char *value);
    void ReadPar(const char *fileName);
    fmat ReadOneGatherData(char const *filename, int sxid);
    void WriteOneGatherData(fmat data2d, char const *filename, int sxid);
    fmat GetReverseLayerDepth(fmat vp2dRes,float vel,int n1, int n2, float nSmoothForDepth);
    void SrmmRayPathMultiShotMultiLayer(char const *fileout, char const *filein, \
        fcube layer3d_xyz, fvec imp, int nthread,int leftExpand, int rightExpand);

    fmat GetCoordx(int nx,float dx,float x0=0.0);
    fmat GetCoordy(int ny,float dy,float y0=0.0);
    fmat GetElasticImpedance(int n=1);
    fmat GetVpReverse();
    void DatSeriesToOneSUandTimeReSampling( char const *fileout, \
        char const *filein, int nt2, int ncpu);
    void OneSUToDatSeriesAndTimeReSampling(char const *fileout, \
        char const *filein, int nt2, int ntOrig, int ncpu);
    void OneDatToDatSeriesAndTimeReSampling(char const *fileout, \
        char const *filein, int nt2, int ntOrig, int ncpu);
    void RemoveDirectMultiShot(\
        char const *fileout, char const *filein);
    fmat GetOneCommOffsetGatherFromCS(\
        char const *fileCOout,int sxOffset);
    void GetMultiCommOffsetGatherFromCS(\
        char const *fileCOout, char const *fileCSin);
    void GetMultiCommShotGatherFromCO(\
        char const *fileCSout, char const *fileCOin);
    fmat GetMatchingDataWithModelOneGather(\
        fmat& timeShift2d, fmat origData2d, fmat modelData2d, \
        int dataTimeLen, int filterLen, int dataSpaceLen,\
        int dataTimeSlide, float wienerTikhonov, int phaseNum,\
        int num_shift, int d_shift, float weight_shift);
    int GetMatchingDataWithModelMultiGather(\
        char const *fileMatch, char const *fileResultData, char const *fileExpected,\
        char const *fileOrigData, char const *fileModel, \
        int nthread, int numLoop, int numLevel, int numAntiLoop, int dataTimeLen, \
        int filterLen, int dataSpaceLen, int dataTimeSlide, float wienerTikhonov, \
        int phaseNum, int num_shift, int d_shift, float weight_shift);
    int GetMatchingDataWithMultiModelMultiGather(\
        char const *fileMatch, char const *fileResultData, char const *fileOutModel, \
        char const *fileExpected, char const *fileOrigData, \
        string * fileModel, fvec modelWeight, int nthread, \
        int numModelUpdate, int numLoop, int numLevel, int numAntiLoop, \
        int dataTimeLen, int filterLen, int dataSpaceLen, int dataTimeSlide, \
        float wienerTikhonov, int phaseNum, int num_shift, int d_shift, \
        float weight_shift,bool doAgc, bool doST);
    int GetMatchingDataWithModelMultiGather(\
        char const *fileMatch, char const *fileResultData,\
        char const *fileOrigData, char const *fileModel, \
        int nthread, int numLoop,  int dataTimeLen, int filterLen, \
        int dataSpaceLen, int dataTimeSlide, float wienerTikhonov, \
        int phaseNum, int num_shift, int d_shift, float weight_shift);
    fmat GetMatchingDataWithModelAgcOneGather(\
        fmat data2d, fmat model2d,\
        int numLoop, int dataTimeLen, int filterLen, \
        int dataSpaceLen, int dataTimeSlide, float wienerTikhonov, \
        int phaseNum, int num_shift, int d_shift, float weight_shift);
    fmat DeNoiseByBeamForming2D(char const *filein, int npx=1001);
    void ChangeFrequenceMultiShot(char const *fileout, float nPow, int nthread);
    fmat AgcMatchPow2d(const fmat& dataFix2d, int nw1, int nw2,float maxAgc);
    
    void ForwardOBNOMultiShot(char const *fileCSoutP, char const *fileCSoutVz,
        int isPmlSurface, int expandModelUp, int nthread,float f0);
    void ForwardOBNOneShot(int sz, int freeiz, int sxid, \
        int isPmlSurface,int expandModelUp, char const *fileCSoutP, char const *fileCSoutVz);
    void acoustic2dOBNforward(elastic3D_ARMA& obj, char const *fileCSoutP, char const *fileCSoutVz, \
        int sx, int sz, int freeiz, float f0, float waveletType);
    fmat acousticFDSRMOneShot(fmat &suf2d, int sxid, float cutSlopeVelMin, \
        float cutSlopeVelMax, float cutDelayTime,int expandModelUp);
    fmat RTMofOBNOneShot(fmat& l2d, fmat& suf2d, int sxid, int sz, int rz, \
        int upExpand, int downExpand, int maxOffset, int ntSave);
    fmat ForwardTowOneShot(int sxid, int isPmlSurface,int expandModelUp,float f0);
    void ForwardTowMultiShot(char const *fileCSout,
        int isPmlSurface, int expandModelUp, int nthread,float f0);
    void ForwardBornTowMultiShot(char const *fileCSout,
        int isPmlSurface, int expandModelUp, int nthread,float f0);
    fmat RTMofTowOneShot(char const *fileCSin, int sxid, \
        int upExpand, int downExpand, int maxOffset, int ntSave);
    fmat RTMofTowMultiShot(char const *fileCSin,\
        int upExpand, int downExpand, int maxOffset, \
        int ntSave, int nthread);

    fvec GetLayerDepth(float vel, float nSmoothForDepth=8);
    fmat GetLayerDepth(fvec dep, int nWide=10);
    fmat GetWaterBottemPrimaryRayTimeOneShot(int sxid,float delayTime,int wide);
    fmat GetWaterBottemPrimaryBornOneShot(fmat scatter2d,\
        int sxid,float dxIn,float dzIn,float delayTime);
    fmat GetLayerPrimaryRayTimeCS(int sxid, float delayTime=0.05, int wide=10);
    fmat RemoveDirectOneShot(fmat data2d, int sxid);
    fmat MWDBornOneShot(fmat suf2d,float velMax, \
        int nSmoothVp=1,int expandModelUp=0);
    void MWDBornMultiShot(\
        char const *fileout, char const *filein, \
        float velMax, int nSmoothVp, int nthread);
    void ISSInterMulBornMultiShot(\
        char const *fileout, char const *filein, \
        fmat scatterUp, fmat scatterDown, int nWinLeft, int nWinRight, \
        float cutSlopeVelMin, float cutSlopeVelMax, int ntCutWin, float cutDelayTime, \
        int nSmoothVpBorn, int expandModelUp, int nthread);
    fmat MWDBornPhaseShiftOneShot(fmat suf2d, \
        float velMax, int nSmoothVp, int nDz, float nxEx);
    fmat SrmmBornOneShot(fmat suf2d, int nSmoothVp, \
        int expandModelUp, bool sourceModeling);
    fmat SrmmBornOneShot(fmat suf2d, fmat scatterOrig, \
        int nSmoothVp, int expandModelUp, bool sourceModeling);
    void SrmmRayPathMultiShot(\
        char const *fileout, char const *filein,
        int nthread);
    fmat SrmmRayPathOneShot(cx_fmat& timeBase, fmat data2d, \
        fmat csCoordx,fmat csCoordy, bool isOBNsys=false);

//Test for Interbed multiple modeling:
    //Ray Method:
    void ISSRayPathMultiShot(char const *fileout, char const *filein, \
        fmat layerUp, fmat layerDown,  int nthread);
    fmat IrmmRayPathOneShot(cx_fmat& timeBase, fmat data2d, \
        fmat csCoordx,fmat csCoordy,fmat layerDepthUp,fmat layerDepthDown);
    //fmat ISSInterMulBornOneShotTwoLayer(fmat suf2d, \
        fmat CutDepth2d, fmat scatterUp, fmat scatterDown, \
        int sxid, int nSmoothVp, int expandModelUp);
    fmat ISSInterMulBornOneShotTwoLayerUp(\
        fmat suf2d, fmat scatterUp,\
        int sxid, int nSmoothVp, int expandModelUp,\
        int nWinLeft, int nWinRight);
    fmat ISSInterMulBornOneShotTwoLayerDown(\
        fmat suf2d, fmat scatterDown, fmat& suf2dCut, float delayTime, \
        int sxid, int nSmoothVp, int expandModelUp,\
        float cutSlopeVelMin, float cutSlopeVelMax, int ntCutWin);
    fmat InterMulBornOneShot(fmat& sufPri2d, fmat scatterPriDown,\
        fmat scatterUp, fmat scatterDown, int sxid, \
        int nSmoothVp=1, int expandModelUp=0, float f0=30.0);

    void ISSInterMulSSFBornMultiShot(\
        char const *fileout, char const *filein, \
        fmat scatterUp, fmat scatterDown, float cutSlopeVelMin, \
        float cutSlopeVelMax, int ntCutWin, float cutDelayTime, \
        int nProcessThread=1,  int nWinLeft=300, int nWinRight=300);
    fmat ISSInterMulSSFBornOneShotTwoLayer(\
        fmat suf2d, fmat scatterUp, fmat scatterDown,\
        float cutSlopeVelMin, float cutSlopeVelMax, int ntCutWin, \
        float cutDelayTime, int sxid);
    //fmat InterMulModelingSSFBornOneShot(fmat& sufPri2d, \
        fmat scatterPriDown,fmat scatterUp, fmat scatterDown, \
        int sxid, int nSmoothVp=1, float f0=30.0);
};
fmat CRMD2d::GetVpReverse()
{
    vp2dReverse=vp2d;
    vp2dReverse=vp2d.max()-vp2d;
    return vp2dReverse;
}

fmat CRMD2d::GetCoordx(int nx, float dx,float x0)
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
fmat CRMD2d::GetCoordy(int ny, float dy,float y0)
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
CRMD2d::CRMD2d()
{
    nIndxNum=6;
    nt=1,nx=1,ny=1,nz=1,ncpu=1,nSmooth=1;
    sxBeg=2,sxEnd=1,sxGap=1,sxNow=1;
    offsetBeg=2,offsetEnd=1,offsetGap=1;
    dx=1,dy=1,dz=1,dt=0.001,sx=1,sy=1;
    fmin=1,fmax=1,df=1,velWater=1500;
    izFreeSurface=1;
    (fileDirectBeg="null");
    (fileDirectEnd="null");
    (filePath="");
    (fileCOdemul="null");
    (fileCOmatch="null");
    (fileCOmul="null");
    (fileCOorig="null");
    (fileCSdemul="null");
    (fileCSmatch="null");
    (fileCSmul="null");
    (fileCSorig="null");
    (fileOutSwap="null");
    (filevp="null");
    (filevs="null");
    (filerho="null");
}
void CRMD2d::GetOnePar(const char *name, const char *value)
{
    //parList:
    //nt,nx,nz,nsx,ncpu,dx,dz,dt,fmin,fmax,nSmooth,izFreeSurface;
    //filevp,filevs,filerho,filedata,filemul,velWater;
    if (strcmp(name,"fileRootPath")==0){
        (this->filePath=value);
        if(strcmp(value,"null")==0){this->filePath="";}
    }
    else if(strcmp(name,"nxCommShot")==0) {this->nx=atoi(value);}
    else if (strcmp(name,"nt")==0) {this->nt=atoi(value);}
    else if (strcmp(name,"nz")==0) {this->nz=atoi(value);}
    else if (strcmp(name,"sxBeg")==0) {this->sxBeg=atoi(value);}
    else if (strcmp(name,"sxEnd")==0) {this->sxEnd=atoi(value);}
    else if (strcmp(name,"sxGap")==0) {this->sxGap=atoi(value);}
    else if (strcmp(name,"sxNow")==0) {this->sxNow=atoi(value);}
    else if (strcmp(name,"offsetBeg")==0) {this->offsetBeg=atoi(value);}
    else if (strcmp(name,"offsetEnd")==0) {this->offsetEnd=atoi(value);}
    else if (strcmp(name,"offsetGap")==0) {this->offsetGap=atoi(value);}
    else if (strcmp(name,"ncpu")==0) {this->ncpu=atoi(value);}
    else if (strcmp(name,"nSmooth")==0) {this->nSmooth=atoi(value);}
    else if (strcmp(name,"izFreeSurface")==0) {this->izFreeSurface=atoi(value);}
    else if (strcmp(name,"dx")==0) {this->dx=atof(value);}
    else if (strcmp(name,"dz")==0) {this->dz=atof(value);}
    else if (strcmp(name,"dt")==0) {this->dt=atof(value);}
    else if (strcmp(name,"fmin")==0) {this->fmin=atof(value);}
    else if (strcmp(name,"fmax")==0) {this->fmax=atof(value);}
    else if (strcmp(name,"velWater")==0) {this->velWater=atof(value);}
    else if (strcmp(name,"filevp")==0) {(this->filevp=value);}
    else if (strcmp(name,"filevs")==0){(this->filevs=value);}
    else if (strcmp(name,"filerho")==0){(this->filerho=value);}
    else if (strcmp(name,"fileCOdemul")==0){(this->fileCOdemul=value);}
    else if (strcmp(name,"fileCOmatch")==0){(this->fileCOmatch=value);}
    else if (strcmp(name,"fileCOmul")==0){(this->fileCOmul=value);}
    else if (strcmp(name,"fileCOorig")==0){(this->fileCOorig=value);}
    else if (strcmp(name,"fileCSdemul")==0){(this->fileCSdemul=value);}
    else if (strcmp(name,"fileCSmatch")==0){(this->fileCSmatch=value);}
    else if (strcmp(name,"fileCSmul")==0){(this->fileCSmul=value);}
    else if (strcmp(name,"fileCSorig")==0) {(this->fileCSorig=value);}
    else if (strcmp(name,"fileOutSwap")==0){(this->fileOutSwap=value);}
    else if (strcmp(name,"fileDirectBeg")==0){(this->fileDirectBeg=value);}
    else if (strcmp(name,"fileDirectEnd")==0){(this->fileDirectEnd=value);}
    else{std::cout<<"Warning: Non-Existent Parameter!"<<name<<std::endl;}
}
void CRMD2d::ReadPar(const char *fileName){
    ifstream parin;
    char name[1024],value[1024];
    string line;
    parin.open(fileName);
    if(parin){
        while(getline(parin,line)){
            int n1,n2;
            //cout<<line<<endl;
            n1=line.find_first_of('=');
            n2=line.find_first_of(';');
            if(n1<1){continue;}
            for(int k=0;k<n1;k++){
                name[k]=line.at(k);
            }
            name[n1]='\0';
            for(int k=n1+1;k<n2;k++){
                value[k-n1-1]=line.at(k);
            }
            value[n2-n1-1]='\0';
            if(strcmp(name,"exit")==0){break;}
            this->GetOnePar(name,value);
        }
    }else{
        cout<<"Not find Par file!"<<endl;
    }
    df=1.0/dt/nt;
    layerDepth.zeros(nx,ny);
    vp2d.zeros(nz,nx);
    vs2d.zeros(nz,nx);
    rho2d.zeros(nz,nx);
    directBeg2d.zeros(nt,nx);
    directEnd2d.zeros(nt,nx);
    dataOrig2d.zeros(nt,nx);
    mul2d.zeros(nt,nx);
    coordx.zeros(nx,ny);
    coordy.zeros(nx,ny);
    sx=sxNow*dx;

    fileCOdemul=filePath+fileCOdemul;
    fileCOmatch=filePath+fileCOmatch;
    fileCOmul=filePath+fileCOmul;
    fileCOorig=filePath+fileCOorig;
    fileCSdemul=filePath+fileCSdemul;
    fileCSmatch=filePath+fileCSmatch;
    fileCSmul=filePath+fileCSmul;
    fileCSorig=filePath+fileCSorig;
    fileOutSwap=filePath+fileOutSwap;
    if(strcmp(fileDirectBeg.c_str(),"null")==0){
        directBeg2d.fill(0.0);
    }else{
        fileDirectBeg=filePath+fileDirectBeg;
        dataread(directBeg2d,fileDirectBeg.c_str());
    }
    if(strcmp(fileDirectEnd.c_str(),"null")==0){
        directEnd2d.fill(0.0);
    }else{
        fileDirectEnd=filePath+fileDirectEnd;
        dataread(directEnd2d,fileDirectEnd.c_str());
    }
    if(strcmp(filevp.c_str(),"null")==0){
        vp2d.fill(velWater);
    }else{
        filevp=filePath+filevp;
        dataread(vp2d,filevp.c_str());
    }
    if(strcmp(filevs.c_str(),"null")==0){
        vs2d.fill(0.0);
    }else{
        filevs=filePath+filevs;
        dataread(vs2d,filevs.c_str());
    }
    if(strcmp(filerho.c_str(),"null")==0){
        rho2d.fill(1000.0);
    }else{
        filerho=filePath+filerho;
        dataread(rho2d,filerho.c_str());
    }
}
CRMD2d::~CRMD2d()
{
    
}

fmat CRMD2d::GetElasticImpedance(int n)
{
    fmat Impedance(nz,nx),Imp(nz,nx);
    Impedance.fill(0.0);
    Imp.fill(0.0);
    for(int ix=0;ix<nx;ix++){
    for(int iz=0;iz<nz-1;iz++){
        float imp=(vp2d(iz+1,ix)*rho2d(iz+1,ix)-vp2d(iz,ix)*rho2d(iz,ix))\
            /(vp2d(iz+1,ix)*rho2d(iz+1,ix)+vp2d(iz,ix)*rho2d(iz,ix)+0.000001);
        Impedance(iz,ix)=imp;
    }}
    for(int ix=0;ix<nx;ix++){
    for(int iz=n;iz<nz-1-n;iz++){
        for(int i=-n;i<=n;i++){
            Imp(iz,ix)+=Impedance(iz+i,ix)/(i*i+1.0);
        }
    }}
    return Imp;
}

fmat CRMD2d::AgcMatchPow2d(const fmat& dataFix2d, \
    int nw1, int nw2, float maxAgc=1000.0)
{
    int n1(dataFix2d.n_rows),n2(dataFix2d.n_cols);
    fmat agcelem(n1,n2);
    float maxpow(1.0/maxAgc),winnum(nw1*nw2);
    for(int i=nw1;i<n1-nw1;i++){
    for(int j=nw2;j<n2-nw2;j++){
        float agcpow=0;
        for(int i1=-nw1;i1<=nw1;i1++){
        for(int j1=-nw2;j1<=nw2;j1++){
            agcpow+=(dataFix2d(i+i1,j+j1)*dataFix2d(i+i1,j+j1));
        }}
        agcelem(i,j)=sqrt(agcpow/winnum);
        agcpow=0;
    }}
    for(int i=0;i<nw1;i++)
    {
        agcelem.row(i)=agcelem.row(nw1);
        agcelem.row(n1-1-i)=agcelem.row(n1-1-nw1);
    }
    for(int i=0;i<nw2;i++)
    {
        agcelem.col(i)=agcelem.col(nw2);
        agcelem.col(n2-1-i)=agcelem.col(n2-1-nw2);
    }
    agcelem=agcelem/(agcelem.max());
    return agcelem=agcelem+maxpow;
}
fmat CRMD2d::DeNoiseByBeamForming2D(char const *filein, int npx)
{
    int nf(this->nt),npy(1);
    float dpx(0.000002),dpy(0.000002);

    float frule(50),factor_L2(0.01), factor_L1(0),\
        iterations_num(625), residual_ratio(1e-6);

    float x0(-this->dx*this->nx/2),y0(0.0),\
        px0(-dpx*npx/2),py0(0.0);
    
    fcube data3dtx(this->nx,this->ny,this->nt),\
        data3dtaup(npx,npy,this->nt),\
        data3drecover(this->nx,this->ny,this->nt),\
        data3derr(this->nx,this->ny,this->nt);
    fmat icoordx(this->nx,this->ny),icoordy(this->nx,this->ny);
    
    for(int i=0;i<nx;i++){
    for(int j=0;j<ny;j++){
        icoordx(i,j)=this->dx*i+x0;
        icoordy(i,j)=this->dy*j+y0;
    }}

    dataread3d_bycol_transpose(data3dtx,this->nt,this->nx,filein);
    BeamformingCGDeNoise2D(data3dtaup, data3drecover, data3derr,\
        data3dtx, icoordx, icoordy, this->nt, this->nx, this->ny, this->dt,\
        npx, px0, dpx, npy, py0, dpy, this->fmax, frule,\
        this->ncpu, factor_L2, factor_L1, iterations_num, residual_ratio,\
        true, true, false);

    data3derr=data3drecover-data3dtx;
    char fileout[1024];
    fileout[0]='\0';
    strcat(fileout,filein);
    strcat(fileout,".err");
    datawrite3d_bycol_transpose(data3derr,nt,nx,fileout);
    fileout[0]='\0';
    strcat(fileout,filein);
    strcat(fileout,".recover");
    datawrite3d_bycol_transpose(data3drecover,nt,nx,fileout);
    fileout[0]='\0';
    strcat(fileout,filein);
    strcat(fileout,".tp");
    datawrite3d_bycol_transpose(data3dtaup,nt,npx,fileout);
    fmat data2dDeno=data3drecover.col(0);
    return data2dDeno.st();
}
int CRMD2d::GetMatchingDataWithMultiModelMultiGather(\
    char const *fileMatch, char const *fileResultData, char const *fileOutModel, \
    char const *fileExpected, char const *fileOrigData, \
    string * fileModel, fvec modelWeight, int nthread=1, \
    int numModelUpdate=1, int numLoop=1, int numLevel=0, int numAntiLoop=1, \
    int dataTimeLen=100, int filterLen=5, int dataSpaceLen=25, \
    int dataTimeSlide=10, float wienerTikhonov=0.01, \
    int phaseNum=4, int num_shift=0, int d_shift=1, float weight_shift=-0.1, \
    bool doAgc=false, bool doST=false)
{
    cout<<"Now is running: Get Matching Data With Model Multi-Gather."<<endl;
    int ibeg,iend,igap,nxSwap;
    if(this->fileCSdemul==fileOrigData ||\
        this->fileCSmatch==fileOrigData ||\
        this->fileCSmul==fileOrigData ||\
        this->fileCSorig==fileOrigData){
        ibeg=this->sxBeg;
        iend=this->sxEnd;
        igap=this->sxGap;
    }else if(this->fileCOdemul==fileOrigData ||\
        this->fileCOmatch==fileOrigData ||\
        this->fileCOmul==fileOrigData||\
        this->fileCOorig==fileOrigData){
        ibeg=this->offsetBeg;
        iend=this->offsetEnd;
        igap=this->offsetGap;
        nxSwap=this->nx;
        this->nx=round(float(this->sxEnd-this->sxBeg)\
            /float(this->sxGap)+1.0);
    }else{
        cout<<"Error: Data file is not exit!"<<endl;
        return -1;
    }
omp_set_num_threads(nthread);
#pragma omp parallel for
    for(int ki=ibeg;ki<=iend;ki+=igap){
        cout<<ki<<endl;
        fmat data2d=this->ReadOneGatherData(fileOrigData,ki);
        if(doST)data2d=data2d.st();
        fmat agcElem2d=getAgcElem2d(data2d,dataTimeLen,dataSpaceLen,nthread);
        if(doAgc)deAgc2d(data2d,agcElem2d);
        fmat model2d=this->ReadOneGatherData(fileModel[0].c_str(),ki);
        if(doST)model2d=model2d.st();
        model2d=model2d*modelWeight(0);
        for(int km=1;km<modelWeight.n_elem;km++){
            fmat model2dAdd=this->ReadOneGatherData(fileModel[km].c_str(),ki);
            if(doST)model2dAdd=model2dAdd.st();
            model2dAdd=model2dAdd*modelWeight(km);
            model2d+=model2dAdd;
        }
        this->WriteOneGatherData(model2d,fileOutModel,ki);
        fmat expect2d;
        if(strcmp(fileExpected,"null")!=0){
            expect2d=this->ReadOneGatherData(fileExpected,ki);
            if(doST)expect2d=expect2d.st();
        }else{
            expect2d=data2d;
        }
        fmat timeShift2d,match2d,err2d;
        timeShift2d.copy_size(model2d);
        match2d.copy_size(model2d);
        match2d.fill(0.0);
        fmat orig2d=data2d;
        for(int kmulupdate=0;kmulupdate<numModelUpdate;kmulupdate++){
            //data2d=orig2d;
            //match2d.fill(0.0);
            for(int kloop=0;kloop<numLoop;kloop++){
                err2d=this->GetMatchingDataWithModelOneGather(\
                    timeShift2d,data2d,model2d, \
                    dataTimeLen, filterLen, dataSpaceLen,\
                    dataTimeSlide, wienerTikhonov, phaseNum,\
                    num_shift, d_shift, weight_shift);
                for(int klevel=0;klevel<numLevel;klevel++){
                    err2d=this->GetMatchingDataWithModelOneGather(\
                        timeShift2d,err2d,model2d, \
                        dataTimeLen, filterLen, dataSpaceLen,\
                        dataTimeSlide, wienerTikhonov, phaseNum,\
                        num_shift, d_shift, weight_shift);
                }
                data2d=data2d-err2d;
                match2d=match2d+err2d;
                if(strcmp(fileExpected,"null")==0){
                    expect2d=data2d;
                for(int kloop2=0;kloop2<numAntiLoop;kloop2++){
                    fmat err2dAnti=this->GetMatchingDataWithModelOneGather(\
                        timeShift2d,err2d,expect2d, \
                        dataTimeLen, filterLen, dataSpaceLen/4,\
                        dataTimeSlide, wienerTikhonov, 4,\
                        0, d_shift, weight_shift);
                    data2d=data2d+err2dAnti;
                    match2d=match2d-err2dAnti;
                }}
            }
            //model2d=this->GetMatchingDataWithModelOneGather(\
                timeShift2d,model2d,match2d, \
                dataTimeLen, filterLen, dataSpaceLen/4,\
                dataTimeSlide, wienerTikhonov, 4,\
                num_shift, d_shift, weight_shift);
            model2d=match2d;
        }
        match2d=orig2d-data2d;
        if(strcmp(fileExpected,"null")!=0){
        //if(false){
            for(int kloop2=0;kloop2<numAntiLoop;kloop2++){
                fmat err2dAnti=this->GetMatchingDataWithModelOneGather(\
                    timeShift2d,match2d,expect2d, \
                    dataTimeLen, filterLen, dataSpaceLen,\
                    dataTimeSlide, wienerTikhonov, phaseNum,\
                    num_shift, d_shift, weight_shift);
                data2d=data2d+err2dAnti;
                match2d=match2d-err2dAnti;
            }
        }
        if(doAgc)match2d=fmatmul(match2d,agcElem2d);
        if(doAgc)data2d=fmatmul(data2d,agcElem2d);
        if(doST)data2d=data2d.st();
        if(doST)match2d=match2d.st();
        this->WriteOneGatherData(match2d,fileMatch,ki);
        this->WriteOneGatherData(data2d,fileResultData,ki);
    }
    this->nx=nxSwap;
    return 0;
}
int CRMD2d::GetMatchingDataWithModelMultiGather(\
    char const *fileMatch, char const *fileResultData, \
    char const *fileExpected, char const *fileOrigData, char const *fileModel, \
    int nthread=1, int numLoop=1,int numLevel=1, int numAntiLoop=1, int dataTimeLen=100, \
    int filterLen=5, int dataSpaceLen=25, int dataTimeSlide=10, float wienerTikhonov=0.01, \
    int phaseNum=4, int num_shift=0, int d_shift=1, float weight_shift=-0.1)
{
    cout<<"Now is running: Get Matching Data With Model Multi-Gather."<<endl;
    int ibeg,iend,igap,nxSwap;
    nxSwap=this->nx;
    if(strcmp(this->fileCSdemul.c_str(),fileOrigData)==0 ||\
        strcmp(this->fileCSmatch.c_str(),fileOrigData)==0 ||\
        strcmp(this->fileCSmul.c_str(),fileOrigData)==0 ||\
        strcmp(this->fileCSorig.c_str(),fileOrigData)==0){
        ibeg=this->sxBeg;
        iend=this->sxEnd;
        igap=this->sxGap;
    }else if(strcmp(this->fileCOdemul.c_str(),fileOrigData)==0 ||\
        strcmp(this->fileCOmatch.c_str(),fileOrigData)==0 ||\
        strcmp(this->fileCOmul.c_str(),fileOrigData)==0||\
        strcmp(this->fileCOorig.c_str(),fileOrigData)==0){
        ibeg=this->offsetBeg;
        iend=this->offsetEnd;
        igap=this->offsetGap;
        this->nx=round(float(this->sxEnd-this->sxBeg)\
            /float(this->sxGap)+1.0);
    }else{
        cout<<"Error: Data file is not exit!"<<endl;
        return -1;
    }
omp_set_num_threads(nthread);
#pragma omp parallel for
    for(int ki=ibeg;ki<=iend;ki+=igap){
        cout<<ki<<endl;
        fmat data2d=this->ReadOneGatherData(fileOrigData,ki);
        fmat model2d=this->ReadOneGatherData(fileModel,ki);
        fmat expect2d;
        if(strcmp(fileExpected,"null")!=0){
            expect2d=this->ReadOneGatherData(fileExpected,ki);
        }else{
            expect2d=data2d;
        }
        fmat timeShift2d,match2d;
        timeShift2d.copy_size(model2d);
        match2d.copy_size(model2d);
        match2d.fill(0.0);
        timeShift2d.fill(0.0);
        for(int kloop=0;kloop<numLoop;kloop++){
            fmat err2d=this->GetMatchingDataWithModelOneGather(\
                timeShift2d,data2d,model2d, \
                dataTimeLen, filterLen, dataSpaceLen,\
                dataTimeSlide, wienerTikhonov, phaseNum,\
                num_shift, d_shift, weight_shift);
                if(weight_shift>=0.0){
                    this->WriteOneGatherData(timeShift2d,"./ts.dat",ki);
                    weight_shift=-1.0;
                    num_shift=0;
                }
                for(int klevel=0;klevel<numLevel;klevel++){
                    err2d=this->GetMatchingDataWithModelOneGather(\
                        timeShift2d,err2d,model2d, \
                        dataTimeLen, filterLen, dataSpaceLen,\
                        dataTimeSlide, wienerTikhonov, phaseNum,\
                        num_shift, d_shift, weight_shift);
                }
            data2d=data2d-err2d;
            match2d=match2d+err2d;
            if(strcmp(fileExpected,"null")==0){expect2d=data2d;}
            for(int kloop2=0;kloop2<numAntiLoop;kloop2++){
                fmat err2dAnti=this->GetMatchingDataWithModelOneGather(\
                    timeShift2d,err2d,expect2d, \
                    dataTimeLen, filterLen, dataSpaceLen,\
                    dataTimeSlide, wienerTikhonov, 4,\
                    0, d_shift, weight_shift);
                data2d=data2d+err2dAnti;
                match2d=match2d-err2dAnti;
            }
        }
        this->WriteOneGatherData(match2d,fileMatch,ki);
        this->WriteOneGatherData(data2d,fileResultData,ki);
    }
    this->nx=nxSwap;
    return 0;
}
int CRMD2d::GetMatchingDataWithModelMultiGather(\
    char const *fileMatch, char const *fileResultData,\
    char const *fileOrigData, char const *fileModel, int nthread=1, \
    int numLoop=1, int dataTimeLen=100, int filterLen=5, \
    int dataSpaceLen=25, int dataTimeSlide=10, float wienerTikhonov=0.01, \
    int phaseNum=4, int num_shift=0, int d_shift=1, float weight_shift=-0.1)
{
    cout<<"Now is running: Get Matching Data With Model Multi-Gather."<<endl;
    int ibeg,iend,igap,nxSwap;
    if(this->fileCSdemul==fileOrigData ||\
        this->fileCSmatch==fileOrigData ||\
        this->fileCSmul==fileOrigData ||\
        this->fileCSorig==fileOrigData){
        ibeg=this->sxBeg;
        iend=this->sxEnd;
        igap=this->sxGap;
    }else if(this->fileCOdemul==fileOrigData ||\
        this->fileCOmatch==fileOrigData ||\
        this->fileCOmul==fileOrigData||\
        this->fileCOorig==fileOrigData){
        ibeg=this->offsetBeg;
        iend=this->offsetEnd;
        igap=this->offsetGap;
        nxSwap=this->nx;
        this->nx=round(float(this->sxEnd-this->sxBeg)\
            /float(this->sxGap)+1.0);
    }else{
        cout<<"Error: Data file is not exit!"<<endl;
        return -1;
    }
omp_set_num_threads(nthread);
#pragma omp parallel for
    for(int ki=ibeg;ki<=iend;ki+=igap){
        cout<<ki<<endl;
        fmat data2d=this->ReadOneGatherData(fileOrigData,ki);
        fmat model2d=this->ReadOneGatherData(fileModel,ki);
        fmat noise2d=data2d;
        noise2d.randu(data2d.n_rows,data2d.n_cols);
        noise2d*=(data2d.max()*0.005);
        data2d+=noise2d;
        fmat timeShift2d,match2d;
        timeShift2d.copy_size(model2d);
        match2d.copy_size(model2d);
        match2d.fill(0.0);
        for(int kloop=0;kloop<numLoop;kloop++){
            fmat err2d=this->GetMatchingDataWithModelOneGather(\
                timeShift2d,data2d,model2d, \
                dataTimeLen, filterLen, dataSpaceLen,\
                dataTimeSlide, wienerTikhonov, phaseNum,\
                num_shift, d_shift, weight_shift);
            data2d=data2d-err2d;
            match2d=match2d+err2d;
        }
        this->WriteOneGatherData(match2d,fileMatch,ki);
        this->WriteOneGatherData(data2d,fileResultData,ki);
    }
    this->nx=nxSwap;
    return 0;
}
fmat CRMD2d::GetMatchingDataWithModelAgcOneGather(\
    fmat data2d, fmat model2d,\
    int numLoop, int dataTimeLen, int filterLen, \
    int dataSpaceLen, int dataTimeSlide, float wienerTikhonov, \
    int phaseNum, int num_shift, int d_shift, float weight_shift)
{
    fmat elemData2d=this->AgcMatchPow2d(data2d,200,2);
    fmat elemModel2d=this->AgcMatchPow2d(model2d,200,2);
    data2d=fmatdiv(data2d,elemData2d);
    model2d=fmatdiv(model2d,elemModel2d);
    datawrite(data2d,this->fileOutSwap.c_str());
    fmat timeShift2d;
    timeShift2d.copy_size(model2d);
    for(int kloop=0;kloop<numLoop;kloop++){
        fmat err2d=this->GetMatchingDataWithModelOneGather(\
            timeShift2d,data2d,model2d, \
            dataTimeLen, filterLen, dataSpaceLen,\
            dataTimeSlide, wienerTikhonov, phaseNum,\
            num_shift, d_shift, weight_shift);
        data2d=data2d-err2d;
    }
    data2d=fmatmul(data2d,elemData2d);
    return data2d;
}

fmat CRMD2d::GetMatchingDataWithModelOneGather(
    fmat& timeShift2d, fmat origData2d, fmat modelData2d,\
    int dataTimeLen=100, int filterLen=5, int dataSpaceLen=25,\
    int dataTimeSlide=10, float wienerTikhonov=0.01, int phaseNum=4,\
    int num_shift=0, int d_shift=1, float weight_shift=-0.1)
{
    fmat matchingData2d;
    matchingData2d.copy_size(modelData2d);
    matchingData2d.fill(0.0);
    //timeShift2d=matchingData2d;
    int slideLength=min(dataTimeLen-filterLen*2,dataTimeSlide);
    //fmat demultiple2d=AdaptiveRemoveMultiple2d(origData2d, modelData2d, \
        timeShift2d, this->dt, filterLen, wienerTikhonov, dataTimeLen, \
        slideLength, dataSpaceLen, num_shift, d_shift, weight_shift, \
        this->ncpu);
    fmat demultiple2d=AdaptiveRemoveMultipleMultiShift2d(origData2d, modelData2d, \
        timeShift2d, this->dt, filterLen, wienerTikhonov, dataTimeLen, \
        slideLength, dataSpaceLen, phaseNum, num_shift, d_shift, weight_shift, \
        this->ncpu);
    //data.col(iline)=demultiple2d.st();
    matchingData2d=origData2d-demultiple2d;
    return matchingData2d;
}

void CRMD2d::GetMultiCommOffsetGatherFromCS(\
    char const *fileCOout, char const *fileCSin)
{
    cout<<"Now is running: Get Multi-CommOffset Gather."<<endl;
    int nfile=round(float(sxEnd-sxBeg)/float(sxGap))+1;
    int noffset=round(float(offsetEnd-offsetBeg)/float(offsetGap))+1;
    fcube comOffset3d(nt,nfile,noffset,fill::zeros);
    int kfile=(-1);
    for(int ksx=sxBeg;ksx<=sxEnd;ksx+=sxGap){
        cout<<ksx<<endl;
        fmat suf2d=this->ReadOneGatherData(fileCSin,ksx);
        kfile++;
        int koffset=(-1);
        for(int kof=offsetBeg;kof<=offsetEnd;kof+=offsetGap){
            int kd=ksx+kof;
            koffset++;
            if(kd<0){continue;}
            if(kd>=(suf2d.n_cols)){continue;}
            //comOffset2d.col(kfile)=suf2d.col(kd);
            comOffset3d(span::all,span(kfile,kfile),span(koffset,koffset))=\
                suf2d(span::all,span(kd,kd));
        }
    }
    int koffset=(-1);
    fmat comOffset2d(nt,nfile,fill::zeros);
    for(int kof=offsetBeg;kof<=offsetEnd;kof+=offsetGap){
        koffset++;
        comOffset2d=comOffset3d.slice(koffset);
        this->WriteOneGatherData(comOffset2d,fileCOout,kof);
    }
}
void CRMD2d::GetMultiCommShotGatherFromCO(\
    char const *fileCSout, char const *fileCOin)
{
    cout<<"Now is running: Get Multi-CommShot Gather."<<endl;
    int nsx=round(float(sxEnd-sxBeg)/float(sxGap))+1;
    int nof=round(float(offsetEnd-offsetBeg)/float(offsetGap))+1;
    fcube comShot3d(nt,this->nx,nsx,fill::zeros);
    int nxCS=this->nx;
    this->nx=nsx;
    int koffset=(-1);
    for(int kof=offsetBeg;kof<=offsetEnd;kof+=offsetGap){
        cout<<kof<<endl;
        fmat suf2d=this->ReadOneGatherData(fileCOin,kof);
        koffset++;
        int ksxFile=-1;
        int kfile=-1;
        for(int ksx=sxBeg;ksx<=sxEnd;ksx+=sxGap){
            ksxFile++;
            kfile++;
            int kd=ksx+kof;
            if(kd<0){continue;}
            if(kd>=(comShot3d.n_cols)){continue;}
            comShot3d(span::all,span(kd,kd),span(kfile,kfile))=\
                suf2d(span::all,span(ksxFile,ksxFile));
        }
    }
    int kfile=-1;
    this->nx=nxCS;
    fmat comShot2d(nt,this->nx,fill::zeros);
    for(int k=sxBeg;k<=sxEnd;k+=sxGap){
        kfile++;
        comShot2d=comShot3d.slice(kfile);
        this->WriteOneGatherData(comShot2d,fileCSout,k);
    }
}
fmat CRMD2d::GetOneCommOffsetGatherFromCS(\
    char const *fileCOout,int sxOffset)
{
    int nfile=round(float(sxEnd-sxBeg)/float(sxGap))+1;
    fmat comOffset2d(nt,nfile,fill::zeros);
    int kfile=(-1);
    for(int ksx=sxBeg;ksx<=sxEnd;ksx+=sxGap){
        fmat suf2d=this->ReadOneGatherData(fileCOout,ksx);
        kfile++;
        int kd=ksx+sxOffset;
        if(kd<0){continue;}
        if(kd>=(suf2d.n_cols)){continue;}
        comOffset2d.col(kfile)=suf2d.col(kd);
    }
    return comOffset2d;
}

fmat CRMD2d::RemoveDirectOneShot(fmat data2d, int sxid)
{
    if(sxid>=data2d.n_cols){return data2d;}
    for(int kx=0;kx<sxid;kx++){
        int offsetData=kx-sxid;
        int kd=sxEnd+offsetData;
        if(kd<0){break;}
        data2d.col(kx)-=directEnd2d.col(kd);
    }
    for(int kx=sxid;kx<data2d.n_cols;kx++){
        int offsetData=kx-sxid;
        int kd=sxBeg+offsetData;
        if(kd>=directEnd2d.n_cols){break;}
        data2d.col(kx)-=directBeg2d.col(kd);
    }
    return data2d;
}
void CRMD2d::ChangeFrequenceMultiShot(\
    char const *fileout, float nPow, int nthread=1)
{
    cout<<"Now is running: Remove Direct Multi-Shot."<<endl;
omp_set_num_threads(nthread);
#pragma omp parallel for
    for(int ksx=sxBeg;ksx<=sxEnd;ksx+=sxGap){
        cout<<ksx<<endl;
        fmat bornMul2d=this->ReadOneGatherData(fileout,ksx);
        cx_fmat cxMul2d;
        cxMul2d.copy_size(bornMul2d);
        for(int k2=0;k2<cxMul2d.n_cols;k2++){
            cxMul2d.col(k2)=fft(bornMul2d.col(k2));
        }
        for(int k1=1;k1<cxMul2d.n_rows;k1++){
            if(df*k1>1.0){
                float wf=2.0*3.1415926*df*k1;
                //wf=wf*wf;
                wf=pow(wf,nPow);
                if(k1<cxMul2d.n_rows/2)
                    cxMul2d.row(k1)=cxMul2d.row(k1)*wf;
                else
                    cxMul2d.row(k1)*=0.0;
            }
        }
        for(int k2=0;k2<cxMul2d.n_cols;k2++){
            bornMul2d.col(k2)=2.0*real(ifft(cxMul2d.col(k2)));
        }
        this->WriteOneGatherData(bornMul2d,fileout,ksx);
    }
}
void suhead_writeonetrace_tofile(segyhead & head, ofstream & outf){
    outf.write((char *)(&head.head2), sizeof(head.head2));
    datawrite(head.data, outf);
}
void suhead_readonetrace(segyhead & head, ifstream & inf){
    inf.read((char *)(&head.head2), sizeof(head.head2));
    dataread(head.data, inf);
}
void CRMD2d::DatSeriesToOneSUandTimeReSampling(\
    char const *fileout, char const *filein, int nt2, int ncpu=1)
{
    cout<<"Now is running: Transform .dat to .su and resampling Multi-Shot."<<endl;
    segyhead head;
    head.head2={};
    head.data.zeros(nt2,1);
    head.head2.sy=0.0;
    head.head2.gy=0.0;
    head.head2.ns=nt2;
    head.head2.dt=dt*float(this->nt)/float(nt2)*1e6;
    ofstream outf(fileout);
    for(int ksx=sxBeg;ksx<=sxEnd;ksx+=sxGap){
        cout<<ksx<<endl;
        head.head2.sx=(ksx)*dx;
        fmat suf2d=this->ReadOneGatherData(filein,ksx);
        fcube suf3d(suf2d.n_cols,1,suf2d.n_rows);
        suf3d.col(0)=suf2d.st();
        datafftCompress(suf3d,nt2,ncpu);
        suf2d=suf3d.col(0);
        suf2d=suf2d.st();
        for(int ix=0;ix<suf2d.n_cols;ix++){
            head.data=suf2d(span::all,span(ix,ix));
            head.head2.gx=(ix)*this->dx;
            suhead_writeonetrace_tofile(head, outf);
        }
    }
    outf.close();
}
void CRMD2d::OneSUToDatSeriesAndTimeReSampling(char const *fileout, \
    char const *filein, int nt2, int ntOrig, int ncpu=1)
{
    cout<<"Now is running: Transform .dat to .su and resampling Multi-Shot."<<endl;
    segyhead head;
    head.head2={};
    head.data.zeros(ntOrig,1);
    ifstream inf(filein);
    for(int ksx=sxBeg;ksx<=sxEnd;ksx+=sxGap){
        cout<<ksx<<endl;
        fmat suf2d(ntOrig,this->nx);
        for(int ix=0;ix<suf2d.n_cols;ix++){
            suhead_readonetrace(head, inf);
            suf2d(span::all,span(ix,ix))=head.data;
        }
        fcube suf3d(suf2d.n_cols,1,suf2d.n_rows);
        suf3d.col(0)=suf2d.st();
        datafftCompress(suf3d,nt2,ncpu);
        suf2d=suf3d.col(0);
        suf2d=suf2d.st();
        this->WriteOneGatherData(suf2d,fileout,ksx);
    }
    inf.close();
}

void CRMD2d::OneDatToDatSeriesAndTimeReSampling(char const *fileout, \
    char const *filein, int nt2, int ntOrig, int ncpu=1)
{
    cout<<"Now is running: Transform .dat to .su and resampling Multi-Shot."<<endl;
    segyhead head;
    head.head2={};
    head.data.zeros(ntOrig,1);
    ifstream inf(filein);
    for(int ksx=sxBeg;ksx<=sxEnd;ksx+=sxGap){
        cout<<ksx<<endl;
        fmat suf2d(ntOrig,this->nx);
        dataread(suf2d, inf);
        fcube suf3d(suf2d.n_cols,1,suf2d.n_rows);
        suf3d.col(0)=suf2d.st();
        datafftCompress(suf3d,nt2,ncpu);
        suf2d=suf3d.col(0);
        suf2d=suf2d.st();
        this->WriteOneGatherData(suf2d,fileout,ksx);
    }
    inf.close();
}
void CRMD2d::RemoveDirectMultiShot(\
    char const *fileout, char const *filein)
{
    cout<<"Now is running: Remove Direct Multi-Shot."<<endl;
    for(int ksx=sxBeg;ksx<=sxEnd;ksx+=sxGap){
        cout<<ksx<<endl;
        fmat suf2d=this->ReadOneGatherData(filein,ksx);
        suf2d=this->RemoveDirectOneShot(suf2d,ksx);
        this->WriteOneGatherData(suf2d,fileout,ksx);
    }

}
fmat CRMD2d::RTMofTowMultiShot(char const *fileCSin,\
    int upExpand, int downExpand, int maxOffset, \
    int ntSave=0, int nthread=1)
{
    cout<<"Now is running: RTM of Multi-Shot."<<endl;
    int nsx=round(1+(sxEnd-sxBeg)/sxGap);
    fcube image3d(nz+upExpand+downExpand,nx,nsx);
omp_set_num_threads(nthread);
#pragma omp parallel for
    for(int ksx=sxBeg;ksx<=sxEnd;ksx+=sxGap){
        fmat image2d=this->RTMofTowOneShot(fileCSin, ksx, \
            upExpand, downExpand, maxOffset, ntSave);
        image3d.slice(round((ksx-sxBeg)/sxGap))=image2d;
    }
    fmat stack2d(nz+upExpand+downExpand,nx,fill::zeros);
    for(int k=0;k<nsx;k++){
        stack2d+=image3d.slice(k);
    }
    fmat image2d=stack2d(span(upExpand,upExpand+nz-1),span::all);
    return image2d;
}
fmat CRMD2d::RTMofTowOneShot(char const *fileCSin, int sxid, \
    int upExpand, int downExpand, int maxOffset, int ntSave)
{
    ntSave=max(ntSave,this->nt);
    fmat suf2d=this->ReadOneGatherData(fileCSin,sxid);
////////////////////////////////////////////////////////
    fcube vp3d(nx,ny,nz),vs3d(nx,ny,nz),rho3d(nx,ny,nz);
    fmat vp2dSwap(nx,nz);
    vp3d.col(0)=this->vp2d.st();
    for(int k=0;k<this->izFreeSurface;k++){
        vp3d.slice(k)=vp3d.slice(izFreeSurface);
    }
    vp3d.col(0)=fmatsmooth(vp2dSwap=vp3d.col(0),nx,nz,nSmooth);
    vs3d.fill(0.0),rho3d.fill(1000.0);
    //vp3d.fill(1500.0);
    vp3d=model3dExtendSliceUp(vp3d,upExpand);
    vp3d=model3dExtendSliceDown(vp3d,downExpand);
    vs3d=model3dExtendSliceUp(vs3d,upExpand);
    vs3d=model3dExtendSliceDown(vs3d,downExpand);
    rho3d=model3dExtendSliceUp(rho3d,upExpand);
    rho3d=model3dExtendSliceDown(rho3d,downExpand);
    int sz(this->izFreeSurface),rz(this->izFreeSurface);
    int nzEx=nz+upExpand+downExpand;
    sz=sz+upExpand;rz=rz+upExpand;
    
    maxOffset=max(maxOffset,101);
    for(int k=0;k<suf2d.n_cols;k++){
    if(abs(k-sxid)>maxOffset){suf2d.col(k).fill(0.0);}
    else if(abs(k-sxid)>(maxOffset-50)){
        suf2d.col(k)=suf2d.col(k)*Blackman(maxOffset-abs(k-sxid),50.0);
    }}
    
    int isPMLSurface=1;
    int nthreadobj=this->ncpu;
    elastic3D_ARMA objSave,objInv;
    elastic3dParModify( objSave, \
        vp3d, vs3d, rho3d, \
        nx, ny, nzEx, ntSave, dx, dy, dz, dt, \
        rz,isPMLSurface, nthreadobj, 8, 50, 9.0);
    elastic3dParModify( objInv, \
        vp3d, vs3d, rho3d, \
        nx, ny, nzEx, this->nt, dx, dy, dz, dt, \
        rz,isPMLSurface, nthreadobj, 8, 50, 9.0);

    acousticRTM2dTowforward(objSave,sxid,30.0,2.0);
    fmat rtmImage2d;
    acousticRTM2dTowInverse(rtmImage2d,objSave,\
        objInv,suf2d,sxid,30.0,2.0);
    fmat rtmImageSwap2d=rtmImage2d;
    //fmat rtmImageSwap2d(nz,nx);
    //rtmImageSwap2d(span::all,span::all)=\
        rtmImage2d(span(upExpand,nzEx-downExpand-1),span::all);
    return rtmImageSwap2d;
}
fmat CRMD2d::RTMofOBNOneShot(fmat& l2d, fmat& suf2d, \
    int sxid, int sz, int rz, \
    int upExpand, int downExpand, int maxOffset, int ntSave)
{
    ntSave=max(ntSave,this->nt);
    int nthread=(this->ncpu);
////////////////////////////////////////////////////////
    fcube vp3d(nx,ny,nz),vs3d(nx,ny,nz),rho3d(nx,ny,nz);
    fmat vp2dSwap(nx,nz);
    vp3d.col(0)=this->vp2d.st();
    for(int k=0;k<this->izFreeSurface;k++){
        vp3d.slice(k)=vp3d.slice(izFreeSurface);
    }
    vp3d.col(0)=fmatsmooth(vp2dSwap=vp3d.col(0),nx,nz,nSmooth);
    vs3d.fill(0.0),rho3d.fill(1000.0);
    //vp3d.fill(1500.0);
    vp3d=model3dExtendSliceUp(vp3d,upExpand);
    vp3d=model3dExtendSliceDown(vp3d,downExpand);
    vs3d=model3dExtendSliceUp(vs3d,upExpand);
    vs3d=model3dExtendSliceDown(vs3d,downExpand);
    rho3d=model3dExtendSliceUp(rho3d,upExpand);
    rho3d=model3dExtendSliceDown(rho3d,downExpand);
    int nzEx=nz+upExpand+downExpand;
    sz=sz+upExpand;rz=rz+upExpand;
    
    maxOffset=max(maxOffset,101);
    for(int k=0;k<suf2d.n_cols;k++){
    if(abs(k-sxid)>maxOffset){suf2d.col(k).fill(0.0);}
    else if(abs(k-sxid)>(maxOffset-50)){
        suf2d.col(k)=suf2d.col(k)*Blackman(maxOffset-abs(k-sxid),50.0);
    }}
    
    int isPMLSurface=1;
    int nthreadobj=this->ncpu;
    elastic3D_ARMA objSave,objInv;
    elastic3dParModify( objSave, \
        vp3d, vs3d, rho3d, \
        nx, ny, nzEx, ntSave, dx, dy, dz, dt, \
        this->izFreeSurface,isPMLSurface, nthreadobj, 8, 50, 9.0);
    elastic3dParModify( objInv, \
        vp3d, vs3d, rho3d, \
        nx, ny, nzEx, this->nt, dx, dy, dz, dt, \
        this->izFreeSurface,isPMLSurface, nthreadobj, 8, 50, 9.0);

    acousticRTM2dOBNforward(objSave,sxid,sz,30.0,2.0);
    fmat rtmImage2d;
    fmat light2d=acousticRTM2dOBNInverse(rtmImage2d,objSave,\
        objInv,suf2d,sxid,sz,rz,30.0,2.0);
    fmat rtmImageSwap2d=rtmImage2d;
    //fmat rtmImageSwap2d(nz,nx);
    //rtmImageSwap2d(span::all,span::all)=\
        rtmImage2d(span(upExpand,nzEx-downExpand-1),span::all);
    //return light2d;
    l2d=light2d;
    return rtmImageSwap2d;
}
void CRMD2d::ForwardBornTowMultiShot(char const *fileCSout,
    int isPmlSurface, int expandModelUp=0, int nthread=1,float f0=30.0)
{
    cout<<"Now is running: Forward Modeling Multi-Shot."<<endl;
omp_set_num_threads(nthread);
#pragma omp parallel for
    for(int ksx=sxBeg;ksx<=sxEnd;ksx+=sxGap){
        fmat suf2d(nt,nx);
        suf2d.fill(0.0);
        for(int kt=0;kt<nt;kt++){
            suf2d(kt,ksx)+=1e6*wavelet02(kt,dt,f0,0.0);
        }
        fmat data2d=this->SrmmBornOneShot(suf2d, nSmooth, expandModelUp, true);
        this->WriteOneGatherData(data2d,fileCSout,ksx);
    }
}
void CRMD2d::ForwardTowMultiShot(char const *fileCSout,
    int isPmlSurface, int expandModelUp=0, int nthread=1,float f0=30.0)
{
    cout<<"Now is running: Forward Modeling Multi-Shot."<<endl;
omp_set_num_threads(nthread);
#pragma omp parallel for
    for(int ksx=sxBeg;ksx<=sxEnd;ksx+=sxGap){
        fmat data2d=this->ForwardTowOneShot(ksx, \
            isPmlSurface,expandModelUp,f0);
        this->WriteOneGatherData(data2d,fileCSout,ksx);
    }
}
fmat CRMD2d::ForwardTowOneShot(int sxid, \
    int isPmlSurface,int expandModelUp=0,float f0=30.0)
{
    int sz(izFreeSurface),rz(izFreeSurface);
    int nthread=ncpu;
    int UpExpand=max(int(70-izFreeSurface),0);
    expandModelUp=max(UpExpand,expandModelUp);
    int nzEx=nz+expandModelUp,freeSufaceZ=izFreeSurface;
    fmat modelvp(nz,nx),modelvs(nz,nx),modelrho(nz,nx);
    modelvp=vp2d;modelvs=vs2d;modelrho=rho2d;
    for(int k1=0;k1<freeSufaceZ;k1++){
        modelvp.row(k1).fill(modelvp(freeSufaceZ,0));
        modelvs.row(k1).fill(modelvs(freeSufaceZ,0));
        modelrho.row(k1).fill(modelrho(freeSufaceZ,0));
    }
    modelvp=model2dExpandUp(modelvp,expandModelUp);
    modelvs=model2dExpandUp(modelvs,expandModelUp);
    modelrho=model2dExpandUp(modelrho,expandModelUp);
    datawrite(modelvp,"./movie/vp.dat");
    datawrite(modelrho,"./movie/rho.dat");
////////////////////////////////////////////////////////
    elastic3D_ARMA objSave;
    elastic2dModelModify(objSave, \
        modelvp, modelvs, modelrho, nx, ny, nzEx,\
        dx, dy, dz, dt, izFreeSurface+expandModelUp, \
        isPmlSurface, nthread);
    objSave.nt=nt;
    fmat data2d=acoustic2dTowforward(objSave,sxid,f0,2.0-isPmlSurface);
    //fmat data2d=acoustic2dTowSimulation(objSave,sxid,f0,2.0-isPmlSurface);
    return data2d;
}
fmat CRMD2d::acousticFDSRMOneShot(fmat &suf2d, int sxid, \
    float cutSlopeVelMin, float cutSlopeVelMax, float cutDelayTime,\
    int expandModelUp=0)
{
    int isPmlSurface=0;
    int sz(izFreeSurface),rz(izFreeSurface);
    int nthread=ncpu;
    int UpExpand=max(int(70-izFreeSurface),0);
    expandModelUp=max(UpExpand,expandModelUp);
    int nzEx=nz+expandModelUp,freeSufaceZ=izFreeSurface;
    fmat modelvp(nz,nx),modelvs(nz,nx),modelrho(nz,nx);
    modelvp=vp2d;modelvs=vs2d;modelrho=rho2d;
    for(int k1=0;k1<freeSufaceZ;k1++){
        modelvp.row(k1).fill(modelvp(freeSufaceZ,0));
        modelvs.row(k1).fill(modelvs(freeSufaceZ,0));
        modelrho.row(k1).fill(modelrho(freeSufaceZ,0));
    }
    modelvp=model2dExpandUp(modelvp,expandModelUp);
    modelvs=model2dExpandUp(modelvs,expandModelUp);
    modelrho=model2dExpandUp(modelrho,expandModelUp);
    datawrite(modelvp,"./movie/vp.dat");
    datawrite(modelrho,"./movie/rho.dat");
    //modelrho.fill(1.0);
    modelrho/=modelrho.max();
////////////////////////////////////////////////////////
    elastic3D_ARMA objOrig, objOrigDirect, objSRM, objDirect;
    elastic2dModelModify(objOrig, \
        modelvp, modelvs, modelrho, nx, ny, nzEx,\
        dx, dy, dz, dt, izFreeSurface+expandModelUp, \
        0, nthread);
    elastic2dModelModify(objOrigDirect, \
        modelvp, modelvs, modelrho, nx, ny, nzEx,\
        dx, dy, dz, dt, izFreeSurface+expandModelUp, \
        0, nthread);
    elastic2dModelModify(objSRM, \
        modelvp, modelvs, modelrho, nx, ny, nzEx,\
        dx, dy, dz, dt, izFreeSurface+expandModelUp, \
        1, nthread);
    elastic2dModelModify(objDirect, \
        modelvp, modelvs, modelrho, nx, ny, nzEx,\
        dx, dy, dz, dt, izFreeSurface+expandModelUp, \
        1, nthread);
    objOrig.nt=nt;objSRM.nt=nt;
    objDirect.nt=nt;objOrigDirect.nt=nt;
    objOrig.mpar_1=objOrig.mpar_1_dec_ro;
    objSRM.mpar_1=objSRM.mpar_1_dec_ro;
    objDirect.mpar_1=objDirect.mpar_1_dec_ro;
    objOrigDirect.mpar_1=objOrigDirect.mpar_1_dec_ro;

    fmat data2d=acoustic2dTowSimulation(objOrig,objOrigDirect,sxid,30.0,true);
    suf2d=data2d;

    data2d=acoustic2dSRMTowSimulation(objSRM,objDirect,objOrig.dataSaveUp2d);
    return data2d;

}   
void CRMD2d::ForwardOBNOMultiShot(char const *fileCSoutP, char const *fileCSoutVz,
    int isPmlSurface, int expandModelUp=0, int nthread=1,float f0=30.0)
{
    cout<<"Now is running: Forward Modeling Multi-Shot."<<endl;
omp_set_num_threads(nthread);
#pragma omp parallel for
    for(int ksx=sxBeg;ksx<=sxEnd;ksx+=sxGap){
        this->ForwardOBNOneShot( this->izFreeSurface, this->izFreeSurface, \
            ksx, isPmlSurface, expandModelUp, fileCSoutP, fileCSoutVz);
    }
}
void CRMD2d::ForwardOBNOneShot(int sz, int freeiz, int sxid, \
    int isPmlSurface,int expandModelUp, char const *fileCSoutP, char const *fileCSoutVz)
{
    int nthread=ncpu;
    int UpExpand=min(int(70-izFreeSurface),0);
    if(isPmlSurface==0){UpExpand=min(int(10-izFreeSurface),0);}
    expandModelUp=max(UpExpand,expandModelUp);
    int nzEx=nz+expandModelUp,freeSufaceZ=izFreeSurface;
    fmat modelvp(nz,nx),modelvs(nz,nx),modelrho(nz,nx);
    modelvp=vp2d;modelvs=vs2d;modelrho=rho2d;
    for(int k1=0;k1<freeSufaceZ;k1++){
        modelvp.row(k1).fill(modelvp(freeSufaceZ,0));
        modelvs.row(k1).fill(modelvs(freeSufaceZ,0));
        modelrho.row(k1).fill(modelrho(freeSufaceZ,0));
    }
    modelvp=model2dExpandUp(modelvp,expandModelUp);
    modelvs=model2dExpandUp(modelvs,expandModelUp);
    modelrho=model2dExpandUp(modelrho,expandModelUp);
    freeiz=freeiz+expandModelUp;
    sz=sz+expandModelUp;
////////////////////////////////////////////////////////
    elastic3D_ARMA objSave;
    elastic2dModelModify(objSave, \
        modelvp, modelvs, modelrho, nx, ny, nzEx,\
        dx, dy, dz, dt, izFreeSurface+expandModelUp, \
        isPmlSurface, nthread);
    objSave.nt=nt;
    acoustic2dOBNforward(objSave,fileCSoutP,fileCSoutVz,sxid,sz,freeiz,30.0,1.0);
}
void CRMD2d::acoustic2dOBNforward(elastic3D_ARMA& obj, \
    char const *fileCSoutP, char const *fileCSoutVz, \
    int sx, int sz, int freeiz, float f0, float waveletType)
{
    fmat irz=(this->layerDepth/this->dz);
    int nx(obj.nx),ny(obj.ny),nz(obj.nz),nCoe(obj.nCoe);
    int sy(0);
    int (*fptr)(class elastic3D_ARMA&);
    float (*fptrWavelet)(int k, float DT, float hz,float det2);
    if(waveletType<1.5){
        fptrWavelet=wavelet01;
    }else if(waveletType>=1.5 && waveletType<2.5){
        fptrWavelet=wavelet02;
    }
    if(this->ncpu>1){
        fptr=TimeSliceCal_elastic2D_MultiThread;
    }else{
        fptr=TimeSliceCal_elastic2D_OneThread;
    }
    fmat suftzz2d(obj.nt,obj.nx);
    fmat sufvz2d(obj.nt,obj.nx);
    fmat vsp2dp(obj.nt,obj.nz);
    fmat vsp2dvz(obj.nt,obj.nz);
    for(int k=0;k<(obj.nt);k++)
    {
        obj.tzz(sx,sy,sz)+=1e6*(*fptrWavelet)(k,obj.dt,f0,0.0);
        //Using internal parallelism
        (*fptr)(obj);
        for(int i=0;i<nx;i++){
            suftzz2d(k,i)=obj.tzz(i,0,int(irz(i,0)+freeiz));
            sufvz2d(k,i)=obj.vz(i,0,int(irz(i,0)+freeiz));
        }
        for(int i=0;i<nz;i++){
            vsp2dp(k,i)=obj.tzz(sx+60,0,i);
            vsp2dvz(k,i)=obj.vz(sx+60,0,i);
        }
        if(k%1000==0){
            cout<<"now is running : "<<k<<endl;
        }
    }
    this->WriteOneGatherData(suftzz2d,fileCSoutP,sx);
    this->WriteOneGatherData(sufvz2d,fileCSoutVz,sx);
    this->WriteOneGatherData(vsp2dp,"./vsp.p.dat",sx);
    this->WriteOneGatherData(vsp2dvz,"./vsp.vz.dat",sx);

    cout<<"Completed: "<<sx<<endl;
}
fvec CRMD2d::GetLayerDepth(float vel,float nSmoothForDepth)
{
    float izdepth=izFreeSurface*dz;
    fvec dep(nx);
    fmat vp2dInter=LinearInterpolateMat2dByRow(vp2d);
    vp2dInter=LinearInterpolateMat2dByRow(vp2dInter);
    vp2dInter=LinearInterpolateMat2dByRow(vp2dInter);
    vp2dInter=LinearInterpolateMat2dByRow(vp2dInter);
    float dzInter=dz/16.0;
    vp2dInter=fmatsmooth(vp2dInter,vp2dInter.n_rows,nx,nSmoothForDepth);
    for(int k2=0;k2<vp2dInter.n_cols;k2++){
    for(int k1=1;k1<vp2dInter.n_rows;k1++){
        if(vp2dInter(k1,k2)>vel && vp2dInter(k1-1,k2)<=vel){
            layerDepth(k2,0)=k1*dzInter-izdepth;
            dep(k2)=layerDepth(k2,0);
            break;
        }
    }}
    //layerDepth.print();
    return dep;
}
fmat CRMD2d::GetReverseLayerDepth(fmat vp2dRes,float vel,int n1, int n2, float nSmoothForDepth)
{
    float izdepth=izFreeSurface*dz;
    fmat dep(nx,ny);
    fmat vp2dInter=LinearInterpolateMat2dByRow(vp2dRes);
    vp2dInter=LinearInterpolateMat2dByRow(vp2dInter);
    vp2dInter=LinearInterpolateMat2dByRow(vp2dInter);
    vp2dInter=LinearInterpolateMat2dByRow(vp2dInter);
    float dzInter=dz/16.0;
    vp2dInter=fmatsmooth(vp2dInter,vp2dInter.n_rows,nx,nSmoothForDepth);
    for(int k2=0;k2<vp2dInter.n_cols;k2++){
    for(int k1=n2*16;k1>n1*16;k1--){
        if(vp2dInter(k1,k2)>vel && vp2dInter(k1+1,k2)<=vel){
            dep(k2,0)=k1*dzInter-izdepth;
            //dep(k2)=layerDepth(k2,0);
            break;
        }
    }}
    //layerDepth.print();
    return dep;
}

void CRMD2d::MWDBornMultiShot(\
    char const *fileout, char const *filein, \
    float velMax, int nSmoothVp, int nthread)
{
    cout<<"Now is running: MWD Born of Multi-Shot."<<endl;
omp_set_num_threads(nthread);
#pragma omp parallel for
    for(int ksx=sxBeg;ksx<=sxEnd;ksx+=sxGap){
        cout<<ksx<<endl;
        fmat suf2d=this->ReadOneGatherData(filein,ksx);
        fmat sufMul2d=this->MWDBornOneShot(suf2d,velMax,nSmoothVp);
        this->WriteOneGatherData(sufMul2d,fileout,ksx);
    }
}
fmat CRMD2d::MWDBornOneShot(fmat suf2d, float velMax, \
    int nSmoothVp, int expandModelUp)
{
    int nthread(this->ncpu),isPMLSurface(1);
    int autoUpExpand=max(int(70-izFreeSurface),0);
    expandModelUp=max(autoUpExpand,expandModelUp);
    int nzEx(this->nz+expandModelUp);
    //freeSuface, source and obn Z-coord in model
    int freeSufaceZ(izFreeSurface),sz(izFreeSurface),\
        rz(izFreeSurface);
    fmat modelvp(nz,nx),modelvs(nz,nx),modelrho(nz,nx);
    modelvp=vp2d;modelvs=vs2d;modelrho=rho2d;
    for(int k1=0;k1<freeSufaceZ;k1++){
        modelvp.row(k1).fill(modelvp(freeSufaceZ,0));
        modelvs.row(k1).fill(modelvs(freeSufaceZ,0));
        modelrho.row(k1).fill(modelrho(freeSufaceZ,0));
    }
    for(int k1=0;k1<modelvp.n_rows;k1++){
    for(int k2=0;k2<modelvp.n_cols;k2++){
        modelvp(k1,k2)=min(float(modelvp(k1,k2)),velMax);
    }}
    modelvp=model2dExpandUp(modelvp,expandModelUp);
    modelvs=model2dExpandUp(modelvs,expandModelUp);
    modelrho=model2dExpandUp(modelrho,expandModelUp);
    freeSufaceZ+=expandModelUp;
    sz+=expandModelUp;
    rz+=expandModelUp;
    fmat scatter(nzEx,nx),backrho(nzEx,nx),backvp(nzEx,nx);
    backrho=fmatsmooth(modelrho,nzEx,nx,nSmoothVp);
    //modelvp=fmatsmooth(modelvp,nzEx,nx,nSmoothVp);
    backvp=fmatsmooth(modelvp,nzEx,nx,nSmoothVp);
    scatter.fill(0.0);
    for(int k1=0;k1<nzEx;k1++){
    for(int k2=0;k2<nx;k2++){
        scatter(k1,k2)=2.0*((modelvp(k1,k2)\
            -backvp(k1,k2))/backvp(k1,k2)\
            +(modelrho(k1,k2)-backrho(k1,k2))\
            /backrho(k1,k2));
        //scatter(k1,k2)=(1.0/modelvp(k1,k2)/modelvp(k1,k2)\
            -1.0/backvp(k1,k2)/backvp(k1,k2))\
            /(1.0/backvp(k1,k2)/backvp(k1,k2));
    }}
    //datawrite(scatter,"scatter.dat");
    backvp.fill(this->velWater);
    elastic3D_ARMA backGround;
    elastic3D_ARMA scatterField;
    elastic2dModelModify(backGround, \
        backvp, modelvs, backrho, nx, ny, nzEx,\
        dx, dy, dz, dt, freeSufaceZ, \
        isPMLSurface, nthread);
    elastic2dModelModify(scatterField, \
        backvp, modelvs, backrho, nx, ny, nzEx,\
        dx, dy, dz, dt, freeSufaceZ, \
        1, nthread);
/////////////////////////////////////
    fmat bornMul2d=Born2dSRME(\
        backGround, suf2d, scatterField, scatter,\
        freeSufaceZ, nthread);
    return bornMul2d;
}
fmat CRMD2d::MWDBornPhaseShiftOneShot(fmat suf2d, \
    float velMax, int nSmoothVp, int nDz, float nxEx=2.0)
{
    int nthread(this->ncpu),isPMLSurface(1);
    //freeSuface, source and obn Z-coord in model
    int freeSufaceZ=this->izFreeSurface;
    fmat modelvp(nz,nx),modelrho(nz,nx);
    modelvp=vp2d;modelrho=rho2d;
    for(int k1=0;k1<freeSufaceZ;k1++){
        modelvp.row(k1).fill(modelvp(freeSufaceZ,0));
        modelrho.row(k1).fill(modelrho(freeSufaceZ,0));
    }
    for(int k1=0;k1<modelvp.n_rows;k1++){
    for(int k2=0;k2<modelvp.n_cols;k2++){
        modelvp(k1,k2)=min(float(modelvp(k1,k2)),velMax);
    }}
    fmat scatter(nz,nx),backrho(nz,nx),backvp(nz,nx);
    backrho=fmatsmooth(modelrho,nz,nx,this->nSmooth);
    backvp=fmatsmooth(modelvp,nz,nx,nSmoothVp);
    scatter.fill(0.0);
    for(int k1=0;k1<nz;k1++){
    for(int k2=0;k2<nx;k2++){
        scatter(k1,k2)=2.0*((modelvp(k1,k2)\
            -backvp(k1,k2))/backvp(k1,k2)\
            +(modelrho(k1,k2)-backrho(k1,k2))\
            /backrho(k1,k2));
        //scatter(k1,k2)=(1.0/modelvp(k1,k2)/modelvp(k1,k2)\
            -1.0/backvp(k1,k2)/backvp(k1,k2))\
            /(1.0/backvp(k1,k2)/backvp(k1,k2));
        //scatter(k1,k2)=(1.0/modelvp(k1,k2)/modelvp(k1,k2)\
            -1.0/backvp(k1,k2)/backvp(k1,k2));
    }}
    backrho.clear();
    backvp.clear();
/////////////////////////////////////
    int ntSwap(nt*2.0),nxSwap(nx);
    ntSwap=getfftnum(ntSwap);
    nxSwap=getfftnum(round(nxEx*nxSwap));
    fmat suf2dSwap(ntSwap,nxSwap);
    suf2dSwap.fill(0.0);
    suf2dSwap(span(0,suf2d.n_rows-1),\
        span((nxSwap-nx)/2,(nxSwap-nx)/2+suf2d.n_cols-1))=
        suf2d(span::all,span::all);
    fmat scatterSwap(nz,nxSwap);
    scatterSwap.fill(0.0);
    scatterSwap(span::all,span((nxSwap-nx)/2,(nxSwap-nx)/2+suf2d.n_cols-1))
        =scatter(span::all,span::all);
    float dfSwap=1.0/dt/ntSwap;
    float dkx=2.0*3.1415926/dx/nxSwap;
    float v0=velWater;
    int nDf=fmax/dfSwap;
    nDf=min(nDf,ntSwap/2);
    cx_fmat suf2dcx=fft2(suf2dSwap)(span(0,nDf),span::all);
    suf2dSwap.clear();

    fcube mul3d(nt,nx,nthread);
    mul3d.fill(0.0);
    cx_fcube b(nxSwap/2,nDf,nDz);
    cx_float a;
    a.imag(1.0);a.real(0.0);
    b.fill(0.0);
    for(int jz=0;jz<nDz;jz++){
        float dep=(jz+1)*dz;
    for(int jf=0;jf<nDf;jf++){
        float wf=2.0*3.1415926*dfSwap*jf;
    for(int jk=0;jk<nxSwap/2;jk++){
        float kx=dkx*jk;
        float kz2=wf*wf/v0/v0-kx*kx;
        if(kz2>0.0){
            float kz=sqrt(kz2);
            b(jk,jf,jz)=a*float(-1.0*dep*kz);
            b(jk,jf,jz)=exp(b(jk,jf,jz));
        }
    }}}
    for(int jf=0;jf<nDf;jf++){
        float wf=2.0*3.1415926*dfSwap*jf;
        float wf2=wf*wf;
        suf2dcx.row(jf)=wf*a*suf2dcx.row(jf);
        //suf2dcx.row(jf)=wf2*suf2dcx.row(jf);
    }
omp_set_num_threads(nthread);
#pragma omp parallel for
for(int jn=0;jn<nthread;jn++){
    int zbeg=round(jn*((nDz+0.001)/nthread));
    int zend=round((jn+1)*((nDz+0.001)/nthread));
    for(int jz=zbeg;jz<zend;jz++){
        if(jz%10==0)cout<<jz<<"/"<<nDz<<endl;
        cx_fmat mul2dcx,suf2dcxBorn;
        mul2dcx.zeros(ntSwap,nxSwap);
        for(int jf=1;jf<nDf;jf++){
            mul2dcx(jf,0)=suf2dcx(jf,0)*b(0,jf,jz);
            float wf=2.0*3.1415926*dfSwap*jf;
            for(int jk=1;jk<nxSwap/2;jk++){
                float kx=dkx*jk;
                float kz=wf/v0-kx;
                if(kz>0.0){
                    mul2dcx(jf,jk)=suf2dcx(jf,jk)*b(jk,jf,jz);
                    mul2dcx(jf,nxSwap-jk)=suf2dcx(jf,nxSwap-jk)*b(jk,jf,jz);
                }
            }
            mul2dcx.row(jf)=ifft(mul2dcx.row(jf));
        }
        for(int jx=0;jx<nxSwap;jx++){
            mul2dcx.col(jx)=mul2dcx.col(jx)*scatterSwap(freeSufaceZ+jz,jx);
        }
        suf2dcxBorn.zeros(nDf,suf2dcx.n_cols);
        for(int jf=1;jf<nDf;jf++){
            suf2dcxBorn.row(jf)=fft(mul2dcx.row(jf));
        }
        mul2dcx.fill(0.0);
        for(int jf=1;jf<nDf;jf++){
            mul2dcx(jf,0)+=suf2dcxBorn(jf,0)*b(0,jf,jz);
            float wf=2.0*3.1415926*dfSwap*jf;
            for(int jk=1;jk<nxSwap/2;jk++){
                float kx=dkx*jk;
                float kz=wf/v0-kx;
                if(kz>0.0){
                    mul2dcx(jf,jk)+=suf2dcxBorn(jf,jk)*b(jk,jf,jz);
                    mul2dcx(jf,nxSwap-jk)+=suf2dcxBorn(jf,nxSwap-jk)*b(jk,jf,jz);
                }
            }
        }
        for(int jf=1;jf<nDf;jf++){
            suf2dcxBorn.row(jf)=ifft(mul2dcx.row(jf));
        }
        cx_fmat swap2dcx;
        swap2dcx.zeros(ntSwap,suf2d.n_cols);
        swap2dcx(span(0,nDf-1),span::all)=suf2dcxBorn(span::all,\
            span((nxSwap-nx)/2,(nxSwap-nx)/2+suf2d.n_cols-1));
        suf2dcxBorn.clear();
        fmat swap2d(ntSwap,suf2d.n_cols);
        for(int jf=0;jf<suf2d.n_cols;jf++){
            swap2d.col(jf)=real(ifft(swap2dcx.col(jf)));
        }
        mul3d.slice(jn)+=(swap2d(span(0,nt-1),span::all));
    }
}
    fmat bornMul2d=suf2d;
    bornMul2d.fill(0.0);
    for(int jn=0;jn<nthread;jn++){
        bornMul2d+=mul3d.slice(jn);
    }
    bornMul2d=get_blackman_leftwin2d(bornMul2d,50.0);
    bornMul2d=get_blackman_rightwin2d(bornMul2d,50.0);
    bornMul2d=get_blackman_downwin2d(bornMul2d,10.0);

    cx_fmat cxBornMul2d=fft2(bornMul2d);
    float dfOrig=1.0/dt/nt;
    float dkxOrig=2.0*3.1415926/dx/nx;
    for(int jf=1;jf<nt;jf++){
        if(jf<nt/2){
            float wf=2.0*3.1415926*dfOrig*jf;
            for(int jk=0;jk<nx/2;jk++){
                float kx=dkxOrig*jk;
                float kf2=wf*wf/v0/v0;
                float kz2=wf*wf/v0/v0-kx*kx;
                if(kz2>0.08*kf2){
                    cxBornMul2d(jf,jk)=wf*wf*cxBornMul2d(jf,jk)/kz2;
                    cxBornMul2d(jf,nx-jk-1)=wf*wf*cxBornMul2d(jf,nx-jk-1)/kz2;
                }
            }
        }else{cxBornMul2d.row(jf).fill(0.0);}
    }
    bornMul2d=real(ifft2(cxBornMul2d));
    return bornMul2d;
}
fmat CRMD2d::SrmmBornOneShot(fmat suf2d, int nSmoothVp=1, \
    int expandModelUp=0, bool sourceModeling=false)
{
    int nthread(this->ncpu),isPMLSurface(1);
    int autoUpExpand=max(int(70-izFreeSurface),0);
    expandModelUp=max(autoUpExpand,expandModelUp);
    int nzEx(this->nz+expandModelUp);
    //freeSuface, source and obn Z-coord in model
    int freeSufaceZ(izFreeSurface),sz(izFreeSurface),\
        rz(izFreeSurface);
    fmat modelvp(nz,nx),modelvs(nz,nx),modelrho(nz,nx);
    modelvp=vp2d;modelvs=vs2d;modelrho=rho2d;
    for(int k1=0;k1<freeSufaceZ;k1++){
        modelvp.row(k1)=modelvp.row(freeSufaceZ);
        modelvs.row(k1)=modelvs.row(freeSufaceZ);
        modelrho.row(k1)=modelrho.row(freeSufaceZ);
    }
    modelvp=model2dExpandUp(modelvp,expandModelUp);
    modelvs=model2dExpandUp(modelvs,expandModelUp);
    modelrho=model2dExpandUp(modelrho,expandModelUp);
    freeSufaceZ+=expandModelUp;
    sz+=expandModelUp;
    rz+=expandModelUp;
    fmat scatter(nzEx,nx),backrho(nzEx,nx),backvp(nzEx,nx);
    backrho=fmatsmooth(modelrho,nzEx,nx,this->nSmooth);
    backvp=fmatsmooth(modelvp,nzEx,nx,this->nSmooth);
    scatter.fill(0.0);
    for(int k1=0;k1<nzEx;k1++){
    for(int k2=0;k2<nx;k2++){
        scatter(k1,k2)=2.0*((modelvp(k1,k2)\
            -backvp(k1,k2))/backvp(k1,k2)\
            +(modelrho(k1,k2)-backrho(k1,k2))\
            /backrho(k1,k2));
        //scatter(k1,k2)=(1.0/modelvp(k1,k2)/modelvp(k1,k2)\
            -1.0/backvp(k1,k2)/backvp(k1,k2))\
            /(1.0/backvp(k1,k2)/backvp(k1,k2));
    }}
    //datawrite(scatter,"scatter.dat");
    backvp=fmatsmooth(modelvp,nzEx,nx,nSmoothVp);
    elastic3D_ARMA backGround;
    elastic3D_ARMA scatterField;
    elastic2dModelModify(backGround, \
        backvp, modelvs, backrho, nx, ny, nzEx,\
        dx, dy, dz, dt, freeSufaceZ, \
        isPMLSurface, nthread);
    elastic2dModelModify(scatterField, \
        backvp, modelvs, backrho, nx, ny, nzEx,\
        dx, dy, dz, dt, freeSufaceZ, \
        isPMLSurface, nthread);
/////////////////////////////////////
    fmat bornMul2d=Born2dSRME(\
        backGround, suf2d, scatterField, scatter,\
        freeSufaceZ, nthread, sourceModeling);
    return bornMul2d;
}
fmat CRMD2d::SrmmBornOneShot(fmat suf2d, fmat scatterOrig, \
    int nSmoothVp=1, int expandModelUp=0, bool sourceModeling=true)
{
    int nthread(this->ncpu),isPMLSurface(1);
    int autoUpExpand=max(int(70-izFreeSurface),0);
    expandModelUp=max(autoUpExpand,expandModelUp);
    int nzEx(this->nz+expandModelUp);
    //freeSuface, source and obn Z-coord in model
    int freeSufaceZ(izFreeSurface),sz(izFreeSurface),\
        rz(izFreeSurface);
    fmat modelvp(nz,nx),modelvs(nz,nx),modelrho(nz,nx),\
        scatter=scatterOrig;
    modelvp=vp2d;modelvs=vs2d;modelrho=rho2d;
    for(int k1=0;k1<freeSufaceZ;k1++){
        modelvp.row(k1)=modelvp.row(freeSufaceZ);
        modelvs.row(k1)=modelvs.row(freeSufaceZ);
        modelrho.row(k1)=modelrho.row(freeSufaceZ);
    }
    scatter=model2dExpandUp(scatter,expandModelUp);
    modelvp=model2dExpandUp(modelvp,expandModelUp);
    modelvs=model2dExpandUp(modelvs,expandModelUp);
    modelrho=model2dExpandUp(modelrho,expandModelUp);
    freeSufaceZ+=expandModelUp;
    sz+=expandModelUp;
    rz+=expandModelUp;
    fmat backrho(nzEx,nx),backvp(nzEx,nx);
    backrho=fmatsmooth(modelrho,nzEx,nx,this->nSmooth);
    backvp=fmatsmooth(modelvp,nzEx,nx,this->nSmooth);

    //datawrite(scatter,"scatter.dat");
    backvp=fmatsmooth(modelvp,nzEx,nx,nSmoothVp);
    elastic3D_ARMA backGround;
    elastic3D_ARMA scatterField;
    elastic2dModelModify(backGround, \
        backvp, modelvs, backrho, nx, ny, nzEx,\
        dx, dy, dz, dt, freeSufaceZ, \
        isPMLSurface, nthread);
    elastic2dModelModify(scatterField, \
        backvp, modelvs, backrho, nx, ny, nzEx,\
        dx, dy, dz, dt, freeSufaceZ, \
        isPMLSurface, nthread);
/////////////////////////////////////
    fmat bornMul2d=Born2dSRME(\
        backGround, suf2d, scatterField, scatter,\
        freeSufaceZ, nthread, sourceModeling);
    return bornMul2d;
}
void CRMD2d::ISSInterMulSSFBornMultiShot(\
    char const *fileout, char const *filein, \
    fmat scatterUp, fmat scatterDown, float cutSlopeVelMin, \
    float cutSlopeVelMax, int ntCutWin, float cutDelayTime, \
    int nProcessThread, int nWinLeft, int nWinRight)
{
    cout<<"Now is running: ISS SSF-Born of Multi-Shot."<<endl;
omp_set_num_threads(nProcessThread);
#pragma omp parallel for
    for(int ksx=sxBeg;ksx<=sxEnd;ksx+=sxGap){
        cout<<ksx<<endl;
        fmat suf2d=this->ReadOneGatherData(filein,ksx);
        for(int i=0;i<ksx-nWinLeft;i++){
            suf2d.col(i).fill(0.0);
        }
        for(int i=max(0,ksx-nWinLeft);i<ksx;i++){
            suf2d.col(i)=suf2d.col(i)*Blackman(nWinLeft-(ksx-i),nWinLeft);
        }
        for(int i=ksx;i<min(int(suf2d.n_cols-1),ksx+nWinRight);i++){
            suf2d.col(i)=suf2d.col(i)*Blackman(nWinRight-(i-ksx),nWinRight);
        }
        for(int i=ksx+nWinRight;i<suf2d.n_cols;i++){
            suf2d.col(i).fill(0.0);
        }
        fmat mul2d=this->ISSInterMulSSFBornOneShotTwoLayer(\
            suf2d, scatterUp, scatterDown,\
            cutSlopeVelMin, cutSlopeVelMax, ntCutWin, \
            cutDelayTime, ksx);
        this->WriteOneGatherData(mul2d,fileout,ksx);
    }
}
fmat CRMD2d::ISSInterMulSSFBornOneShotTwoLayer(\
    fmat suf2d, fmat scatterUp, fmat scatterDown,\
    float cutSlopeVelMin, float cutSlopeVelMax, int ntCutWin, \
    float cutDelayTime, int sxid)
{
    SSF_BORN_2d ssfObj;
    ssfObj.getParams(this->nz,this->nx,this->nt*2,\
        this->dz,this->dx,this->dt,this->fmax);
    ssfObj.nthread=this->ncpu;
    ssfObj.izBeg=this->izFreeSurface;
    ssfObj.izEnd=this->nz-1;
    ssfObj.vp2d=this->vp2d;
    ssfObj.wfPow=1.0;

    fmat csCoordx, csCoordy;
    this->NewDataMatInfo(csCoordx,csCoordy,sxid);
    fcube data3dMultiple(suf2d.n_cols,1,suf2d.n_rows);
    suf2d=suf2d.st();
    data3dMultiple.col(0)=suf2d(span::all,span(0,suf2d.n_cols-1));
    cutDataBySlope(data3dMultiple,csCoordx,csCoordy,\
        this->dt, cutSlopeVelMin, cutSlopeVelMax, cutDelayTime);
    cutDataBySlopeWin(data3dMultiple,csCoordx,csCoordy,\
        this->dt, cutSlopeVelMax, cutDelayTime, ntCutWin);
    suf2d=data3dMultiple.col(0);
    suf2d=suf2d.st();

    suf2d=GetReverRowFmat(suf2d);
    ssfObj.inpData2d(span(0,this->nt-1),span::all)=suf2d;
    for(int iz=0;iz<this->izFreeSurface;iz++){
        ssfObj.vp2d.row(iz)=this->vp2d.row(this->izFreeSurface);
    }
    ssfObj.vp2d=fmatsmooth(ssfObj.vp2d,this->nz,this->nx,this->nSmooth);

    ssfObj.scatterDown2d=scatterUp;
    ssfObj.scatterUp2d.fill(0.0);
    for(int iz=(scatterUp.n_rows-52);iz>this->izFreeSurface;iz--){
        if(accu(abs(scatterUp.row(iz)))>0.00001){
            ssfObj.izEnd=iz+1;
            cout<<"izEnd up: "<<ssfObj.izEnd<<endl;
            break;
        }
    }
    ssfObj.ssf2dUpToDown();
    ssfObj.ssf2dDownToUp();
    //ssfObj.ffd2dTwoOrderUpToDown(sxid*this->dx);
    //ssfObj.ffd2dTwoOrderDownToUp(sxid*this->dx);
    ssfObj.outData2d=GetReverRowFmat(ssfObj.outData2d);
    //this->WriteOneGatherData(ssfObj.outData2d,this->fileOutSwap.c_str(),sxid);

    suf2d=ssfObj.outData2d(span(this->nt,2*this->nt-1),span::all);
    suf2d=suf2d.st();
    data3dMultiple.col(0)=suf2d(span::all,span(0,suf2d.n_cols-1));
    cutDataBySlope(data3dMultiple,csCoordx,csCoordy,\
        this->dt, cutSlopeVelMin, cutSlopeVelMax, cutDelayTime);
    cutDataBySlopeWin(data3dMultiple,csCoordx,csCoordy,\
        this->dt, cutSlopeVelMax, cutDelayTime, ntCutWin);
    suf2d=data3dMultiple.col(0);
    suf2d=suf2d.st();
    data3dMultiple.clear();

    ssfObj.getParams(this->nz,this->nx,this->nt,\
        this->dz,this->dx,this->dt,this->fmax);
    ssfObj.inpData2d=suf2d;
    ssfObj.upKzKxKf3d.fill(0.0);
    ssfObj.scatterDown2d=scatterDown;
    ssfObj.scatterUp2d.fill(0.0);
    for(int iz=(scatterDown.n_rows-52);iz>this->izFreeSurface;iz--){
        if(accu(abs(scatterDown.row(iz)))>0.00001){
            ssfObj.izEnd=iz+1;
            cout<<"izEnd down: "<<ssfObj.izEnd<<endl;
            break;
        }
    }
    ssfObj.ssf2dUpToDown();
    ssfObj.ssf2dDownToUp();
    //ssfObj.ffd2dTwoOrderUpToDown(sxid*this->dx);
    //ssfObj.ffd2dTwoOrderDownToUp(sxid*this->dx);
    return ssfObj.outData2d;
}
void CRMD2d::ISSInterMulBornMultiShot(\
    char const *fileout, char const *filein, \
    fmat scatterUp, fmat scatterDown, int nWinLeft, int nWinRight, \
    float cutSlopeVelMin, float cutSlopeVelMax, int ntCutWin, float cutDelayTime, \
    int nSmoothVpBorn, int expandModelUp=0, int nthread=1)
{
    cout<<"Now is running: ISS Born of Multi-Shot."<<endl;
omp_set_num_threads(nthread);
#pragma omp parallel for
    for(int ksx=sxBeg;ksx<=sxEnd;ksx+=sxGap){
        cout<<ksx<<endl;
        fmat suf2d=this->ReadOneGatherData(filein,ksx);
        fmat suf2dSwap=this->ISSInterMulBornOneShotTwoLayerUp(\
            suf2d, scatterUp,  ksx, nSmoothVpBorn, \
            expandModelUp,nWinLeft, nWinRight);
        this->WriteOneGatherData(suf2dSwap,"./4.swap/iss.swap.dat",ksx);

        fmat cut2d,mul2d;
        mul2d=this->ISSInterMulBornOneShotTwoLayerDown(\
            suf2dSwap, scatterDown, cut2d, cutDelayTime, ksx, \
            nSmoothVpBorn, expandModelUp, cutSlopeVelMin, cutSlopeVelMax,ntCutWin);
        this->WriteOneGatherData(cut2d,"./4.swap/iss.cut.dat",ksx);
        this->WriteOneGatherData(mul2d,fileout,ksx);
    }
}
fmat CRMD2d::ISSInterMulBornOneShotTwoLayerUp(\
    fmat suf2d, fmat scatterUp, \
    int sxid, int nSmoothVp, int expandModelUp,\
    int nWinLeft, int nWinRight)
{
    //suf2d=get_blackman_leftwin2d(suf2d,nWinLeft);
    //suf2d=get_blackman_rightwin2d(suf2d,nWinRight);
    for(int i=0;i<sxid-nWinLeft;i++){
        suf2d.col(i).fill(0.0);
    }
    for(int i=max(0,sxid-nWinLeft);i<sxid;i++){
        suf2d.col(i)=suf2d.col(i)*Blackman(nWinLeft-(sxid-i),nWinLeft);
    }
    for(int i=sxid;i<min(int(suf2d.n_cols-1),sxid+nWinRight);i++){
        suf2d.col(i)=suf2d.col(i)*Blackman(nWinRight-(i-sxid),nWinRight);
    }
    for(int i=sxid+nWinRight;i<suf2d.n_cols;i++){
        suf2d.col(i).fill(0.0);
    }
    fmat suf2dInvTime(suf2d.n_rows*2,suf2d.n_cols);
    suf2dInvTime.fill(0.0);
    for(int i=0;i<suf2d.n_rows;i++){
        suf2dInvTime.row(i)=suf2d.row(suf2d.n_rows-1-i);
    }

    int nthread(this->ncpu),isPMLSurface(1);
    int autoUpExpand=max(int(70-izFreeSurface),0);
    expandModelUp=max(autoUpExpand,expandModelUp);
    int nzEx(this->nz+expandModelUp);
    //freeSuface, source and obn Z-coord in model
    int freeSufaceZ(izFreeSurface),sz(izFreeSurface),\
        rz(izFreeSurface);
    fmat modelvp(nz,nx),modelvs(nz,nx),modelrho(nz,nx);
    modelvp=vp2d;modelvs=vs2d;modelrho=rho2d;
    for(int k1=0;k1<freeSufaceZ;k1++){
        modelvp.row(k1).fill(modelvp(freeSufaceZ,0));
        modelvs.row(k1).fill(modelvs(freeSufaceZ,0));
        modelrho.row(k1).fill(modelrho(freeSufaceZ,0));
    }
    modelvp=model2dExpandUp(modelvp,expandModelUp);
    modelvs=model2dExpandUp(modelvs,expandModelUp);
    modelrho=model2dExpandUp(modelrho,expandModelUp);
    scatterUp.row(0).fill(0.0);
    scatterUp=model2dExpandUp(scatterUp,expandModelUp);
    freeSufaceZ+=expandModelUp;
    sz+=expandModelUp;
    rz+=expandModelUp;
    fmat backrho(nzEx,nx),backvp(nzEx,nx);

    backvp=fmatsmooth(modelvp,nzEx,nx,nSmoothVp);
    datawrite(backvp,"./movie/backvp.dat");
    elastic3D_ARMA backGround;
    elastic3D_ARMA scatterField;
    elastic2dModelModify(backGround, \
        backvp, modelvs, backrho, nx, ny, nzEx,\
        dx, dy, dz, dt, freeSufaceZ, \
        isPMLSurface, nthread);
    elastic2dModelModify(scatterField, \
        backvp, modelvs, backrho, nx, ny, nzEx,\
        dx, dy, dz, dt, freeSufaceZ, \
        isPMLSurface, nthread);
    suf2d=Born2dSRME(\
        backGround, suf2dInvTime, scatterField, scatterUp,\
        freeSufaceZ, nthread);

    suf2dInvTime.fill(0.0);
    for(int i=0;i<suf2d.n_rows;i++){
        suf2dInvTime.row(i)=suf2d.row(suf2d.n_rows-1-i);
    }
    return suf2dInvTime;
}

fmat CRMD2d::ISSInterMulBornOneShotTwoLayerDown(\
    fmat suf2d, fmat scatterDown, fmat& suf2dCut, float cutDelayTime, \
    int sxid, int nSmoothVp, int expandModelUp,\
    float cutSlopeVelMin, float cutSlopeVelMax, int ntCutWin)
{
    fmat csCoordx, csCoordy;
    fcube data3dMultiple(suf2d.n_cols,1,suf2d.n_rows/2);
    suf2d=suf2d.st();
    data3dMultiple.col(0)=suf2d(span::all,span(suf2d.n_cols/2,suf2d.n_cols-1));
    this->NewDataMatInfo(csCoordx,csCoordy,sxid);
    cutDataBySlope(data3dMultiple,csCoordx,csCoordy,\
        this->dt, cutSlopeVelMin, cutSlopeVelMax, cutDelayTime);
    cutDataBySlopeWin(data3dMultiple,csCoordx,csCoordy,\
        this->dt, cutSlopeVelMax, cutDelayTime, ntCutWin);
    suf2d=data3dMultiple.col(0);
    suf2d=suf2d.st();
    suf2dCut=suf2d;

    int nthread(this->ncpu),isPMLSurface(1);
    int autoUpExpand=max(int(70-izFreeSurface),0);
    expandModelUp=max(autoUpExpand,expandModelUp);
    int nzEx(this->nz+expandModelUp);
    //freeSuface, source and obn Z-coord in model
    int freeSufaceZ(izFreeSurface),sz(izFreeSurface),\
        rz(izFreeSurface);
    fmat modelvp(nz,nx),modelvs(nz,nx),modelrho(nz,nx);
    modelvp=vp2d;modelvs=vs2d;modelrho=rho2d;
    for(int k1=0;k1<freeSufaceZ;k1++){
        modelvp.row(k1).fill(modelvp(freeSufaceZ,0));
        modelvs.row(k1).fill(modelvs(freeSufaceZ,0));
        modelrho.row(k1).fill(modelrho(freeSufaceZ,0));
    }
    modelvp=model2dExpandUp(modelvp,expandModelUp);
    modelvs=model2dExpandUp(modelvs,expandModelUp);
    modelrho=model2dExpandUp(modelrho,expandModelUp);
    scatterDown.row(0).fill(0.0);
    scatterDown=model2dExpandUp(scatterDown,expandModelUp);
    freeSufaceZ+=expandModelUp;
    sz+=expandModelUp;
    rz+=expandModelUp;
    fmat backrho(nzEx,nx),backvp(nzEx,nx);

    backvp=fmatsmooth(modelvp,nzEx,nx,nSmoothVp);
    elastic3D_ARMA backGround;
    elastic3D_ARMA scatterField;
    elastic2dModelModify(backGround, \
        backvp, modelvs, backrho, nx, ny, nzEx,\
        dx, dy, dz, dt, freeSufaceZ, \
        isPMLSurface, nthread);
    elastic2dModelModify(scatterField, \
        backvp, modelvs, backrho, nx, ny, nzEx,\
        dx, dy, dz, dt, freeSufaceZ, \
        isPMLSurface, nthread);
    fmat mul2d=Born2dSRME(\
        backGround, suf2d, scatterField, scatterDown,\
        freeSufaceZ, nthread);

    return mul2d;
}

fmat CRMD2d::InterMulBornOneShot(fmat& sufPri2d, \
    fmat scatterPriDown, fmat scatterUp, fmat scatterDown, \
    int sxid, int nSmoothVp, int expandModelUp, float f0)
{
    int nthread(this->ncpu),isPMLSurface(1);
    int autoUpExpand=max(int(70-izFreeSurface),0);
    expandModelUp=max(autoUpExpand,expandModelUp);
    int nzEx(this->nz+expandModelUp);
    //freeSuface, source and obn Z-coord in model
    int freeSufaceZ(izFreeSurface),sz(izFreeSurface),\
        rz(izFreeSurface);
    fmat modelvp(nz,nx),modelvs(nz,nx),modelrho(nz,nx);
    modelvp=vp2d;modelvs=vs2d;modelrho=rho2d;
    for(int k1=0;k1<freeSufaceZ;k1++){
        modelvp.row(k1).fill(modelvp(freeSufaceZ,0));
        modelvs.row(k1).fill(modelvs(freeSufaceZ,0));
        modelrho.row(k1).fill(modelrho(freeSufaceZ,0));
    }
    modelvp=model2dExpandUp(modelvp,expandModelUp);
    modelvs=model2dExpandUp(modelvs,expandModelUp);
    modelrho=model2dExpandUp(modelrho,expandModelUp);
    scatterUp.row(0).fill(0.0);
    scatterDown.row(0).fill(0.0);
    scatterPriDown.row(0).fill(0.0);
    scatterUp=model2dExpandUp(scatterUp,expandModelUp);
    datawrite(scatterUp,"./movie/scatterUp.dat");
    scatterDown=model2dExpandUp(scatterDown,expandModelUp);
    datawrite(scatterDown,"./movie/scatterDown.dat");
    scatterPriDown=model2dExpandUp(scatterPriDown,expandModelUp);
    datawrite(scatterPriDown,"./movie/scatterPriDown.dat");
    freeSufaceZ+=expandModelUp;
    sz+=expandModelUp;
    rz+=expandModelUp;
    fmat backrho(nzEx,nx),backvp(nzEx,nx);
    backvp=fmatsmooth(modelvp,nzEx,nx,nSmoothVp);
    datawrite(backvp,"./movie/backvp.dat");

    elastic3D_ARMA sourceBackGround;
    elastic3D_ARMA primaryDown;
    elastic3D_ARMA scatterUpField;
    elastic3D_ARMA scatterDownField;
    elastic2dModelModify(sourceBackGround, \
        backvp, modelvs, backrho, nx, ny, nzEx,\
        dx, dy, dz, dt, freeSufaceZ, \
        isPMLSurface, nthread);
    elastic2dModelModify(primaryDown, \
        backvp, modelvs, backrho, nx, ny, nzEx,\
        dx, dy, dz, dt, freeSufaceZ, \
        isPMLSurface, nthread);
    elastic2dModelModify(scatterUpField, \
        backvp, modelvs, backrho, nx, ny, nzEx,\
        dx, dy, dz, dt, freeSufaceZ, \
        isPMLSurface, nthread);
    elastic2dModelModify(scatterDownField, \
        backvp, modelvs, backrho, nx, ny, nzEx,\
        dx, dy, dz, dt, freeSufaceZ, \
        isPMLSurface, nthread);

    char fileMovie[1024];
    fileMovie[0]='\0';
    //strcat(fileMovie,"./movie/movie");
    elastic3D_ARMA** pfield;
    pfield=new elastic3D_ARMA*[4];
    pfield[0]=&sourceBackGround;
    pfield[1]=&primaryDown;
    pfield[2]=&scatterDownField;
    pfield[3]=&scatterUpField;
    fmat suftzz(nt,nx);
    suftzz.fill(0.0);
{
    int (*fptr[4])(class elastic3D_ARMA&);
    if(nthread<=1){
        fptr[0]=TimeSliceCal_acoustic2D_OneThread;
        fptr[1]=TimeSliceCal_acoustic2D_OneThread;
        fptr[2]=TimeSliceCal_acoustic2D_OneThread;
        fptr[3]=TimeSliceCal_acoustic2D_OneThread;
    }else{
        fptr[0]=TimeSliceCal_acoustic2D_MultiThread;
        fptr[1]=TimeSliceCal_acoustic2D_MultiThread;
        fptr[2]=TimeSliceCal_acoustic2D_MultiThread;
        fptr[3]=TimeSliceCal_acoustic2D_MultiThread;
    }
    int *recz;
    recz=new int[nx];
    for(int i=0;i<nx;i++){
        recz[i]=rz;
    }
    scatterUp=scatterUp.st();
    scatterDown=scatterDown.st();
    scatterPriDown=scatterPriDown.st();
////////////////////////Multiple Modeling/////////////////////////
    sufPri2d=suftzz;
    for(int k=0;k<nt;k++)
    {
        sourceBackGround.tzz(sxid,0,sz)\
            +=1e6*wavelet01(k,dt,f0);
        primaryDown.tzz.col(0)+=fmatmul\
            (scatterPriDown,sourceBackGround.tzz.col(0),nthread);
        //Interbed Multiple:
        //scatterDownField.tzz.col(0)+=fmatmul\
            (scatterDown,sourceBackGround.tzz.col(0),nthread);
        scatterDownField.tzz.col(0)+=fmatmul\
            (scatterDown,scatterUpField.tzz.col(0),nthread);
        scatterUpField.tzz.col(0)+=fmatmul\
            (scatterUp,primaryDown.tzz.col(0),nthread);
        scatterUpField.tzz.col(0)+=fmatmul\
            (scatterUp,scatterDownField.tzz.col(0),nthread);

    //omp_set_num_threads(4);
    //#pragma omp parallel for
        for(int kp=0;kp<4;kp++){
            fptr[kp](pfield[kp][0]);
        }

        for(int i=0;i<nx;i++){
            suftzz(k,i)+=scatterDownField.tzz(i,0,recz[i]);
            sufPri2d(k,i)+=primaryDown.tzz(i,0,recz[i]);
        }
        if(k%1000==0)
        {std::cout<<"now is running : "<<k<<endl;}
        if(fileMovie[0]!='\0' && k%200==0){
            char str[1024];
            fmat swap2d=scatterUpField.tzz.col(0);
            swap2d=swap2d.st();
            str[0]='\0';
            strcat(str,fileMovie);
            strcat(str,".up.");
            strcat(str,numtostr(k,8));
            datawrite(swap2d,str);

            swap2d=scatterDownField.tzz.col(0);
            swap2d=swap2d.st();
            str[0]='\0';
            strcat(str,fileMovie);
            strcat(str,".down.");
            strcat(str,numtostr(k,8));
            datawrite(swap2d,str);

            swap2d=sourceBackGround.tzz.col(0);
            swap2d=swap2d.st();
            str[0]='\0';
            strcat(str,fileMovie);
            strcat(str,".back.");
            strcat(str,numtostr(k,8));
            datawrite(swap2d,str);

            swap2d=primaryDown.tzz.col(0);
            swap2d=swap2d.st();
            str[0]='\0';
            strcat(str,fileMovie);
            strcat(str,".pri.");
            strcat(str,numtostr(k,8));
            datawrite(swap2d,str);
        }
    }
    std::cout<<"finished"<<endl;
    delete [] recz;
    recz=nullptr;
}
    delete [] pfield;
    pfield=nullptr;

    return suftzz;
}
fmat CRMD2d::GetWaterBottemPrimaryBornOneShot(fmat scatter2d,\
    int sxid,float dxIn,float dzIn,float delayTime)
{
    float pi=3.1415926;
    float df=1.0/dt/nt;
    float dtn=dt*0.1;
    float maxTime=min(float(dt*nt),8.0f);
    scatter2d=scatter2d*1000.0;
    cx_fmat timeBase=getTimeCodeingBase(\
        maxTime, df, 1, fmax/df, ncpu,dtn);
    fmat priBorn(nt,scatter2d.n_cols),coordxIn(scatter2d.n_cols,1);
    for(int ix=0;ix<scatter2d.n_cols;ix++){
        coordxIn(ix,0)=dxIn*ix-sxid*dx;
    }
    cx_fmat priBornFX(nt,scatter2d.n_cols);
    int nf=fmax/df;
    int nf1=1;
    int ntMax=timeBase.n_rows-1;
    priBornFX.fill(0.0);
omp_set_num_threads(ncpu);
#pragma omp parallel for
    for(int jf=nf1;jf<nf;jf++){
        cout<<"now is running : "<<jf<<"/"<<nf<<endl;
        float w=2.0*pi*df*jf;
        float w2=1.0;
        //if(jf>nf/2)
        w2=Blackman(jf+nf,nf);
        for(int iz=1;iz<scatter2d.n_rows;iz++){
            float dep=iz*dzIn;
        for(int ix=0;ix<scatter2d.n_cols;ix++){
            float offset1=abs(coordxIn(ix,0));
            float len1=sqrt(dep*dep+offset1*offset1);
            float w3=w2*scatter2d(iz,ix)/len1/pi/4.0;
        for(int ix2=0;ix2<scatter2d.n_cols;ix2++){
            float offset2=abs(coordxIn(ix2,0)-coordxIn(ix,0));
            float len2=sqrt(dep*dep+offset2*offset2);
            float len=len1+len2;
            float travel=1.0/len2/pi/4.0;
            float t=delayTime+len/velWater;
            //cx_float a;
            //a.real(0.0);
            //a.imag(-t*w);
            int kt=round(t/dtn);
            priBornFX(jf,ix2)+=w3*travel\
                *timeBase(min(kt,ntMax),jf-nf1);
        }}}
        priBornFX.row(jf)=w*w*priBornFX.row(jf);
    }
    for(int ix=0;ix<scatter2d.n_cols;ix++){
        priBorn.col(ix)=real(ifft(priBornFX.col(ix)));
    }
    return priBorn; 
}

fmat CRMD2d::GetWaterBottemPrimaryRayTimeOneShot(int sxid, \
    float delayTime=0.05, int wide=10)
{
    fmat timeWaterBottem(nt,nx);
    timeWaterBottem.fill(0.0);
    for(int j1=0;j1<nx;j1++){
        float tMin=1e30;
        for(int j2=0;j2<nx;j2++){
            float dep2=layerDepth(j2)*layerDepth(j2);
            float offsetSx2=abs(j2-sxid)*dx;
            float offsetRx2=abs(j2-j1)*dx;
            offsetSx2=offsetSx2*offsetSx2;
            offsetRx2=offsetRx2*offsetRx2;
            float rayLenth=sqrt(dep2+offsetSx2)+sqrt(dep2+offsetRx2);
            float t0=rayLenth/velWater;
            if(t0<tMin){tMin=t0;}
        }
        int jt=round((tMin+delayTime)/dt);
        if(jt<nt-wide && jt>wide){
            for(int j=jt-wide;j<jt+wide;j++){
                timeWaterBottem(j,j1)=1.0;
            }
        }
    }
    return timeWaterBottem;
}
void CRMD2d::SrmmRayPathMultiShot(\
    char const *fileout, char const *filein,
    int nthread=1)
{
    float maxTime=max(float(dt*nt),8.0f);
    cx_fmat timeBase=getTimeCodeingBase(\
        maxTime, df, fmin/df, fmax/df, ncpu);
    cout<<"Now is running: SRMM Ray-Path of Multi-Shot."<<endl;
omp_set_num_threads(nthread);
#pragma omp parallel for
    for(int ksx=sxBeg;ksx<=sxEnd;ksx+=sxGap){
        cout<<ksx<<endl;
        fmat csCoordx, csCoordy;
        NewDataMatInfo(csCoordx,csCoordy,ksx);
        fmat suf2d=this->ReadOneGatherData(filein,ksx);
        fmat sufMul2d=this->SrmmRayPathOneShot(timeBase, suf2d,\
            csCoordx, csCoordy);
        this->WriteOneGatherData(sufMul2d,fileout,ksx);
    }
}
void CRMD2d::SrmmRayPathMultiShotMultiLayer(\
    char const *fileout, char const *filein,
    fcube layer3d_xyz, fvec imp,int nthread=1,\
    int leftExpand=0, int rightExpand=0)
{
    float maxTime=max(float(dt*nt),8.0f);
    cx_fmat timeBase=getTimeCodeingBase(\
        maxTime, df, fmin/df, fmax/df, ncpu);
    cout<<"Now is running: SRMM Ray-Path of Multi-Shot."<<endl;
omp_set_num_threads(nthread);
#pragma omp parallel for
    for(int ksx=sxBeg;ksx<=sxEnd;ksx+=sxGap){
        cout<<ksx<<endl;
        fmat csCoordx, csCoordy;
        NewDataMatInfo(csCoordx,csCoordy,ksx);
        fmat suf2d=this->ReadOneGatherData(filein,ksx);
        fmat suf2dex(this->nt,this->nx,fill::zeros);
        suf2dex(span::all,span(leftExpand,nx-rightExpand-1))\
            =suf2d(span::all,span(0,nx-leftExpand-rightExpand-1));
        this->layerDepth=layer3d_xyz.slice(0);
        fmat sufMul2d=imp(0)*this->SrmmRayPathOneShot(timeBase, suf2dex,\
            csCoordx, csCoordy);
        for(int k=1;k<layer3d_xyz.n_slices;k++){
            this->layerDepth=layer3d_xyz.slice(k);
            sufMul2d+=imp(k)*this->SrmmRayPathOneShot(timeBase, suf2dex,\
                csCoordx, csCoordy);
        }
        fmat mul2d=sufMul2d(span::all,span(leftExpand,nx-rightExpand-1));
        this->WriteOneGatherData(mul2d,fileout,ksx);
    }
}
fmat CRMD2d::SrmmRayPathOneShot(cx_fmat& timeBase, fmat data2d,\
    fmat csCoordx, fmat csCoordy, bool isOBNsys)
{
    cx_fcube data3dFx1(nx,ny,nt), data3dFx2(nx,ny,nt);
    fcube data3dOrig(nx,ny,nt),data3dMultiple(nx,ny,nt),\
        velLayerOrig3d(nx,ny,nz);
    fmat vrms2d(nx,ny), t0Double2d(nx,ny), vp2dswap(nz,nx);
    data3dOrig.col(0)=data2d.st();
    vp2dswap(span(0,nz-izFreeSurface-1),span::all)\
        =vp2d(span(izFreeSurface,nz-1),span::all);
    vp2dswap=fmatsmooth(vp2dswap,nz,nx,3);
    velLayerOrig3d.col(0)=vp2dswap.st();

    tx2fx3dCompress(data3dFx1,data3dOrig, fmax/df+1, ncpu);
    data3dOrig.zeros(1,1,1);

    float scoordx(0.0), scoordy(0.0), eccentricity(1.0), \
        maxAperture(10.0*nx*dx), \
        aModify(nx/2), bModify(nx/2), weightPow(1.5), \
        offsetMidMove(0.25), focusExpand(1.0), weightMax(0.99);
    if(isOBNsys){offsetMidMove=0.16;}
    
    getLayerVrms2d(vrms2d, t0Double2d, velLayerOrig3d, layerDepth, dz);
    multiplePrediction3dMWD(\
        data3dFx2, csCoordx, csCoordy, \
        data3dFx1, csCoordx, csCoordy, \
        vrms2d, t0Double2d,\
        layerDepth, timeBase, velWater, \
        scoordx, scoordy, eccentricity, maxAperture, \
        aModify, bModify, weightPow, \
        offsetMidMove, focusExpand, weightMax, \
        df, fmin/df, fmax/df, ncpu);

    data3dFx1.zeros(1,1,1);
    fx2tx_3d_thread(data3dMultiple,data3dFx2, ncpu);
    fmat mulRay2d=data3dMultiple.col(0);

    mulRay2d=mulRay2d.st();
/*    
    cx_fmat suf2dcx2(mulRay2d.n_rows,mulRay2d.n_cols);
    cx_fmat suf2dcx=fft2(mulRay2d);
    suf2dcx2.fill(0.0);
    float df=1/dt/nt;
    float pi=3.1415926;
    float dkx=1.0/dx/nx;
    cx_float a,b,c;
    a.real(0.0);
    for(int jf=0;jf<nt/2;jf++){
        float wf=2.0*pi*df*jf;
    for(int jk=0;jk<nx/2;jk++){
        float kx=2.0*pi*dkx*jk;
        float kz=wf*wf/1500/1500-kx*kx;
        if(kz>0.0){
            kz=sqrt(kz);
            a.imag(kz);
            suf2dcx2(jf,jk)=a*suf2dcx(jf,jk);
            suf2dcx2(jf,nx-jk-1)=a*suf2dcx(jf,nx-jk-1);
        }
    }
    }
    mulRay2d=real(ifft2(suf2dcx2))*2.0;*/
    return mulRay2d;
}

fmat CRMD2d::GetLayerPrimaryRayTimeCS(int sxid, float delayTime, int wide)
{
    fcube velLayerOrig3d(nx,ny,nz);
    velLayerOrig3d.fill(0.0);
    velLayerOrig3d.col(0)=vp2d.st();
    velLayerOrig3d(span::all,span::all,span(0,nz-1-izFreeSurface))\
        =velLayerOrig3d(span::all,span::all,span(izFreeSurface,nz-1));

    fmat vrms2d(nx,ny,fill::zeros),t0Double2d(nx,ny,fill::zeros),\
        weight(nx,ny);
    weight.fill(1.0);
    getLayerVrms2d(vrms2d, t0Double2d, velLayerOrig3d, layerDepth, dz);

    fmat travelTime(nx,ny),nullx(nx,ny),nully(nx,ny);
    fmat coordxin=GetCoordx(nx,dx);
    fmat coordyin=GetCoordy(ny,dy);
    getMultipleLayerMinimumTravelTimeMWD(\
        travelTime, nullx, nullx, \
        coordxin, coordyin, weight, \
        vrms2d, t0Double2d, layerDepth, \
        coordxin(sxid,0), coordyin(sxid,0), velWater);

    fmat timeWaterBottem(nt,nx);
    timeWaterBottem.fill(0.0);
    for(int j1=0;j1<nx;j1++){
        int jt=round((delayTime+travelTime(j1,0))/dt);
        if(jt<nt-wide && jt>wide){
            for(int j=jt-wide;j<jt+wide;j++){
                timeWaterBottem(j,j1)=1.0;
            }
        }
    }
    return timeWaterBottem;
}
fmat CRMD2d::GetLayerDepth(fvec dep, int nWide)
{
    fmat layerDepthMat(nz,nx);
    layerDepthMat.fill(0.0);
    for(int k2=0;k2<nx;k2++){
        int k1=round(dep(k2)/dz)+izFreeSurface;
        if(k1<nz-nWide && k1>nWide){
            for(int j=k1-nWide;j<k1+nWide;j++){
                layerDepthMat(j,k2)=1.0;
            }
        }
    }
    return layerDepthMat;
}
void CRMD2d::ISSRayPathMultiShot(\
    char const *fileout, char const *filein, \
    fmat layerUp, fmat layerDown,  int nthread=1)
{
    cout<<"Now is running: ISS Ray of Multi-Shot."<<endl;
    float maxTime=max(float(this->dt*this->nt),8.0f);
    this->nt=this->nt*2;this->df=1/this->dt/this->nt;
    cx_fmat timeBase=getTimeCodeingBase(\
        maxTime, this->df, this->fmin/this->df, this->fmax/this->df,this->ncpu);
    this->nt=this->nt/2;this->df=1/this->dt/this->nt;
    timeBase.row(0).fill(0.0);
omp_set_num_threads(nthread);
#pragma omp parallel for
    for(int ksx=sxBeg;ksx<=sxEnd;ksx+=sxGap){
        fmat csCoordx, csCoordy;
        this->NewDataMatInfo(csCoordx,csCoordy,ksx);
        fmat suf2d=this->ReadOneGatherData(filein,ksx);
        fmat interMul=this->IrmmRayPathOneShot(timeBase,suf2d,\
            csCoordx,csCoordy,layerUp,layerDown);
        this->WriteOneGatherData(interMul,fileout,ksx);
    }
}
fmat CRMD2d::IrmmRayPathOneShot(cx_fmat& timeBase, fmat data2d,\
    fmat csCoordx,fmat csCoordy,fmat layerDepthUp,fmat layerDepthDown)
{
    fmat data2d2t;
    data2d2t.zeros(2*nt,nx);
    data2d2t(span(nt,2*nt-1),span::all)=data2d(span::all,span::all);
    int ntIn=nt*2;
    float dfIn=1/dt/nt;
    cx_fcube data3dFx1(nx,ny,ntIn), data3dFx2(nx,ny,ntIn);
    fcube data3dOrig(nx,ny,ntIn),data3dMultiple(nx,ny,ntIn),\
        velLayerOrig3d(nx,ny,nz);
    fmat vrms2d(nx,ny), t0Double2d(nx,ny), vp2dswap(nz,nx);
    data3dOrig.col(0)=data2d2t.st();
    vp2dswap(span(0,nz-izFreeSurface-1),span::all)\
        =vp2d(span(izFreeSurface,nz-1),span::all);
    vp2dswap=fmatsmooth(vp2dswap,nz,nx,1);
    velLayerOrig3d.col(0)=vp2dswap.st();

    tx2fx3dCompress(data3dFx1,data3dOrig, fmax/dfIn+1, ncpu);
    data3dOrig.zeros(1,1,1);

    float scoordx(0.0), scoordy(0.0), eccentricity(1.0), \
        maxAperture(10.0*nx*dx), \
        aModify(nx), bModify(nx), weightPow(1.5), \
        offsetMidMove(0.25), focusExpand(1.0), weightMax(0.99);
    getLayerVrms2d(vrms2d, t0Double2d, velLayerOrig3d, layerDepthUp, dz);
    timeBase=conj(timeBase);
    multiplePrediction3dMWD(\
        data3dFx2, csCoordx, csCoordy, \
        data3dFx1, csCoordx, csCoordy, \
        vrms2d, t0Double2d,\
        layerDepthUp, timeBase, velWater, \
        scoordx, scoordy, eccentricity, maxAperture, \
        aModify, bModify, weightPow, \
        offsetMidMove, focusExpand, weightMax, \
        dfIn, fmin/dfIn, fmax/dfIn, ncpu);
    //cout<<"ok"<<endl;
    data3dMultiple.copy_size(data3dFx2);
    fx2tx_3d_thread(data3dMultiple,data3dFx2, ncpu);
    data2d2t=data3dMultiple.col(0);
    //datawrite(data2d2t=data2d2t.st(),"./4.swap/interbed.swap.dat");
    cutDataBySlope(data3dMultiple,csCoordx,csCoordy,\
        dt, vp2dswap(round(layerDepthUp(0,0)/dz)-2,0), \
        vp2dswap(round(layerDepthUp(0,0)/dz)-2,0)+900, dt*ntIn/2+0.10);
    cutDataBySlopeWin(data3dMultiple,csCoordx,csCoordy,\
        dt, vp2dswap(round(layerDepthUp(0,0)/dz)-2,0)+900, dt*ntIn/2+0.10, 200);
    data2d2t=data3dMultiple.col(0);
    //datawrite(data2d2t=data2d2t.st(),"./4.swap/interbed.swap.cut.dat");

    tx2fx3dCompress(data3dFx1,data3dMultiple, fmax/dfIn+1, ncpu);
    getLayerVrms2d(vrms2d, t0Double2d, velLayerOrig3d, layerDepthDown, dz);
    timeBase=conj(timeBase);
    multiplePrediction3dMWD(\
        data3dFx2, csCoordx, csCoordy, \
        data3dFx1, csCoordx, csCoordy, \
        vrms2d, t0Double2d,\
        layerDepthDown, timeBase, velWater, \
        scoordx, scoordy, eccentricity, maxAperture, \
        aModify, bModify, weightPow, \
        offsetMidMove, focusExpand, weightMax, \
        dfIn, fmin/dfIn, fmax/dfIn, ncpu);
    data3dFx1.zeros(1,1,1);
    fx2tx_3d_thread(data3dMultiple,data3dFx2, ncpu);
    fmat mulRay2d=data3dMultiple.col(0);
    mulRay2d=mulRay2d.st();
    
    data2d(span::all,span::all)=mulRay2d(span(nt,2*nt-1),span::all);
    return data2d;
}
void CRMD2d::NewDataMatInfo(fmat& csCoordx,fmat& csCoordy,int ksx)
{
    csCoordx.zeros(nx,ny);
    csCoordy.zeros(nx,ny);
    float sxSite=ksx*dx;
    for(int k=0;k<nx;k++){
        csCoordx(k,0)=k*dx-sxSite;
    }
}
void CRMD2d::ClearData()
{
    mul2d.fill(0.0);
}
fmat CRMD2d::ReadOneGatherData(char const *filename, int sxid)
{
    char fileName[1024];
    fmat data2d(nt,nx);
    if(strcmp(filename,"null")==0){
        data2d.fill(0.0);
    }else{
        fileName[0]='\0';
        strcat(fileName,filename);
        if(sxid<0){strcat(fileName,"-");sxid=(-sxid);}
        strcat(fileName,numtostr(sxid,nIndxNum));
        dataread(data2d,fileName);
    }
    return data2d;
}

void CRMD2d::WriteOneGatherData(fmat data2d, \
    char const *filename, int sxid)
{
    char file[1024];
    if(strcmp(filename,"null")==0){
        file[0]='\0';
        strcat(file,"cs.swap");
        if(sxid<0){strcat(file,"-");sxid=(-sxid);}
        strcat(file,numtostr(sxid,nIndxNum));
        datawrite(data2d,file);
    }else{
        file[0]='\0';
        strcat(file,filename);
        if(sxid<0){strcat(file,"-");sxid=(-sxid);}
        strcat(file,numtostr(sxid,nIndxNum));
        datawrite(data2d,file);
    }
}

/**************************************************/
fmat Born2dSRME(\
    elastic3D_ARMA& backGround,fmat data2d,\
    elastic3D_ARMA& scatterField, fmat scatter,\
    int izFreeSurface,int nthread, bool sourceModeling)
{
/*
    fmat suf2ddir=data2d;
    data2d.fill(0.0);
    data2d(span(14,data2d.n_rows-1),span::all)\
        =suf2ddir(span(0,data2d.n_rows-15),span::all);
*/
    char fileMovie[1024];
    fileMovie[0]='\0';
    //strcat(fileMovie,"./movie/movie.dat");

    int (*fptr)(class elastic3D_ARMA&);
    if(nthread<=1){
        fptr=TimeSliceCal_acoustic2D_OneThread;
    }else{
        fptr=TimeSliceCal_acoustic2D_MultiThread;
    }
    int nx=backGround.nx;
    int nt=data2d.n_rows;
    int *recz;
    recz=new int[nx];
    for(int i=0;i<nx;i++){
        recz[i]=izFreeSurface;
    }
    datawrite(scatter,"./movie/scatter.dat");
    scatter=scatter.st();
////////////////////////////////////////////////
    fmat suftzz(nt,nx);
    suftzz.fill(0.0);
    for(int k=0;k<nt;k++)
    {
        if(sourceModeling){
            for(int i=0;i<nx;i++){
                backGround.tzz(i,0,izFreeSurface)\
                    +=data2d(k,i);
                backGround.tzz(i,0,izFreeSurface-2)\
                    -=data2d(k,i);
            }
        }else{
            for(int i=0;i<nx;i++){
                backGround.tzz(i,0,izFreeSurface-1)\
                    =data2d(k,i);
            }
        }
        //Using internal parallelism
        scatterField.tzz.col(0)+=fmatmul\
            (scatter,backGround.tzz.col(0),nthread);
        (*fptr)(backGround);
        (*fptr)(scatterField);
        for(int i=0;i<nx;i++){
            suftzz(k,i)=scatterField.tzz(i,0,recz[i]);
        }
        if(k%1000==0)
        {std::cout<<"now is running : "<<k<<endl;}
        if(fileMovie[0]!='\0' && k%200==0){
            char str[1024];
            fmat swap2d=scatterField.tzz.col(0);
            swap2d=swap2d.st();
            str[0]='\0';
            strcat(str,fileMovie);
            strcat(str,".up.");
            strcat(str,numtostr(k,8));
            datawrite(swap2d,str);
            swap2d=backGround.tzz.col(0);
            swap2d=swap2d.st();
            str[0]='\0';
            strcat(str,fileMovie);
            strcat(str,".down.");
            strcat(str,numtostr(k,8));
            datawrite(swap2d,str);
        }
    }
    //for(int i=0;i<=seriesNum;i++){outf[i].close();}
    //datawrite3d_bycol_transpose(sufall,nt,nx,fileGather);
    std::cout<<"finished"<<endl;
    delete [] recz;
/*
    cx_fmat suf2dcx2(suftzz.n_rows,suftzz.n_cols);
    cx_fmat suf2dcx=fft2(suftzz);
    suf2dcx2.fill(0.0);
    float df=1/backGround.dt/nt;
    float pi=3.1415926;
    float dkx=1.0/backGround.dx/backGround.nx;
    cx_float a,b,c;
    a.real(0.0);
    for(int jf=0;jf<nt/2;jf++){
        float wf=2.0*pi*df*jf;
    for(int jk=0;jk<backGround.nx/2;jk++){
        float kx=2.0*pi*dkx*jk;
        float kz=wf*wf/1500/1500-kx*kx;
        if(kz>0.0){
            kz=sqrt(kz);
            a.imag(kz);
            suf2dcx2(jf,jk)=a*suf2dcx(jf,jk);
            suf2dcx2(jf,backGround.nx-jk-1)=a*suf2dcx(jf,backGround.nx-jk-1);
        }
    }
    }
    suftzz=-real(ifft2(suf2dcx2))*2.0;
    */
    return suftzz;
}
void elastic2dModelModify(elastic3D_ARMA& obj, \
    fmat modelvp, fmat modelvs, fmat modelrho, \
    int nx, int ny, int nz, \
    float dx, float dy, float dz, float dt,\
    int izFreeSurface, int isPMLSurface, int nthread)
{
//initialize, do first
    ny=1;
    obj.initialize(nx,ny,nz);
//maxThreadNum: Number of Thread for each source modeling;
    obj.maxThreadNum=nthread;
    obj.dz=dz;obj.dx=dx;
    obj.dy=dy;obj.dt=dt;
    obj.isPMLSurface=isPMLSurface;
    obj.nzSampleOfFreeSurface=izFreeSurface;
    obj.mpar_vp.col(0)=modelvp.st();
    //density-model
    obj.mpar_ro.col(0)=modelrho.st();
    //model-vs
    //obj.mpar_vs.col(0)=modelvs.st();
    obj.mpar_vs.col(0).fill(0.0);

    for(int i=0;i<izFreeSurface;i++){
        if(obj.isPMLSurface<=0.5){
            //free surface, air density, suface=0.0
            obj.mpar_ro.slice(i)=obj.mpar_ro.slice(izFreeSurface);
            obj.mpar_vp.slice(i).fill(0.0);
            obj.mpar_vs.slice(i).fill(0.0);
        }
        else if(obj.isPMLSurface>0.5){
            //PML surface, water density, suface=1.0
            obj.mpar_ro.slice(i)=obj.mpar_ro.slice(izFreeSurface);
            obj.mpar_vp.slice(i)=obj.mpar_vp.slice(izFreeSurface);
            obj.mpar_vs.slice(i)=obj.mpar_vs.slice(izFreeSurface);
        }
    }
    //obj.PML_wide=50;
    obj.PML_wide=60;
    obj.R=12;
    obj.nCoe=8;
    //update class obj, do before modeling
    obj.updatepar();
}

void modelConstantVel2d(fmat &vp, fmat& vs,\
    fmat& rho, float vp0, float vs0)
{
    fmat impedance;
    impedance.copy_size(vp);
    vs.fill(vs0);
    int n1(vp.n_rows),n2(vp.n_cols);
    for(int k2=0;k2<n2;k2++){
    for(int k1=0;k1<n1;k1++){
        impedance(k1,k2)=vp(k1,k2)*rho(k1,k2);
    }}
    vp.fill(vp0);
    for(int k2=0;k2<n2;k2++){
    for(int k1=0;k1<n1;k1++){
        rho(k1,k2)=impedance(k1,k2)/vp0;
        rho(k1,k2)=max(1000.0f,float(rho(k1,k2)));
    }}
}

fmat model2dExpandUp(fmat model, int nExpandRow)
{
    int n1(model.n_rows),n2(model.n_cols);
    fmat modelEx(n1+nExpandRow,n2);
    modelEx(span(nExpandRow,n1+nExpandRow-1),span::all)\
        =model(span::all,span::all);
    for(int k=0;k<nExpandRow;k++){
        modelEx.row(k)=modelEx.row(nExpandRow);
    }
    return modelEx;
}
fcube model3dExpandN1(fcube model, int nExpand)
{
    int n1(model.n_rows),n2(model.n_cols),n3(model.n_slices);
    fcube modelEx(n1,n2,n3+nExpand);
    modelEx(span::all,span::all,span(nExpand,n3+nExpand-1))\
        =model(span::all,span::all,span::all);
    for(int k=0;k<nExpand;k++){
        modelEx.slice(k)=modelEx.slice(nExpand);
    }
    return modelEx;
}
fmat LinearInterpolateMat2dByRow(fmat data2d)
{
    int n1(data2d.n_rows),n2(data2d.n_cols),i;
    fmat data2dInter;
    int nInter(n1*2-1);
    data2dInter.zeros(nInter,n2);
    data2dInter.row(0)=data2d.row(0);
    for(i=1;i<n1;i++){
        int kinter=2*i;
        data2dInter.row(kinter)=data2d.row(i);
        data2dInter.row(kinter-1)=(data2dInter.row(kinter)\
            +data2dInter.row(kinter-2));
        data2dInter.row(kinter-1)=data2dInter.row(kinter-1)/2.0;
    }
    return data2dInter;
}
fmat getScatterFromVp(fmat vp2d, int nSmooth)
{
    fmat backvp=fmatsmooth(vp2d,vp2d.n_rows,vp2d.n_cols,nSmooth);
    fmat scatter=vp2d;
    scatter.fill(0.0);
    for(int k1=0;k1<vp2d.n_rows;k1++){
    for(int k2=0;k2<vp2d.n_cols;k2++){
        scatter(k1,k2)=2.0*((vp2d(k1,k2)\
            -backvp(k1,k2))/backvp(k1,k2));
    }}
    return scatter;
}
fmat fillNumLessThan(fmat data2d, float num)
{
    for(int i1=0;i1<data2d.n_rows;i1++){
    for(int i2=0;i2<data2d.n_cols;i2++){
        if(data2d(i1,i2)<num)data2d(i1,i2)=num;
    }}
    return data2d;
}
fmat fillNumGreaterThan(fmat data2d, float num)
{
    for(int i1=0;i1<data2d.n_rows;i1++){
    for(int i2=0;i2<data2d.n_cols;i2++){
        if(data2d(i1,i2)>num)data2d(i1,i2)=num;
    }}
    return data2d;
}
fmat ChangeTimeSampling(fmat& datain, int nt2)
{
    int n1(datain.n_rows),n2(datain.n_cols);
    cx_fmat datafx(n1,n2);
    for(int i2=0;i2<n2;i2++)
    {
        datafx.col(i2)=fft(datain.col(i2));
    }
    
    fmat dataout(nt2,n2);
    cx_fmat dataoutfx(nt2,n2,fill::zeros);
    for(int i=0;i<min(nt2/2,n1/2);i++){
        dataoutfx.row(i)=datafx.row(i);
    }
    for(int i2=0;i2<n2;i2++){
        dataout.col(i2)=2.0*real(ifft(dataoutfx.col(i2)));
    }    
    return dataout;
}

fmat GetLayerDepth2dFromIndex(fmat layerIndex, float dz, int nSmoothForDepth=1)
{
    fmat dep(layerIndex.n_cols,1);
    dep.fill(0.0);
    /*
    for(int k2=0;k2<layerIndex.n_cols;k2++){
    for(int k1=0;k1<layerIndex.n_rows;k1++){
        if(layerIndex(k1,k2)>0.5){
            dep(k2,0)=k1*dz;
        }
    }}*/
    for(int k2=0;k2<layerIndex.n_cols;k2++){
        int iz=layerIndex.col(k2).index_max();
        dep(k2,0)=iz*dz;
    }
    fmat dep2(layerIndex.n_cols,1);
    dep2=dep;
    for(int kn=0;kn<nSmoothForDepth;kn++){
    for(int k=1;k<dep.n_rows-1;k++){
        if(dep(k-1,0)>0.01 && dep(k+1,0)>0.01 && dep(k,0)>0.01){
            dep2(k,0)=0.25*dep(k+1,0)+0.25*dep(k-1,0)+0.5*dep(k,0);
        }else if(dep(k-1,0)<=0.01 && dep(k+1,0)>0.01 && dep(k,0)>0.01){
            dep2(k,0)=(0.5*dep(k+1,0)+1.0*dep(k,0))/1.5;
        }else if(dep(k-1,0)>0.01 && dep(k+1,0)<=0.01 && dep(k,0)>0.01){
            dep2(k,0)=(0.5*dep(k-1,0)+1.0*dep(k,0))/1.5;
        }else if(dep(k-1,0)<=0.01 && dep(k+1,0)<=0.01 && dep(k,0)>0.01){
            dep2(k,0)=dep(k,0);
        }else if(dep(k,0)<=0.01){
            dep2(k,0)=dep(k,0);
        }else {
            dep2(k,0)=dep(k,0);
        }
    }
        dep=dep2;
    }
    for(int k=0;k<dep.n_rows;k++){
        if(dep(k,0)<=0.01){
            dep(k,0)=(layerIndex.n_rows-3)*dz;
        }
    }
    return dep;
}

fvec GetBornConstScatter(int n, float vp1, float vp2, int nSmooth)
{
    fmat vp2d(n,800);
    vp2d.fill(vp1);
    for(int i=n/2;i<n;i++){
        vp2d.row(i).fill(vp2);
    }
    fmat scatter=getScatterFromVp(vp2d,nSmooth);
    fvec s1(n);
    for(int i=0;i<n;i++){
        s1(i)=scatter(i,400);
    }
    return s1;
}

fmat GetLayerScatter(fmat layer2d, int nSmooth, \
    float dz, float vp1, float vp2, int nSmoothForDepth=25)
{
    fmat layerDep=GetLayerDepth2dFromIndex(layer2d,dz,nSmoothForDepth);
    int n=layer2d.n_rows;
    fvec s1d=GetBornConstScatter(n*2,vp1,vp2,nSmooth);
    fmat scatter=layer2d;
    scatter.fill(0.0);
    for(int k2=0;k2<layer2d.n_cols;k2++){
        if(layerDep(k2,0)>0.01){
            int z1=floor(layerDep(k2,0)/dz);
            int z2=ceil(layerDep(k2,0)/dz);
            float w1=1.0/(0.000001+abs(layerDep(k2,0)-z1*dz));
            float w2=1.0/(0.000001+abs(layerDep(k2,0)-z2*dz));
            float wAll=w1+w2;
            w1=w1/wAll;
            w2=w2/wAll;
            for(int k=z1;k>=0;k--){
                scatter(k,k2)+=w1*s1d(n-(z1-k));
            }
            for(int k=z2;k>=0;k--){
                scatter(k,k2)+=w2*s1d(n-(z2-k));
            }
            for(int k=z1+1;k<n;k++){
                scatter(k,k2)+=w1*s1d(n-(z1-k));
            }
            for(int k=z2+1;k<n;k++){
                scatter(k,k2)+=w2*s1d(n-(z2-k));
            }
        }
    }
    return scatter;
}
void write2dtoSUfile(fmat data2d, string filename, \
    float dx=5.0, float gx0=0.0, float dz=5.0)
{
    int n1(data2d.n_rows),n2(data2d.n_cols);
    segyhead head;
    head.head2={};
    head.data.zeros(n1,1);
    head.head2.sy=0.0;
    head.head2.gy=0.0;
    head.head2.sx=0.0;
    head.head2.ns=n1;
    head.head2.dt=dz*1e3;
    ofstream outf(filename);
    for(int ix=0;ix<data2d.n_cols;ix++){
        head.data=data2d(span::all,span(ix,ix));
        head.head2.gx=(ix)*dx+gx0;
        suhead_writeonetrace_tofile(head, outf);
    }
    outf.close();
}

fcube Born3dSRME(\
    elastic3D_ARMA& backGround,fcube data3d,\
    elastic3D_ARMA& scatterField, fcube scatter,\
    int izFreeSurface,int nthread, bool sourceModeling=true)
{

    int (*fptr)(class elastic3D_ARMA&);
    if(nthread<=1){
        fptr=TimeSliceCal_acoustic3D_OneThread;
    }else{
        fptr=TimeSliceCal_acoustic3D_MultiThread;
    }
    int nx=backGround.nx;
    int ny=backGround.ny;
    int nt=data3d.n_slices;

////////////////////////////////////////////////
    fcube suftzz(nx,ny,nt),scatterW;
    scatterW.copy_size(scatter);
    suftzz.fill(0.0);
    scatterW.fill(0.0);
    for(int k=0;k<nt;k++){
        if(sourceModeling){
            for(int i=0;i<nx;i++){
            for(int j=0;j<ny;j++){
                backGround.tzz(i,j,izFreeSurface)\
                    +=data3d(i,j,k);
                //backGround.tzz(i,j,izFreeSurface-2)\
                    -=data3d(i,j,k);
            }}
        }else{
            for(int i=0;i<nx;i++){
            for(int j=0;j<ny;j++){
                backGround.tzz(i,j,izFreeSurface)\
                    =data3d(i,j,k);
            }}
        }
        //Using internal parallelism
        fcubemul(scatterW,scatter,backGround.tzz,nthread);
        scatterField.tzz+=scatterW;
        (*fptr)(backGround);
        (*fptr)(scatterField);
        for(int i=0;i<nx;i++){
        for(int j=0;j<ny;j++){
            suftzz(i,j,k)=scatterField.tzz(i,j,izFreeSurface);
        }}
        if(k%200==0)
        {std::cout<<"now is running : "<<k<<endl;}
    }
    //for(int i=0;i<=seriesNum;i++){outf[i].close();}
    //datawrite3d_bycol_transpose(sufall,nt,nx,fileGather);
    std::cout<<"finished"<<endl;

    return suftzz;
}
fcube MWDBornOneShot3d(fcube suf3d, elastic3D_ARMA& objOrig, \
    float velWater, float velMax,int nSmoothVp, int ncpu)
{
    int nthread(ncpu),isPMLSurface(1);
    int expandModelUp=max(int(70-objOrig.nzSampleOfFreeSurface),0);
    int nzEx(objOrig.nz+expandModelUp);
    //freeSuface, source and obn Z-coord in model
    int freeSufaceZ(objOrig.nzSampleOfFreeSurface),sz(freeSufaceZ),\
        rz(freeSufaceZ);
    fcube modelvp=objOrig.mpar_vp;
    fcube modelvs=objOrig.mpar_vs;
    fcube modelrho=objOrig.mpar_ro;
    for(int k1=0;k1<freeSufaceZ;k1++){
        modelvp.slice(k1)=modelvp.slice(freeSufaceZ);
        modelvs.slice(k1)=modelvs.slice(freeSufaceZ);
        modelrho.slice(k1)=modelrho.slice(freeSufaceZ);
    }
    for(int k1=0;k1<modelvp.n_rows;k1++){
    for(int k2=0;k2<modelvp.n_cols;k2++){
    for(int k3=0;k3<modelvp.n_slices;k3++){
        modelvp(k1,k2,k3)=min(float(modelvp(k1,k2,k3)),velMax);
    }}}
    modelvp=model3dExpandN1(modelvp,expandModelUp);
    modelvs=model3dExpandN1(modelvs,expandModelUp);
    modelrho=model3dExpandN1(modelrho,expandModelUp);
    freeSufaceZ+=expandModelUp;
    sz+=expandModelUp;
    rz+=expandModelUp;
    fcube scatter,backrho,backvp;
    scatter.copy_size(modelvp);
    backrho.copy_size(modelvp);
    backvp.copy_size(modelvp);
    backrho=fcubesmooth(modelrho,nSmoothVp);
    //modelvp=fmatsmooth(modelvp,nzEx,nx,nSmoothVp);
    backvp=fcubesmooth(modelvp,nSmoothVp);
    scatter.fill(0.0);
    for(int k3=0;k3<nzEx;k3++){
    for(int k1=0;k1<objOrig.nx;k1++){
    for(int k2=0;k2<objOrig.ny;k2++){
        scatter(k1,k2,k3)=2.0*((modelvp(k1,k2,k3)\
            -backvp(k1,k2,k3))/backvp(k1,k2,k3)\
            +(modelrho(k1,k2,k3)-backrho(k1,k2,k3))\
            /backrho(k1,k2,k3));
        //scatter(k1,k2)=(1.0/modelvp(k1,k2)/modelvp(k1,k2)\
            -1.0/backvp(k1,k2)/backvp(k1,k2))\
            /(1.0/backvp(k1,k2)/backvp(k1,k2));
    }}}
    //datawrite(scatter,"scatter.dat");
    backvp.fill(velWater);
    elastic3D_ARMA backGround;
    elastic3D_ARMA scatterField;
    elastic2dModelModify(backGround, backvp, modelvs, backrho, \
        objOrig.nx, objOrig.ny, nzEx, objOrig.dx, objOrig.dy, objOrig.dz, \
        objOrig.dt, freeSufaceZ, isPMLSurface, nthread);
    elastic2dModelModify(scatterField, backvp, modelvs, backrho, \
        objOrig.nx, objOrig.ny, nzEx,objOrig.dx, objOrig.dy, objOrig.dz, \
        objOrig.dt, freeSufaceZ, 1, nthread);
/////////////////////////////////////
    fcube bornMul3d=Born3dSRME(\
        backGround, suf3d, scatterField, scatter,\
        freeSufaceZ, nthread, true);
    return bornMul3d;
}

#endif
