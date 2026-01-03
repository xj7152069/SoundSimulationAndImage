/*
    Multiple wave prediction and suppression of seismic data.
    Clear up in 2022.11.05, by Xiang Jian, WPI.
New
*/

#ifndef CRMD_HPP
#define CRMD_HPP

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

//////////////////////// Struct /////////////////////////
struct parMWD
{
    char fileinVelLayer[1024],fileinLayer[1024],\
         fileinData[1024],fileoutModel[1024],\
         fileoutDemultiple[1024];
    char fileoutModelSwap[1024],\
         fileoutDemultipleSwap[1024];
    char keySx[19], keySy[19], keyGx[19], keyGy[19],\
        keyWaterDepth[19], keyNodeDepth[19];

    int nx, ny, nz, nt, nlayer;
    int ilayerUpRef, ilayerDownRef;
    float dx, dy, dz, dt;
    int interTimesByRow, interTimesByCol;
    float maxAperture, weightMax, \
        eccentricity, focusEnhance, \
        offsetEnhance, aModify, \
        bModify, weightPow, \
        cutSlopeMax, cutSlopeMin, \
        fmin, fmax, systemDelayTime, \
        waterVelocity, theta, \
        timeUnitTrans, spaceUnitTrans;
    int wienerFilterLength, wienerTimeSlide, \
        wienerDataLength, halfWinx;
    float wienerTikhonov;
    int ncpu; 
    int dataGroupBegIndex, dataGroupGap;
//The data processing process and output that need to be executed {0,1}:
    int step0DataFilter;
    int output0DataFilter;
    int step1MultiplePrediction;
    int output1MultipleModelOri;
    int step2MultipleModelFilter;
    int output2MultipleModelFiltered;
    int step3AdaptiveSubtraction;
    int output3AdaptiveSubtractionResult;
    int step4CatData;
    int NumCoordxExtra;
    int dataMeshSizeModify;
    int dataIsSuFile;
    float VrmsGrad,\
        nxLeftWin,nxRightWin,\
        nyUpWin,nyDownWin;
    int matchLoopNum,matchLevelNum;
};
///////////////////// Function declaration ///////////////////////
void cxfcubeLinearInterpolation3dByCol(cx_fcube& data3d);
void cxfcubeLinearInterpolation3dByCol(cx_fcube& data3d, int ncpu);
void cxfcubeLinearInterpolation3dByRow(cx_fcube& data3d);
void cxfcubeLinearInterpolation3dByRow(cx_fcube& data3d, int ncpu);
void cxfcubeAntiLinearInterpolation3dByRow(cx_fcube& data3d);
void cxfcubeAntiLinearInterpolation3dByCol(cx_fcube& data3d);
void fcubeLinearInterpolation3dByRow(fcube& data3d,int ncpu);
void fcubeLinearInterpolation3dByCol(fcube& data3d,int ncpu);
void fcubeAntiLinearInterpolation3dByCol(fcube& data3d);
void fcubeAntiLinearInterpolation3dByRow(fcube& data3d);
void fmatLinearInterpolation2dByRow(fmat& data2d);
void fmatLinearInterpolation2dByCol(fmat& data2d);
void getSourceIndes(int& sxindex,int& syindex,\
    fmat coordx, fmat coordy);
void getPointIndex2d(int& sxindex,int& syindex,\
 float pointx, float pointy, fmat coordx, fmat coordy);
int getNtr(segyhead &head, int nt, const char *key);
fmat cutDataBySlope(fcube &data3d, fmat &coordx, fmat &coordy,\
    float dt, float cutSlopeMin, float cutSlopeMax, float delayTime=0.0);
double rotateCoordx(double coordx, double coordy, double theta);
double rotateCoordy(double coordx, double coordy, double theta);
void rotateCoord2d(fmat& coordx2d, fmat& coordy2d, \
     fmat coordx2dOri, fmat coordy2dOri, float theta);
bool ifstreamFloatEndOfFile(ifstream & infile);
float getCoordRotateTheta(fmat coordx, fmat coordy, float theta0=0.0);
///////////////////////////////////////////////////////////
void InitialParMWD(struct parMWD& par)
{
    par.fileinLayer[0]='\0';
    par.fileinVelLayer[0]='\0';
    par.fileinData[0]='\0';
    par.fileoutModel[0]='\0';
    par.fileoutDemultiple[0]='\0';
    par.fileoutModelSwap[0]='\0';
    par.fileoutDemultipleSwap[0]='\0';
    par.keySx[0]='\0';
    par.keySy[0]='\0'; 
    par.keyGx[0]='\0';
    par.keyGy[0]='\0';
    par.keyWaterDepth[0]='\0';
    par.keyNodeDepth[0]='\0';
    par.nx=1, par.ny=1, par.nt=1;
    par.nz=1, par.nlayer=1;
    par.dx=1, par.dy=1, par.dz=1; 
    par.dt=0.001;
    par.ilayerUpRef=0;
    par.ilayerDownRef=0;
    par.interTimesByRow=0;
    par.interTimesByCol=0;
    par.systemDelayTime=0.001;
    par.maxAperture=3000000.0;
    par.weightMax=0.99;
    par.eccentricity=1.0;
    par.focusEnhance=1.0;
    par.offsetEnhance=0.25;
    par.aModify=150.0;
    par.bModify=150.0;
    par.weightPow=1.5;
    par.cutSlopeMax=1450;
    par.cutSlopeMin=1250;
    par.fmin=1.0;
    par.fmax=150.0;
    par.waterVelocity=1520.0;
    par.theta=0.0;
    par.timeUnitTrans=1000000.0;
    par.spaceUnitTrans=1.0;
    par.wienerFilterLength=3;
    par.wienerTimeSlide=10; 
    par.wienerDataLength=30; 
    par.halfWinx=5;
    par.wienerTikhonov=0.01;
    par.ncpu=1;
    par.dataGroupBegIndex=0;
    par.dataGroupGap=0;
    par.step0DataFilter=0;
    par.output0DataFilter=0;
    par.step1MultiplePrediction=0;
    par.output1MultipleModelOri=0;
    par.step2MultipleModelFilter=0;
    par.output2MultipleModelFiltered=0;
    par.step3AdaptiveSubtraction=0;
    par.output3AdaptiveSubtractionResult=0;
    par.step4CatData=0;
    par.NumCoordxExtra=0;
    par.dataMeshSizeModify=1;
    par.dataIsSuFile=0;
    par.matchLoopNum=1;
    par.matchLevelNum=0;
    strcat(par.keySx,"sx");
    strcat(par.keySy,"sy");
    strcat(par.keyGx,"gx");
    strcat(par.keyGy,"gy");
    strcat(par.keyWaterDepth,"swdep");
    strcat(par.keyNodeDepth,"gelev");
    par.VrmsGrad=200.0;
    par.nxLeftWin=-10.0;
    par.nxRightWin=-10.0;
    par.nyUpWin=-10.0;
    par.nyDownWin=-10.0;
}
void getParMWD(struct parMWD& par, \
    const char *name, const char *value, bool outlog=false);
void getParMWD(struct parMWD& par, \
    const char *name, const char *value, bool outlog)
{
    if(outlog){cout<<name<<"="<<value<<endl;}
    if(strcmp(name,"InputData")==0) 
        { strcat(par.fileinData,value);}
    else if (strcmp(name,"InputVelModel")==0) 
        { strcat(par.fileinVelLayer,value);}
    else if (strcmp(name,"InputLayerModel")==0) 
        { strcat(par.fileinLayer,value);}
    else if (strcmp(name,"OutputMultipleModelResult")==0) 
        { strcat(par.fileoutModel,value);}
    else if (strcmp(name,"OutputMultipleModelSwap")==0) 
        { strcat(par.fileoutModelSwap,value);}
    else if (strcmp(name,"OutputDemultipleResult")==0) 
        { strcat(par.fileoutDemultiple,value);}
    else if (strcmp(name,"OutputDemultipleSwap")==0) 
        { strcat(par.fileoutDemultipleSwap,value);}
    else if (strcmp(name,"nx")==0) 
        { par.nx=atoi(value);}
    else if (strcmp(name,"ny")==0) 
        { par.ny=atoi(value);}
    else if (strcmp(name,"nz")==0) 
        { par.nz=atoi(value);}
    else if (strcmp(name,"nlayer")==0) 
        { par.nlayer=atoi(value);}
    else if (strcmp(name,"ilayerDowngoing")==0) 
        { par.ilayerDownRef=atoi(value);}
    else if (strcmp(name,"ilayerUpgoing")==0) 
        { par.ilayerUpRef=atoi(value);}
    else if (strcmp(name,"nt")==0) 
        { par.nt=atoi(value);}
    else if (strcmp(name,"dx")==0) 
        { par.dx=atof(value);}
    else if (strcmp(name,"dy")==0) 
        { par.dy=atof(value);}
    else if (strcmp(name,"dz")==0) 
        { par.dz=atof(value);}
    else if (strcmp(name,"dt")==0) 
        { par.dt=atof(value);}
    else if (strcmp(name,"nxInterTimes")==0) 
        { par.interTimesByRow=atoi(value);}
    else if (strcmp(name,"nyInterTimes")==0) 
        { par.interTimesByCol=atoi(value);}
    else if (strcmp(name,"systemDelayTime")==0) 
        { par.systemDelayTime=atof(value);}
    else if (strcmp(name,"maxAperture")==0) 
        { par.maxAperture=atof(value);}
    else if (strcmp(name,"weightMax")==0) 
        { par.weightMax=atof(value);}
    else if (strcmp(name,"eccentricity")==0) 
        { par.eccentricity=atof(value);}
    else if (strcmp(name,"focusExpandOffset")==0) 
        { par.focusEnhance=atof(value);}
    else if (strcmp(name,"focusMoveOffset")==0) 
        { par.offsetEnhance=atof(value);}
    else if (strcmp(name,"xMinAperture")==0) 
        { par.aModify=atof(value);}
    else if (strcmp(name,"yMinAperture")==0) 
        { par.bModify=atof(value);}
    else if (strcmp(name,"weightPow")==0) 
        { par.weightPow=atof(value);}
    else if (strcmp(name,"cutSlopeMax")==0) 
        { par.cutSlopeMax=atof(value);}
    else if (strcmp(name,"cutSlopeMin")==0) 
        { par.cutSlopeMin=atof(value);}
    else if (strcmp(name,"fmin")==0) 
        { par.fmin=atof(value);}
    else if (strcmp(name,"fmax")==0) 
        { par.fmax=atof(value);}
    else if (strcmp(name,"waterVelocity")==0) 
        { par.waterVelocity=atof(value);}
    else if (strcmp(name,"theta")==0) 
        { par.theta=atof(value);}
    else if (strcmp(name,"timeUnitTrans")==0) 
        { par.timeUnitTrans=atof(value);}
    else if (strcmp(name,"spaceUnitTrans")==0) 
        { par.spaceUnitTrans=atof(value);}
    else if (strcmp(name,"wienerFilterLength")==0) 
        { par.wienerFilterLength=atoi(value);}
    else if (strcmp(name,"wienerTimeSlide")==0) 
        { par.wienerTimeSlide=atoi(value);}
    else if (strcmp(name,"wienerDataTimeLength")==0) 
        { par.wienerDataLength=atoi(value);}
    else if (strcmp(name,"wienerDataSpaceLength")==0) 
        { par.halfWinx=atoi(value);}
    else if (strcmp(name,"wienerTikhonov")==0) 
        { par.wienerTikhonov=atof(value);}
    else if (strcmp(name,"VrmsGrad")==0) 
        { par.VrmsGrad=atof(value);}
    else if (strcmp(name,"nxLeftWin")==0) 
        { par.nxLeftWin=atof(value);}
    else if (strcmp(name,"nxRightWin")==0) 
        { par.nxRightWin=atof(value);}
    else if (strcmp(name,"nyUpWin")==0) 
        { par.nyUpWin=atof(value);}
    else if (strcmp(name,"nyDownWin")==0) 
        { par.nyDownWin=atof(value);}
    else if (strcmp(name,"ncpu")==0) 
        { par.ncpu=atoi(value);}
    else if (strcmp(name,"dataGatherBegIndex")==0) 
        { par.dataGroupBegIndex=atoi(value);}
    else if (strcmp(name,"dataGatherNum")==0) 
        { par.dataGroupGap=atoi(value);}
    else if (strcmp(name,"step0DataFilter")==0) 
        { par.step0DataFilter=atoi(value);}
    else if (strcmp(name,"output0DataFilter")==0) 
        { par.output0DataFilter=atoi(value);}
    else if (strcmp(name,"doMultiplePrediction")==0) 
        { par.step1MultiplePrediction=atoi(value);}
    else if (strcmp(name,"outputSwapMultipleModel")==0) 
        { par.output1MultipleModelOri=atoi(value);}
    else if (strcmp(name,"step2MultipleModelFilter")==0) 
        { par.step2MultipleModelFilter=atoi(value);}
    else if (strcmp(name,"output2MultipleModelFiltered")==0) 
        { par.output2MultipleModelFiltered=atoi(value);}
    else if (strcmp(name,"doAdaptiveSubtraction")==0) 
        { par.step3AdaptiveSubtraction=atoi(value);}
    else if (strcmp(name,"outputSwapAdaptiveSubtraction")==0) 
        { par.output3AdaptiveSubtractionResult=atoi(value);}
    else if (strcmp(name,"doOutputResult")==0) 
        { par.step4CatData=atoi(value);}
    else if (strcmp(name,"nxZeroOffsetExtra")==0) 
        { par.NumCoordxExtra=atoi(value);}
    else if (strcmp(name,"dataMeshSizeModify")==0) 
        { par.dataMeshSizeModify=atoi(value);}
    else if (strcmp(name,"dataIsSuFile")==0) 
        { par.dataIsSuFile=atoi(value);}
    else if (strcmp(name,"matchLoopNum")==0) 
        { par.matchLoopNum=atoi(value);}
    else if (strcmp(name,"matchLevelNum")==0) 
        { par.matchLevelNum=atoi(value);}
    else if (strcmp(name,"keySx")==0) 
        { par.keySx[0]='\0'; strcat(par.keySx,value);}
    else if (strcmp(name,"keySy")==0) 
        { par.keySy[0]='\0'; strcat(par.keySy,value);}
    else if (strcmp(name,"keyGx")==0) 
        { par.keyGx[0]='\0'; strcat(par.keyGx,value);}
    else if (strcmp(name,"keyGy")==0) 
        { par.keyGy[0]='\0'; strcat(par.keyGy,value);}
    else if (strcmp(name,"keyWaterDepth")==0) 
        { par.keyWaterDepth[0]='\0'; strcat(par.keyWaterDepth,value);}
    else if (strcmp(name,"keyNodeDepth")==0) 
        { par.keyNodeDepth[0]='\0'; strcat(par.keyNodeDepth,value);} 
    else{std::cout<<"Warning: Non-Existent Parameter!"<<std::endl;}
}
void readParMWD(struct parMWD& par, const char *file, bool parlog)
{
    ifstream parin;
    char name[1024],value[1024];
    string line;
    parin.open(file);
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
            getParMWD((&par)[0],name,value,parlog);
        }
    }else{
        cout<<"Not find Par file!"<<endl;
    } 

}
/******* Coding predicts multiple waves, 3D, beta*******/
void getLayerVrms2d(fmat &vrms2d, fmat &t0Double2d,\
    fcube& layerVel3d, fmat layerDepth, float dz, int ncpu=1)
{
    //float maxDepth=layerDepth.max();
    //int n3=round(maxDepth/dz)+2;
    //cout<<"Max depth is: "<<layerDepth.max()<<endl;
    int n1(layerVel3d.n_rows),n2(layerVel3d.n_cols);
    vrms2d.zeros(n1,n2);
    t0Double2d.zeros(n1,n2);
ncpu=min(ncpu,n1);
omp_set_num_threads(ncpu);
#pragma omp parallel for
    for(int i1=0;i1<n1;i1++){
    for(int i2=0;i2<n2;i2++){
        int nz=floor(layerDepth(i1,i2)/dz);
        float t0=0.0;
        float vrms=0.0;
    for(int i3=0;i3<=nz;i3++){
        float vi=layerVel3d(i1,i2,i3);
        float ti=dz/vi;
        t0+=ti;
        vrms+=(ti*vi*vi);
    }
        float vi=layerVel3d(i1,i2,nz+1);
        float ti=(layerDepth(i1,i2)-nz*dz)/vi;
        t0+=ti;
        vrms+=(ti*vi*vi);
        vrms2d(i1,i2)=sqrt(vrms/t0);
        t0Double2d(i1,i2)=2.0*t0;
    }}
}
void getMultipleLayerMiddlePointTravelTimeMWD(\
    fmat& travelTime, \
    fmat& coordxData, fmat& coordyData, fmat& weightWin, \
    fmat& vrms2d, fmat& t0Double2d, fmat& layerDepth, \
    float coordxMul, float coordyMul, float waterVel)
{
    int n1(travelTime.n_rows),n2(travelTime.n_cols);
    travelTime.fill(0.0);
    float l_min(0.001);
    int iMulx, jMuly;
    getPointIndex2d(iMulx, jMuly,\
        coordxMul, coordyMul,\
        coordxData, coordyData);
    ivec begix(n2),endix(n2);
    for(int j=0;j<n2;j++){
        begix(j)=0;
        endix(j)=0;
    for(int i=0;i<n1-1;i++){
        if(weightWin(i,j)>0.000001 && begix(j)<0.5){
            begix(j)=i;
            endix(j)=n1;
        }
        if(weightWin(i,j)>0.000001 && weightWin(i+1,j)<=0.000001){
            endix(j)=i+1;
        }
    }}
    for(int j=0;j<n2;j++){
        int j1=round((jMuly+j)/2.0);
        j1=max(j1,0);j1=min(j1,n2-1);
    for(int i=begix(j);i<endix(j);i++){
        int i1=round((iMulx+i)/2.0);
        i1=max(i1,0);i1=min(i1,n1-1);
        float fi=(coordxMul+coordxData(i,j))/2.0;
        float fj=(coordyMul+coordyData(i,j))/2.0;
        float half_offset=(fi-coordxMul)*(fi-coordxMul)\
            +(fj-coordyMul)*(fj-coordyMul);
        //float deepth=layerDepth(i1,j1);
        //float l_trace=2.0*sqrt(half_offset+deepth*deepth);
        //travelTime(i,j)=l_trace/waterVel;
        float vrms=vrms2d(i1,j1);
        float t0=t0Double2d(i1,j1);
        travelTime(i,j)=sqrt(t0*t0+4.0*half_offset/vrms/vrms);
    }}
}
void getMultipleLayerMinimumTravelTimeMWD( \
    fmat& travelTime, fmat& incidentPx, fmat& incidentPy, \
    fmat coordxData, fmat coordyData, fmat& weightWin, \
    fmat& vrms2d, fmat& t0Double2d, fmat& layerDepth, \
    float coordxMul, float coordyMul, float waterVel)
{
    int n1(travelTime.n_rows),n2(travelTime.n_cols),iMulx, jMuly;
    int dn1=round(50.0/(abs(coordxData.max()-coordxData.min())/n1));
    int dn2=round(50.0/(abs(coordyData.max()-coordyData.min())/n2));
    dn1=min(dn1,n1-1);dn2=min(dn2,n2-1);
    dn1=max(dn1,1);dn2=max(dn2,1);
    //cout<<dn1<<","<<dn2<<endl;
    travelTime.fill(0.0);
    getPointIndex2d(iMulx, jMuly,\
        coordxMul, coordyMul,\
        coordxData, coordyData);
    ivec begix(n2),endix(n2);
    for(int j=0;j<n2;j++){
        begix(j)=0;
        endix(j)=0;
    for(int i=0;i<n1-1;i++){
        if(weightWin(i,j)>0.000001 && begix(j)<0.5){
            begix(j)=i;
            endix(j)=n1;
        }
        if(weightWin(i,j)>0.000001 && weightWin(i+1,j)<=0.000001){
            endix(j)=i+1;
        }
    }}
    for(int j=0;j<n2;j++){
        int fj=round((jMuly+j)/2.0);
        fj=max(fj,0);fj=min(fj,n2-1);
    for(int i=begix(j);i<endix(j);i++){  
        int fi=round((iMulx+i)/2.0);
        fi=max(fi,0);fi=min(fi,n1-1);
        float mulOffsetX=coordxData(fi,fj)-coordxMul;
        float mulOffsetY=coordyData(fi,fj)-coordyMul;
        float offsetMulMin=mulOffsetX*mulOffsetX\
            +mulOffsetY*mulOffsetY;
        float dataOffsetX=coordxData(fi,fj)-coordxData(i,j);
        float dataOffsetY=coordyData(fi,fj)-coordyData(i,j);
        float offsetDataMin=dataOffsetX*dataOffsetX\
            +dataOffsetY*dataOffsetY;
        //float deepthMin=layerDepth(fi,fj);
        //float l_min=sqrt(offsetMulMin+deepthMin*deepthMin)\
            +sqrt(offsetDataMin+deepthMin*deepthMin);
        float vrms=vrms2d(fi,fj);
        float t0=t0Double2d(fi,fj);
        float l_min=0.5*sqrt(t0*t0+4.0*offsetMulMin/vrms/vrms)\
            +0.5*sqrt(t0*t0+4.0*offsetDataMin/vrms/vrms);
        for(int j1=0;j1<n2;j1=j1+dn2){
        for(int i1=begix(j1);i1<endix(j1);i1=i1+dn1){
            float offsetMul=(coordxData(i1,j1)-coordxMul)\
                *(coordxData(i1,j1)-coordxMul)\
                +(coordyData(i1,j1)-coordyMul)\
                *(coordyData(i1,j1)-coordyMul);
            float offsetData=(coordxData(i1,j1)-coordxData(i,j))\
                *(coordxData(i1,j1)-coordxData(i,j))\
                +(coordyData(i1,j1)-coordyData(i,j))\
                *(coordyData(i1,j1)-coordyData(i,j));
            //float l_trace=sqrt(offsetMul+d11*d11)+sqrt(offsetData+d11*d11);
            vrms=vrms2d(i1,j1);
            t0=t0Double2d(i1,j1);
            float l_trace=0.5*sqrt(t0*t0+4.0*offsetMul/vrms/vrms)\
                +0.5*sqrt(t0*t0+4.0*offsetData/vrms/vrms);
            if(l_trace<l_min){
                l_min=l_trace;
                offsetMulMin=offsetMul;
                offsetDataMin=offsetData;
                //deepthMin=d11;
                fi=i1;fj=j1;
            }
        }}
        int minix(fi),minjy(fj);
        for(int j2=max(fj-dn2,0);j2<=min(fj+dn2,n2-1);j2++){
        for(int i2=max(fi-dn1,0);i2<=min(fi+dn1,n1-1);i2++){    
            float offsetMul=(coordxData(i2,j2)-coordxMul)\
                *(coordxData(i2,j2)-coordxMul)\
                +(coordyData(i2,j2)-coordyMul)\
                *(coordyData(i2,j2)-coordyMul);
            float offsetData=(coordxData(i2,j2)-coordxData(i,j))\
                *(coordxData(i2,j2)-coordxData(i,j))\
                +(coordyData(i2,j2)-coordyData(i,j))\
                *(coordyData(i2,j2)-coordyData(i,j));
            //float l_trace=sqrt(offsetMul+d11*d11)+sqrt(offsetData+d11*d11);
            vrms=vrms2d(i2,j2);
            t0=t0Double2d(i2,j2);
            float l_trace=0.5*sqrt(t0*t0+4.0*offsetMul/vrms/vrms)\
                +0.5*sqrt(t0*t0+4.0*offsetData/vrms/vrms);
            if(l_trace<l_min){
                l_min=l_trace;
                offsetMulMin=offsetMul;
                offsetDataMin=offsetData;
                //deepthMin=d11;
                minix=i2;
                minjy=j2;
            }
        }}
    //travelTime(i,j)=l_min/waterVel;
        travelTime(i,j)=l_min;
/*
    offsetMulMin=sqrt(offsetMulMin);
    offsetDataMin=sqrt(offsetDataMin);
    deepthMin=abs(deepthMin);
    float theta(0.0);
    theta=atan(offsetMulMin/deepthMin)\
        +atan(offsetDataMin/deepthMin);
    theta=cos(theta*0.5);
    weightWin(i,j)=min(float(weightWin(i,j)),float(theta));
    
    theta=(offsetDataMin/sqrt(deepthMin*deepthMin\
        +offsetDataMin*offsetDataMin+1e-12))/waterVel;
    incidentPx(i,j)=theta*(coordxData(i,j)\
        -coordxData(minix,minjy))/(offsetDataMin+1e-12);
    incidentPy(i,j)=theta*(coordyData(i,j)\
        -coordyData(minix,minjy))/(offsetDataMin+1e-12);
*/
    }}
}
void getMultiplePredictionAperture(fmat& weightWin,\
    fmat coordxData, fmat coordyData, \
    float coordxMul, float coordyMul, \
    float sourceCoordx, float sourceCoordy, \
    float eccentricity, float maxAperture, \
    float aModify, float bModify, float weightPow, \
    float offsetEnhance,float focusEnhance,float weightMax)
{
    int n1(coordxData.n_rows),n2(coordxData.n_cols);
    weightWin.copy_size(coordxData);
    float gx=sourceCoordx;
    float gy=sourceCoordy;
    float gx2=coordxMul;
    float gy2=coordyMul;
    float dgx=(gx-gx2);
    if(abs(dgx)<0.001){
        if(dgx<0){dgx=-0.001;}
        else{dgx=0.001;}
    }
    float theta=-atan((gy-gy2)/dgx);
    fmat coordxRota=cos(theta)*coordxData-sin(theta)*coordyData;
    fmat coordyRota=sin(theta)*coordxData+cos(theta)*coordyData;
    float gxrota=cos(theta)*gx-sin(theta)*gy;
    float gyrota=sin(theta)*gx+cos(theta)*gy;
    float gx2rota=cos(theta)*gx2-sin(theta)*gy2;
    float gy2rota=sin(theta)*gx2+cos(theta)*gy2;
    float c=focusEnhance*abs(gx2rota-gxrota)/2.0;
    c=min(c,float(maxAperture));
    float a=eccentricity*c;
    float b=sqrt(a*a-c*c);
    a=a+aModify;
    b=b+bModify;
    float gxmid=((gx2rota+gxrota)*0.5\
        +(gx2rota-gxrota)*offsetEnhance);
    //cout<<(gx2rota+gxrota)*0.5<<" | "<<gxmid<<endl;
    float gymid=gyrota;
    coordxRota=coordxRota-gxmid;
    coordyRota=coordyRota-gymid;
    a=a*a;b=b*b;
    for(int k2=0;k2<n2;k2++){
    for(int k1=0;k1<n1;k1++){
        coordxRota(k1,k2)=(coordxRota(k1,k2)*coordxRota(k1,k2))/a;
        coordyRota(k1,k2)=(coordyRota(k1,k2)*coordyRota(k1,k2))/b;
        weightWin(k1,k2)=(1.0-coordxRota(k1,k2)-coordyRota(k1,k2));
        if(weightWin(k1,k2)<0.0){
            weightWin(k1,k2)=0.0;
        }else if(weightWin(k1,k2)>weightMax){
            weightWin(k1,k2)=weightMax;
        }
    }}
    //weightWin-=weightWin.min();
    weightWin/=weightWin.max();
    weightWin=pow(weightWin,weightPow);
    for(int k2=0;k2<n2;k2++){
    for(int k1=0;k1<n1;k1++){
        weightWin(k1,k2)=Blackman(weightWin(k1,k2),1.0);
    }}
}
float Gauss1d(float n, float N)
{
    float f=exp(-4.5*n*n/N/N);
    return f;
}
void multiplePrediction3dMWD_onepoint_allfrequence(\
 cx_fcube* uMulti, int indexMul, float coordxMul, float coordyMul,\
 cx_fcube* uData, fmat* vrms2d, fmat* t0Double2d, fmat* layerDepth,\
 fmat *coordxData,fmat *coordyData,cx_fmat* timeBase, \
 float sourceCoordx, float sourceCoordy, \
 float eccentricity, float maxAperture, \
 float aModify, float bModify, float weightPow, \
 float offsetEnhance,float focusEnhance,float weightMax, \
 float df, int fn1, int fn2, float waterVel, \
 float dtn, bool *end_of_thread)
{
    int n1(uData[0].n_rows),n2(uData[0].n_cols),n3(uData[0].n_slices);
    int ntr=uMulti->n_cols*uMulti->n_rows;
    fmat travelTime(n1,n2),weightWin(n1,n2);
    fmat incidentPx(n1,n2),incidentPy(n1,n2);
    getMultiplePredictionAperture(weightWin,\
        coordxData[0], coordyData[0], \
        coordxMul, coordyMul, \
        sourceCoordx, sourceCoordy, \
        eccentricity, maxAperture, \
        aModify, bModify, weightPow, \
        offsetEnhance, focusEnhance, weightMax);
    if(abs(layerDepth[0].max()-layerDepth[0].min())<15.0){
        getMultipleLayerMiddlePointTravelTimeMWD(travelTime, \
            coordxData[0], coordyData[0], weightWin, \
            vrms2d[0], t0Double2d[0], layerDepth[0], \
            coordxMul, coordyMul, waterVel);
    }else{
        getMultipleLayerMinimumTravelTimeMWD(\
            travelTime, incidentPx, incidentPy, \
            coordxData[0], coordyData[0], weightWin, \
            vrms2d[0], t0Double2d[0], layerDepth[0], \
            coordxMul, coordyMul, waterVel);
    }
    //incidentPx.fill(0.0);incidentPy.fill(0.0);
    imat timeIndex(n1,n2,fill::zeros);
    ivec begix(n2),endix(n2);
    for(int j=0;j<n2;j++){
        begix(j)=0; endix(j)=0;
    for(int i=0;i<n1-1;i++){
        if(travelTime(i,j)>0.0001 && begix(j)<0.5){
            begix(j)=i;
            endix(j)=n1;
        }
        if(travelTime(i,j)>0.0001 && travelTime(i+1,j)<=0.0001){
            endix(j)=i+1;
        }
        timeIndex(i,j)=round(travelTime(i,j)/dtn);
        float wr=travelTime(i,j)*3000.0;
        weightWin(i,j)*=(1.0/wr);
        if(timeIndex(i,j)>(timeBase[0].n_rows-1)){timeIndex(i,j)=0;}
    }}

    cx_float a,b,c;
    c.real(0.0);c.imag(1.0);
    for(int k=fn1;k<fn2;k++){
        float w=-2.0*3.1415926*df*k;
        b=c*w;
        int kfn=k-fn1;
    for(int j=0;j<n2;j++){
    for(int i=begix(j);i<endix(j);i++){
        a=timeBase[0](timeIndex(i,j),kfn)*weightWin(i,j);
        uMulti[0](indexMul,0,k)+=(a*uData[0](i,j,k));
    }}
        uMulti[0](indexMul,0,k)=uMulti[0](indexMul,0,k)*b;
    }
    end_of_thread[0]=true;
}
cx_fmat getTimeCodeingBase(float maxTime,\
    float df, int fn1, int fn2, \
    int ncpu=1,float dtn=1e-4)
{
    int nfn=fn2-fn1;
    int ntn=maxTime/dtn;
    cx_fmat timeBase(ntn,nfn);
    timeBase.fill(0.0);
omp_set_num_threads(ncpu);
#pragma omp parallel for
    for(int k=fn1;k<fn2;k++){
        cx_float a,b;
        float w=-2.0*3.1415926*df*k;
        a.real(0.0);a.imag(1.0);
        for(int i=1;i<ntn;i++){
            b=a*w*dtn*float(i);
            timeBase(i,k-fn1)=exp(b);
            //timeBase(i,k-fn1)=conj(exp(b));
        }
    }
    return timeBase;
}
void multiplePrediction3dMWD(\
 cx_fcube& multiple3d, fmat& coordxMul, fmat& coordyMul, \
 cx_fcube& data3d, fmat& coordxData, fmat& coordyData, \
 fmat& vrms2d, fmat& t0Double2d, \
 fmat& layerDepth2d, cx_fmat& timeBase,float waterVel, \
 float sourceCoordx, float sourceCoordy, \
 float eccentricity, float maxAperture, \
 float aModify, float bModify, float weightPow, \
 float offsetEnhance,float focusEnhance,float weightMax, \
 float df, int fn1, int fn2, int ncpu, float dtn=1e-4)
{
    int ntr=multiple3d.n_rows;
    ncpu=max(ncpu,1);
    ncpu=min(ncpu,ntr);
    multiple3d.fill(0.0);
    cx_fcube *pMul(&multiple3d);
    cx_fcube *pData(&data3d);
    fmat *pCoordx(&coordxData);
    fmat *pCoordy(&coordyData);
    fmat *pVrms(&vrms2d);
    fmat *pT0Double(&t0Double2d);
    fmat *pLayer(&layerDepth2d);
    thread *pcal;
    bool *end_of_thread;
    end_of_thread=new bool[ncpu];
    pcal=new thread[ncpu];

    for(int kcpu=0;kcpu<ncpu;kcpu++){
        end_of_thread[kcpu]=false;
        pcal[kcpu]=thread(multiplePrediction3dMWD_onepoint_allfrequence,\
            pMul,kcpu,coordxMul(kcpu,0),coordyMul(kcpu,0),\
            pData,pVrms,pT0Double,pLayer,pCoordx,pCoordy,&timeBase, \
            sourceCoordx,sourceCoordy,eccentricity,maxAperture, \
            aModify, bModify, weightPow, \
            offsetEnhance, focusEnhance, weightMax, \
            df,fn1,fn2,waterVel,dtn,&(end_of_thread[kcpu]));
    }
    int taskCount=ncpu;
    int percentage;
    while(taskCount<ntr){
        for(int kcpu=0;kcpu<ncpu;kcpu++){
        if(!end_of_thread[kcpu]){continue;}
        else if(end_of_thread[kcpu] && pcal[kcpu].joinable())
        {
            pcal[kcpu].join();
            end_of_thread[kcpu]=false;
            if(taskCount%(ntr/10)==0)
            {
                percentage=round(100*float(taskCount)/ntr);
                cout<<percentage<<"%";
            }
            pcal[kcpu]=thread(multiplePrediction3dMWD_onepoint_allfrequence,\
                pMul,taskCount,coordxMul(taskCount,0),coordyMul(taskCount,0),\
                pData,pVrms,pT0Double,pLayer,pCoordx,pCoordy,&timeBase, \
                sourceCoordx,sourceCoordy,eccentricity,maxAperture, \
                aModify, bModify, weightPow, \
                offsetEnhance, focusEnhance, weightMax, \
                df,fn1,fn2,waterVel,dtn,&(end_of_thread[kcpu]));
            taskCount++;
        }
        if(taskCount>=ntr){break;}
        }
    }
    for(int kcpu=0;kcpu<ncpu;kcpu++){
        if(pcal[kcpu].joinable()){
        pcal[kcpu].join();}
    }
    delete[] pcal;
    pcal=nullptr;
    delete[] end_of_thread;
    end_of_thread=nullptr;
}
////////////////////////////////////////////////////////////
/******* Common Functions *******/
void vrmsSpaceInterData3dtxByCol(fcube& data3dtxRegular, \
    fmat& coordxRegular, fmat& coordyRegular, \
    fcube& Vrms3dRegular, fcube& data3dtxOri, \
    fcube& baseDepthVelRegular, float dt, int ncpu)
{
    int nx1(data3dtxOri.n_rows),ny1(data3dtxOri.n_cols),\
        nx2(coordxRegular.n_rows),ny2(coordxRegular.n_cols),\
        nt(data3dtxOri.n_slices),isx,jsy;
    if(ny2!=(ny1*2-1) || nx1!=nx2)
        {cout<<"Error: Coordinate mismatch of regularized data!"<<endl;}
    data3dtxRegular.zeros(nx2,ny2,nt);
    fcube& vrmsRegular=Vrms3dRegular;
/*
omp_set_num_threads(ncpu);
#pragma omp parallel for
    for(int i=0;i<nx2;i++){
    for(int j=0;j<ny2;j++){
        float td=2.0*seaDepth(i,j)/velWater;
    for(int k=0;k<nt;k++){
        if(k*dt<=td){
            vrmsRegular(i,j,k)=velWater;
        }else{
            vrmsRegular(i,j,k)=velWater+(k*dt-td)*velDt;
        }
    }}}*/
    getSourceIndes(isx,jsy,coordxRegular,coordyRegular);
    data3dtxRegular.col(0)=data3dtxOri.col(0);
    for(int j=1;j<ny1;j++){
        data3dtxRegular.col(2*j)=data3dtxOri.col(j);
    }
omp_set_num_threads(ncpu);
#pragma omp parallel for
    for(int j=1;j<ny1;j++){
        int jInter=2*j-1;
        fmat dataTrace(nt,1);
        for(int i=0;i<nx1;i++){
            float dtBase=0.5*(baseDepthVelRegular(i,jInter+1,0)\
                -baseDepthVelRegular(i,jInter-1,0))/(baseDepthVelRegular(i,jInter,1));
            int ixMid=round(float(i+isx)/2.0);
            int jyMid=round(float(jInter+jsy)/2.0);
            float offset=(coordxRegular(i,jInter)*coordxRegular(i,jInter)\
                +coordyRegular(i,jInter)*coordyRegular(i,jInter));
            float offset1=(coordxRegular(i,jInter-1)*coordxRegular(i,jInter-1)\
                +coordyRegular(i,jInter-1)*coordyRegular(i,jInter-1));
            float offset2=(coordxRegular(i,jInter+1)*coordxRegular(i,jInter+1)\
                +coordyRegular(i,jInter+1)*coordyRegular(i,jInter+1));
            dataTrace.fill(0.0);
            for(int k=-nt;k<nt;k++){
                int kAnti=max(k,1);
                float kSign=sqrt(float(k*k+1e-8));
                float t0=k*kSign*dt*dt;
                float t=(t0+offset/vrmsRegular(ixMid,jyMid,kAnti)\
                    /vrmsRegular(ixMid,jyMid,kAnti));
                if(t<1e-8f){continue;}
                else{t=sqrt(t);}
                float t1=(dtBase)+(t0+offset1/vrmsRegular(ixMid,jyMid,kAnti)\
                    /vrmsRegular(ixMid,jyMid,kAnti));
                if(t1<1e-8f){continue;}
                else{t1=sqrt(t1);}
                float t2=(-dtBase)+(t0+offset2/vrmsRegular(ixMid,jyMid,kAnti)\
                    /vrmsRegular(ixMid,jyMid,kAnti));
                if(t2<1e-8f){continue;}
                else{t2=sqrt(t2);}
                float kt(t/dt),kt1(t1/dt),kt2(t2/dt);
                int dkt(floor(t/dt)),dkt1(floor(t1/dt)),dkt2(floor(t2/dt));
                int ukt(ceil(t/dt)),ukt1(ceil(t1/dt)),ukt2(ceil(t2/dt));
                float weight(1.0),dat1,dat2,wu,wd;
                if(ukt>=nt || ukt1>=nt || ukt2>=nt || \
                   dkt<2 || dkt1<2 || dkt2<2){continue;}
                else{
                    wd=1.0/sqrt((dkt1-kt1)*(dkt1-kt1)+1e-8);
                    wu=1.0/sqrt((ukt1-kt1)*(ukt1-kt1)+1e-8);
                    dat1=(data3dtxRegular(i,jInter-1,dkt1)*wd\
                        +data3dtxRegular(i,jInter-1,ukt1)*wu)\
                        /(wu+wd);
                    wd=1.0/sqrt((dkt2-kt2)*(dkt2-kt2)+1e-8);
                    wu=1.0/sqrt((ukt2-kt2)*(ukt2-kt2)+1e-8);
                    dat2=(data3dtxRegular(i,jInter+1,dkt2)*wd\
                        +data3dtxRegular(i,jInter+1,ukt2)*wu)\
                        /(wu+wd);
                    wd=1.0/sqrt((dkt-kt)*(dkt-kt)+1e-8);
                    wu=1.0/sqrt((ukt-kt)*(ukt-kt)+1e-8);
                    data3dtxRegular(i,jInter,ukt)+=\
                        wu*(dat1+dat2)/2.0;
                    data3dtxRegular(i,jInter,dkt)+=\
                        wd*(dat1+dat2)/2.0;
                    dataTrace(ukt,0)+=(wu);
                    dataTrace(dkt,0)+=(wd);
                }
            }
            for(int k=1;k<nt;k++){
                if(dataTrace(k,0)<0.5){
                    data3dtxRegular(i,jInter,k)=(data3dtxRegular(i,jInter-1,k)\
                        +data3dtxRegular(i,jInter+1,k))/2.0;
                }else{
                    data3dtxRegular(i,jInter,k)/=dataTrace(k,0);
                }
            }
        }
    }
}

//Interpolation is performed between rows of the data matrix
void vrmsSpaceInterData3dtxByRow(fcube& data3dtxRegular, \
    fmat& coordxRegular, fmat& coordyRegular, \
    fcube& Vrms3dRegular, fcube& data3dtxOri,  \
    fcube& baseDepthVelRegular, float dt,  int ncpu)
{
    int nx0(data3dtxOri.n_rows),ny0(data3dtxOri.n_cols),\
        nt(data3dtxOri.n_slices);
    fcube data3dtxOriSwap(ny0,nx0,nt),data3dtxRegularSwap,\
        Vrms3dSwap(Vrms3dRegular.n_cols,Vrms3dRegular.n_rows,nt),\
        baseDepthVelSwap(Vrms3dRegular.n_cols,Vrms3dRegular.n_rows,2);
    fmat coordxSwap(ny0,nx0),coordySwap(ny0,nx0);
    coordxSwap=coordxRegular.st();
    coordySwap=coordyRegular.st();
    for(int k=0;k<nt;k++){
        data3dtxOriSwap.slice(k)=data3dtxOri.slice(k).st();
        Vrms3dSwap.slice(k)=Vrms3dRegular.slice(k).st();
    }
    for(int k=0;k<2;k++){
        baseDepthVelSwap.slice(k)=baseDepthVelRegular.slice(k).st();
    }
    vrmsSpaceInterData3dtxByCol(data3dtxRegularSwap, \
        coordxSwap, coordySwap, Vrms3dSwap, data3dtxOriSwap,  \
        baseDepthVelSwap, dt,  ncpu);
    coordxSwap.clear(); coordySwap.clear();
    Vrms3dSwap.clear();data3dtxOriSwap.clear();
    baseDepthVelSwap.clear();
    int nx1(data3dtxRegularSwap.n_rows),ny1(data3dtxRegularSwap.n_cols);
    data3dtxRegular.zeros(ny1,nx1,nt);
    for(int k=0;k<nt;k++){
        data3dtxRegular.slice(k)=data3dtxRegularSwap.slice(k).st();
    }
}
void cxfcubeLinearInterpolation3dByCol(cx_fcube& data3d)
{
    int n1(data3d.n_rows),n2(data3d.n_cols),n3(data3d.n_slices),i;
    cx_fcube data3dInter;
    int nInter(n2*2-1);
    data3dInter.zeros(n1,nInter,n3);

    data3dInter.col(0)=data3d.col(0);
    for(i=1;i<n2;i++){
        int kinter=2*i;
        data3dInter.col(kinter)=data3d.col(i);
        data3dInter.col(kinter-1)=(data3dInter.col(kinter)\
            +data3dInter.col(kinter-2));
        data3dInter.col(kinter-1)=data3dInter.col(kinter-1)/2.0;
    }
    data3d=data3dInter;
}
void cxfcubeLinearInterpolation3dByCol(cx_fcube& data3d, int ncpu)
{
    int n1(data3d.n_rows),n2(data3d.n_cols),n3(data3d.n_slices),i;
    cx_fcube data3dInter;
    int nInter(n2*2-1);
    data3dInter.zeros(n1,nInter,n3);
    data3dInter.col(0)=data3d.col(0);
    for(i=1;i<n2;i++){
        int kinter=2*i;
        data3dInter.col(kinter)=data3d.col(i);
    }
    ncpu=min(ncpu,n2-1);
    ncpu=max(ncpu,1);
omp_set_num_threads(ncpu);
#pragma omp parallel for
    for(i=1;i<n2;i++){
        int kinter=2*i;
        data3dInter.col(kinter-1)=(data3dInter.col(kinter)\
            +data3dInter.col(kinter-2));
        data3dInter.col(kinter-1)=data3dInter.col(kinter-1)/2.0;
    }
    data3d=data3dInter;
}
void cxfcubeLinearInterpolation3dByRow(cx_fcube& data3d)
{
    int n1(data3d.n_rows),n2(data3d.n_cols),n3(data3d.n_slices),i;
    cx_fcube data3dInter;
    int nInter(n1*2-1);
    data3dInter.zeros(nInter,n2,n3);

    data3dInter.row(0)=data3d.row(0);
    for(i=1;i<n1;i++){
        int kinter=2*i;
        data3dInter.row(kinter)=data3d.row(i);
        data3dInter.row(kinter-1)=(data3dInter.row(kinter)\
            +data3d.row(i-1));
        data3dInter.row(kinter-1)=data3dInter.row(kinter-1)/2.0;
    }
    data3d=data3dInter;
}
void cxfcubeLinearInterpolation3dByRow(cx_fcube& data3d, int ncpu)
{
    int n1(data3d.n_rows),n2(data3d.n_cols),n3(data3d.n_slices),i;
    cx_fcube data3dInter;
    int nInter(n1*2-1);
    data3dInter.zeros(nInter,n2,n3);
    data3dInter.row(0)=data3d.row(0);
    for(i=1;i<n1;i++){
        int kinter=2*i;
        data3dInter.row(kinter)=data3d.row(i);
    }
    ncpu=min(ncpu,n1-1);
    ncpu=max(ncpu,1);
omp_set_num_threads(ncpu);
#pragma omp parallel for
    for(i=1;i<n1;i++){
        int kinter=2*i;
        data3dInter.row(kinter-1)=(data3dInter.row(kinter)\
            +data3d.row(i-1));
        data3dInter.row(kinter-1)=data3dInter.row(kinter-1)/2.0;
    }
    data3d=data3dInter;
}
void cxfcubeAntiLinearInterpolation3dByRow(cx_fcube& data3d)
{
    int n1(data3d.n_rows),n2(data3d.n_cols),n3(data3d.n_slices),kinter;
    cx_fcube data3dInter;
    int nAntiInter((n1+1)/2);
    data3dInter.zeros(nAntiInter,n2,n3);

    for(kinter=0;kinter<nAntiInter;kinter++){
        int i=2*kinter;
        data3dInter.row(kinter)=data3d.row(i);
    }
    data3d=data3dInter;
}
void cxfcubeAntiLinearInterpolation3dByCol(cx_fcube& data3d)
{
    int n1(data3d.n_rows),n2(data3d.n_cols),n3(data3d.n_slices),kinter;
    cx_fcube data3dInter;
    int nAntiInter((n2+1)/2);
    data3dInter.zeros(n1,nAntiInter,n3);

    for(kinter=0;kinter<nAntiInter;kinter++){
        int i=2*kinter;
        data3dInter.col(kinter)=data3d.col(i);
    }
    data3d=data3dInter;
}
void fcubeLinearInterpolation3dByRow(fcube& data3d, int ncpu=1)
{
    int n1(data3d.n_rows),n2(data3d.n_cols),n3(data3d.n_slices);
    fcube data3dInter;
    int nInter(n1*2-1);
    data3dInter.zeros(nInter,n2,n3);
    data3dInter.row(0)=data3d.row(0);

ncpu=min(ncpu,n1-1);
omp_set_num_threads(ncpu);
#pragma omp parallel for
    for(int i=1;i<n1;i++){
        int kinter=2*i;
        data3dInter.row(kinter)=data3d.row(i);
        data3dInter.row(kinter-1)=(data3dInter.row(kinter)\
            +data3d.row(i-1));
        data3dInter.row(kinter-1)=data3dInter.row(kinter-1)/2.0;
    }
    data3d=data3dInter;
}
void fcubeLinearInterpolation3dBySlice(fcube& data3d, int ncpu=1)
{
    int n1(data3d.n_rows),n2(data3d.n_cols),n3(data3d.n_slices),i;
    fcube data3dInter;
    int nInter(n3*2-1);
    data3dInter.zeros(n1,n2,nInter);
    data3dInter.slice(0)=data3d.slice(0);

ncpu=min(ncpu,n3-1);
omp_set_num_threads(ncpu);
#pragma omp parallel for
    for(i=1;i<n3;i++){
        int kinter=2*i;
        data3dInter.slice(kinter)=data3d.slice(i);
        data3dInter.slice(kinter-1)=(data3dInter.slice(kinter)\
            +data3d.slice(i-1));
        data3dInter.slice(kinter-1)=data3dInter.slice(kinter-1)/2.0;
    }
    data3d=data3dInter;
}
void fcubeLinearInterpolation3dByCol(fcube& data3d, int ncpu=1)
{
    int n1(data3d.n_rows),n2(data3d.n_cols),n3(data3d.n_slices),i;
    fcube data3dInter;
    int nInter(n2*2-1);
    data3dInter.zeros(n1,nInter,n3);
    data3dInter.col(0)=data3d.col(0);

ncpu=min(ncpu,n2-1);
omp_set_num_threads(ncpu);
#pragma omp parallel for
    for(i=1;i<n2;i++){
        int kinter=2*i;
        data3dInter.col(kinter)=data3d.col(i);
        data3dInter.col(kinter-1)=(data3dInter.col(kinter)\
            +data3d.col(i-1));
        data3dInter.col(kinter-1)=data3dInter.col(kinter-1)/2.0;
    }
    data3d=data3dInter;
}
void fmatAntiLinearInterpolation2dByCol(fmat& data2d)
{
    int n1(data2d.n_rows),n2(data2d.n_cols),kinter;
    fmat data2dInter;
    int nAntiInter((n2+1)/2);
    data2dInter.zeros(n1,nAntiInter);
    for(kinter=0;kinter<nAntiInter;kinter++){
        int i=2*kinter;
        data2dInter.col(kinter)=data2d.col(i);
    }
    data2d=data2dInter;
}
void fmatDownSampleLinearInterpol2dByCol(fmat& data2d)
{
    int n1(data2d.n_rows),n2(data2d.n_cols),kinter;
    fmat data2dInter;
    int nAntiInter((n2+1)/2);
    data2dInter.zeros(n1,nAntiInter);
    for(kinter=0;kinter<nAntiInter;kinter++){
        int i=2*kinter;
        int ki=min(i+1,n2-1);
        data2d.col(i)+=data2d.col(ki);
        data2dInter.col(kinter)=data2d.col(i)*0.5;
    }
    data2d=data2dInter;
}
void fmatDownSampleLinearInterpol2dByRow(fmat& data2d)
{
    int n1(data2d.n_rows),n2(data2d.n_cols),kinter;
    fmat data2dInter;
    int nAntiInter((n1+1)/2);
    data2dInter.zeros(nAntiInter,n2);
    for(kinter=0;kinter<nAntiInter;kinter++){
        int i=2*kinter;
        int ki=min(i+1,n1-1);
        data2d.row(i)+=data2d.row(ki);
        data2dInter.row(kinter)=data2d.row(i)*0.5;
    }
    data2d=data2dInter;
}
void fcubeAntiLinearInterpolation3dBySlice(fcube& data3d)
{
    int n1(data3d.n_rows),n2(data3d.n_cols),n3(data3d.n_slices),kinter;
    fcube data3dInter;
    int nAntiInter((n3+1)/2);
    data3dInter.zeros(n1,n2,nAntiInter);

    for(kinter=0;kinter<nAntiInter;kinter++){
        int i=2*kinter;
        data3dInter.slice(kinter)=data3d.slice(i);
    }
    data3d=data3dInter;
}
void fcubeAntiLinearInterpolation3dByCol(fcube& data3d)
{
    int n1(data3d.n_rows),n2(data3d.n_cols),n3(data3d.n_slices),kinter;
    fcube data3dInter;
    int nAntiInter((n2+1)/2);
    data3dInter.zeros(n1,nAntiInter,n3);

    for(kinter=0;kinter<nAntiInter;kinter++){
        int i=2*kinter;
        data3dInter.col(kinter)=data3d.col(i);
    }
    data3d=data3dInter;
}
void fmatAntiLinearInterpolation2dByRow(fmat& data2d)
{
    int n1(data2d.n_rows),n2(data2d.n_cols),kinter;
    fmat data2dInter;
    int nAntiInter((n1+1)/2);
    data2dInter.zeros(nAntiInter,n2);
    for(kinter=0;kinter<nAntiInter;kinter++){
        int i=2*kinter;
        data2dInter.row(kinter)=data2d.row(i);
    }
    data2d=data2dInter;
}
void fcubeAntiLinearInterpolation3dByRow(fcube& data3d)
{
    int n1(data3d.n_rows),n2(data3d.n_cols),n3(data3d.n_slices),kinter;
    fcube data3dInter;
    int nAntiInter((n1+1)/2);
    data3dInter.zeros(nAntiInter,n2,n3);
    for(kinter=0;kinter<nAntiInter;kinter++){
        int i=2*kinter;
        data3dInter.row(kinter)=data3d.row(i);
    }
    data3d=data3dInter;
}
void fmatLinearInterpolation2dByRow(fmat& data2d)
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
    data2d=data2dInter;
}
void fmatLinearInterpolation2dByCol(fmat& data2d)
{
    int n1(data2d.n_rows),n2(data2d.n_cols),i;
    fmat data2dInter;
    int nInter(n2*2-1);
    data2dInter.zeros(n1,nInter);
    data2dInter.col(0)=data2d.col(0);
    for(i=1;i<n2;i++){
        int kinter=2*i;
        data2dInter.col(kinter)=data2d.col(i);
        data2dInter.col(kinter-1)=(data2dInter.col(kinter)\
            +data2dInter.col(kinter-2));
        data2dInter.col(kinter-1)=data2dInter.col(kinter-1)/2.0;
    }
    data2d=data2dInter;
}
uvec sortDataNtrByOffset(fcube& data3dSort, \
    fcube data3dOri, fmat coordxNtr, fmat coordyNtr)
{
    int ntr=data3dOri.n_rows;
    fvec offset(ntr,fill::zeros);
    data3dSort.copy_size(data3dOri);
    data3dSort.fill(0.0);
    for(int k=0; k<ntr; k++){
        offset(k)=sqrt(coordxNtr(k,0)*coordxNtr(k,0) \
            +coordyNtr(k,0)*coordyNtr(k,0));
    }
    uvec indexSort = sort_index(offset);
    for(int k=0; k<ntr; k++){
        data3dSort.row(k)=data3dOri.row(indexSort(k));
    }
    return indexSort;
}
void getSourceIndes(int& sxindex,int& syindex,\
    fmat coordx, fmat coordy)
{
    int i,j;
    float offset,d(1e30);
    for(j=0;j<coordx.n_cols;j++){
    for(i=0;i<coordx.n_rows;i++){
        offset=coordx(i,j)*coordx(i,j)+coordy(i,j)*coordy(i,j);
        if(offset<d){
            d=offset;
            sxindex=i;
            syindex=j;
    }}}
}
void getPointIndex2d(int& sxindex,int& syindex,\
 float pointx, float pointy,\
 fmat coordx, fmat coordy)
{
    int i,j;
    float offset,d(1e30),dx,dy;
    for(j=0;j<coordx.n_cols;j++){
    for(i=0;i<coordx.n_rows;i++){
        dx=coordx(i,j)-pointx;
        dy=coordy(i,j)-pointy;
        offset=sqrt(dx*dx+dy*dy);
        if(offset<d){
            d=offset;
            sxindex=i;
            syindex=j;
    }}}
}
void readSuDataDirect(segyhead2 *suHeadArray1d, fcube &data3dNtr,\
    int nt, int ntr, segyhead &headp)
{
    int i,j,k,nt0(nt);
    data3dNtr.zeros(ntr,1,nt);
    for(i=0;i<ntr;i++){
        segyhead_readonetrace_tofmat(headp,headp.data);
        suHeadArray1d[i]=headp.head2;
        for(k=0;k<min(nt,headp.nz);k++){
            data3dNtr(i,0,k)=headp.data(k,0);
        }
    }
}
void readSuDataDirect(fcube &data3dNtr,\
    fmat &coordxNtr, fmat &coordyNtr,\
    const char *keyGX, const char *keyGY,\
    float spaceScaleNum,bool doEndianSwap,\
    int nt, int ntr, segyhead &headp)
{
    int i,j,k;
    coordxNtr.zeros(ntr,1);
    coordyNtr.zeros(ntr,1);
    for(i=0;i<ntr;i++){
        segyhead_readonetrace_tofmat(headp,headp.data);
        for(k=0;k<min(nt,headp.nz);k++){
            data3dNtr(i,0,k)=headp.data(k,0);
        }
        coordxNtr(i,0)=getSuHeadKey(headp.head2,\
            keyGX,doEndianSwap)/spaceScaleNum;
        coordyNtr(i,0)=getSuHeadKey(headp.head2,\
            keyGY,doEndianSwap)/spaceScaleNum;
    }
}
void matchVelLayerNtr(fcube& velLayerNtr3d, fmat &coordxNtr, fmat &coordyNtr, \
    fcube& velAll3d, fmat &coordxAll, fmat &coordyAll, int ncpu=1)
{
    int n1all(velAll3d.n_rows),n2all(velAll3d.n_cols);
    int n1ntr(velLayerNtr3d.n_rows),n2ntr(velLayerNtr3d.n_cols);
    int n3=min(int(velAll3d.n_slices),int(velLayerNtr3d.n_slices));
    velLayerNtr3d.fill(0.0);
ncpu=min(ncpu,n1ntr);
omp_set_num_threads(ncpu);
#pragma omp parallel for
    for(int ntri1=0;ntri1<n1ntr;ntri1++){
    for(int ntri2=0;ntri2<n2ntr;ntri2++){
        float minOffset=1e30;
        int n1min(0),n2min(0);
        float xntr(coordxNtr(ntri1,ntri2)),\
            yntr(coordyNtr(ntri1,ntri2));
    for(int alli1=0;alli1<n1all;alli1++){
    for(int alli2=0;alli2<n2all;alli2++){
        float offset=(coordxAll(alli1,alli2)-xntr)\
            *(coordxAll(alli1,alli2)-xntr)\
            +(coordyAll(alli1,alli2)-yntr)\
            *(coordyAll(alli1,alli2)-yntr);
        if(offset<minOffset){
            minOffset=offset;
            n1min=alli1;
            n2min=alli2;
        }
    }}
        for(int k=0;k<n3;k++){
            velLayerNtr3d(ntri1,ntri2,k)=velAll3d(n1min,n2min,k);
        }
    }}
}

void readSuDataDirect(segyhead2 *suHeadArray1d, fcube &data3dNtr,\
    fmat &coordxNtr, fmat &coordyNtr, fmat &depthNtr, \
    float &sxCoord, float &syCoord, const char *keySX, const char *keySY,\
    const char *keyGX, const char *keyGY, const char *keyDepth,\
    float dtUnitTrans, float spaceScaleNum, bool doEndianSwap,\
    int nt, int ntr, segyhead &headp)
{
    int i,j,k,nt0(nt);
    for(i=0;i<ntr;i++){
        segyhead_readonetrace_tofmat(headp,headp.data);
        suHeadArray1d[i]=headp.head2;
        for(k=0;k<min(nt,headp.nz);k++){
            data3dNtr(i,0,k)=headp.data(k,0);
        }
    }
    for(i=0;i<ntr;i++){
        coordxNtr(i,0)=getSuHeadKey(suHeadArray1d[i],\
            keyGX,doEndianSwap)/spaceScaleNum;
        coordyNtr(i,0)=getSuHeadKey(suHeadArray1d[i],\
            keyGY,doEndianSwap)/spaceScaleNum;
        depthNtr(i,0)=getSuHeadKey(suHeadArray1d[i],keyDepth,\
            doEndianSwap)/spaceScaleNum;
    }
    sxCoord=getSuHeadKey(suHeadArray1d[1],\
        keySX,doEndianSwap)/spaceScaleNum;
    syCoord=getSuHeadKey(suHeadArray1d[1],\
        keySY,doEndianSwap)/spaceScaleNum;
}
void readSwapDataDirect(segyhead2 *suHeadArray1d, fcube &data3dNtr,\
    int nt, int ntr, segyhead &headp)
{
    int k,i;
    float a;
    for(i=0;i<ntr;i++){
        headp.infile.read((char *)(&headp.head2), sizeof(headp.head2));
        suHeadArray1d[i]=headp.head2;
        for(k=0;k<nt;k++){
            headp.infile.read((char *)&a, 4); 
            data3dNtr(i,0,k)=a;
        }
    }
}
int getSwapFileNtr(segyhead &head, int nt)
{
    int ntr=0;
    while(1){
        head.infile.seekg(sizeof(head.head2),ios::cur);
        head.infile.seekg(nt*sizeof(float),ios::cur);    
        ntr++;
        if(ifstreamFloatEndOfFile(head.infile))
        {break;}
    }
    return ntr;
}
int getNtr(segyhead &head, \
    int nt, const char *key)
{
    bool doEndianSwap(false);
    if(head.endian=='b'){doEndianSwap=true;}
    float num1,num2;
    segyhead_readonetrace_tofmat(head,head.data);
    num1=getSuHeadKey(head.head2,key,doEndianSwap);
    int ntr=1;
    while(1){
        head.infile.read((char *)(&head.head2), sizeof(head.head2));
        num2=getSuHeadKey(head.head2,key,doEndianSwap);
        if(abs(num2-num1)<0.001){
            ntr++;
            head.infile.seekg(nt*sizeof(float),ios::cur);    
            if(ifstreamFloatEndOfFile(head.infile))
            {break;}
        }
        else{
            head.infile.seekg(-240,ios::cur);
            break;
        }
    }
    return ntr;
}
void catSudata(const char *outName, \
    const char *inName, int fileNum ,int idNum, \
    int fileGap, int fileBeg, int nt,\
    bool doEndianSwap=false, bool doIEEEtoIBM=false);
void catSudata(const char *outName, \
    const char *inName, int fileNum ,int idNum, \
    int fileGap, int fileBeg, int nt, \
    bool doEndianSwap, bool doIEEEtoIBM)
{
    segyhead head, headswap;
    ofstream outf;
    segyhead2 *suHeadArray1d;
    float *float_p;
    int *int_p;
    float IBMFloatBytes;
    float_p=(float *)(&IBMFloatBytes);
    int_p=(int *)(&IBMFloatBytes);
    suHeadArray1d=nullptr;
    fcube data3d;
    outf.open(outName,ios::app);
if(outf){
    for(int kfile=fileBeg;kfile<fileNum;kfile=kfile+1+fileGap){
        head.filename[0]='\0';
        strcat(head.filename,inName);
        strcat(head.filename,numtostr(kfile,idNum));
        headswap.filename[0]='\0';
        strcat(headswap.filename,head.filename);
        segyhead_open(head,true);
        segyhead_open(headswap,true);
        if(!head.infile || !headswap.infile){continue;}
        int ntr=getSwapFileNtr(headswap,nt);
        data3d.zeros(ntr,1,nt);
        suHeadArray1d=new segyhead2[ntr];
        readSwapDataDirect(suHeadArray1d, data3d,\
            nt, ntr, head);
        head.data.zeros(nt,1);
    cout<<"  Cat File Info: "<<head.filename\
        <<"; (ntr,nt) = ("<<ntr<<","<<nt<<")"<<endl;
        for(int i=0;i<data3d.n_rows;i++){
            outf.write((char *)(&suHeadArray1d[i]), sizeof(head.head2));
            for(int k=0;k<nt;k++){
                if(doIEEEtoIBM){
                    IBMFloatBytes=data3d(i,0,k);
                    float_to_ibm(int_p,int_p,1,1);
                    data3d(i,0,k)=*float_p;
                }
                if(doEndianSwap){
                    head.data(k,0)=getendianchange(float(data3d(i,0,k)));
                }
                else{head.data(k,0)=data3d(i,0,k);}
            }
            datawrite(head.data,nt,1,outf);
        }
        head.infile.close();
        headswap.infile.close();
        delete [] suHeadArray1d;
        suHeadArray1d=nullptr;
    }
    outf.close();
}
}
void writeSudata3d(fcube& data3d ,ofstream& outf, \
    segyhead2 *suHeadArray1d , int nt, \
    bool doEndianSwap, bool doIEEEtoIBM)
{
    fmat datahead(nt,1);
    segyhead head;
    float *float_p;
    int *int_p;
    float IBMFloatBytes;
    float_p=(float *)(&IBMFloatBytes);
    int_p=(int *)(&IBMFloatBytes);
    for(int i=0;i<data3d.n_rows;i++){
        outf.write((char *)(&suHeadArray1d[i]), sizeof(head.head2));
        for(int k=0;k<nt;k++){
            datahead(k,0)=data3d(i,0,k);
            if(doIEEEtoIBM){
                IBMFloatBytes=datahead(k,0);
                float_to_ibm(int_p,int_p,1,1);
                datahead(k,0)=*float_p;
            }
            if(doEndianSwap){
                datahead(k,0)=getendianchange(float(datahead(k,0)));
            }
            outf.write((char*)&datahead(k,0),sizeof(float));
        }
    }
}
void getBestCoord(fmat &coordxSort, fmat &coordySort, \
    const fmat &coordx, const fmat &coordy, \
    int nx, int ny, float dx, float dy)
{
    float sxCoordmin=coordx.min();
    float syCoordmin=coordy.min();
    float sxCoordmax=coordx.max();
    float syCoordmax=coordy.max();
    float sxSortmax=(sxCoordmin+nx*dx-dx);
    float sySortmax=(syCoordmin+ny*dy-dy);
    float ibeg=((sxSortmax-sxCoordmax)/2.0);
    float jbeg=((sySortmax-syCoordmax)/2.0);
    ibeg=max(ibeg,0.0f);jbeg=max(jbeg,0.0f);
//////////////////////////////////////////////////////////////////
    int k,i,j,ntr(coordx.n_rows);
    for(j=0;j<ny;j++){
    for(i=0;i<nx;i++){
        coordxSort(i,j)=sxCoordmin+(i)*dx-ibeg-dx;
        coordySort(i,j)=syCoordmin+(j)*dy-jbeg-dy;
    }}
    fmat coordxMoid,coordyMoid;
    coordxMoid=coordxSort;
    coordyMoid=coordySort;
    float coordErr,errmin(1e30);
    for(k=0;k<8*dx;k++){
        //if(coordxMoid.max()>sxCoordmax){break;}
        coordxMoid+=0.25;
        float coord0=coordxMoid.min();
        coordErr=0;
        j=0;
        for(i=0;i<ntr;i++){
            float coordMap=(coordx(i,j)-coord0)/dx;
            int ix=round(coordMap);
            ix=min(ix,nx-1);ix=max(ix,0);
            coordErr+=abs(ix-coordMap);
        }
        if(coordErr<errmin){
            errmin=coordErr;
            coordxSort=coordxMoid;
        }
    }
    errmin=(1e30);
    for(k=0;k<8*dy;k++){
        //if(coordyMoid.max()>syCoordmax){break;}
        coordyMoid+=0.25;
        float coord0=coordyMoid.min();
        coordErr=0;
        j=0;
        for(i=0;i<ntr;i++){
            float coordMap=(coordy(i,j)-coord0)/dy;
            int jy=round(coordMap);
            jy=min(jy,ny-1);jy=max(jy,0);
            coordErr+=abs(jy-coordMap);
        }
        if(coordErr<errmin){
            //cout<<k<<",";
            errmin=coordErr;
            coordySort=coordyMoid;
        }
    }
}
fmat sortNtrDataByCoord(fcube &data3dSort,\
    fcube &Vrms3dSort, fmat& coordxOut, fmat& coordyOut, \
    fcube &data3dOrig, fcube &Vrms3dOrig, fmat coordx, fmat coordy,\
    int ntr, int n1, int n2, int nt, float d1, float d2, float dt,int ncpu=1);
fmat sortNtrDataByCoord(fcube &data3dSort,\
    fcube &Vrms3dSort, fmat& coordxOut, fmat& coordyOut, \
    fcube &data3dOrig, fcube &Vrms3dOrig, fmat coordx, fmat coordy,\
    int ntr, int n1, int n2, int nt, float d1, float d2, float dt,int ncpu)
{
    int nx(n1),ny(n2);
    float dx(d1),dy(d2);
    fmat coordxSort(nx,ny,fill::zeros),coordySort(nx,ny,fill::zeros);
    fmat foldSort(nx,ny,fill::zeros),offsetSort(nx,ny,fill::zeros);
    float dxy=min(dx,dy);
    coordxOut=coordxSort;
    coordyOut=coordySort;
////////////////data sort by sx-sy///////////////////////////////////
    getBestCoord(coordxSort,coordySort,\
        coordx,coordy,nx,ny,dx,dy);
    float sxCoordmin=coordxSort.min();
    float syCoordmin=coordySort.min();
    float sxCoordmax=coordxSort.max();
    float syCoordmax=coordySort.max();
///////////////////////////////////////////////////////////////////////
    foldSort.fill(0.0);
    offsetSort.fill(0.0);
    for(int i=0;i<ntr;i++){
        int j=0;
        int ix=round((coordx(i,j)-sxCoordmin)/dx);
        int jy=round((coordy(i,j)-syCoordmin)/dy);
        ix=min(ix,nx-1),ix=max(ix,0);
        jy=min(jy,ny-1),jy=max(jy,0);
        if(foldSort(ix,jy)<0.5){
            coordxOut(ix,jy)=coordx(i,j);
            coordyOut(ix,jy)=coordy(i,j);
            for(int k=0;k<nt;k++){
                data3dSort(ix,jy,k)=data3dOrig(i,j,k);
                Vrms3dSort(ix,jy,k)=Vrms3dOrig(i,j,k);
            }
            offsetSort(ix,jy)=(coordx(i,j)-coordxSort(ix,jy))\
                *(coordx(i,j)-coordxSort(ix,jy))\
                +(coordy(i,j)-coordySort(ix,jy))\
                *(coordy(i,j)-coordySort(ix,jy));
            foldSort(ix,jy)=1;
        }else{
            float offset=(coordx(i,j)-coordxSort(ix,jy))\
                *(coordx(i,j)-coordxSort(ix,jy))\
                +(coordy(i,j)-coordySort(ix,jy))\
                *(coordy(i,j)-coordySort(ix,jy));
            if(offset<offsetSort(ix,jy)){
                coordxOut(ix,jy)=coordx(i,j);
                coordyOut(ix,jy)=coordy(i,j);
                for(int k=0;k<nt;k++){
                    data3dSort(ix,jy,k)=data3dOrig(i,j,k);
                    Vrms3dSort(ix,jy,k)=Vrms3dOrig(i,j,k);
                }
                offsetSort(ix,jy)=offset;
            }
        }
    }

ncpu=min(ncpu,nx);
omp_set_num_threads(ncpu);
#pragma omp parallel for
    for(int i=0;i<nx;i++){
    for(int j=0;j<ny;j++){
    if(foldSort(i,j)>0.5){continue;}
    else{
        float d(1e30);
        int jy(0),kix(-1);
        for(int ix=0;ix<ntr;ix++){
            float offset=(coordx(ix,jy)-coordxSort(i,j))\
                *(coordx(ix,jy)-coordxSort(i,j))\
                +(coordy(ix,jy)-coordySort(i,j))\
                *(coordy(ix,jy)-coordySort(i,j));
            if(offset<d){
                d=offset;
                kix=ix;
            }
        }
        if(kix>=0){
            coordxOut(i,j)=coordx(kix,jy);
            coordyOut(i,j)=coordy(kix,jy);
            for(int k=0;k<nt;k++){
                data3dSort(i,j,k)=data3dOrig(kix,jy,k);
                Vrms3dSort(i,j,k)=Vrms3dOrig(kix,jy,k);
            }
            if(sqrt(d)<=dxy){foldSort(i,j)=1.0;}
            else{foldSort(i,j)=-1.0;}
        }
    }}}

/**/
    fcube& vrms3d=Vrms3dSort;
    int isx,jsy;
    getSourceIndes(isx,jsy,coordxOut,coordyOut);
omp_set_num_threads(ncpu);
#pragma omp parallel for
    for(int i=0;i<nx;i++){
    for(int j=0;j<ny;j++){
    //if(foldSort(i,j)<0.0){
        fmat dataTrace(nt,1),dataWeight(nt,1);
        dataTrace.fill(0.0);
        dataWeight.fill(0.0);
        int ixMid=round(float(i+isx)/2.0);
        int jyMid=round(float(j+jsy)/2.0);
        float offset1=(coordxOut(i,j)*coordxOut(i,j)\
            +coordyOut(i,j)*coordyOut(i,j));
        float offset=(coordxSort(i,j)*coordxSort(i,j)\
            +coordySort(i,j)*coordySort(i,j));
        for(int k=-nt;k<nt;k++){
            int kAnti=max(k,1);
            float kSign=sqrt(float(k*k+1e-8));
            float t=(k*kSign*dt*dt+offset/\
                vrms3d(ixMid,jyMid,kAnti)/vrms3d(ixMid,jyMid,kAnti));
            if(t<1e-8f){continue;}
            else{t=sqrt(t);}
            float t1=(k*kSign*dt*dt+offset1/\
                vrms3d(ixMid,jyMid,kAnti)/vrms3d(ixMid,jyMid,kAnti));
            if(t1<1e-8f){continue;}
            else{t1=sqrt(t1);}
            float kt(t/dt),kt1(t1/dt);
            int dkt(floor(t/dt)),dkt1(floor(t1/dt));
            int ukt(ceil(t/dt)),ukt1(ceil(t1/dt));
            float weight(1.0),dat1,wu,wd;
            if(ukt>=nt || ukt1>=nt || dkt<2 || dkt1<2){continue;}
            else{
                wd=1.0/sqrt((dkt1-kt1)*(dkt1-kt1)+1e-8);
                wu=1.0/sqrt((ukt1-kt1)*(ukt1-kt1)+1e-8);
                dat1=(data3dSort(i,j,dkt1)*wd\
                    +data3dSort(i,j,ukt1)*wu)\
                    /(wu+wd);
                wd=1.0/sqrt((dkt-kt)*(dkt-kt)+1e-8);
                wu=1.0/sqrt((ukt-kt)*(ukt-kt)+1e-8);
                dataTrace(ukt,0)+=wu*(dat1);
                dataTrace(dkt,0)+=wd*(dat1);
                dataWeight(ukt,0)+=(wu);
                dataWeight(dkt,0)+=(wd);
            }
        }
        coordxOut(i,j)=coordxSort(i,j);
        coordyOut(i,j)=coordySort(i,j);
        for(int k=1;k<nt;k++){
            if(dataWeight(k,0)<0.5){
                data3dSort(i,j,k)=0.0;
            }else{
                data3dSort(i,j,k)=dataTrace(k,0)/dataWeight(k,0);
            }
        }
    }}
    return foldSort;
}

//Compensate for near offset missing data;
//Using the 1d RMS-Velocity model to estimate the missing data;
void dataMatColExtra(fcube & data3dtx, fcube & vel3d,\
    fmat& coordx, fmat& coordy, fcube& baseDepthVel, \
    float dx, float dt, int NumCoordxExtra, \
    fcube &Vrms3dOrig, fmat& mulDepth)
{
    int nx(coordx.n_rows),ny(coordx.n_cols),nz(vel3d.n_slices),\
        nxExtra(nx+NumCoordxExtra),nt(data3dtx.n_slices);
    fmat vrms2d(nt,ny);
    fmat coordxExtra(nxExtra,ny);
    fmat coordyExtra(nxExtra,ny);
    fmat mulDepthExtra(nxExtra,ny);
    fcube baseDepthVelExtra(nxExtra,ny,2);
    fcube data3dtxExtra(nxExtra,ny,data3dtx.n_slices);
    fcube vel3dExtra(nxExtra,ny,vel3d.n_slices);
    fcube Vrms3dExtra(nxExtra,ny,Vrms3dOrig.n_slices);
    if(abs(coordx(nx-1,0))<abs(coordx(0,0))){
        coordyExtra(span(0,nx-1),span::all)=coordy;
        coordxExtra(span(0,nx-1),span::all)=coordx;
        mulDepthExtra(span(0,nx-1),span::all)=mulDepth;
        baseDepthVelExtra(span(0,nx-1),span::all,span::all)=baseDepthVel;
        data3dtxExtra(span(0,nx-1),span::all,span::all)=data3dtx;
        Vrms3dExtra(span(0,nx-1),span::all,span::all)=Vrms3dOrig;
        vel3dExtra(span(0,nx-1),span::all,span::all)=vel3d;
        int isx,jsy;
        fmat dataTrace(nt,1);
        for(int k=nx;k<nxExtra;k++){
            vel3dExtra.row(k)=vel3d.row(nx-1);
            Vrms3dExtra.row(k)=Vrms3dOrig.row(nx-1);
            mulDepthExtra.row(k)=mulDepthExtra.row(nx-1);
            coordyExtra.row(k)=coordyExtra.row(nx-1);
            baseDepthVelExtra.row(k)=baseDepthVel.row(nx-1);
            coordxExtra.row(k)=coordxExtra.row(k-1)+dx;
        }
        getSourceIndes(isx,jsy,coordxExtra,coordyExtra);
        for(int i=0;i<ny;i++){
        for(int j=0;j<nt;j++){
            vrms2d(j,i)=Vrms3dOrig(nx-1,i,j);
        }}
        for(int k=nx;k<nxExtra;k++){
        for(int i=0;i<ny;i++){
            dataTrace.fill(0.0);
            int kxMid=round(float(k+isx)/2.0);
            int iyMid=round(float(i+jsy)/2.0);
            float offset=(coordxExtra(k,i)*coordxExtra(k,i)\
                +coordyExtra(k,i)*coordyExtra(k,i));
            float offset1=(coordxExtra(nx-1,i)*coordxExtra(nx-1,i)\
                +coordyExtra(nx-1,i)*coordyExtra(nx-1,i));
            for(int j=1;j<nt;j++){
                float t=sqrt(j*j*dt*dt+offset/\
                    vrms2d(j,iyMid)/vrms2d(j,iyMid));
                float t1=sqrt(j*j*dt*dt+offset1/\
                    vrms2d(j,iyMid)/vrms2d(j,iyMid));
                float kt(t/dt),kt1(t1/dt);
                int dkt(floor(t/dt)),dkt1(floor(t1/dt));
                int ukt(ceil(t/dt)),ukt1(ceil(t1/dt));
                float weight(1.0),dat1,wu,wd;
                if(ukt>=nt || ukt1>=nt){continue;}
                else{
                    wd=1.0/sqrt((dkt1-kt1)*(dkt1-kt1)+1e-8);
                    wu=1.0/sqrt((ukt1-kt1)*(ukt1-kt1)+1e-8);
                    dat1=(data3dtxExtra(nx-1,i,dkt1)*wd\
                        +data3dtxExtra(nx-1,i,ukt1)*wu)\
                        /(wu+wd);
                    wd=1.0/sqrt((dkt-kt)*(dkt-kt)+1e-8);
                    wu=1.0/sqrt((ukt-kt)*(ukt-kt)+1e-8);
                    data3dtxExtra(k,i,ukt)+=wu*(dat1);
                    data3dtxExtra(k,i,dkt)+=wd*(dat1);
                    dataTrace(ukt,0)+=(wu);
                    dataTrace(dkt,0)+=(wd);
                }
            }
            for(int j=1;j<nt;j++){
                if(dataTrace(j,0)<0.5){
                    //data3dtxExtra(k,i,j)=data3dtxExtra(nx-1,i,j);
                    data3dtxExtra(k,i,j)=0.0;
                }else{
                    data3dtxExtra(k,i,j)/=dataTrace(j,0);
                }
            }
        }
        }
    }else{
        coordyExtra(span(NumCoordxExtra,nxExtra-1),span::all)=coordy;
        coordxExtra(span(NumCoordxExtra,nxExtra-1),span::all)=coordx;
        baseDepthVelExtra(span(NumCoordxExtra,nxExtra-1),span::all,span::all)=baseDepthVel;
        data3dtxExtra(span(NumCoordxExtra,nxExtra-1),span::all,span::all)=data3dtx;
        Vrms3dExtra(span(NumCoordxExtra,nxExtra-1),span::all,span::all)=Vrms3dOrig;
        vel3dExtra(span(NumCoordxExtra,nxExtra-1),span::all,span::all)=vel3d;
        mulDepthExtra(span(NumCoordxExtra,nxExtra-1),span::all)=mulDepth;
        int isx,jsy;
        fmat dataTrace(nt,1);
        for(int k=NumCoordxExtra-1;k>=0;k--){
            vel3dExtra.row(k)=vel3d.row(NumCoordxExtra);
            Vrms3dExtra.row(k)=Vrms3dOrig.row(NumCoordxExtra);
            mulDepthExtra.row(k)=mulDepthExtra.row(NumCoordxExtra);
            coordyExtra.row(k)=coordyExtra.row(NumCoordxExtra);
            baseDepthVelExtra.row(k)=baseDepthVel.row(NumCoordxExtra);
            coordxExtra.row(k)=coordxExtra.row(k+1)-dx;
        }
        getSourceIndes(isx,jsy,coordxExtra,coordyExtra);
        for(int i=0;i<ny;i++){
        for(int j=0;j<nt;j++){
            vrms2d(j,i)=Vrms3dOrig(NumCoordxExtra,i,j);
        }}
        for(int k=NumCoordxExtra-1;k>=0;k--){
        for(int i=0;i<ny;i++){
            dataTrace.fill(0.0);
            int kxMid=round(float(k+isx)/2.0);
            int iyMid=round(float(i+jsy)/2.0);
            float offset=(coordxExtra(k,i)*coordxExtra(k,i)\
                +coordyExtra(k,i)*coordyExtra(k,i));
            float offset1=(coordxExtra(NumCoordxExtra,i)*coordxExtra(NumCoordxExtra,i)\
                +coordyExtra(NumCoordxExtra,i)*coordyExtra(NumCoordxExtra,i));
            for(int j=1;j<nt;j++){
                float t=sqrt(j*j*dt*dt+offset/\
                    vrms2d(j,iyMid)/vrms2d(j,iyMid));
                float t1=sqrt(j*j*dt*dt+offset1/\
                    vrms2d(j,iyMid)/vrms2d(j,iyMid));
                float kt(t/dt),kt1(t1/dt);
                int dkt(floor(t/dt)),dkt1(floor(t1/dt));
                int ukt(ceil(t/dt)),ukt1(ceil(t1/dt));
                float weight(1.0),dat1,wu,wd;
                if(ukt>=nt || ukt1>=nt){continue;}
                else{
                    wd=1.0/sqrt((dkt1-kt1)*(dkt1-kt1)+1e-8);
                    wu=1.0/sqrt((ukt1-kt1)*(ukt1-kt1)+1e-8);
                    dat1=(data3dtxExtra(NumCoordxExtra,i,dkt1)*wd\
                        +data3dtxExtra(NumCoordxExtra,i,ukt1)*wu)\
                        /(wu+wd);
                    wd=1.0/sqrt((dkt-kt)*(dkt-kt)+1e-8);
                    wu=1.0/sqrt((ukt-kt)*(ukt-kt)+1e-8);
                    data3dtxExtra(k,i,ukt)+=wu*(dat1);
                    data3dtxExtra(k,i,dkt)+=wd*(dat1);
                    dataTrace(ukt,0)+=(wu);
                    dataTrace(dkt,0)+=(wd);
                }
            }
            for(int j=1;j<nt;j++){
                if(dataTrace(j,0)<0.5){
                    //data3dtxExtra(k,i,j)=data3dtxExtra(NumCoordxExtra,i,j);
                    data3dtxExtra(k,i,j)=0.0;
                }else{
                    data3dtxExtra(k,i,j)/=dataTrace(j,0);
                }
            }
        }
        }
    }
    coordx=coordxExtra;
    coordy=coordyExtra;
    baseDepthVel=baseDepthVelExtra;
    data3dtx=data3dtxExtra;
    Vrms3dOrig=Vrms3dExtra;
    vel3d=vel3dExtra;
    mulDepth=mulDepthExtra;
}

fmat cutDataBySlope(fcube &data3d, fmat &coordx, fmat &coordy,\
    float dt, float cutSlopeMin, float cutSlopeMax, float delayTime)
{
    int nx(data3d.n_rows),ny(data3d.n_cols),nt(data3d.n_slices),\
        delayNt(delayTime/dt),i,j,k;
    fmat coordfold(nx,ny);
    coordfold.fill(1.0);
    for(j=0;j<ny;j++){
    for(i=0;i<nx;i++){
        float offset=sqrt(coordx(i,j)*coordx(i,j)\
            +coordy(i,j)*coordy(i,j)+1e-6);
        int nCutMin(floor(offset/dt/cutSlopeMax)+delayNt);
        int nCutMax(floor(offset/dt/cutSlopeMin)+1+delayNt);
        for(k=max(nCutMin,1);k<min(nCutMax,nt);k++){
            //data3dCut(i,j,k)=data3dOrig(i,j,k);
            float nslope=(float(k-delayNt)/offset)*dt;
            nslope=1.0/(nslope+1e-9);
            nslope=min(nslope,float(cutSlopeMax));
            nslope=max(nslope,float(cutSlopeMin));
            float w_black=Blackman(cutSlopeMax-nslope,\
                abs(cutSlopeMax-cutSlopeMin));
            data3d(i,j,k)*=w_black;
        }
        for(k=0;k<min(nt,nCutMin);k++){
            data3d(i,j,k)=0;
        }
        if(nCutMin>=nt)
            coordfold(i,j)=0;
    }}
    return coordfold;
}
fmat cutDataBySlopeWin(fcube &data3d, fmat &coordx, fmat &coordy,\
    float dt, float cutSlopeVelMax, float delayTime,int nWin)
{
    int nx(data3d.n_rows),ny(data3d.n_cols),nt(data3d.n_slices),\
        delayNt(delayTime/dt),i,j,k;
    fmat coordfold(nx,ny);
    coordfold.fill(1.0);
    for(j=0;j<ny;j++){
    for(i=0;i<nx;i++){
        float offset=sqrt(coordx(i,j)*coordx(i,j)\
            +coordy(i,j)*coordy(i,j)+1e-6);
        int nCutMin(floor(offset/dt/cutSlopeVelMax)+1+delayNt);
        int nk=nWin;
        for(k=min(nCutMin+nWin,nt-1);k>=max(nCutMin,0);k--){
            //data3dCut(i,j,k)=data3dOrig(i,j,k);
            float w_black=Blackman(nk,nWin);
            data3d(i,j,k)*=w_black;
            nk--;
        }
        for(k=0;k<min(nt,nCutMin);k++){
            data3d(i,j,k)=0;
        }
        if(nCutMin>=nt)
            coordfold(i,j)=0;
    }}
    return coordfold;
}
double rotateCoordx(double coordx, double coordy, double theta)
{
    double xRota=cos(theta)*coordx-sin(theta)*coordy;
    return xRota;
}
double rotateCoordy(double coordx, double coordy, double theta)
{
    double yRota=sin(theta)*coordx+cos(theta)*coordy;
    return yRota;
}

void rotateCoord2d(fmat& coordx2d, fmat& coordy2d, \
    fmat coordx2dOri, fmat coordy2dOri, float theta)
{
    int n1(coordx2dOri.n_rows),n2(coordx2dOri.n_cols);
    float xRota1xCos=cos(theta);
    float xRota2ySin=-sin(theta);
    float yRota1xSin=sin(theta);
    float yRota2yCos=cos(theta);
    coordx2d.copy_size(coordx2dOri);
    coordy2d.copy_size(coordy2dOri);
    int k1,k2;
    for(k2=0;k2<n2;k2++){
    for(k1=0;k1<n1;k1++){
        coordx2d(k1,k2)=xRota1xCos*coordx2dOri(k1,k2)+
            xRota2ySin*coordy2dOri(k1,k2);
        coordy2d(k1,k2)=yRota1xSin*coordx2dOri(k1,k2)+
            yRota2yCos*coordy2dOri(k1,k2);
    }}
}
/*
bool ifstreamFloatEndOfFile(ifstream & infile)
{
    float readtest;
    bool endOfFile(false);
    infile.read((char *)(&readtest), sizeof(readtest));
    if(!infile.is_open()){endOfFile=true;cout<<"File has been closed!"<<endl;}
    else if(infile.eof()){endOfFile=true;}
    else{infile.seekg(-sizeof(readtest),ios::cur);}
    return endOfFile;
}
*/
void tx2fx3dCompress(cx_fcube &data3dfx,fcube &data3dtx,int fn,int ncpu=1)
{
    int n1(data3dtx.n_rows),n2(data3dtx.n_cols),n3(data3dtx.n_slices);
    data3dfx.zeros(n1,n2,fn);
    int nf0=min(n3,fn);

omp_set_num_threads(ncpu);
#pragma omp parallel for
    for(int i=0;i<n1;i++){
        fmat data2dtx=data3dtx.row(i);
        cx_fmat data2dfx;
        data2dfx.copy_size(data2dtx);
        for(int j=0;j<n2;j++){
            data2dfx.row(j)=fft(data2dtx.row(j));
        }
        data3dfx(span(i,i),span::all,span(0,nf0-1))\
            =data2dfx(span::all,span(0,nf0-1));
    }
}
float getCoordRotateTheta(fmat coordx, fmat coordy, float theta0)
{
    fmat xNoise,yNoise;
    xNoise.randu(coordx.n_rows,coordx.n_cols);
    yNoise.randu(coordx.n_rows,coordx.n_cols);
    xNoise*=0.001;yNoise*=0.001;
    coordx+=xNoise;
    coordy+=yNoise;
    double theta(theta0);
    int ntr(coordx.n_rows);
    fmat x(ntr,2),h(2,2);
    double avgx=accu(coordx)/coordx.n_elem;
    double avgy=accu(coordy)/coordx.n_elem;
    coordx-=avgx;
    coordy-=avgy;
    x.col(0)=coordx.col(0);
    x.col(1)=coordy.col(0);
    h=x.t()*x;
    double a=h(0,0);
    double b=h(0,1);
    double c=h(1,1);
    double m=(a+c)/2.0;
    double p=a*c-b*b;
    if(m*m-p>=0.0){
        double lmd=m+sqrt(m*m-p);
        double x1=b/((c-lmd)*(a-lmd)-b*b);
        double x2=1.0/(b*b/(a-lmd)-(c-lmd));
        theta = -1.0*atan(x2/x1);
        theta=theta*180.0/3.1415926;
    }
    if(theta!=theta){
        cout<<"  Warning: System Rotate Theta Can't Calculate!"<<endl;
        theta=theta0;
    }
    return theta0=theta;
}
int getfftNumMWD(int num){
    int n(1);
    while(n<num){
        n*=2;
    }
    n=n/2;
    return n;
}
void datafftCompress(fcube& data3dNtr, int nt2, int ncpu=1)
{
    cx_fcube data3dfx;
    tx2fx3dCompress(data3dfx,data3dNtr, nt2, ncpu);
    for(int k=nt2/2;k<nt2;k++){
        data3dfx.slice(k).fill(0.0);
    }
    data3dNtr.copy_size(data3dfx);
    fx2tx_3d_thread(data3dNtr,data3dfx,ncpu);
    data3dNtr*=2.0;
}
float dataNtrfftTrans(fcube& data3dNtr, float dt, int ncpu=1)
{
    fcube data3d;
    int nt=data3dNtr.n_slices;
    int nt2,det(round(0.05*nt));
    nt2=getfftNumMWD(nt);
    int nt3=nt2*2;
    if(nt==(nt3)){return dt;}
    if((nt-nt2)<det && nt>nt2){
        data3d.zeros(data3dNtr.n_rows,data3dNtr.n_cols,nt2);
        data3d(span::all,span::all,span(0,nt2-1))\
            =data3dNtr(span::all,span::all,span(0,nt2-1));
        data3dNtr=data3d;
    }else if((nt3-nt)<det && nt3>nt){
        data3d.zeros(data3dNtr.n_rows,data3dNtr.n_cols,nt3);
        data3d(span::all,span::all,span(0,nt-1))\
            =data3dNtr(span::all,span::all,span(0,nt-1));
        data3dNtr=data3d;
    }else if(nt3>nt){
        cx_fcube data3dfx2,data3dfx;
        data3dfx.copy_size(data3dNtr);
        tx2fx_3d_thread(data3dfx,data3dNtr, ncpu);
        data3dfx2.zeros(data3dNtr.n_rows,data3dNtr.n_cols,nt3);
        data3dfx2(span::all,span::all,span(0,nt/2-1))\
            =data3dfx(span::all,span::all,span(0,nt/2-1));
        data3dfx.zeros(1,1,1);
        data3d.copy_size(data3dfx2);
        fx2tx_3d_thread(data3d,data3dfx2,ncpu);
        data3dNtr=data3d*2.0;
        double dt2=double(dt)*double(nt)/double(nt3);
        dt=float(dt2);
    }
    return dt;
}

fmat GetLayerVrmsRefTravelTime2d( \
    fmat coordxData, fmat coordyData, int nt, float dt, \
    fmat vp2d, fvec layerDep, float dz, float coordSx, float coordSy,\
    float delayTime=0.05, int wide=10)
{
    fmat vrms2d, t0Double2d;
    fcube velLayerOrig3d(vp2d.n_cols,1,vp2d.n_rows);
    velLayerOrig3d.col(0)=vp2d.st();
    fmat layerDepth(layerDep.n_elem,1);
    for(int i=0;i<layerDep.n_elem;i++){
        layerDepth(i,0)=layerDep(i)-dz;
    }
    getLayerVrms2d(vrms2d, t0Double2d, velLayerOrig3d, layerDepth, dz);
    int n1(coordxData.n_rows),n2(coordxData.n_cols),iMulx, jMuly;
    int dn1=round(50.0/(abs(coordxData.max()-coordxData.min())/n1));
    int dn2=round(50.0/(abs(coordyData.max()-coordyData.min())/n2));
    dn1=min(dn1,n1-1);dn2=min(dn2,n2-1);
    dn1=max(dn1,1);dn2=max(dn2,1);
    //cout<<dn1<<","<<dn2<<endl;
    fmat travelTime;
    travelTime.zeros(n1,n2);
    getPointIndex2d(iMulx, jMuly,coordSx, coordSy,\
        coordxData, coordyData);
    ivec begix(n2),endix(n2);
    for(int j=0;j<n2;j++){
        begix(j)=0;
        endix(j)=n1;
    }
    for(int j=0;j<n2;j++){
        int fj=round((jMuly+j)/2.0);
        fj=max(fj,0);fj=min(fj,n2-1);
    for(int i=begix(j);i<endix(j);i++){      
        int fi=round((iMulx+i)/2.0);
        fi=max(fi,0);fi=min(fi,n1-1);
        float offsetMulMin=(coordxData(fi,fj)-coordSx)\
            *(coordxData(fi,fj)-coordSx)\
            +(coordyData(fi,fj)-coordSy)\
            *(coordyData(fi,fj)-coordSy);
        float offsetDataMin=(coordxData(fi,fj)-coordxData(i,j))\
            *(coordxData(fi,fj)-coordxData(i,j))\
            +(coordyData(fi,fj)-coordyData(i,j))\
            *(coordyData(fi,fj)-coordyData(i,j));
        //float deepthMin=layerDepth(fi,fj);
        //float l_min=sqrt(offsetMulMin+deepthMin*deepthMin)\
            +sqrt(offsetDataMin+deepthMin*deepthMin);
        float vrms=vrms2d(fi,fj);
        float t0=t0Double2d(fi,fj);
        float l_min=0.5*sqrt(t0*t0+4.0*offsetMulMin/vrms/vrms)\
            +0.5*sqrt(t0*t0+4.0*offsetDataMin/vrms/vrms);
    for(int j1=0;j1<n2;j1=j1+dn2){
    for(int i1=begix(j1);i1<endix(j1);i1=i1+dn1){
        float d11=layerDepth(i1,j1);
        float offsetMul=(coordxData(i1,j1)-coordSx)\
            *(coordxData(i1,j1)-coordSx)\
            +(coordyData(i1,j1)-coordSy)\
            *(coordyData(i1,j1)-coordSy);
        float offsetData=(coordxData(i1,j1)-coordxData(i,j))\
            *(coordxData(i1,j1)-coordxData(i,j))\
            +(coordyData(i1,j1)-coordyData(i,j))\
            *(coordyData(i1,j1)-coordyData(i,j));
        //float l_trace=sqrt(offsetMul+d11*d11)+sqrt(offsetData+d11*d11);
        vrms=vrms2d(i1,j1);
        t0=t0Double2d(i1,j1);
        float l_trace=0.5*sqrt(t0*t0+4.0*offsetMul/vrms/vrms)\
            +0.5*sqrt(t0*t0+4.0*offsetData/vrms/vrms);
        if(l_trace<l_min){
            l_min=l_trace;
            offsetMulMin=offsetMul;
            offsetDataMin=offsetData;
            //deepthMin=d11;
            fi=i1;fj=j1;
        }
    }}
    int minix(fi),minjy(fj);
    for(int j2=max(fj-dn2,0);j2<=min(fj+dn2,n2-1);j2++){
    for(int i2=max(fi-dn1,0);i2<=min(fi+dn1,n1-1);i2++){    
        float d11=layerDepth(i2,j2);
        float offsetMul=(coordxData(i2,j2)-coordSx)\
            *(coordxData(i2,j2)-coordSx)\
            +(coordyData(i2,j2)-coordSy)\
            *(coordyData(i2,j2)-coordSy);
        float offsetData=(coordxData(i2,j2)-coordxData(i,j))\
            *(coordxData(i2,j2)-coordxData(i,j))\
            +(coordyData(i2,j2)-coordyData(i,j))\
            *(coordyData(i2,j2)-coordyData(i,j));
        //float l_trace=sqrt(offsetMul+d11*d11)+sqrt(offsetData+d11*d11);
        vrms=vrms2d(i2,j2);
        t0=t0Double2d(i2,j2);
        float l_trace=0.5*sqrt(t0*t0+4.0*offsetMul/vrms/vrms)\
            +0.5*sqrt(t0*t0+4.0*offsetData/vrms/vrms);
        if(l_trace<l_min){
            l_min=l_trace;
            offsetMulMin=offsetMul;
            offsetDataMin=offsetData;
            //deepthMin=d11;
            minix=i2;
            minjy=j2;
        }
    }}
        //travelTime(i,j)=l_min/waterVel;
        travelTime(i,j)=l_min;
    }}
    fmat timeLine(nt,n1,fill::zeros);
    for(int i=0;i<n1;i++){
    int jt=round((travelTime(i,0)+delayTime)/dt);
    if(jt<nt-wide && jt>wide){
        for(int j=jt-wide;j<jt+wide;j++){
            timeLine(j,i)=1.0;
        }
    }}
    return timeLine;
}
//////////////////////////CRMD3d Added Function/////////////////////
//////////////////fileIO.hpp//////////////////
void MakeGlobalCoordToLocalCoord(\
    fmat& localCoordx, fmat& localCoordy, \
    fmat globalCoordx, fmat globalCoordy, \
    float originCoordx, float originCoordy, \
    float theta\
)
{
    rotateCoord2d(localCoordx, localCoordy,\
        globalCoordx, globalCoordy, theta);
    localCoordx=localCoordx-originCoordx;
    localCoordy=localCoordy-originCoordy;
}

void WriteSegyHeadToExistFileAte(segyhead& head, const char *filename)
{
    ofstream outf;
    outf.open(filename,ios::in|ios::out|ios::ate|ios::binary);
    if(outf.is_open()){
        outf.seekp(0,ios::beg);
        outf.write((char *)(&head.head0), sizeof(head.head0));
        outf.write((char *)(&head.head1), sizeof(head.head1));
        outf.close();
    }
}
void ReadOneShotSegyDataAndCoord(int ntr,int nt,\
    segyhead& head, segyhead2 *suHeadArray1d,\
    fcube& data3dNtr, fmat& depthNtr,\
    fmat& coordxNtr, fmat& coordyNtr,\
    float& sxCoord, float& syCoord,\
    const char *keyAntiSx, const char *keyAntiSy,\
    const char *keyAntiGx, const char *keyAntiGy, \
    const char *keyWaterDepth, const char *keyNodeDepth,\
    float dtUnitTrans, bool spaceScaleNum,  bool doEndianSwap\
)
{
    data3dNtr.zeros(ntr,1,nt);
    depthNtr.zeros(ntr,1);
    coordxNtr.zeros(ntr,1);
    coordyNtr.zeros(ntr,1);
    readSuDataDirect(suHeadArray1d, data3dNtr,\
        coordxNtr, coordyNtr, depthNtr, \
        sxCoord, syCoord, keyAntiSx, keyAntiSy,\
        keyAntiGx, keyAntiGy, keyWaterDepth,\
        dtUnitTrans, spaceScaleNum,  doEndianSwap,\
        nt, ntr, head);
}
void ReadVelAndLayerModel(\
    fcube &layerAll3d, fcube &velAll3d, \
    fmat &velCoordxAll, fmat &velCoordyAll,\
    const char *fileinLayerDepth,\
    const char *fileinVel,\
    const char *keyGx,\
    const char *keyGy,\
    int nzVel, int nlayer, \
    float spaceUnitTrans, \
    bool doEndianSwap, bool sufile\
)
{
// Input Vel&Layer model:
    segyhead headLayerSwap;
    headLayerSwap.filename[0]='\0';
    strcat(headLayerSwap.filename,fileinVel);
    segyhead_open(headLayerSwap,sufile);
    if(headLayerSwap.infile.is_open()){
        int ntrVel=getNtr(headLayerSwap,nzVel,"ns");
        headLayerSwap.infile.close();
        layerAll3d.zeros(ntrVel,1,max(nlayer,1));
        velAll3d.zeros(ntrVel,1,max(nzVel,1));
        velCoordxAll.zeros(ntrVel,1);
        velCoordyAll.zeros(ntrVel,1);

        segyhead headVel;
        segyhead headLayer;
        headLayer.filename[0]='\0';
        headVel.filename[0]='\0';
        strcat(headLayer.filename,fileinLayerDepth);
        strcat(headVel.filename,fileinVel);

        segyhead_open(headVel,sufile);
        readSuDataDirect(velAll3d, velCoordxAll, velCoordyAll,\
            keyGx, keyGy, spaceUnitTrans,\
            doEndianSwap, nzVel, ntrVel, headVel);
        headVel.infile.close();
    
        segyhead_open(headLayer,sufile);
        readSuDataDirect(layerAll3d, velCoordxAll, velCoordyAll,\
            keyGx, keyGy, spaceUnitTrans,\
            doEndianSwap, nlayer, ntrVel, headLayer);
        headLayer.infile.close();
    }else{
        layerAll3d.zeros(1,1,max(nlayer,1));
        velAll3d.zeros(1,1,max(nzVel,1));
        velCoordxAll.zeros(1,1);
        velCoordyAll.zeros(1,1);
        cout<<"  Error: Cannot open Vel&Layer file: "<<fileinVel<<endl;
    }
}

//////////////////predict.hpp//////////////////
fcube VelGradientToVrms3d(fmat &waterDepthNtr, float VrmsGrad, \
    float waterVel, float nt, float dt, int ncpu=1)
{
    int ny(waterDepthNtr.n_cols), nx(waterDepthNtr.n_rows);
    fcube vrms3d(nx,ny,nt);
ncpu=min(ncpu,nx);
omp_set_num_threads(ncpu);
#pragma omp parallel for
    for(int i=0;i<nx;i++){
    for(int j=0;j<ny;j++){
        float td=2.0*waterDepthNtr(i,j)/waterVel;
    for(int k=0;k<nt;k++){
        if(k*dt<=td){
            vrms3d(i,j,k)=waterVel;
        }else{
            vrms3d(i,j,k)=waterVel+(k*dt-td)*VrmsGrad;
        }
    }}}
    return vrms3d;
}

fcube VelDepthToVrms3d(fcube &vel3d, int nt, float dt, float dz, \
    fcube &baseDepthVel, int ncpu=1)
{
    int nz(vel3d.n_slices),ny(vel3d.n_cols),\
        nx(vel3d.n_rows);
    int izBeg=round(baseDepthVel.slice(0).min()/dz);
    fcube t03d,vrmsDep3d;
    t03d.zeros(nx,ny,nz);
    vrmsDep3d.zeros(nx,ny,nz);
ncpu=min(ncpu,nx);
omp_set_num_threads(ncpu);
#pragma omp parallel for
    for(int ix=0;ix<nx;ix++){
    for(int iy=0;iy<ny;iy++){
        baseDepthVel(ix,iy,1)=vel3d(ix,iy,ceil(baseDepthVel(ix,iy,0)/dz)+1);
        float t0=0.0;
        float vrms=0.0;
    for(int iz=izBeg;iz<nz;iz++){
        float vi=vel3d(ix,iy,iz);
        float ti=dz/vi;
        t0+=ti;
        vrms+=(ti*vi*vi);
        vrmsDep3d(ix,iy,iz)=sqrt(vrms/t0);
        t03d(ix,iy,iz)=2.0*t0;
    }}}
    baseDepthVel+=1e-6;
    
    fcube vrms3d,t0Weight3d;
    t0Weight3d.zeros(nx,ny,nt);
    vrms3d.zeros(nx,ny,nt);
omp_set_num_threads(ncpu);
    for(int ix=0;ix<nx;ix++){
    for(int iy=0;iy<ny;iy++){
    for(int iz=izBeg;iz<nz;iz++){
        float t0max,t0min;
        if(iz==izBeg)t0min=0.0;
        else t0min=t03d(ix,iy,iz-1);
        if(iz==(nz-1))t0max=t03d(ix,iy,(nz-1));
        else t0max=t03d(ix,iy,iz+1);
        for(int it=int(t0min/dt);it<=int(t0max/dt);it++){
            float dt0=dt*it-t03d(ix,iy,iz);
            float w=1.0/(dt0*dt0+1e-8);
            vrms3d(ix,iy,it)+=w*vrmsDep3d(ix,iy,iz);
            t0Weight3d(ix,iy,it)+=w;
        }
    }}}
    t0Weight3d+=1e-6;
omp_set_num_threads(ncpu);
    for(int ix=0;ix<nx;ix++){
    for(int iy=0;iy<ny;iy++){
    for(int it=0;it<nt;it++){
        vrms3d(ix,iy,it)=vrms3d(ix,iy,it)/t0Weight3d(ix,iy,it);
    }}}
omp_set_num_threads(ncpu);
    for(int ix=0;ix<nx;ix++){
    for(int iy=0;iy<ny;iy++){
    for(int it=1;it<nt;it++){
        if(vrms3d(ix,iy,it)<1.0)vrms3d(ix,iy,it)=vrms3d(ix,iy,it-1);
    }}}
    return  vrms3d;
}
void mul3dLayerCode(\
    fcube& data3dCut, float fmin, float fmax, bool useHeadWaterDepthOnly,\
    fcube& vel3d, fcube& layerRefDepth3d, fmat& velCoordxNtr,fmat& velCoordyNtr,\
    fcube& datumDepth3d, fmat& coordxNtr,fmat& coordyNtr, \
    int ntOrig, float dtOrig, int ntr, int nx, int ny, float dx, float dy,float dz,\
    int interTimesByRow, int interTimesByCol, int dataMeshSizeModify,\
    float xLeftWinWide, float xRightWinWide,float yUpWinWide,float yDownWinWide,\
    float waterVelocity, float VrmsGrad, int NumCoordxExtra, \
    float cutSlopeMin, float cutSlopeMax, float delayTime,\
    float maxAperture, float aModify, float bModify, \
    float eccentricity, float offsetEnhance, float focusEnhance,\
    float weightMax, float weightPow, bool doBaseConjFocus, int ncpu)
{
//cout<<"Step - <Data matrixing> is running."<<endl;
    fcube VrmsForRegularization,VrmsFromDepthVel;
    float dt=dataNtrfftTrans(data3dCut,dtOrig,ncpu);
    int nt=data3dCut.n_slices;
    if(nt!=ntOrig || dt!=dtOrig){
        cout<<"  Check Swap Data Time Sampling: (nt,dt)?=("\
            <<nt<<","<<dt<<")"<<endl;
    }
    if(useHeadWaterDepthOnly){
        fmat seabaseDepthNtr=layerRefDepth3d.slice(0);
        VrmsForRegularization=VelGradientToVrms3d(seabaseDepthNtr,VrmsGrad,\
            waterVelocity,nt,dt,ncpu);
    }else{
        VrmsFromDepthVel=VelDepthToVrms3d(vel3d,nt,dt,dz,datumDepth3d,ncpu);
        VrmsForRegularization.copy_size(data3dCut);
        matchVelLayerNtr(VrmsForRegularization,coordxNtr,coordyNtr,\
            VrmsFromDepthVel,velCoordxNtr,velCoordyNtr,ncpu);
    }

    if(dataMeshSizeModify){
        nx=5+round((coordxNtr.max()-coordxNtr.min())/dx);
        ny=5+round((coordyNtr.max()-coordyNtr.min())/dy);
    }
    fmat coordxOrig(nx,ny,fill::zeros),coordyOrig(nx,ny,fill::zeros),\
        vrms2d(nx,ny,fill::zeros),t0Double2d(nx,ny,fill::zeros),\
        coordx,coordy,mulLayerDepth(nx,ny);
    fcube data3dOrig,velOrig3d,layerRefDepthOrig3d,\
        Vrms3dForInterpolation,datumDepthOrig;
    datumDepthOrig.zeros(nx,ny,2);
    data3dOrig.zeros(nx,ny,nt);
    Vrms3dForInterpolation.zeros(nx,ny,nt);
    layerRefDepthOrig3d.zeros(nx,ny,1);
    velOrig3d.zeros(nx,ny,vel3d.n_slices);
    sortNtrDataByCoord(\
        data3dOrig,Vrms3dForInterpolation,coordxOrig,coordyOrig,\
        data3dCut, VrmsForRegularization, coordxNtr, coordyNtr, \
        ntr, nx, ny, nt, dx, dy, dt, ncpu);
    VrmsForRegularization.clear();

    if(!useHeadWaterDepthOnly){
        matchVelLayerNtr(Vrms3dForInterpolation,coordxOrig,coordyOrig,\
            VrmsFromDepthVel,velCoordxNtr,velCoordyNtr,ncpu);
    }
    VrmsFromDepthVel.clear();
    matchVelLayerNtr(datumDepthOrig,coordxOrig,coordyOrig,\
        datumDepth3d,velCoordxNtr,velCoordyNtr,ncpu);
    matchVelLayerNtr(velOrig3d,coordxOrig,coordyOrig,\
        vel3d,velCoordxNtr,velCoordyNtr,ncpu);
    matchVelLayerNtr(layerRefDepthOrig3d,coordxOrig,coordyOrig,\
        layerRefDepth3d,velCoordxNtr,velCoordyNtr,ncpu);
    mulLayerDepth=layerRefDepthOrig3d.slice(0);
    layerRefDepthOrig3d.clear();
    coordx=coordxOrig;
    coordy=coordyOrig;

    //datawrite(coordxOrig,"./swap/coordx.regular.dat");
    //datawrite(coordyOrig,"./swap/coordy.regular.dat");
    //datawrite3d_bycol_transpose(Vrms3dForInterpolation,\
        Vrms3dForInterpolation.n_slices,Vrms3dForInterpolation.n_rows,"./swap/vrms.dat");

    if(NumCoordxExtra>0){
        dataMatColExtra(data3dOrig, velOrig3d, \
            coordx, coordy, datumDepthOrig, dx, dt, \
            NumCoordxExtra, Vrms3dForInterpolation, mulLayerDepth);
        coordxOrig=coordx;
        coordyOrig=coordy;
    }
    getLayerVrms2d(vrms2d, t0Double2d, velOrig3d, mulLayerDepth, dz);
    velOrig3d.clear();

    float dxInter(dx),dyInter(dy);
    for(int k=0;k<interTimesByRow;k++){
        fmatLinearInterpolation2dByRow(coordx);
        fmatLinearInterpolation2dByRow(coordy);
        fmatLinearInterpolation2dByRow(vrms2d);
        fmatLinearInterpolation2dByRow(t0Double2d);
        fcubeLinearInterpolation3dByRow(Vrms3dForInterpolation, ncpu);
        fcubeLinearInterpolation3dByRow(datumDepthOrig, ncpu);
        vrmsSpaceInterData3dtxByRow(data3dOrig, \
            coordx, coordy, Vrms3dForInterpolation, data3dOrig, \
            datumDepthOrig, dt, ncpu);
        coordxOrig=coordx;
        coordyOrig=coordy;
        dxInter*=0.5;
    }
    for(int k=0;k<interTimesByCol;k++){
        fcube data3dOrigSwap;
        fmatLinearInterpolation2dByCol(coordx);
        fmatLinearInterpolation2dByCol(coordy);
        fmatLinearInterpolation2dByCol(vrms2d);
        fmatLinearInterpolation2dByCol(t0Double2d);
        fcubeLinearInterpolation3dByCol(Vrms3dForInterpolation, ncpu);
        fcubeLinearInterpolation3dByCol(datumDepthOrig, ncpu);
        vrmsSpaceInterData3dtxByCol(data3dOrigSwap, \
            coordx, coordy, Vrms3dForInterpolation, data3dOrig, \
            datumDepthOrig, dt, ncpu);
        coordxOrig=coordx;
        coordyOrig=coordy;
        data3dOrig=data3dOrigSwap;
        dyInter*=0.5;
    }
    datumDepthOrig.clear();
    Vrms3dForInterpolation.clear();
    for(int k=interTimesByRow;k<0;k++){
        fcubeAntiLinearInterpolation3dByRow(data3dOrig);
        fmatAntiLinearInterpolation2dByRow(coordx);
        fmatAntiLinearInterpolation2dByRow(coordy);
        fmatAntiLinearInterpolation2dByRow(vrms2d);
        fmatAntiLinearInterpolation2dByRow(t0Double2d);
        coordxOrig=coordx;
        coordyOrig=coordy;
        dxInter*=2.0;
    }
    for(int k=interTimesByCol;k<0;k++){
        fcubeAntiLinearInterpolation3dByCol(data3dOrig);
        fmatAntiLinearInterpolation2dByCol(coordx);
        fmatAntiLinearInterpolation2dByCol(coordy);
        fmatAntiLinearInterpolation2dByCol(vrms2d);
        fmatAntiLinearInterpolation2dByCol(t0Double2d);
        coordxOrig=coordx;
        coordyOrig=coordy;
        dyInter*=2.0;
    }

    //datawrite3d_bycol_transpose(data3dOrig,nt,\
        data3dOrig.n_rows,"./swap/data.inter.dat");
    //datawrite(coordxOrig,"./swap/coordx.inter.dat");
    //datawrite(coordyOrig,"./swap/coordy.inter.dat");
/////////////////MWD predicts multiple waves/////////////////
//cout<<"  sub-step - <data tx2fx> is running."<<endl;
    cx_fcube data3dFx1, data3dFx2;
    float df=1.0/dt/nt;
    int reSize(1);
    if(doBaseConjFocus){reSize=2;}
    df=df/reSize;
    data3dOrig.resize(data3dOrig.n_rows,data3dOrig.n_cols,\
        reSize*data3dOrig.n_slices);
    data3dFx1.copy_size(data3dOrig);
    tx2fx3dCompress(data3dFx1,data3dOrig, fmax/df+1, ncpu);
    data3dOrig.zeros(1,1,1);

    data3dFx2.set_size(data3dCut.n_rows,data3dCut.n_cols,\
        reSize*data3dCut.n_slices);
    data3dCut.zeros(1,1,1);

    for(int k=0;k<data3dFx1.n_slices;k++){
        data3dFx1.slice(k)=get_blackman_upwin2d\
            (data3dFx1.slice(k),xLeftWinWide/dxInter);
        data3dFx1.slice(k)=get_blackman_downwin2d\
            (data3dFx1.slice(k),xRightWinWide/dxInter);
        data3dFx1.slice(k)=get_blackman_leftwin2d\
            (data3dFx1.slice(k),yUpWinWide/dyInter);
        data3dFx1.slice(k)=get_blackman_rightwin2d\
            (data3dFx1.slice(k),yDownWinWide/dyInter);
    }
    data3dFx2.fill(0.0);

    int sxIndex,syIndex;
    //getSourceIndes(sxIndex,syIndex,coordx,coordy);
//cout<<"  Check for Source-Coord: (iSx,iSy)?=("\
        <<sxIndex<<","<<syIndex<<")"<<endl;
//cout<<"Step - <Predict Multiple> is running."<<endl;
    float maxTime=max(float(dt*nt),8.0f);
    nt=nt*reSize; ntOrig=ntOrig*reSize;
    cx_fmat timeBase=getTimeCodeingBase(\
        maxTime, df, fmin/df, fmax/df, ncpu);
    if(doBaseConjFocus){timeBase=conj(timeBase);}
    multiplePrediction3dMWD(\
        data3dFx2, coordxNtr, coordyNtr, \
        data3dFx1, coordx, coordy, \
        vrms2d, t0Double2d,\
        mulLayerDepth, timeBase, waterVelocity, \
        0.0, 0.0, eccentricity, maxAperture, \
        aModify, bModify, weightPow, \
        offsetEnhance, focusEnhance, weightMax, \
        df, fmin/df, fmax/df, ncpu);
    
    if(dt!=dtOrig){
        data3dFx1.zeros(data3dFx2.n_rows,data3dFx2.n_cols,ntOrig);
        data3dFx1(span::all,span::all,span(0,ntOrig/2-1))\
            =data3dFx2(span::all,span::all,span(0,ntOrig/2-1));
        data3dFx2=data3dFx1;
        dt=dtOrig;nt=ntOrig;
    }
    data3dFx1.zeros(1,1,1);
    data3dCut.copy_size(data3dFx2);
    fx2tx_3d_thread(data3dCut,data3dFx2, ncpu);
    data3dFx2.zeros(1,1,1);
    if(data3dCut.n_slices!=ntOrig){
        int ntSwap=min(int(data3dCut.n_slices),ntOrig);
        fcube dataSwap(data3dCut.n_rows,data3dCut.n_cols,ntOrig);
        dataSwap.fill(0.0);
        dataSwap(span::all,span::all,span(0,ntSwap-1))\
            =data3dCut(span::all,span::all,span(0,ntSwap-1));
        data3dCut=dataSwap;
        nt=ntOrig;
    }
cout<<"  Check Mul Time Sampling: (nt,dt)?=("<<nt<<","<<dt<<")"<<endl;
}

fcube PredictMultipleKirchhoff(\
    fcube& data3dNtrOrig, float fmin, float fmax,\
    float sxCoordOri, float syCoordOri,\
    fmat& coordxNtr,fmat& coordyNtr,\
    fmat& headWaterDepthNtr,\
    float waterVelocity, float VrmsGrad,\
    fcube& vel3d, fcube& layerDepth3d, \
    fmat& velCoordxNtr,fmat& velCoordyNtr,\
    int ntOrig, float dtOrig, int ntr, \
    int nx, int ny, float dx, float dy, float dz,\
    int interTimesByRow, int interTimesByCol, \
    float xLeftWinWide, float xRightWinWide,\
    float yUpWinWide,float yDownWinWide,\
    float nearOffsetExpand, float receiverDepth,\
    float cutSlopeMin, float cutSlopeMax, \
    float systemDelayTime, float cutTimeWinWide,\
    float minAperture, float offsetEnhance, int nthread,\
    int iDeepRef, int iShallowRef,\
    int dataMeshSizeModify,\
    int useHeadWaterDepthOnly,\
    int predictIndexLayerOnly,\
    int doSufaceMultiplePrediction,\
    int doInterbedMultiplePrediction\
)
{
    float maxAperture=800000000.0;
    float aModify=minAperture;
    float bModify=minAperture;
    float eccentricity=1.0; 
    float focusEnhance=1.0;
    float weightPow=1.5;
    float weightMax=0.99;

    fmat coordxSwap=coordxNtr;
    fmat coordySwap=coordyNtr;
    float theta0=getCoordRotateTheta(coordxNtr,coordyNtr,0.0);
    float theta=theta0*3.1415926/180.0;
    float sxCoord=rotateCoordx(sxCoordOri, syCoordOri, theta);
    float syCoord=rotateCoordy(sxCoordOri, syCoordOri, theta);
    MakeGlobalCoordToLocalCoord(coordxSwap,coordySwap,\
        coordxNtr,coordyNtr,sxCoord,syCoord,theta);

    float minSRx=abs(coordxSwap.min()-sxCoord);
    float maxSRx=abs(coordxSwap.max()-sxCoord);
    float minSRy=abs(coordySwap.min()-syCoord);
    float maxSRy=abs(coordySwap.max()-syCoord);
    if(minSRy/maxSRy<minSRx/maxSRx){
        cout<<"  Check for Rotate Coord: (theta0)?=("<<theta0<<")"<<endl;
        theta0=theta0+90.0;
    }
    cout<<"  Check for Rotate Coord: (theta0)?=("<<theta0<<")"<<endl;
    theta=theta0*3.1415926/180.0;
    sxCoord=rotateCoordx(sxCoordOri, syCoordOri, theta);
    syCoord=rotateCoordy(sxCoordOri, syCoordOri, theta);
    MakeGlobalCoordToLocalCoord(coordxSwap,coordySwap,\
        coordxNtr,coordyNtr,sxCoord,syCoord,theta);
    MakeGlobalCoordToLocalCoord(velCoordxNtr,velCoordyNtr,\
        velCoordxNtr,velCoordyNtr,sxCoord,syCoord,theta);
    coordxNtr=coordxSwap;
    coordyNtr=coordySwap;

    for(int i=0;i<interTimesByRow;i++){nearOffsetExpand*=2.0;}
    int NumCoordxExtra=round(nearOffsetExpand/dx);
    float delayTime=systemDelayTime+receiverDepth/waterVelocity;
    delayTime=max(delayTime,dtOrig);
    fcube data3dMul=data3dNtrOrig;
    if(doInterbedMultiplePrediction<1){
        cutDataBySlope(data3dNtrOrig,coordxNtr,coordyNtr,\
            dtOrig, cutSlopeMin, cutSlopeMax, delayTime);
        cutDataBySlopeWin(data3dNtrOrig,coordxNtr,coordyNtr,\
            dtOrig, cutSlopeMax, delayTime, round(cutTimeWinWide/dtOrig));
    }

    fcube layerRefDepth3d,datumDepth3d;
    datumDepth3d.zeros(layerDepth3d.n_rows,layerDepth3d.n_cols,2);
    datumDepth3d.slice(1).fill(waterVelocity);

    if(useHeadWaterDepthOnly>=1){
        fcube data3dNtrIO=data3dNtrOrig;

        if(doInterbedMultiplePrediction>=1){
            datumDepth3d.zeros(headWaterDepthNtr.n_rows,1,2);
            datumDepth3d.slice(1).fill(waterVelocity);
            layerRefDepth3d.zeros(headWaterDepthNtr.n_rows,1,1);
            for(int i=0;i<headWaterDepthNtr.n_rows;i++){
                layerRefDepth3d(i,0,0)=headWaterDepthNtr(i,0);
            }
            vel3d.zeros(headWaterDepthNtr.n_rows,1,\
                5+ceil(headWaterDepthNtr.max()/dz));
            vel3d.fill(waterVelocity);
            velCoordxNtr=coordxNtr;
            velCoordyNtr=coordyNtr;
            mul3dLayerCode(\
                data3dNtrIO, fmin, fmax, useHeadWaterDepthOnly,\
                vel3d, layerRefDepth3d, velCoordxNtr,velCoordyNtr,\
                datumDepth3d, coordxNtr, coordyNtr, \
                ntOrig, dtOrig, ntr, nx, ny, dx, dy, dz,\
                interTimesByRow, interTimesByCol, dataMeshSizeModify,\
                xLeftWinWide, xRightWinWide, yUpWinWide, yDownWinWide,\
                waterVelocity, VrmsGrad, NumCoordxExtra, \
                cutSlopeMin, cutSlopeMax, delayTime,\
                maxAperture, aModify, bModify, \
                eccentricity, offsetEnhance, focusEnhance,\
                weightMax, weightPow, true, nthread);

            data3dNtrIO=data3dNtrIO/data3dNtrIO.max();
            datawrite3d_bycol_transpose(data3dNtrIO, data3dNtrIO.n_slices,\
                data3dNtrIO.n_rows,"./swap/data.swap.dat");
            data3dNtrIO=data3dNtrIO(span::all,span::all,\
                span(0,data3dNtrIO.n_slices/2-1));
            cutDataBySlope(data3dNtrIO,coordxNtr,coordyNtr,\
                dtOrig, cutSlopeMin, cutSlopeMax, delayTime);
            cutDataBySlopeWin(data3dNtrIO,coordxNtr,coordyNtr,\
                dtOrig, cutSlopeMax, delayTime, round(cutTimeWinWide/dtOrig));
        }
        datumDepth3d.zeros(headWaterDepthNtr.n_rows,1,2);
        datumDepth3d.slice(1).fill(waterVelocity);
        layerRefDepth3d.zeros(headWaterDepthNtr.n_rows,1,1);
        for(int i=0;i<headWaterDepthNtr.n_rows;i++){
            layerRefDepth3d(i,0,0)=headWaterDepthNtr(i,0);
        }
        vel3d.zeros(headWaterDepthNtr.n_rows,1,\
            5+ceil(headWaterDepthNtr.max()/dz));
        vel3d.fill(waterVelocity);
        velCoordxNtr=coordxNtr;
        velCoordyNtr=coordyNtr;
        mul3dLayerCode(\
            data3dNtrIO, fmin, fmax, useHeadWaterDepthOnly,\
            vel3d, layerRefDepth3d, velCoordxNtr,velCoordyNtr,\
            datumDepth3d, coordxNtr, coordyNtr, \
            ntOrig, dtOrig, ntr, nx, ny, dx, dy, dz,\
            interTimesByRow, interTimesByCol, dataMeshSizeModify,\
            xLeftWinWide, xRightWinWide, yUpWinWide, yDownWinWide,\
            waterVelocity, VrmsGrad, NumCoordxExtra, \
            cutSlopeMin, cutSlopeMax, delayTime,\
            maxAperture, aModify, bModify, \
            eccentricity, offsetEnhance, focusEnhance,\
            weightMax, weightPow, false, nthread);
        cutDataBySlope(data3dNtrIO,coordxNtr,coordyNtr,\
            dtOrig, cutSlopeMin, cutSlopeMax, delayTime);
        data3dMul=data3dNtrIO;
    }else if(doInterbedMultiplePrediction>=1){
        int iShallowBeg,iShallowEnd,iDeepBeg,iDeepEnd;
        int ilayerDeepRef=iDeepRef;
        int ilayerShallowRef=iShallowRef;
        if(predictIndexLayerOnly>=1){
            iShallowBeg=iShallowRef;
            iShallowEnd=iShallowRef+1;
            iDeepBeg=iDeepRef;
            iDeepEnd=iDeepRef+1;
        }else{
            iShallowBeg=0;
            iShallowEnd=layerDepth3d.n_slices-1;
            iDeepBeg=1+ilayerShallowRef;
            iDeepEnd=layerDepth3d.n_slices;
        }
        data3dMul.fill(0.0);
        for(ilayerShallowRef=iShallowBeg;ilayerShallowRef<iShallowEnd;ilayerShallowRef++){
        cout<<"  Interbed Multiple Prediction Shallow Ref: "<<ilayerShallowRef<<endl;
            fcube data3dNtrIO=data3dNtrOrig;
            layerRefDepth3d.zeros(layerDepth3d.n_rows,layerDepth3d.n_cols,1);
            layerRefDepth3d.slice(0)=layerDepth3d.slice(ilayerShallowRef);
            mul3dLayerCode(\
                data3dNtrIO, fmin, fmax, useHeadWaterDepthOnly,\
                vel3d, layerRefDepth3d, velCoordxNtr,velCoordyNtr,\
                datumDepth3d, coordxNtr, coordyNtr, \
                ntOrig, dtOrig, ntr, nx, ny, dx, dy, dz,\
                interTimesByRow, interTimesByCol, dataMeshSizeModify,\
                xLeftWinWide, xRightWinWide, yUpWinWide, yDownWinWide,\
                waterVelocity, VrmsGrad, NumCoordxExtra, \
                cutSlopeMin, cutSlopeMax, delayTime,\
                maxAperture, aModify, bModify, \
                eccentricity, offsetEnhance, focusEnhance,\
                weightMax, weightPow, true, nthread);
                    data3dNtrIO=data3dNtrIO/data3dNtrIO.max();
            //datawrite3d_bycol_transpose(data3dNtrIO, data3dNtrIO.n_slices,\
                data3dNtrIO.n_rows,"./swap/data.swap.dat");
            data3dNtrIO=data3dNtrIO(span::all,span::all,\
                span(0,data3dNtrIO.n_slices/2-1));
            cutDataBySlope(data3dNtrIO,coordxNtr,coordyNtr,\
                dtOrig, cutSlopeMin, cutSlopeMax, delayTime);
            cutDataBySlopeWin(data3dNtrIO,coordxNtr,coordyNtr,\
                dtOrig, cutSlopeMax, delayTime, round(cutTimeWinWide/dtOrig));
            fcube data3dSwap=data3dNtrIO;
            if(predictIndexLayerOnly<1){
                iDeepBeg=1+ilayerShallowRef;
                iDeepEnd=layerDepth3d.n_slices;
            }
            for(ilayerDeepRef=iDeepBeg;ilayerDeepRef<iDeepEnd;ilayerDeepRef++){
            cout<<"    Interbed Multiple Prediction Deep Ref: "<<ilayerDeepRef<<endl;
                data3dNtrIO=data3dSwap;
                layerRefDepth3d.zeros(layerDepth3d.n_rows,layerDepth3d.n_cols,1);
                layerRefDepth3d.slice(0)=layerDepth3d.slice(ilayerDeepRef);
                datumDepth3d.slice(0)=layerDepth3d.slice(ilayerShallowRef);
                mul3dLayerCode(\
                    data3dNtrIO, fmin, fmax, useHeadWaterDepthOnly,\
                    vel3d, layerRefDepth3d, velCoordxNtr,velCoordyNtr,\
                    datumDepth3d, coordxNtr, coordyNtr, \
                    ntOrig, dtOrig, ntr, nx, ny, dx, dy, dz,\
                    interTimesByRow, interTimesByCol, dataMeshSizeModify,\
                    xLeftWinWide, xRightWinWide, yUpWinWide, yDownWinWide,\
                    waterVelocity, VrmsGrad, NumCoordxExtra, \
                    cutSlopeMin, cutSlopeMax, delayTime,\
                    maxAperture, aModify, bModify, \
                    eccentricity, offsetEnhance, focusEnhance,\
                    weightMax, weightPow, false, nthread);
                cutDataBySlope(data3dNtrIO,coordxNtr,coordyNtr,\
                    dtOrig, cutSlopeMin, cutSlopeMax, delayTime);
                data3dMul+=data3dNtrIO;
            }
        }
    }else if(doSufaceMultiplePrediction>=1){
        int ilayerDeepRef=iDeepRef;
        int iDeepBeg,iDeepEnd;
        if(predictIndexLayerOnly>=1){
            iDeepBeg=iDeepRef;
            iDeepEnd=iDeepRef+1;
        }else{
            iDeepBeg=0;
            iDeepEnd=layerDepth3d.n_slices;
        }
        for(ilayerDeepRef=iDeepBeg;ilayerDeepRef<iDeepEnd;ilayerDeepRef++){
        cout<<"  Suface Multiple Prediction Ref: "<<ilayerDeepRef<<endl;
            fcube data3dNtrIO=data3dNtrOrig;
            layerRefDepth3d.zeros(layerDepth3d.n_rows,layerDepth3d.n_cols,1);
            layerRefDepth3d.slice(0)=layerDepth3d.slice(ilayerDeepRef);
            mul3dLayerCode(\
                data3dNtrIO, fmin, fmax, useHeadWaterDepthOnly,\
                vel3d, layerRefDepth3d, velCoordxNtr,velCoordyNtr,\
                datumDepth3d, coordxNtr, coordyNtr, \
                ntOrig, dtOrig, ntr, nx, ny, dx, dy, dz,\
                interTimesByRow, interTimesByCol, dataMeshSizeModify,\
                xLeftWinWide, xRightWinWide, yUpWinWide, yDownWinWide,\
                waterVelocity, VrmsGrad, NumCoordxExtra, \
                cutSlopeMin, cutSlopeMax, delayTime,\
                maxAperture, aModify, bModify, \
                eccentricity, offsetEnhance, focusEnhance,\
                weightMax, weightPow, false, nthread);
            cutDataBySlope(data3dNtrIO,coordxNtr,coordyNtr,\
                dtOrig, cutSlopeMin, cutSlopeMax, delayTime);
            data3dMul+=data3dNtrIO;
        }
    }
    return data3dMul;
}

////////////////////////////par.hpp////////////////////////
struct CRMD3d_ParList
{
//Input/Output file-path + file-name
    string fileinVel,\
        fileinLayerDepth,\
        fileinDataCS;
    string fileoutMultiplePrediction,\
        fileoutDeMultipleData,\
        fileoutMatchedMultiple;
    string fileDataSwap;

//.su head key words
    string keySx, keySy, keyGx, keyGy,\
        keyWaterDepth, keyNodeDepth;

//Flow process control parameters;
    int isInputSuFile;
    int predictIndexLayerOnly;
    int useHeadWaterDepthOnly;
    int doSufaceMultiplePrediction;
    int doInterbedMultiplePrediction;
    int doAdaptiveSubtraction;
    int doDataGridSizeModify;
    int dataGroupBegIndex, dataGroupGap;
    int dataGroupProcNum;

//Regularization parameters of data grid;
    float timeUnitTrans, spaceUnitTrans;
    int interTimesByRow, interTimesByCol;
    float nearOffsetExpand;
    float VrmsGradient, cutTimeWinWide,\
        xLeftWinWide,xRightWinWide,\
        yUpWinWide,yDownWinWide;
    int nx, ny, nz, nt;
    float dx, dy, dz, dt;

//Multiple prediction parameters;
    int nthread; 
    float cutSlopeMax, cutSlopeMin, \
        fmin, fmax, systemDelayTime, \
        waterVelocity;
    int nlayer, ilayerDeepRef, ilayerShallowRef;

//Data aperture parameters for multiple prediction;
    float maxAperture, weightMax, \
        eccentricity, focusEnhance, \
        offsetEnhance, weightPow, \
        aModify, bModify,minAperture;
        
//Adaptive subtraction parameters
    int matchLoopNum,matchLevelNum;
    int wienerFilterLength, wienerTimeSlideGap, \
        wienerDataTimeLength, wienerMultiTraceNum,\
        wienerMulPhaseShiftNum, wienerMulTimeShiftNum,\
        wienerMulTimeShiftGap;
    float wienerTikhonovFactor;
};

void InitialCRMD3dParList(struct CRMD3d_ParList& par)
{
    par.fileinVel="null";
    par.fileinLayerDepth="null";
    par.fileinDataCS="null";
    par.fileoutMultiplePrediction="null";
    par.fileoutDeMultipleData="null";
    par.fileoutMatchedMultiple="null";
    par.fileDataSwap="null";
    par.keySx="null";
    par.keySy="null";
    par.keyGx="null";
    par.keyGy="null";
    par.keyWaterDepth="null";
    par.keyNodeDepth="null";

    par.isInputSuFile=0;
    par.useHeadWaterDepthOnly=0;
    par.predictIndexLayerOnly=0;
    par.doDataGridSizeModify=1;
    par.doSufaceMultiplePrediction=0;
    par.doInterbedMultiplePrediction=0;
    par.doAdaptiveSubtraction=0;
    par.dataGroupBegIndex=0;
    par.dataGroupGap=1;
    par.dataGroupProcNum=999999999;

    par.timeUnitTrans=1000000.0;  //us -> s
    par.spaceUnitTrans=10.0;  //0.1m -> m
    par.interTimesByRow=0;
    par.interTimesByCol=0;
    par.nearOffsetExpand=0; 
    par.VrmsGradient=200.0;
    par.xLeftWinWide=-10.0;
    par.xRightWinWide=-10.0;
    par.yUpWinWide=-10.0;
    par.yDownWinWide=-10.0;
    par.nx=1, par.ny=1;
    par.nt=1, par.nz=1;
    par.dx=1, par.dy=1;
    par.dz=1, par.dt=0.001;

    par.nthread=1;
    par.nlayer=1;
    par.ilayerDeepRef=-1;
    par.ilayerShallowRef=-1;
    par.cutSlopeMax=1450;
    par.cutSlopeMin=1250;
    par.fmin=0.1;
    par.fmax=150.0;
    par.waterVelocity=1520.0;
    par.cutTimeWinWide=0.1;
    par.systemDelayTime=0.05;

    par.maxAperture=8000000.0;
    par.weightMax=0.99;
    par.eccentricity=1.0;
    par.focusEnhance=1.0;
    par.offsetEnhance=0.25;
    par.weightPow=1.5;
    par.aModify=350.0;
    par.bModify=350.0;
    par.minAperture=350.0;

    par.matchLoopNum=1;
    par.matchLevelNum=0;
    par.wienerFilterLength=3;
    par.wienerTimeSlideGap=10; 
    par.wienerDataTimeLength=30; 
    par.wienerMultiTraceNum=5;
    par.wienerMulPhaseShiftNum=4;
    par.wienerMulTimeShiftNum=0;
    par.wienerMulTimeShiftGap=5;
    par.wienerTikhonovFactor=0.01;
}
void ReadCRMD3dParList(struct CRMD3d_ParList& par, \
    const char *name, const char *value, bool outlog=false);
void ReadCRMD3dParList(struct CRMD3d_ParList& par, \
    const char *name, const char *value, bool outlog)
{
    if(outlog){cout<<name<<"="<<value<<endl;}

    if(strcmp(name,"fileinVel")==0)
        {par.fileinVel=value;}
    else if (strcmp(name,"fileinLayerDepth")==0)
        {par.fileinLayerDepth=value;}
    else if (strcmp(name,"fileinDataCS")==0)
        {par.fileinDataCS=value;}
    else if (strcmp(name,"fileoutMultiplePrediction")==0) 
        {par.fileoutMultiplePrediction=value;}
    else if (strcmp(name,"fileoutDeMultipleData")==0) 
        {par.fileoutDeMultipleData=value;}
    else if (strcmp(name,"fileoutMatchedMultiple")==0) 
        {par.fileoutMatchedMultiple=value;}
    else if (strcmp(name,"fileDataSwap")==0) 
        {par.fileDataSwap=value;}
    \
    else if (strcmp(name,"keySx")==0) 
        {par.keySx=value;}
    else if (strcmp(name,"keySy")==0) 
        {par.keySy=value;}
    else if (strcmp(name,"keyGx")==0) 
        {par.keyGx=value;}
    else if (strcmp(name,"keyGy")==0) 
        {par.keyGy=value;}
    else if (strcmp(name,"keyWaterDepth")==0) 
        {par.keyWaterDepth=value;}
    else if (strcmp(name,"keyNodeDepth")==0) 
        {par.keyNodeDepth=value;}
    \
    else if (strcmp(name,"isInputSuFile")==0) 
        {par.isInputSuFile=atoi(value);}
    else if (strcmp(name,"useHeadWaterDepthOnly")==0) 
        {par.useHeadWaterDepthOnly=atoi(value);}
    else if (strcmp(name,"predictIndexLayerOnly")==0) 
        {par.predictIndexLayerOnly=atoi(value);}
    else if (strcmp(name,"doDataGridSizeModify")==0) 
        {par.doDataGridSizeModify=atoi(value);}
    else if (strcmp(name,"doSufaceMultiplePrediction")==0) 
        {par.doSufaceMultiplePrediction=atoi(value);}
    else if (strcmp(name,"doInterbedMultiplePrediction")==0) 
        {par.doInterbedMultiplePrediction=atoi(value);}
    else if (strcmp(name,"doAdaptiveSubtraction")==0) 
        {par.doAdaptiveSubtraction=atoi(value);}
    else if (strcmp(name,"dataGroupBegIndex")==0) 
        {par.dataGroupBegIndex=atoi(value);}
    else if (strcmp(name,"dataGroupGap")==0) 
        {par.dataGroupGap=atoi(value);}
    else if (strcmp(name,"dataGroupProcNum")==0) 
        {par.dataGroupProcNum=atoi(value);}
    \
    else if (strcmp(name,"timeUnitTrans")==0) 
        {par.timeUnitTrans=atof(value);}
    else if (strcmp(name,"spaceUnitTrans")==0) 
        {par.spaceUnitTrans=atof(value);}
    else if (strcmp(name,"interTimesByRow")==0) 
        {par.interTimesByRow=atoi(value);}
    else if (strcmp(name,"interTimesByCol")==0) 
        {par.interTimesByCol=atoi(value);}
    else if (strcmp(name,"nearOffsetExpand")==0) 
        {par.nearOffsetExpand=atof(value);}
    else if (strcmp(name,"VrmsGradient")==0) 
        {par.VrmsGradient=atof(value);}
    else if (strcmp(name,"cutTimeWinWide")==0) 
        {par.cutTimeWinWide=atof(value);}
    else if (strcmp(name,"xLeftWinWide")==0) 
        {par.xLeftWinWide=atof(value);}
    else if (strcmp(name,"xRightWinWide")==0) 
        {par.xRightWinWide=atof(value);}
    else if (strcmp(name,"yUpWinWide")==0) 
        {par.yUpWinWide=atof(value);}
    else if (strcmp(name,"yDownWinWide")==0) 
        {par.yDownWinWide=atof(value);}
    else if (strcmp(name,"nx")==0) 
        {par.nx=atoi(value);}
    else if (strcmp(name,"ny")==0) 
        {par.ny=atoi(value);}
    else if (strcmp(name,"nz")==0) 
        {par.nz=atoi(value);}
    else if (strcmp(name,"nt")==0) 
        {par.nt=atoi(value);}
    else if (strcmp(name,"dx")==0) 
        {par.dx=atof(value);}
    else if (strcmp(name,"dy")==0) 
        {par.dy=atof(value);}
    else if (strcmp(name,"dz")==0) 
        {par.dz=atof(value);}
    else if (strcmp(name,"dt")==0) 
        {par.dt=atof(value);}
    \
    else if (strcmp(name,"nthread")==0) 
        { par.nthread=atoi(value);}
    else if (strcmp(name,"cutSlopeMax")==0) 
        { par.cutSlopeMax=atof(value);}
    else if (strcmp(name,"cutSlopeMin")==0) 
        { par.cutSlopeMin=atof(value);}
        else if (strcmp(name,"fmin")==0) 
        { par.fmin=atof(value);}
    else if (strcmp(name,"fmax")==0) 
        { par.fmax=atof(value);}
    else if (strcmp(name,"systemDelayTime")==0) 
        { par.systemDelayTime=atof(value);}
    else if (strcmp(name,"waterVelocity")==0) 
        { par.waterVelocity=atof(value);}
    else if (strcmp(name,"nlayer")==0) 
        { par.nlayer=atoi(value);}
    else if (strcmp(name,"ilayerDeepRef")==0) 
        { par.ilayerDeepRef=atoi(value);}
    else if (strcmp(name,"ilayerShallowRef")==0) 
        { par.ilayerShallowRef=atoi(value);}
    \
    else if (strcmp(name,"maxAperture")==0) 
        { par.maxAperture=atof(value);}
    else if (strcmp(name,"minAperture")==0) 
        { par.minAperture=atof(value);}
    else if (strcmp(name,"weightMax")==0) 
        { par.weightMax=atof(value);}
    else if (strcmp(name,"eccentricity")==0) 
        { par.eccentricity=atof(value);}
    else if (strcmp(name,"focusEnhance")==0) 
        { par.focusEnhance=atof(value);}
    else if (strcmp(name,"offsetEnhance")==0) 
        { par.offsetEnhance=atof(value);}
    else if (strcmp(name,"weightPow")==0) 
        { par.weightPow=atof(value);}
    else if (strcmp(name,"aModify")==0) 
        { par.aModify=atof(value);}
    else if (strcmp(name,"bModify")==0) 
        { par.bModify=atof(value);}
    \
    else if (strcmp(name,"matchLoopNum")==0) 
        { par.matchLoopNum=atoi(value);}
    else if (strcmp(name,"matchLevelNum")==0) 
        { par.matchLevelNum=atoi(value);}
    else if (strcmp(name,"wienerFilterLength")==0) 
        { par.wienerFilterLength=atoi(value);}
    else if (strcmp(name,"wienerTimeSlideGap")==0) 
        { par.wienerTimeSlideGap=atoi(value);}
    else if (strcmp(name,"wienerDataTimeLength")==0) 
        { par.wienerDataTimeLength=atoi(value);}
    else if (strcmp(name,"wienerMultiTraceNum")==0) 
        { par.wienerMultiTraceNum=atoi(value);}
    else if (strcmp(name,"wienerMulPhaseShiftNum")==0) 
        { par.wienerMulPhaseShiftNum=atoi(value);}
    else if (strcmp(name,"wienerMulTimeShiftNum")==0) 
        { par.wienerMulTimeShiftNum=atoi(value);}
    else if (strcmp(name,"wienerMulTimeShiftGap")==0) 
        { par.wienerMulTimeShiftGap=atoi(value);}
    else if (strcmp(name,"wienerTikhonovFactor")==0) 
        { par.wienerTikhonovFactor=atof(value);}
    \
    else{std::cout<<"Warning: illegal parameter name: "<<name<<std::endl;}
}
CRMD3d_ParList ReadCRMD3dParList(const char *file, bool parlog=false);
CRMD3d_ParList ReadCRMD3dParList(const char *file, bool parlog)
{
    CRMD3d_ParList par;
    InitialCRMD3dParList(par);

    ifstream parin;
    char name[1024],value[1024];
    string line;
    parin.open(file);
    if(parin){
        while(getline(parin,line)){
            int n1,n2;
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
            ReadCRMD3dParList(par,name,value,parlog);
        }
    }else{
        cout<<"Not find Par file!"<<endl;
    } 
    return par;
}

#endif
