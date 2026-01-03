
#ifndef WIENERMATCH_2D_HPP
#define WIENERMATCH_2D_HPP

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
class WienerMatch{ 
private:
public:
    string fileOrig,fileMul,\
        fileDemul,fileMatch,\
        filePath,fileOutSwap;
    fmat dataOrig2d,mul2d,filter;
    fmat match2d,demul2d;
    int nt,nx,ny,nz,ncpu;
    float dx,dy,dz,dt,sx,sy;
    int numLoop, numLevel, numAntiLoop, dataTimeLen, filterLen, \
        dataSpaceLen, dataTimeSlide, phaseNum, num_shift, d_shift;
    float wienerTikhonov;
    int nIndxNum=5;

    WienerMatch2d();
    ~WienerMatch2d();
    void ClearData();
    int SolveWienerFilter();
};

int SolveWienerFilter(){
    fmat matd(psubmat, datanum,1); 
    dmat x2(nw,1,fill::zeros);
    //matcopy(x2,w);
    x2=solveCG_real<dmat>\
        (mat1, x2, matd, lmd, 0.000000000001, nw*nw);
    matcopy(w,x2);

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
        for(int kloop=0;kloop<numLoop;kloop++){
            fmat err2d=this->GetMatchingDataWithModelOneGather(\
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


fmat CRMD2d::GetMatchingDataWithModelOneGather(
    fmat& timeShift2d, fmat origData2d, fmat modelData2d,\
    int dataTimeLen=100, int filterLen=5, int dataSpaceLen=25,\
    int dataTimeSlide=10, float wienerTikhonov=0.01, int phaseNum=4,\
    int num_shift=0, int d_shift=1, float weight_shift=-0.1)
{
    fmat matchingData2d;
    matchingData2d.copy_size(modelData2d);
    matchingData2d.fill(0.0);
    timeShift2d=matchingData2d;
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


#endif
