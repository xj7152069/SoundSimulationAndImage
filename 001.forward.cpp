#define ARMA_DONT_USE_WRAPPER
#define ARMA_DONT_USE_BLAS
#define ARMA_DONT_USE_OPENMP
#define ARMA_OPENMP_THREADS 1
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <iomanip>
#include <math.h>
#include "./include/xjc.h"
//#include "../include/xjc.h"

using namespace arma;
using namespace std;

int main(int argc , char *argv[])
{
    CRMD2d obj;
    obj.ReadPar(argv[1]);
    int maxThread(obj.ncpu);
    obj.nIndxNum=6;

/*Modeling the direct wave: vp->dr*/
    obj.ncpu=maxThread;
    obj.vp2d.fill(obj.velWater);
    obj.rho2d.fill(1000.0);
    obj.dataOrig2d=obj.ForwardTowOneShot(obj.sxBeg,1);
    datawrite(obj.dataOrig2d,obj.fileDirectBeg.c_str());
    obj.dataOrig2d=obj.ForwardTowOneShot(obj.sxEnd,1);
    datawrite(obj.dataOrig2d,obj.fileDirectEnd.c_str());


/*Modeling of Tow Multi-Shot: vp->cs*/
    obj.ReadPar(argv[1]);
    obj.ncpu=maxThread/(abs(obj.sxEnd-obj.sxBeg)/obj.sxGap+1);
    obj.ncpu=max(obj.ncpu,1);
    //Modeling the PML-Surface tow data:
    obj.ForwardTowMultiShot(obj.fileCSorig.c_str(),1,0,maxThread);

/*Remove the direct wave: cs->cs.outSwap*/
    obj.RemoveDirectMultiShot(obj.fileOutSwap.c_str(),obj.fileCSorig.c_str());

/*RTM of Tow Multi-Shot: cs.demul->image*/
    fmat image2d=obj.RTMofTowMultiShot(\
        obj.fileOutSwap.c_str(),100,100,300,obj.nt,maxThread/2);
    datawrite(image2d,"image.dat");

/*RTM Image2d result post-process:*/
    for(int k=obj.izFreeSurface+10;k<image2d.n_rows;k++){
        image2d.row(k)/=sqrt(1.0/obj.dz/float(k-obj.izFreeSurface+11.0));
    }
    datawrite(image2d,"image.modify.dat");
    image2d=Laplace(image2d);
    datawrite(image2d,"image.la.dat");

    return 0;
}
