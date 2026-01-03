
#ifndef TRAVELTIME_3D_HPP
#define TRAVELTIME_3D_HPP

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

/////////////////////////////////////////////////////
void cubecopy(fcube& cube2, dcube& cube1)
{
    cube2.copy_size(cube1);
    for(int i=0;i<cube1.n_rows;i++){
    for(int j=0;j<cube1.n_cols;j++){
    for(int k=0;k<cube1.n_slices;k++){
        cube2(i,j,k)=cube1(i,j,k);
    }}}
}

double GetThisTimeAndNextDepthRayTrace(\
    double& x1, double& y1, const fcube& vel3d, int& ix, int& iy, int& iz,\
    double& v0, double x0, double y0, double z0, double sinTheta0, \
    double dx, double dy, double dz, double projTX, double projTY)
{
    double tanTheta0=sinTheta0/(sqrt(1.0-sinTheta0*sinTheta0));
    double dAdd=tanTheta0*dz;
    double xAdd=dAdd*projTX;
    double yAdd=dAdd*projTY;
    x1=x0+xAdd;
    y1=y0+yAdd;
    double lAdd=dAdd/sinTheta0;
    double tAdd=lAdd/v0;
    iz=round(z0/dz);
    iz=max(iz,1);iz=min(iz,int(vel3d.n_slices-1));
    ix=round(x1/dx), iy=round(y1/dy);
    ix=max(ix,0);ix=min(ix,int(vel3d.n_rows-1));
    iy=max(iy,0);iy=min(iy,int(vel3d.n_cols-1));
    double vModify=vel3d(ix,iy,iz);
    if(abs(vModify-v0)>1.0){
        tAdd=2.0*lAdd/(v0+vModify);
        v0=vModify;
    }
    return tAdd;
}
double GetSimplifiedRayTraceTimeDownToUp(\
    const fcube& vel3d, double& gx0, double& gy0, double gz, \
    double sx, double sy, double sz, double dx, double dy, double dz, \
    double thetaAngle, double thetaAzimuth)
{
    if(sz<=gz){cout<<"Ray-Trace Down To Up Error: sz<=gz !"<<endl; return 0.0;}
    double pi=3.1415926;
    thetaAngle=thetaAngle*pi/180.0;
    double offsetZ_1=dz*tan(thetaAngle);
    double projTX=cos(pi*thetaAzimuth/180.0);
    double projTY=sin(pi*thetaAzimuth/180.0);
    double x0=sx, y0=sy, z0=sz;
    double sinTheta0=sin(thetaAngle);

    double t0=0.0;
    double v0=vel3d(round(x0/dx),round(y0/dy),round(z0/dz));
    while(true){
        double x1(0.0),y1(0.0);
        int iz(0),ix(0),iy(0);
        t0+=GetThisTimeAndNextDepthRayTrace(x1, y1, vel3d, \
            ix, iy, iz, v0, x0, y0, z0, sinTheta0, dx, dy, dz, \
            projTX, projTY);
        x0=x1; y0=y1;

        double v1=vel3d(ix,iy,iz-1);
        if(abs(v1-v0)>1.0){
            sinTheta0=sinTheta0*(v1/v0);
            v0=v1;
        }
        if(abs(x0)>=1e7){break;} 
        if(abs(y0)>=1e7){break;} 
        if(sinTheta0>=0.99 || sinTheta0<=-0.99){\
            x0=-1e7, y0=-1e7;
            t0=0.0; break;
        }
        z0=z0-dz;
        if(z0<=gz){
            float d1=(z0+dz)/(sqrt(1.0-sinTheta0*sinTheta0));
            t0=t0+d1/v0;
            break;
        }
    }
    gx0=x0; gy0=y0;
    //cout<<gy0<<",";
    //cout<<projTY<<"||";
    return t0;
}
int UpdateSufaceTime(dcube& sufTime, dcube&  sufWeight,\
    double dx, double dy, double gx, double gy, \
    double oneTime, double disErr)
{
    double disRange=max(disErr,double(25.5));
    int nkx=ceil(disRange/dx);
    int nky=ceil(disRange/dy);
    int kx=round(gx/dx), ky=round(gy/dy);
    for(int j=max(ky-nky,0);j<=min(ky+nky,int(sufTime.n_cols-1));j++){
    for(int i=max(kx-nkx,0);i<=min(kx+nkx,int(sufTime.n_rows-1));i++){
        double disX=i*dx-gx;
        double disY=j*dy-gy;
        double dis=disErr/(1e-12+sqrt(disX*disX+disY*disY));
        if(dis>1.0){
        for(int k=0;k<sufTime.n_slices;k++){
            if(sufWeight(i,j,k)<dis){
                sufTime(i,j,k)=oneTime;
                sufWeight(i,j,k)=dis;
                break;
            }
        }}
    }}
    return 0;
}
fmat GetSimplifiedSufaceRayTraceTime(\
    const fcube& vel3d, double thetaAddDis, double maxAngle,double disErr, \
    double gz, double sx, double sy, double sz, double dx, double dy, double dz)
{
    int nx=vel3d.n_rows, ny=vel3d.n_cols, nz=vel3d.n_slices;
    dcube sufWeight(nx,ny,5,fill::zeros),sufTime(nx,ny,5,fill::zeros);
    double oneTime, pi=3.1415926;
    double tanThetaAdd_0=thetaAddDis/sz;
    //cout<<"("<<tanThetaAdd_0<<")"<<endl;

//do Ray-Trace:
    int countX=0, countY=0, \
        countYmax=0;
    double thetaAngleOld=0;
    double thetaAngleInp=0;
    while(true){
        countX++;
        int countXbreak=0;
        double tanThetaOld=tan(thetaAngleOld);
        double thetaAdd=atan(tanThetaAdd_0/(1.0+tanThetaOld*tanThetaOld\
            +abs(tanThetaOld*tanThetaAdd_0)));
        thetaAdd=max(thetaAdd,double(0.00001));
        thetaAdd=min(thetaAdd,double(0.001));
        double thetaAngle=thetaAngleOld+thetaAdd;
        thetaAngleInp=thetaAngle*180.0/pi;
        thetaAngleOld=thetaAngle;
        if(abs(thetaAngleInp)>=maxAngle){break;}

        countY=0;
        double gx0=sx;
        while(true){
            double gy0=sy,gz0=gz;
            double thetaAzimuth=(thetaAdd*countY)*180.0/pi;
            if(thetaAzimuth>=90.0){break;}
            oneTime=GetSimplifiedRayTraceTimeDownToUp(\
                vel3d, gx0, gy0, gz0, sx, sy, sz, dx, dy, dz, \
                thetaAngleInp, thetaAzimuth);
            if(gy0<=(ny-1)*dy && gx0<=(nx-1)*dx\
                && gy0>=0 && gx0>=0){
                countXbreak++;
                UpdateSufaceTime(sufTime, sufWeight, dx, dy, \
                    gx0, gy0, 1000.0*oneTime, disErr);
            }
            if(gx0<=(nx-1)*dx && (gy0>(ny-1)*dy || gy0<0)){break;}
            if(gx0>=0 && (gy0>(ny-1)*dy || gy0<0)){break;}
            countY++;
        }
        countYmax=max(countYmax,countY);

        countY=1;
        gx0=sx;
        while(true){
            double gy0=sy,gz0=gz;
            double thetaAzimuth=-(thetaAdd*countY)*180.0/pi;
            if(thetaAzimuth<=90.0){break;}
            oneTime=GetSimplifiedRayTraceTimeDownToUp(\
                vel3d, gx0, gy0, gz0, sx, sy, sz, dx, dy, dz, \
                thetaAngleInp, thetaAzimuth);
            if(gy0<=(ny-1)*dy && gx0<=(nx-1)*dx\
                && gy0>=0 && gx0>=0){
                countXbreak++;
                UpdateSufaceTime(sufTime, sufWeight, dx, dy, \
                    gx0, gy0, 1000.0*oneTime, disErr);
            }
            if(gx0<=(nx-1)*dx && (gy0>(ny-1)*dy || gy0<0)){break;}
            if(gx0>=0 && (gy0>(ny-1)*dy || gy0<0)){break;}
            countY++;
        }
        if(countXbreak==0){break;}
        countYmax=max(countYmax,countY);
    }
//cout<<thetaAngleInp<<"-";
//cout<<"("<<countX<<","<<countYmax<<")"<<endl;

    thetaAngleOld=0;
    thetaAngleInp=0;
    while(true){
        countX++;
        int countXbreak=0;
        double tanThetaOld=tan(thetaAngleOld);
        double thetaAdd=atan(tanThetaAdd_0/(1.0+tanThetaOld*tanThetaOld\
            +abs(tanThetaOld*tanThetaAdd_0)));
        thetaAdd=max(thetaAdd,double(0.00001));
        thetaAdd=min(thetaAdd,double(0.001));
        double thetaAngle=thetaAngleOld-thetaAdd;
        thetaAngleInp=thetaAngle*180.0/pi;
        thetaAngleOld=thetaAngle;
        if(thetaAngleInp>=maxAngle){break;}

        countY=0;
        double gx0=sx;
        while(true){
            double gy0=sy,gz0=gz;
            double thetaAzimuth=(thetaAdd*countY)*180.0/pi;
            if(thetaAzimuth>90.0){break;}
            oneTime=GetSimplifiedRayTraceTimeDownToUp(\
                vel3d, gx0, gy0, gz0, sx, sy, sz, dx, dy, dz, \
                thetaAngleInp, thetaAzimuth);
            if(gy0<=(ny-1)*dy && gx0<=(nx-1)*dx\
                && gy0>=0 && gx0>=0){
                countXbreak++;
                UpdateSufaceTime(sufTime, sufWeight, dx, dy, \
                    gx0, gy0, 1000.0*oneTime, disErr);
            }
            if(gx0<=(nx-1)*dx && (gy0>(ny-1)*dy || gy0<0)){break;}
            if(gx0>=0 && (gy0>(ny-1)*dy || gy0<0)){break;}
            countY++;
        }
        //countYmax=max(countYmax,countY);

        countY=1;
        gx0=sx;
        while(true){
            double gy0=sy,gz0=gz;
            double thetaAzimuth=-(thetaAdd*countY)*180.0/pi;
            if(thetaAzimuth<-90.0){break;}
            oneTime=GetSimplifiedRayTraceTimeDownToUp(\
                vel3d, gx0, gy0, gz0, sx, sy, sz, dx, dy, dz, \
                thetaAngleInp, thetaAzimuth);
            if(gy0<=(ny-1)*dy && gx0<=(nx-1)*dx\
                && gy0>=0 && gx0>=0){
                countXbreak++;
                UpdateSufaceTime(sufTime, sufWeight, dx, dy, \
                    gx0, gy0, 1000.0*oneTime, disErr);
            }
            if(gx0<=(nx-1)*dx && (gy0>(ny-1)*dy || gy0<0)){break;}
            if(gx0>=0 && (gy0>(ny-1)*dy || gy0<0)){break;}
            countY++;
        }
        if(countXbreak==0){break;}
        //countYmax=max(countYmax,countY);
    }
//cout<<thetaAngleInp<<"-";
//cout<<"("<<countX<<","<<countYmax<<")"<<endl;

    fmat sufTimeFloat(sufTime.n_rows,sufTime.n_cols,fill::zeros);
    for(int j=0;j<sufTime.n_cols;j++){
    for(int i=0;i<sufTime.n_rows;i++){
        double sumW=0.0;
        for(int k=0;k<sufTime.n_slices;k++){
            sufTimeFloat(i,j)+=sufTime(i,j,k)*sufWeight(i,j,k);
            sumW+=sufWeight(i,j,k);
        }
        sufTimeFloat(i,j)=0.001*sufTimeFloat(i,j)/sumW;
    }}
    return sufTimeFloat;
}

void GetSimplifiedSufaceRayTraceTime2D(\
    fmat* sufTimeAll2d, int sxid, fcube* vel3d, double thetaAddDis, double maxAngle,double disErr, \
    double gz, double sx, double sy, double sz, double dx, double dy, double dz,\
    int interXnum, int interYnum, int interZnum, int nSmoothOfTime)
{
    fcube vel3dInter=vel3d[0];
    for(int k=0;k<interZnum;k++){
        fcubeLinearInterpolation3dBySlice(vel3dInter);
        dz=dz/2.0;
    }
    for(int k=0;k<interYnum;k++){
        fcubeLinearInterpolation3dByCol(vel3dInter);
        dy=dy/2.0;
    }
    for(int k=0;k<interXnum;k++){
        fcubeLinearInterpolation3dByRow(vel3dInter);
        dx=dx/2.0;
    }
    fmat sufTravel=GetSimplifiedSufaceRayTraceTime(\
        vel3dInter, thetaAddDis, maxAngle, disErr, \
        gz, sx, sy, sz, dx, dy, dz);
    
    if(sufTravel.n_cols<5){
    for(int j=0;j<sufTravel.n_cols;j++){
        sufTravel.col(j)=smoothdig(sufTravel.col(j),sufTravel.n_rows,nSmoothOfTime);
    }}else{
        sufTravel=fmatsmooth(sufTravel,sufTravel.n_rows,sufTravel.n_cols,nSmoothOfTime);
    }
    for(int k=0;k<interYnum;k++){fmatAntiLinearInterpolation2dByCol(sufTravel);}
    for(int k=0;k<interXnum;k++){fmatAntiLinearInterpolation2dByRow(sufTravel);}
    sufTimeAll2d[0].col(sxid)=sufTravel.col(0);
}

#endif
