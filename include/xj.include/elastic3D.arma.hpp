/*********(version 1.0)***********/
/*
wave2D.h
    c++ head file: 
*/
/********************************/
#ifndef ELASTIC3D_ARMA_HPP
#define ELASTIC3D_ARMA_HPP

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <thread>
#include <future>
#include "../xjc.h"
#include <omp.h>

using namespace std;
using namespace arma;
/////////////////////////////////////////////////////////////////////////////////////////////
fvec getCoe1st(int nCoe);
class elastic3D_ARMA{ 
private:
public:
//_pd*_: means partial derivative to x.or.y.or.z
//mpar_: model parameter
    fcube mpar_1_dec_ro,mpar_lmd_add_2miu,mpar_lmd,mpar_miu;
    fcube mpar_1_dec_ro_pdx_jc1,mpar_lmd_add_2miu_pdx_jc1,\
        mpar_lmd_pdx_jc1,mpar_miu_pdx_jc1;
    fcube mpar_1_dec_ro_pdy_jc1,mpar_lmd_add_2miu_pdy_jc1,\
        mpar_lmd_pdy_jc1,mpar_miu_pdy_jc1;
    fcube mpar_1_dec_ro_pdz_jc1,mpar_lmd_add_2miu_pdz_jc1,\
        mpar_lmd_pdz_jc1,mpar_miu_pdz_jc1;
    fcube mpar_ro,mpar_vp,mpar_vs,mpar_vp2_square;
    //fcube &mpar_1=mpar_1_dec_ro;
    fcube mpar_1;
    fcube vx,vy,vz,\
        vx_pdx_t1,vx_pdy_t1,vx_pdz_t1,\
        vy_pdx_t1,vy_pdy_t1,vy_pdz_t1,\
        vz_pdx_t1,vz_pdy_t1,vz_pdz_t1,\
        vx_pdx_t2,vx_pdy_t2,vx_pdz_t2,\
        vy_pdx_t2,vy_pdy_t2,vy_pdz_t2,\
        vz_pdx_t2,vz_pdy_t2,vz_pdz_t2;
    fcube txx,tyy,tzz,\
        txx_pdx_t1,tyy_pdx_t1,tzz_pdx_t1,\
        txx_pdy_t1,tyy_pdy_t1,tzz_pdy_t1,\
        txx_pdz_t1,tyy_pdz_t1,tzz_pdz_t1,\
        txx_pdx_t2,tyy_pdx_t2,tzz_pdx_t2,\
        txx_pdy_t2,tyy_pdy_t2,tzz_pdy_t2,\
        txx_pdz_t2,tyy_pdz_t2,tzz_pdz_t2;
    fcube txy,txz,tyz,txy_t2,txz_t2,tyz_t2,\
        txy_pdx_t1,txy_pdy_t1,txy_pdx_t2,txy_pdy_t2,\
        txz_pdx_t1,txz_pdz_t1,txz_pdx_t2,txz_pdz_t2,\
        tyz_pdy_t1,tyz_pdz_t1,tyz_pdy_t2,tyz_pdz_t2;

    fcube &acoustic_p=tzz,\
        &acoustic_p_pdx_t1=tzz_pdx_t1,\
        &acoustic_p_pdy_t1=tzz_pdy_t1,\
        &acoustic_p_pdz_t1=tzz_pdz_t1,\
        &acoustic_p_pdx_t2=tzz_pdx_t2,\
        &acoustic_p_pdy_t2=tzz_pdy_t2,\
        &acoustic_p_pdz_t2=tzz_pdz_t2;
    fvec coe1st,coe2nd;
    fcube dataSaveUp2d,dataSaveDown2d,\
        dataSaveLeft2d,dataSaveRight2d,
        fieldSave01,fieldSave02;
    fcube &dataSuface2d=dataSaveUp2d;

    float dx,dy,dz,dt,PML_wide,isPMLSurface;
    float TheoreticalReflectionCoefficient,\
        PML_AttenuationCapacity_X,\
        PML_AttenuationCapacity_Y,\
        PML_AttenuationCapacity_Z;
    float &R=TheoreticalReflectionCoefficient,\
        &C_X=PML_AttenuationCapacity_X,\
        &C_Y=PML_AttenuationCapacity_Y,\
        &C_Z=PML_AttenuationCapacity_Z;
    int nx,ny,nz,nt,nzSampleOfFreeSurface;
    int ompThreadNum,maxThreadNum,nCoe;

    
    elastic3D_ARMA();
    elastic3D_ARMA(const int x, const int y, const int z);
    ~elastic3D_ARMA();

    void cleardata();
    void updatepar();
    
    void initialize(const int x, const int y, const int z);
    void calx_p(fcube &ut2 , fcube &ut1, const fcube &u,\
         const fcube &m, const int pjc);
    void calx_p_pseudo2d(fcube &ut2 , fcube &ut1, const fcube &u,\
         const fcube &m, const int pjc);
    void caly_p(fcube &ut2 , fcube &ut1, const fcube &u,\
         const fcube &m, const int pjc);
    void calz_p(fcube &ut2 , fcube &ut1, const fcube &u,\
         const fcube &m, const int pjc);
    void calz_p_pseudo2d(fcube &ut2 , fcube &ut1, const fcube &u,\
         const fcube &m, const int pjc);
};
///////////////////////////////////////////////////////////////////////////////////////////
void elastic3D_ARMA::initialize(const int x, const int y, const int z)
{
    nx=x;ny=y;nz=z;
    dx=5.0;dy=5.0;dz=5.0;dt=0.0005;
    PML_wide=50.0;isPMLSurface=1.0;R=9.0;
    nzSampleOfFreeSurface=60,nCoe=8;
    ompThreadNum=1,maxThreadNum=9;
    coe1st=getCoe1st(nCoe);
    C_Y=(R)*3.0/2.0/(PML_wide)/(PML_wide)/(PML_wide)/dy/dy/dy;
    C_X=(R)*3.0/2.0/(PML_wide)/(PML_wide)/(PML_wide)/dx/dx/dx;
    C_Z=(R)*3.0/2.0/(PML_wide)/(PML_wide)/(PML_wide)/dz/dz/dz;
    mpar_1_dec_ro.zeros(nx,ny,nz),mpar_lmd_add_2miu.zeros(nx,ny,nz),\
    mpar_lmd.zeros(nx,ny,nz),mpar_miu.zeros(nx,ny,nz);
    mpar_ro.zeros(nx,ny,nz),mpar_vp.zeros(nx,ny,nz),mpar_vs.zeros(nx,ny,nz);
    mpar_vp2_square.zeros(nx,ny,nz),mpar_1.set_size(nx,ny,nz),mpar_1.fill(1.0);
    mpar_1_dec_ro_pdx_jc1.zeros(nx,ny,nz),mpar_lmd_add_2miu_pdx_jc1.zeros(nx,ny,nz),\
    mpar_lmd_pdx_jc1.zeros(nx,ny,nz),mpar_miu_pdx_jc1.zeros(nx,ny,nz);
    mpar_1_dec_ro_pdy_jc1.zeros(nx,ny,nz),mpar_lmd_add_2miu_pdy_jc1.zeros(nx,ny,nz),\
    mpar_lmd_pdy_jc1.zeros(nx,ny,nz),mpar_miu_pdy_jc1.zeros(nx,ny,nz);
    mpar_1_dec_ro_pdz_jc1.zeros(nx,ny,nz),mpar_lmd_add_2miu_pdz_jc1.zeros(nx,ny,nz),\
    mpar_lmd_pdz_jc1.zeros(nx,ny,nz),mpar_miu_pdz_jc1.zeros(nx,ny,nz);
    vx.zeros(nx,ny,nz),vy.zeros(nx,ny,nz),vz.zeros(nx,ny,nz),\
    vx_pdx_t1.zeros(nx,ny,nz),vx_pdy_t1.zeros(nx,ny,nz),vx_pdz_t1.zeros(nx,ny,nz),\
    vy_pdx_t1.zeros(nx,ny,nz),vy_pdy_t1.zeros(nx,ny,nz),vy_pdz_t1.zeros(nx,ny,nz),\
    vz_pdx_t1.zeros(nx,ny,nz),vz_pdy_t1.zeros(nx,ny,nz),vz_pdz_t1.zeros(nx,ny,nz),\
    vx_pdx_t2.zeros(nx,ny,nz),vx_pdy_t2.zeros(nx,ny,nz),vx_pdz_t2.zeros(nx,ny,nz),\
    vy_pdx_t2.zeros(nx,ny,nz),vy_pdy_t2.zeros(nx,ny,nz),vy_pdz_t2.zeros(nx,ny,nz),\
    vz_pdx_t2.zeros(nx,ny,nz),vz_pdy_t2.zeros(nx,ny,nz),vz_pdz_t2.zeros(nx,ny,nz);
    txx.zeros(nx,ny,nz),tyy.zeros(nx,ny,nz),tzz.zeros(nx,ny,nz),\
    txx_pdx_t1.zeros(nx,ny,nz),tyy_pdx_t1.zeros(nx,ny,nz),tzz_pdx_t1.zeros(nx,ny,nz),\
    txx_pdy_t1.zeros(nx,ny,nz),tyy_pdy_t1.zeros(nx,ny,nz),tzz_pdy_t1.zeros(nx,ny,nz),\
    txx_pdz_t1.zeros(nx,ny,nz),tyy_pdz_t1.zeros(nx,ny,nz),tzz_pdz_t1.zeros(nx,ny,nz),\
    txx_pdx_t2.zeros(nx,ny,nz),tyy_pdx_t2.zeros(nx,ny,nz),tzz_pdx_t2.zeros(nx,ny,nz),\
    txx_pdy_t2.zeros(nx,ny,nz),tyy_pdy_t2.zeros(nx,ny,nz),tzz_pdy_t2.zeros(nx,ny,nz),\
    txx_pdz_t2.zeros(nx,ny,nz),tyy_pdz_t2.zeros(nx,ny,nz),tzz_pdz_t2.zeros(nx,ny,nz);
    txy.zeros(nx,ny,nz),txz.zeros(nx,ny,nz),tyz.zeros(nx,ny,nz),\
    txy_pdx_t1.zeros(nx,ny,nz),txy_pdy_t1.zeros(nx,ny,nz),txy_pdx_t2.zeros(nx,ny,nz),txy_pdy_t2.zeros(nx,ny,nz),\
    txz_pdx_t1.zeros(nx,ny,nz),txz_pdz_t1.zeros(nx,ny,nz),txz_pdx_t2.zeros(nx,ny,nz),txz_pdz_t2.zeros(nx,ny,nz),\
    tyz_pdy_t1.zeros(nx,ny,nz),tyz_pdz_t1.zeros(nx,ny,nz),tyz_pdy_t2.zeros(nx,ny,nz),tyz_pdz_t2.zeros(nx,ny,nz);
}
elastic3D_ARMA::elastic3D_ARMA()
{
    nx=0;ny=0;nz=0;
    dx=5.0;dy=5.0;dz=5.0;dt=0.0005;
    PML_wide=50.0;isPMLSurface=1.0;R=9.0;
    nzSampleOfFreeSurface=60;
    ompThreadNum=1,maxThreadNum=9;
    nCoe=8;coe1st=getCoe1st(nCoe);
    cout<<"Warning: Creat an Empty object-wave_modeling_2D"<<endl;
}

elastic3D_ARMA::elastic3D_ARMA(const int x, const int y, const int z)
{
    nx=x;ny=y;nz=z;
    dx=5.0;dy=5.0;dz=5.0;dt=0.0005;
    PML_wide=50.0;isPMLSurface=1.0;R=9.0;
    nzSampleOfFreeSurface=60;
    ompThreadNum=1,maxThreadNum=9;
    nCoe=8;coe1st=getCoe1st(nCoe);
    C_Y=(R)*3.0/2.0/(PML_wide)/(PML_wide)/(PML_wide)/dy/dy/dy;
    C_X=(R)*3.0/2.0/(PML_wide)/(PML_wide)/(PML_wide)/dx/dx/dx;
    C_Z=(R)*3.0/2.0/(PML_wide)/(PML_wide)/(PML_wide)/dz/dz/dz;
    mpar_1_dec_ro.zeros(nx,ny,nz),mpar_lmd_add_2miu.zeros(nx,ny,nz),\
    mpar_lmd.zeros(nx,ny,nz),mpar_miu.zeros(nx,ny,nz);
    mpar_ro.zeros(nx,ny,nz),mpar_vp.zeros(nx,ny,nz),mpar_vs.zeros(nx,ny,nz);
    mpar_vp2_square.zeros(nx,ny,nz),mpar_1.set_size(nx,ny,nz),mpar_1.fill(1.0);
    mpar_1_dec_ro_pdx_jc1.zeros(nx,ny,nz),mpar_lmd_add_2miu_pdx_jc1.zeros(nx,ny,nz),\
    mpar_lmd_pdx_jc1.zeros(nx,ny,nz),mpar_miu_pdx_jc1.zeros(nx,ny,nz);
    mpar_1_dec_ro_pdy_jc1.zeros(nx,ny,nz),mpar_lmd_add_2miu_pdy_jc1.zeros(nx,ny,nz),\
    mpar_lmd_pdy_jc1.zeros(nx,ny,nz),mpar_miu_pdy_jc1.zeros(nx,ny,nz);
    mpar_1_dec_ro_pdz_jc1.zeros(nx,ny,nz),mpar_lmd_add_2miu_pdz_jc1.zeros(nx,ny,nz),\
    mpar_lmd_pdz_jc1.zeros(nx,ny,nz),mpar_miu_pdz_jc1.zeros(nx,ny,nz);
    vx.zeros(nx,ny,nz),vy.zeros(nx,ny,nz),vz.zeros(nx,ny,nz),\
    vx_pdx_t1.zeros(nx,ny,nz),vx_pdy_t1.zeros(nx,ny,nz),vx_pdz_t1.zeros(nx,ny,nz),\
    vy_pdx_t1.zeros(nx,ny,nz),vy_pdy_t1.zeros(nx,ny,nz),vy_pdz_t1.zeros(nx,ny,nz),\
    vz_pdx_t1.zeros(nx,ny,nz),vz_pdy_t1.zeros(nx,ny,nz),vz_pdz_t1.zeros(nx,ny,nz),\
    vx_pdx_t2.zeros(nx,ny,nz),vx_pdy_t2.zeros(nx,ny,nz),vx_pdz_t2.zeros(nx,ny,nz),\
    vy_pdx_t2.zeros(nx,ny,nz),vy_pdy_t2.zeros(nx,ny,nz),vy_pdz_t2.zeros(nx,ny,nz),\
    vz_pdx_t2.zeros(nx,ny,nz),vz_pdy_t2.zeros(nx,ny,nz),vz_pdz_t2.zeros(nx,ny,nz);
    txx.zeros(nx,ny,nz),tyy.zeros(nx,ny,nz),tzz.zeros(nx,ny,nz),\
    txx_pdx_t1.zeros(nx,ny,nz),tyy_pdx_t1.zeros(nx,ny,nz),tzz_pdx_t1.zeros(nx,ny,nz),\
    txx_pdy_t1.zeros(nx,ny,nz),tyy_pdy_t1.zeros(nx,ny,nz),tzz_pdy_t1.zeros(nx,ny,nz),\
    txx_pdz_t1.zeros(nx,ny,nz),tyy_pdz_t1.zeros(nx,ny,nz),tzz_pdz_t1.zeros(nx,ny,nz),\
    txx_pdx_t2.zeros(nx,ny,nz),tyy_pdx_t2.zeros(nx,ny,nz),tzz_pdx_t2.zeros(nx,ny,nz),\
    txx_pdy_t2.zeros(nx,ny,nz),tyy_pdy_t2.zeros(nx,ny,nz),tzz_pdy_t2.zeros(nx,ny,nz),\
    txx_pdz_t2.zeros(nx,ny,nz),tyy_pdz_t2.zeros(nx,ny,nz),tzz_pdz_t2.zeros(nx,ny,nz);
    txy.zeros(nx,ny,nz),txz.zeros(nx,ny,nz),tyz.zeros(nx,ny,nz),\
    txy_pdx_t1.zeros(nx,ny,nz),txy_pdy_t1.zeros(nx,ny,nz),txy_pdx_t2.zeros(nx,ny,nz),txy_pdy_t2.zeros(nx,ny,nz),\
    txz_pdx_t1.zeros(nx,ny,nz),txz_pdz_t1.zeros(nx,ny,nz),txz_pdx_t2.zeros(nx,ny,nz),txz_pdz_t2.zeros(nx,ny,nz),\
    tyz_pdy_t1.zeros(nx,ny,nz),tyz_pdz_t1.zeros(nx,ny,nz),tyz_pdy_t2.zeros(nx,ny,nz),tyz_pdz_t2.zeros(nx,ny,nz);
}

elastic3D_ARMA::~elastic3D_ARMA()
{
    cout<<"Delete an object-wave_modeling_2D"<<endl;
}

void elastic3D_ARMA::cleardata()
{
    vx.fill(0.0),vy.fill(0.0),vz.fill(0.0),\
    vx_pdx_t1.fill(0.0),vx_pdy_t1.fill(0.0),vx_pdz_t1.fill(0.0),\
    vy_pdx_t1.fill(0.0),vy_pdy_t1.fill(0.0),vy_pdz_t1.fill(0.0),\
    vz_pdx_t1.fill(0.0),vz_pdy_t1.fill(0.0),vz_pdz_t1.fill(0.0),\
    vx_pdx_t2.fill(0.0),vx_pdy_t2.fill(0.0),vx_pdz_t2.fill(0.0),\
    vy_pdx_t2.fill(0.0),vy_pdy_t2.fill(0.0),vy_pdz_t2.fill(0.0),\
    vz_pdx_t2.fill(0.0),vz_pdy_t2.fill(0.0),vz_pdz_t2.fill(0.0);
    txx.fill(0.0),tyy.fill(0.0),tzz.fill(0.0),\
    txx_pdx_t1.fill(0.0),tyy_pdx_t1.fill(0.0),tzz_pdx_t1.fill(0.0),\
    txx_pdy_t1.fill(0.0),tyy_pdy_t1.fill(0.0),tzz_pdy_t1.fill(0.0),\
    txx_pdz_t1.fill(0.0),tyy_pdz_t1.fill(0.0),tzz_pdz_t1.fill(0.0),\
    txx_pdx_t2.fill(0.0),tyy_pdx_t2.fill(0.0),tzz_pdx_t2.fill(0.0),\
    txx_pdy_t2.fill(0.0),tyy_pdy_t2.fill(0.0),tzz_pdy_t2.fill(0.0),\
    txx_pdz_t2.fill(0.0),tyy_pdz_t2.fill(0.0),tzz_pdz_t2.fill(0.0);
    txy.fill(0.0),txz.fill(0.0),tyz.fill(0.0),\
    txy_pdx_t1.fill(0.0),txy_pdy_t1.fill(0.0),\
    txy_pdx_t2.fill(0.0),txy_pdy_t2.fill(0.0),\
    txz_pdx_t1.fill(0.0),txz_pdz_t1.fill(0.0),\
    txz_pdx_t2.fill(0.0),txz_pdz_t2.fill(0.0),\
    tyz_pdy_t1.fill(0.0),tyz_pdz_t1.fill(0.0),\
    tyz_pdy_t2.fill(0.0),tyz_pdz_t2.fill(0.0);
    cout<<"All data has clear!"<<endl;
}
 
void elastic3D_ARMA::updatepar()
{
    int i,j,k;
    coe1st=getCoe1st(nCoe);
    C_Y=(R)*3.0/2.0/(PML_wide)/(PML_wide)/(PML_wide)/dy/dy/dy;
    C_X=(R)*3.0/2.0/(PML_wide)/(PML_wide)/(PML_wide)/dx/dx/dx;
    C_Z=(R)*3.0/2.0/(PML_wide)/(PML_wide)/(PML_wide)/dz/dz/dz;    
    for(i=0;i<nx;i++){ 
    for(j=0;j<ny;j++){
    for(k=0;k<nz;k++){
        mpar_vp2_square(i,j,k)=mpar_vp(i,j,k)*mpar_vp(i,j,k);
        mpar_miu(i,j,k)=mpar_vs(i,j,k)*mpar_vs(i,j,k)*mpar_ro(i,j,k);
        mpar_lmd(i,j,k)=mpar_ro(i,j,k)*(mpar_vp(i,j,k)*mpar_vp(i,j,k)\
            -2.0*mpar_vs(i,j,k)*mpar_vs(i,j,k));
        mpar_lmd_add_2miu(i,j,k)=mpar_lmd(i,j,k)+2.0*mpar_miu(i,j,k);
        mpar_1_dec_ro(i,j,k)=1.0/mpar_ro(i,j,k);
        //mo1[i][j]=(lmd[i][j]+2*miu[i][j])/(2*lmd[i][j]+2*miu[i][j])/ro[i][j];
    }}}
    for(i=0;i<nx-1;i++){
    for(j=0;j<ny-1;j++){
    for(k=0;k<nz-1;k++){
        mpar_1_dec_ro_pdx_jc1(i,j,k)=\
            (mpar_1_dec_ro(i+1,j,k)+mpar_1_dec_ro(i,j,k))*0.5;
        mpar_1_dec_ro_pdy_jc1(i,j,k)=\
            (mpar_1_dec_ro(i,j+1,k)+mpar_1_dec_ro(i,j,k))*0.5;
        mpar_1_dec_ro_pdz_jc1(i,j,k)=\
            (mpar_1_dec_ro(i,j,k+1)+mpar_1_dec_ro(i,j,k))*0.5;
        mpar_lmd_add_2miu_pdx_jc1(i,j,k)=\
            (mpar_lmd_add_2miu(i+1,j,k)+mpar_lmd_add_2miu(i,j,k))*0.5;
        mpar_lmd_add_2miu_pdy_jc1(i,j,k)=\
            (mpar_lmd_add_2miu(i,j+1,k)+mpar_lmd_add_2miu(i,j,k))*0.5;
        mpar_lmd_add_2miu_pdz_jc1(i,j,k)=\
            (mpar_lmd_add_2miu(i,j,k+1)+mpar_lmd_add_2miu(i,j,k))*0.5;
        mpar_lmd_pdx_jc1(i,j,k)=(mpar_lmd(i+1,j,k)+mpar_lmd(i,j,k))*0.5;
        mpar_lmd_pdy_jc1(i,j,k)=(mpar_lmd(i,j+1,k)+mpar_lmd(i,j,k))*0.5;
        mpar_lmd_pdz_jc1(i,j,k)=(mpar_lmd(i,j,k+1)+mpar_lmd(i,j,k))*0.5;
        mpar_miu_pdx_jc1(i,j,k)=(mpar_miu(i+1,j,k)+mpar_miu(i,j,k))*0.5;
        mpar_miu_pdy_jc1(i,j,k)=(mpar_miu(i,j+1,k)+mpar_miu(i,j,k))*0.5;
        mpar_miu_pdz_jc1(i,j,k)=(mpar_miu(i,j,k+1)+mpar_miu(i,j,k))*0.5;
        //mo1[i][j]=(lmd[i][j]+2*miu[i][j])/(2*lmd[i][j]+2*miu[i][j])/ro[i][j];
    }}}
    j=ny-1;{
    for(i=0;i<nx-1;i++){
    for(k=0;k<nz-1;k++){
        mpar_1_dec_ro_pdx_jc1(i,j,k)=\
            (mpar_1_dec_ro(i+1,j,k)+mpar_1_dec_ro(i,j,k))*0.5;
        mpar_1_dec_ro_pdy_jc1(i,j,k)=\
            (mpar_1_dec_ro(i,j,k)+mpar_1_dec_ro(i,j,k))*0.5;
        mpar_1_dec_ro_pdz_jc1(i,j,k)=\
            (mpar_1_dec_ro(i,j,k+1)+mpar_1_dec_ro(i,j,k))*0.5;
        mpar_lmd_add_2miu_pdx_jc1(i,j,k)=\
            (mpar_lmd_add_2miu(i+1,j,k)+mpar_lmd_add_2miu(i,j,k))*0.5;
        mpar_lmd_add_2miu_pdy_jc1(i,j,k)=\
            (mpar_lmd_add_2miu(i,j,k)+mpar_lmd_add_2miu(i,j,k))*0.5;
        mpar_lmd_add_2miu_pdz_jc1(i,j,k)=\
            (mpar_lmd_add_2miu(i,j,k+1)+mpar_lmd_add_2miu(i,j,k))*0.5;
        mpar_lmd_pdx_jc1(i,j,k)=(mpar_lmd(i+1,j,k)+mpar_lmd(i,j,k))*0.5;
        mpar_lmd_pdy_jc1(i,j,k)=(mpar_lmd(i,j,k)+mpar_lmd(i,j,k))*0.5;
        mpar_lmd_pdz_jc1(i,j,k)=(mpar_lmd(i,j,k+1)+mpar_lmd(i,j,k))*0.5;
        mpar_miu_pdx_jc1(i,j,k)=(mpar_miu(i+1,j,k)+mpar_miu(i,j,k))*0.5;
        mpar_miu_pdy_jc1(i,j,k)=(mpar_miu(i,j,k)+mpar_miu(i,j,k))*0.5;
        mpar_miu_pdz_jc1(i,j,k)=(mpar_miu(i,j,k+1)+mpar_miu(i,j,k))*0.5;
        //mo1[i][j]=(lmd[i][j]+2*miu[i][j])/(2*lmd[i][j]+2*miu[i][j])/ro[i][j];
    }}}
    cout<<"updatepar finished!"<<endl;
}

void calx_3d(fcube *ut2 ,fcube *ut1,fcube *u, fcube *m,int pjc,\
    class elastic3D_ARMA *obj)
{
    obj->calx_p(ut2[0],ut1[0],u[0],m[0],pjc);
}
void elastic3D_ARMA::calx_p(fcube &ut2 , fcube &ut1, \
    const fcube &u, const fcube &m, const int pjc)
{
    int Z,coeNum;
    coeNum=this->nCoe;
    Z=this->nz-coeNum;
omp_set_num_threads(this->ompThreadNum);
#pragma omp parallel for
    for(int k=coeNum;k<Z;k++){
        int i,j,jc(pjc);
        float du1(0),du2(0),du(0),cx=this->C_X;
        float DX,DT,xshd,DT1,DX1;
        int X,Y;
        DX=this->dx,DT=this->dt,xshd=this->PML_wide;
        X=this->nx,Y=this->ny,DT1=1.0/DT,DX1=1.0/DX;
        fvec coeff=this->coe1st;
    for(j=coeNum;j<Y-coeNum;j++){
        for(i=coeNum;i<xshd+coeNum;i++){  
            du=0.0;
            for(int kc=0;kc<coeNum;kc++){
                du+=coeff(kc)*(u(i+kc+jc,j,k)-u(i-1-kc+jc,j,k));
            }
            du=m(i,j,k)*du*DX1;
            du1=cx*3000.0*DX*(xshd+coeNum-i)*DX*(xshd+coeNum-i);
            ut2(i,j,k)=((du+ut1(i,j,k)*(DT1-du1*0.5))/(DT1+du1*0.5));
        }
        for(i=xshd+coeNum;i<X-xshd-coeNum;i++)
        {  
            du=0.0;
            for(int kc=0;kc<coeNum;kc++){
                du+=coeff(kc)*(u(i+kc+jc,j,k)-u(i-1-kc+jc,j,k));
            }
            du=m(i,j,k)*du*DX1;
            ut2(i,j,k)=((du)*DT+ut1(i,j,k));
        }
        for(i=X-xshd-coeNum;i<X-coeNum;i++)
        {  
            du=0.0;
            for(int kc=0;kc<coeNum;kc++){
                du+=coeff(kc)*(u(i+kc+jc,j,k)-u(i-1-kc+jc,j,k));
            }
            du=0.5*(m(i+jc,j,k)+m(i,j,k))*du*DX1;
            du1=cx*3000.0*DX*(i-X+xshd+coeNum)*DX*(i-X+xshd+coeNum);
            ut2(i,j,k)=((du+ut1(i,j,k)*(DT1-du1*0.5))/(DT1+du1*0.5));
        }
    }
    }
    ut1.swap(ut2);
}
void calx_3d_pseudo2d(fcube *ut2 ,fcube *ut1,fcube *u, fcube *m,int pjc,\
    class elastic3D_ARMA *obj)
{
    obj->calx_p_pseudo2d(ut2[0],ut1[0],u[0],m[0],pjc);
}
void elastic3D_ARMA::calx_p_pseudo2d(fcube &ut2 , fcube &ut1, \
    const fcube &u, const fcube &m, const int pjc)
{
    int Z,coeNum;
    coeNum=this->nCoe;
    Z=this->nz-coeNum;
omp_set_num_threads(this->ompThreadNum);
#pragma omp parallel for
    for(int k=coeNum;k<Z;k++){
        int i,j,jc(pjc);
        float du1(0),du2(0),du(0);
        float DX,DX2,DT,xshd,DT1,DX1,cx=this->C_X*3000.0;
        int X,Y;
        DX=this->dx,DT=this->dt,xshd=this->PML_wide;
        X=this->nx,Y=this->ny,DT1=1.0/DT,DX1=1.0/DX;
        DX2=DX*DX;
        j=floor(Y/2.0);
        fvec coeff=this->coe1st;
        for(i=coeNum;i<xshd+coeNum;i++){  
            du=0.0;
            for(int kc=0;kc<coeNum;kc++){
                du+=coeff(kc)*(u(i+kc+jc,j,k)-u(i-1-kc+jc,j,k));
            }
            du=m(i,j,k)*du*DX1;
            float ni=(xshd+coeNum-i);
            du1=cx*DX2*ni*ni;
            ut2(i,j,k)=((du+ut1(i,j,k)*(DT1-du1*0.5))/(DT1+du1*0.5));
        }
        for(i=xshd+coeNum;i<X-xshd-coeNum;i++)
        {  
            du=0;
            for(int kc=0;kc<coeNum;kc++){
                du+=coeff(kc)*(u(i+kc+jc,j,k)-u(i-1-kc+jc,j,k));
            }
            du=m(i,j,k)*du*DX1;
            ut2(i,j,k)=((du)*DT+ut1(i,j,k));
        }
        for(i=X-xshd-coeNum;i<X-coeNum;i++)
        {  
            du=0;
            for(int kc=0;kc<coeNum;kc++){
                du+=coeff(kc)*(u(i+kc+jc,j,k)-u(i-1-kc+jc,j,k));
            }
            du=0.5*(m(i+jc,j,k)+m(i,j,k))*du*DX1;
            float ni=(i-X+xshd+coeNum);
            du1=cx*DX2*ni*ni;
            ut2(i,j,k)=((du+ut1(i,j,k)*(DT1-du1*0.5))/(DT1+du1*0.5));
    }
    }
    ut1.swap(ut2);
}

void caly_3d(fcube *ut2 ,fcube *ut1,fcube *u, fcube *m,int pjc,\
    class elastic3D_ARMA *obj)
{
    obj->caly_p(ut2[0],ut1[0],u[0],m[0],pjc);
}
void elastic3D_ARMA::caly_p(fcube &ut2 , fcube &ut1, \
    const fcube &u, const fcube &m, const int pjc)
{
    int X,coeNum;
    coeNum=this->nCoe;
    X=this->nx-coeNum;
omp_set_num_threads(this->ompThreadNum);
#pragma omp parallel for
    for(int i=coeNum;i<X;i++){
        float DY,DT,xshd,DY1,DT1;
        int Y,Z;
        int j,k,jc(pjc);
        float du1(0),du2(0),du(0),cy=this->C_Y;
        DY=this->dy,DT=this->dt,xshd=this->PML_wide;
        Y=this->ny,Z=this->nz,DY1=1.0/DY,DT1=1.0/DT;
        fvec coeff=this->coe1st;
    for(k=coeNum;k<Z-coeNum;k++){
        for(j=coeNum;j<xshd+coeNum;j++){
            du=0.0;
            for(int kc=0;kc<coeNum;kc++){
                du+=coeff(kc)*(u(i,j+kc+jc,k)-u(i,j-1-kc+jc,k));
            }
            du=m(i,j,k)*du*DY1;
            du1=cy*3000.0*DY*(xshd+coeNum-j)*DY*(xshd+coeNum-j);
            ut2(i,j,k)=((du+ut1(i,j,k)*(DT1-du1*0.5))/(DT1+du1*0.5));
        }
        for(j=xshd+coeNum;j<Y-xshd-coeNum;j++)
        {
            du=0.0;
            for(int kc=0;kc<coeNum;kc++){
                du+=coeff(kc)*(u(i,j+kc+jc,k)-u(i,j-1-kc+jc,k));
            }
            du=m(i,j,k)*du*DY1;
            ut2(i,j,k)=(du*DT+ut1(i,j,k));
        }
        for(j=Y-xshd-coeNum;j<Y-coeNum;j++)
        {
            du=0.0;
            for(int kc=0;kc<coeNum;kc++){
                du+=coeff(kc)*(u(i,j+kc+jc,k)-u(i,j-1-kc+jc,k));
            }
            du=m(i,j,k)*du*DY1;
            du1=cy*3000.0*DY*(j-Y+xshd+coeNum)*DY*(j-Y+xshd+coeNum);
            ut2(i,j,k)=((du+ut1(i,j,k)*(DT1-du1*0.5))/(DT1+du1*0.5));
        }
    }}
    ut1.swap(ut2);
}

void calz_3d(fcube *ut2 ,fcube *ut1,fcube *u, fcube *m,int pjc,\
    class elastic3D_ARMA *obj)
{
    obj->calz_p(ut2[0],ut1[0],u[0],m[0],pjc);
}
void elastic3D_ARMA::calz_p(fcube &ut2 , fcube &ut1,\
    const fcube &u, const fcube &m, const int pjc)
{
    int X,coeNum;
    coeNum=this->nCoe;
    X=this->nx-coeNum;
omp_set_num_threads(this->ompThreadNum);
#pragma omp parallel for
    for(int i=coeNum;i<X;i++){
        float DZ,DT,xshd,suface_PML,DZ1,DT1;
        int Y,Z,nSurface;
        int j,k,jc(pjc);
        float du1(0),du2(0),du(0),cz=this->C_Z;
        DZ=this->dz,DT=this->dt,xshd=this->PML_wide;
        DZ1=1.0/DZ,DT1=1.0/DT;
        Y=this->ny,Z=this->nz,suface_PML=this->isPMLSurface;
        nSurface=this->nzSampleOfFreeSurface;
        fvec coeff=this->coe1st;
    for(j=coeNum;j<Y-coeNum;j++){
        for(k=coeNum;k<xshd+coeNum;k++){  
            du=0.0;
            for(int kc=0;kc<coeNum;kc++){
                du+=coeff(kc)*(u(i,j,k+kc+jc)-u(i,j,k-1-kc+jc));
            }
            //take care of the FreeSurface_PML_boundary!!!
            //the freesurface should be: ro=1000.0, vp=0.0
            du=m(i,j,k)*du*DZ1;
            du1=cz*3000.0*DZ*suface_PML*(xshd+coeNum-k)*DZ\
                *suface_PML*(xshd+coeNum-k);
            ut2(i,j,k)=((du+ut1(i,j,k)*(DT1-du1*0.5))/(DT1+du1*0.5));
        }
        for(k=xshd+coeNum;k<Z-xshd-coeNum;k++)
        {  
            du=0.0;
            for(int kc=0;kc<coeNum;kc++){
                du+=coeff(kc)*(u(i,j,k+kc+jc)-u(i,j,k-1-kc+jc));
            }
            //take care of the FreeSurface_PML_boundary!!!
            //the freesurface should be: ro=1000.0, vp=0.0
            du=m(i,j,k)*du*DZ1;
            ut2(i,j,k)=((du)*DT+ut1(i,j,k));
        }
        for(k=Z-xshd-coeNum;k<Z-coeNum;k++)
        {  
            du=0.0;
            for(int kc=0;kc<coeNum;kc++){
                du+=coeff(kc)*(u(i,j,k+kc+jc)-u(i,j,k-1-kc+jc));
            }
            du=m(i,j,k)*du*DZ1;
            du1=cz*3000.0*DZ*(k-Z+xshd+coeNum)*DZ*(k-Z+xshd+coeNum);
            ut2(i,j,k)=((du+ut1(i,j,k)*(DT1-du1*0.5))/(DT1+du1*0.5));
        }
    }
    }
    ut1.swap(ut2);
}
void calz_3d_pseudo2d(fcube *ut2 ,fcube *ut1,fcube *u, fcube *m,int pjc,\
    class elastic3D_ARMA *obj)
{
    obj->calz_p_pseudo2d(ut2[0],ut1[0],u[0],m[0],pjc);
}
void elastic3D_ARMA::calz_p_pseudo2d(fcube &ut2 , fcube &ut1,\
    const fcube &u, const fcube &m, const int pjc)
{
    int X,coeNum;
    coeNum=this->nCoe;
    X=this->nx-coeNum;
omp_set_num_threads(this->ompThreadNum);
#pragma omp parallel for
    for(int i=coeNum;i<X;i++){
        float DZ,DZ2,DT,xshd,suface_PML,DZ1,DT1;
        int Y,Z,nSurface;
        int j,k,jc(pjc);
        float du1(0),du2(0),du(0),cz=this->C_Z*3000.0;
        DZ=this->dz,DT=this->dt,xshd=this->PML_wide;
        DZ1=1.0/DZ,DT1=1.0/DT,DZ2=DZ*DZ;
        Y=this->ny,Z=this->nz,suface_PML=this->isPMLSurface;
        nSurface=this->nzSampleOfFreeSurface;
        j=floor(Y/2.0);
        fvec coeff=this->coe1st;
        for(k=coeNum;k<xshd+coeNum;k++){  
            du=0;
            for(int kc=0;kc<coeNum;kc++){
                du+=coeff(kc)*(u(i,j,k+kc+jc)-u(i,j,k-1-kc+jc));
            }
            //take care of the FreeSurface_PML_boundary!!!
            //the freesurface should be: ro=1000.0, vp=0.0
            du=m(i,j,k)*du*DZ1;
            float ni=(xshd+coeNum-k);
            du1=cz*DZ2*suface_PML*ni*ni;
            ut2(i,j,k)=((du+ut1(i,j,k)*(DT1-du1*0.5))/(DT1+du1*0.5));
        }
        for(k=xshd+coeNum;k<Z-xshd-coeNum;k++)
        {  
            du=0;
            for(int kc=0;kc<coeNum;kc++){
                du+=coeff(kc)*(u(i,j,k+kc+jc)-u(i,j,k-1-kc+jc));
            }
            //take care of the FreeSurface_PML_boundary!!!
            //the freesurface should be: ro=1000.0, vp=0.0
            du=m(i,j,k)*du*DZ1;
            ut2(i,j,k)=((du)*DT+ut1(i,j,k));
        }
        for(k=Z-xshd-coeNum;k<Z-coeNum;k++)
        {  
            du=0;
            for(int kc=0;kc<coeNum;kc++){
                du+=coeff(kc)*(u(i,j,k+kc+jc)-u(i,j,k-1-kc+jc));
            }
            du=m(i,j,k)*du*DZ1;
            float ni=(k-Z+xshd+coeNum);
            du1=cz*DZ2*ni*ni;
            ut2(i,j,k)=((du+ut1(i,j,k)*(DT1-du1*0.5))/(DT1+du1*0.5));
    }
    }
    ut1.swap(ut2);
}
void OneThread_calx_p_pseudo2d(elastic3D_ARMA& obj, \
    fcube &ut2 , fcube &ut1, const fcube &u, \
    const fcube &m, const int pjc)
{
    int Z,coeNum;
    coeNum=obj.nCoe;
    Z=obj.nz-coeNum;
    int i,j,jc(pjc);
    float du1(0),du2(0),du(0);
    float DX,DX2,DT,xshd,DT1,DX1,cx=obj.C_X*3000.0;
    int X,Y;
    DX=obj.dx,DT=obj.dt,xshd=obj.PML_wide;
    X=obj.nx,Y=obj.ny,DT1=1.0/DT,DX1=1.0/DX;
    j=floor(Y/2.0);DX2=DX*DX;
    fvec coeff=obj.coe1st;
    for(int k=coeNum;k<Z;k++){
        for(i=coeNum;i<xshd+coeNum;i++){  
            du=0.0;
            for(int kc=0;kc<coeNum;kc++){
                du+=coeff(kc)*(u(i+kc+jc,j,k)-u(i-1-kc+jc,j,k));
            }
            du=m(i,j,k)*du*DX1;
            float ni=(xshd+coeNum-i);
            du1=cx*DX2*ni*ni;
            ut2(i,j,k)=((du+ut1(i,j,k)*(DT1-du1*0.5))/(DT1+du1*0.5));
        }
        for(i=xshd+coeNum;i<X-xshd-coeNum;i++)
        {  
            du=0;
            for(int kc=0;kc<coeNum;kc++){
                du+=coeff(kc)*(u(i+kc+jc,j,k)-u(i-1-kc+jc,j,k));
            }
            du=m(i,j,k)*du*DX1;
            ut2(i,j,k)=((du)*DT+ut1(i,j,k));
        }
        for(i=X-xshd-coeNum;i<X-coeNum;i++)
        {  
            du=0;
            for(int kc=0;kc<coeNum;kc++){
                du+=coeff(kc)*(u(i+kc+jc,j,k)-u(i-1-kc+jc,j,k));
            }
            du=0.5*(m(i+jc,j,k)+m(i,j,k))*du*DX1;
            float ni=(i-X+xshd+coeNum);
            du1=cx*DX2*ni*ni;
            ut2(i,j,k)=((du+ut1(i,j,k)*(DT1-du1*0.5))/(DT1+du1*0.5));
    }
    }
    ut1.swap(ut2);
}
void OneThread_calz_p_pseudo2d(elastic3D_ARMA& obj, \
    fcube &ut2 , fcube &ut1, const fcube &u, \
    const fcube &m, const int pjc)
{
    int X,coeNum;
    coeNum=obj.nCoe;
    X=obj.nx-coeNum;
    float DZ,DZ2,DT,xshd,suface_PML2,DZ1,DT1;
    int Y,Z,nSurface;
    int j,k,jc(pjc);
    float du1(0),du2(0),du(0),cz=obj.C_Z*3000.0;
    DZ=obj.dz,DT=obj.dt,xshd=obj.PML_wide;
    DZ1=1.0/DZ,DT1=1.0/DT,DZ2=DZ*DZ;
    Y=obj.ny,Z=obj.nz,suface_PML2=obj.isPMLSurface;
    suface_PML2=suface_PML2*suface_PML2;
    nSurface=obj.nzSampleOfFreeSurface;
    j=floor(Y/2.0);
    fvec coeff=obj.coe1st;
    for(int i=coeNum;i<X;i++){
        for(k=coeNum;k<xshd+coeNum;k++){  
            du=0;
            for(int kc=0;kc<coeNum;kc++){
                du+=coeff(kc)*(u(i,j,k+kc+jc)-u(i,j,k-1-kc+jc));
            }
            //take care of the FreeSurface_PML_boundary!!!
            //the freesurface should be: ro=1000.0, vp=0.0
            du=m(i,j,k)*du*DZ1;
            float ni=(xshd+coeNum-k);
            du1=cz*DZ2*suface_PML2*ni*ni;
            ut2(i,j,k)=((du+ut1(i,j,k)*(DT1-du1*0.5))/(DT1+du1*0.5));
        }
        for(k=xshd+coeNum;k<Z-xshd-coeNum;k++)
        {  
            du=0;
            for(int kc=0;kc<coeNum;kc++){
                du+=coeff(kc)*(u(i,j,k+kc+jc)-u(i,j,k-1-kc+jc));
            }
            //take care of the FreeSurface_PML_boundary!!!
            //the freesurface should be: ro=1000.0, vp=0.0
            du=m(i,j,k)*du*DZ1;
            ut2(i,j,k)=((du)*DT+ut1(i,j,k));
        }
        for(k=Z-xshd-coeNum;k<Z-coeNum;k++)
        {  
            du=0;
            for(int kc=0;kc<coeNum;kc++){
                du+=coeff(kc)*(u(i,j,k+kc+jc)-u(i,j,k-1-kc+jc));
            }
            du=m(i,j,k)*du*DZ1;
            float ni=(k-Z+xshd+coeNum);
            du1=cz*DZ2*ni*ni;
            ut2(i,j,k)=((du+ut1(i,j,k)*(DT1-du1*0.5))/(DT1+du1*0.5));
    }
    }
    ut1.swap(ut2);
}
void TimeSliceCal_elastic3D_MultiThread(class elastic3D_ARMA & obj)
{
    int k,jc0(0),jc1(1);
    thread *useThread;
    useThread=new thread[24];
    obj.ompThreadNum=round(obj.maxThreadNum/9.0);
    obj.ompThreadNum=max(obj.ompThreadNum,1);
    useThread[0]=thread(calx_3d,&obj.vx_pdx_t2,&obj.vx_pdx_t1,\
        &obj.txx,&obj.mpar_1_dec_ro_pdx_jc1,jc1,&obj);
    useThread[1]=thread(caly_3d,&obj.vx_pdy_t2,&obj.vx_pdy_t1,\
        &obj.txy,&obj.mpar_1_dec_ro,jc0,&obj);
    useThread[2]=thread(calz_3d,&obj.vx_pdz_t2,&obj.vx_pdz_t1,\
        &obj.txz,&obj.mpar_1_dec_ro,jc0,&obj);
    useThread[3]=thread(calx_3d,&obj.vy_pdx_t2,&obj.vy_pdx_t1,\
        &obj.txy,&obj.mpar_1_dec_ro,jc0,&obj);
    useThread[4]=thread(caly_3d,&obj.vy_pdy_t2,&obj.vy_pdy_t1,\
        &obj.tyy,&obj.mpar_1_dec_ro_pdy_jc1,jc1,&obj);
    useThread[5]=thread(calz_3d,&obj.vy_pdz_t2,&obj.vy_pdz_t1,\
        &obj.tyz,&obj.mpar_1_dec_ro,jc0,&obj);
    useThread[6]=thread(calx_3d,&obj.vz_pdx_t2,&obj.vz_pdx_t1,\
        &obj.txz,&obj.mpar_1_dec_ro,jc0,&obj);
    useThread[7]=thread(caly_3d,&obj.vz_pdy_t2,&obj.vz_pdy_t1,\
        &obj.tyz,&obj.mpar_1_dec_ro,jc0,&obj);
    useThread[8]=thread(calz_3d,&obj.vz_pdz_t2,&obj.vz_pdz_t1,\
        &obj.tzz,&obj.mpar_1_dec_ro_pdz_jc1,jc1,&obj);
    for(k=0;k<9;k++){
        if(useThread[k].joinable())
            useThread[k].join();
    }
    obj.vx=obj.vx_pdx_t1+obj.vx_pdy_t1+obj.vx_pdz_t1;
    obj.vy=obj.vy_pdx_t1+obj.vy_pdy_t1+obj.vy_pdz_t1;
    obj.vz=obj.vz_pdx_t1+obj.vz_pdy_t1+obj.vz_pdz_t1;

    obj.ompThreadNum=round(obj.maxThreadNum/9.0);
    obj.ompThreadNum=max(obj.ompThreadNum,1);
    useThread[9]=thread(calx_3d,&obj.txx_pdx_t2,&obj.txx_pdx_t1,\
        &obj.vx,&obj.mpar_lmd_add_2miu,jc0,&obj);
    useThread[10]=thread(caly_3d,&obj.txx_pdy_t2,&obj.txx_pdy_t1,\
        &obj.vy,&obj.mpar_lmd,jc0,&obj);
    useThread[11]=thread(calz_3d,&obj.txx_pdz_t2,&obj.txx_pdz_t1,\
        &obj.vz,&obj.mpar_lmd,jc0,&obj);
    useThread[12]=thread(caly_3d,&obj.tyy_pdy_t2,&obj.tyy_pdy_t1,\
        &obj.vy,&obj.mpar_lmd_add_2miu,jc0,&obj);
    useThread[13]=thread(calx_3d,&obj.tyy_pdx_t2,&obj.tyy_pdx_t1,\
        &obj.vx,&obj.mpar_lmd,jc0,&obj);
    useThread[14]=thread(calz_3d,&obj.tyy_pdz_t2,&obj.tyy_pdz_t1,\
        &obj.vz,&obj.mpar_lmd,jc0,&obj);
    useThread[15]=thread(calz_3d,&obj.tzz_pdz_t2,&obj.tzz_pdz_t1,\
        &obj.vz,&obj.mpar_lmd_add_2miu,jc0,&obj);
    useThread[16]=thread(calx_3d,&obj.tzz_pdx_t2,&obj.tzz_pdx_t1,\
        &obj.vx,&obj.mpar_lmd,jc0,&obj);
    useThread[17]=thread(caly_3d,&obj.tzz_pdy_t2,&obj.tzz_pdy_t1,\
        &obj.vy,&obj.mpar_lmd,jc0,&obj);
    for(k=9;k<18;k++){
        if(useThread[k].joinable())
            useThread[k].join();
    }
    obj.txx=obj.txx_pdx_t1+obj.txx_pdy_t1+obj.txx_pdz_t1;
    obj.tyy=obj.tyy_pdx_t1+obj.tyy_pdy_t1+obj.tyy_pdz_t1;
    obj.tzz=obj.tzz_pdx_t1+obj.tzz_pdy_t1+obj.tzz_pdz_t1;

    obj.ompThreadNum=round(obj.maxThreadNum/6.0);
    obj.ompThreadNum=max(obj.ompThreadNum,1);
    useThread[18]=thread(caly_3d,&obj.txy_pdy_t2,&obj.txy_pdy_t1,\
        &obj.vx,&obj.mpar_miu_pdy_jc1,jc1,&obj);
    useThread[19]=thread(calx_3d,&obj.txy_pdx_t2,&obj.txy_pdx_t1,\
        &obj.vy,&obj.mpar_miu_pdx_jc1,jc1,&obj);
    useThread[20]=thread(calz_3d,&obj.txz_pdz_t2,&obj.txz_pdz_t1,\
        &obj.vx,&obj.mpar_miu_pdz_jc1,jc1,&obj);
    useThread[21]=thread(calx_3d,&obj.txz_pdx_t2,&obj.txz_pdx_t1,\
        &obj.vz,&obj.mpar_miu_pdx_jc1,jc1,&obj);
    useThread[22]=thread(calz_3d,&obj.tyz_pdz_t2,&obj.tyz_pdz_t1,\
        &obj.vy,&obj.mpar_miu_pdz_jc1,jc1,&obj);
    useThread[23]=thread(caly_3d,&obj.tyz_pdy_t2,&obj.tyz_pdy_t1,\
        &obj.vz,&obj.mpar_miu_pdy_jc1,jc1,&obj);
    for(k=18;k<24;k++){
        if(useThread[k].joinable())
            useThread[k].join();
    }
    obj.txy=obj.txy_pdy_t1+obj.txy_pdx_t1;
    obj.txz=obj.txz_pdz_t1+obj.txz_pdx_t1;
    obj.tyz=obj.tyz_pdz_t1+obj.tyz_pdy_t1;
    delete [] useThread;
}

void TimeSliceCal_elastic3D_PurePwave_MultiThread(class elastic3D_ARMA & obj)
{
    int k,jc0(0),jc1(1);
    thread *useThread;
    useThread=new thread[9];
    obj.ompThreadNum=round(obj.maxThreadNum/3.0);
    obj.ompThreadNum=max(obj.ompThreadNum,1);
    useThread[0]=thread(calx_3d,&obj.vx_pdx_t2,&obj.vx_pdx_t1,\
        &obj.txx,&obj.mpar_1_dec_ro_pdx_jc1,jc1,&obj);
    useThread[1]=thread(caly_3d,&obj.vy_pdy_t2,&obj.vy_pdy_t1,\
        &obj.tyy,&obj.mpar_1_dec_ro_pdy_jc1,jc1,&obj);
    useThread[2]=thread(calz_3d,&obj.vz_pdz_t2,&obj.vz_pdz_t1,\
        &obj.tzz,&obj.mpar_1_dec_ro_pdz_jc1,jc1,&obj);
    for(k=0;k<3;k++){
        if(useThread[k].joinable())
            useThread[k].join();
    }
    obj.vx=obj.vx_pdx_t1;
    obj.vy=obj.vy_pdy_t1;
    obj.vz=obj.vz_pdz_t1;
    obj.ompThreadNum=round(obj.maxThreadNum/9.0);
    obj.ompThreadNum=max(obj.ompThreadNum,1);
    useThread[0]=thread(calx_3d,&obj.txx_pdx_t2,&obj.txx_pdx_t1,\
        &obj.vx,&obj.mpar_lmd_add_2miu,jc0,&obj);
    useThread[1]=thread(caly_3d,&obj.txx_pdy_t2,&obj.txx_pdy_t1,\
        &obj.vy,&obj.mpar_lmd,jc0,&obj);
    useThread[2]=thread(calz_3d,&obj.txx_pdz_t2,&obj.txx_pdz_t1,\
        &obj.vz,&obj.mpar_lmd,jc0,&obj);
    useThread[3]=thread(caly_3d,&obj.tyy_pdy_t2,&obj.tyy_pdy_t1,\
        &obj.vy,&obj.mpar_lmd_add_2miu,jc0,&obj);
    useThread[4]=thread(calx_3d,&obj.tyy_pdx_t2,&obj.tyy_pdx_t1,\
        &obj.vx,&obj.mpar_lmd,jc0,&obj);
    useThread[5]=thread(calz_3d,&obj.tyy_pdz_t2,&obj.tyy_pdz_t1,\
        &obj.vz,&obj.mpar_lmd,jc0,&obj);
    useThread[6]=thread(calz_3d,&obj.tzz_pdz_t2,&obj.tzz_pdz_t1,\
        &obj.vz,&obj.mpar_lmd_add_2miu,jc0,&obj);
    useThread[7]=thread(calx_3d,&obj.tzz_pdx_t2,&obj.tzz_pdx_t1,\
        &obj.vx,&obj.mpar_lmd,jc0,&obj);
    useThread[8]=thread(caly_3d,&obj.tzz_pdy_t2,&obj.tzz_pdy_t1,\
        &obj.vy,&obj.mpar_lmd,jc0,&obj);
    for(k=0;k<9;k++){
        if(useThread[k].joinable())
            useThread[k].join();
    }
    obj.txx=obj.txx_pdx_t1+obj.txx_pdy_t1+obj.txx_pdz_t1;
    obj.tyy=obj.tyy_pdx_t1+obj.tyy_pdy_t1+obj.tyy_pdz_t1;
    obj.tzz=obj.tzz_pdx_t1+obj.tzz_pdy_t1+obj.tzz_pdz_t1;
    delete [] useThread;
}

int TimeSliceCal_elastic2D_MultiThread(class elastic3D_ARMA & obj)
{
    int k,jc0(0),jc1(1);
    thread *useThread;
    useThread=new thread[10];

    obj.ompThreadNum=round(obj.maxThreadNum/4.0);
    obj.ompThreadNum=max(obj.ompThreadNum,1);
    useThread[0]=thread(calx_3d_pseudo2d,&obj.vx_pdx_t2,&obj.vx_pdx_t1,\
        &obj.txx,&obj.mpar_1_dec_ro_pdx_jc1,jc1,&obj);
    useThread[1]=thread(calz_3d_pseudo2d,&obj.vx_pdz_t2,&obj.vx_pdz_t1,\
        &obj.txz,&obj.mpar_1_dec_ro,jc0,&obj);
    useThread[2]=thread(calx_3d_pseudo2d,&obj.vz_pdx_t2,&obj.vz_pdx_t1,\
        &obj.txz,&obj.mpar_1_dec_ro,jc0,&obj);
    useThread[3]=thread(calz_3d_pseudo2d,&obj.vz_pdz_t2,&obj.vz_pdz_t1,\
        &obj.tzz,&obj.mpar_1_dec_ro_pdz_jc1,jc1,&obj);
    for(k=0;k<4;k++){
        if(useThread[k].joinable())
            useThread[k].join();
    }
    obj.vx=obj.vx_pdx_t1+obj.vx_pdz_t1;
    obj.vz=obj.vz_pdx_t1+obj.vz_pdz_t1;

    useThread[4]=thread(calx_3d_pseudo2d,&obj.txx_pdx_t2,&obj.txx_pdx_t1,\
        &obj.vx,&obj.mpar_lmd_add_2miu,jc0,&obj);
    useThread[5]=thread(calz_3d_pseudo2d,&obj.txx_pdz_t2,&obj.txx_pdz_t1,\
        &obj.vz,&obj.mpar_lmd,jc0,&obj);
    useThread[6]=thread(calz_3d_pseudo2d,&obj.tzz_pdz_t2,&obj.tzz_pdz_t1,\
        &obj.vz,&obj.mpar_lmd_add_2miu,jc0,&obj);
    useThread[7]=thread(calx_3d_pseudo2d,&obj.tzz_pdx_t2,&obj.tzz_pdx_t1,\
        &obj.vx,&obj.mpar_lmd,jc0,&obj);
    for(k=4;k<8;k++){
        if(useThread[k].joinable())
            useThread[k].join();
    }
    obj.txx=obj.txx_pdx_t1+obj.txx_pdz_t1;
    obj.tzz=obj.tzz_pdx_t1+obj.tzz_pdz_t1;

    obj.ompThreadNum=round(obj.maxThreadNum/2.0);
    obj.ompThreadNum=max(obj.ompThreadNum,1);
    useThread[8]=thread(calz_3d_pseudo2d,&obj.txz_pdz_t2,&obj.txz_pdz_t1,\
        &obj.vx,&obj.mpar_miu_pdz_jc1,jc1,&obj);
    useThread[9]=thread(calx_3d_pseudo2d,&obj.txz_pdx_t2,&obj.txz_pdx_t1,\
        &obj.vz,&obj.mpar_miu_pdx_jc1,jc1,&obj);
    for(k=8;k<10;k++){
        if(useThread[k].joinable())
            useThread[k].join();
    }
    obj.txz=obj.txz_pdz_t1+obj.txz_pdx_t1;
    delete [] useThread;
    return 0;
}

int TimeSliceCal_elastic2D_OneThread(class elastic3D_ARMA & obj)
{
    int k,jc0(0),jc1(1);
    obj.ompThreadNum=1;
    calx_3d_pseudo2d(&obj.vx_pdx_t2,&obj.vx_pdx_t1,\
        &obj.txx,&obj.mpar_1_dec_ro_pdx_jc1,jc1,&obj);
    calz_3d_pseudo2d(&obj.vx_pdz_t2,&obj.vx_pdz_t1,\
        &obj.txz,&obj.mpar_1_dec_ro,jc0,&obj);
    calx_3d_pseudo2d(&obj.vz_pdx_t2,&obj.vz_pdx_t1,\
        &obj.txz,&obj.mpar_1_dec_ro,jc0,&obj);
    calz_3d_pseudo2d(&obj.vz_pdz_t2,&obj.vz_pdz_t1,\
        &obj.tzz,&obj.mpar_1_dec_ro_pdz_jc1,jc1,&obj);
    obj.vx=obj.vx_pdx_t1+obj.vx_pdz_t1;
    obj.vz=obj.vz_pdx_t1+obj.vz_pdz_t1;

    calx_3d_pseudo2d(&obj.txx_pdx_t2,&obj.txx_pdx_t1,\
        &obj.vx,&obj.mpar_lmd_add_2miu,jc0,&obj);
    calz_3d_pseudo2d(&obj.txx_pdz_t2,&obj.txx_pdz_t1,\
        &obj.vz,&obj.mpar_lmd,jc0,&obj);
    calz_3d_pseudo2d(&obj.tzz_pdz_t2,&obj.tzz_pdz_t1,\
        &obj.vz,&obj.mpar_lmd_add_2miu,jc0,&obj);
    calx_3d_pseudo2d(&obj.tzz_pdx_t2,&obj.tzz_pdx_t1,\
        &obj.vx,&obj.mpar_lmd,jc0,&obj);
    obj.txx=obj.txx_pdx_t1+obj.txx_pdz_t1;
    obj.tzz=obj.tzz_pdx_t1+obj.tzz_pdz_t1;

    calz_3d_pseudo2d(&obj.txz_pdz_t2,&obj.txz_pdz_t1,\
        &obj.vx,&obj.mpar_miu_pdz_jc1,jc1,&obj);
    calx_3d_pseudo2d(&obj.txz_pdx_t2,&obj.txz_pdx_t1,\
        &obj.vz,&obj.mpar_miu_pdx_jc1,jc1,&obj);
    obj.txz=obj.txz_pdz_t1+obj.txz_pdx_t1;
    return 0;
}

void TimeSliceCal_elastic2D_PurePwave_MultiThread(class elastic3D_ARMA & obj)
{
    int k,jc0(0),jc1(1);
    thread *useThread;
    useThread=new thread[6];
    obj.ompThreadNum=round(obj.maxThreadNum/2.0);
    obj.ompThreadNum=max(obj.ompThreadNum,1);
    useThread[0]=thread(calx_3d_pseudo2d,&obj.vx_pdx_t2,&obj.vx_pdx_t1,\
        &obj.txx,&obj.mpar_1_dec_ro_pdx_jc1,jc1,&obj);
    useThread[1]=thread(calz_3d_pseudo2d,&obj.vz_pdz_t2,&obj.vz_pdz_t1,\
        &obj.tzz,&obj.mpar_1_dec_ro_pdz_jc1,jc1,&obj);
    for(k=0;k<2;k++){
        if(useThread[k].joinable())
            useThread[k].join();
    }
    obj.vx=obj.vx_pdx_t1;
    obj.vz=obj.vz_pdz_t1;

    obj.ompThreadNum=round(obj.maxThreadNum/4.0);
    obj.ompThreadNum=max(obj.ompThreadNum,1);
    useThread[2]=thread(calx_3d_pseudo2d,&obj.txx_pdx_t2,&obj.txx_pdx_t1,\
        &obj.vx,&obj.mpar_lmd_add_2miu,jc0,&obj);
    useThread[3]=thread(calz_3d_pseudo2d,&obj.txx_pdz_t2,&obj.txx_pdz_t1,\
        &obj.vz,&obj.mpar_lmd,jc0,&obj);
    useThread[4]=thread(calz_3d_pseudo2d,&obj.tzz_pdz_t2,&obj.tzz_pdz_t1,\
        &obj.vz,&obj.mpar_lmd_add_2miu,jc0,&obj);
    useThread[5]=thread(calx_3d_pseudo2d,&obj.tzz_pdx_t2,&obj.tzz_pdx_t1,\
        &obj.vx,&obj.mpar_lmd,jc0,&obj);
    for(k=2;k<6;k++){
        if(useThread[k].joinable())
            useThread[k].join();
    }
    obj.txx=obj.txx_pdx_t1+obj.txx_pdz_t1;
    obj.tzz=obj.tzz_pdx_t1+obj.tzz_pdz_t1;
    delete [] useThread;
}
int TimeSliceCal_acoustic3D_MultiThread(class elastic3D_ARMA & obj)
{
    int k,jc0(0),jc1(1);
    thread *useThread;
    useThread=new thread[6];
    obj.ompThreadNum=round(obj.maxThreadNum/3.0);
    obj.ompThreadNum=max(obj.ompThreadNum,1);
    useThread[0]=thread(calx_3d,&obj.vx_pdx_t2,&obj.vx_pdx_t1,\
        &obj.acoustic_p,&obj.mpar_1,jc1,&obj);
    useThread[1]=thread(caly_3d,&obj.vy_pdy_t2,&obj.vy_pdy_t1,\
        &obj.acoustic_p,&obj.mpar_1,jc1,&obj);
    useThread[2]=thread(calz_3d,&obj.vz_pdz_t2,&obj.vz_pdz_t1,\
        &obj.acoustic_p,&obj.mpar_1,jc1,&obj);
    for(k=0;k<3;k++){
        if(useThread[k].joinable())
            useThread[k].join();
    }

    useThread[3]=thread(calz_3d,&obj.acoustic_p_pdz_t2,\
        &obj.acoustic_p_pdz_t1,&obj.vz_pdz_t1,&obj.mpar_vp2_square,jc0,&obj);
    useThread[4]=thread(calx_3d,&obj.acoustic_p_pdx_t2,\
        &obj.acoustic_p_pdx_t1,&obj.vx_pdx_t1,&obj.mpar_vp2_square,jc0,&obj);
    useThread[5]=thread(caly_3d,&obj.acoustic_p_pdy_t2,\
        &obj.acoustic_p_pdy_t1,&obj.vy_pdy_t1,&obj.mpar_vp2_square,jc0,&obj);
    for(k=3;k<6;k++){
        if(useThread[k].joinable())
            useThread[k].join();
    }
    obj.acoustic_p=obj.acoustic_p_pdz_t1+obj.acoustic_p_pdx_t1\
        +obj.acoustic_p_pdy_t1;
    delete [] useThread;
    return 0;
}
int TimeSliceCal_acoustic3D_OneThread(class elastic3D_ARMA & obj)
{
    int k,jc0(0),jc1(1);
    obj.ompThreadNum=1;
    calx_3d(&obj.vx_pdx_t2,&obj.vx_pdx_t1,\
        &obj.acoustic_p,&obj.mpar_1,jc1,&obj);
    caly_3d(&obj.vy_pdy_t2,&obj.vy_pdy_t1,\
        &obj.acoustic_p,&obj.mpar_1,jc1,&obj);
    calz_3d(&obj.vz_pdz_t2,&obj.vz_pdz_t1,\
        &obj.acoustic_p,&obj.mpar_1,jc1,&obj);

    calz_3d(&obj.acoustic_p_pdz_t2,\
        &obj.acoustic_p_pdz_t1,&obj.vz_pdz_t1,&obj.mpar_vp2_square,jc0,&obj);
    calx_3d(&obj.acoustic_p_pdx_t2,\
        &obj.acoustic_p_pdx_t1,&obj.vx_pdx_t1,&obj.mpar_vp2_square,jc0,&obj);
    caly_3d(&obj.acoustic_p_pdy_t2,\
        &obj.acoustic_p_pdy_t1,&obj.vy_pdy_t1,&obj.mpar_vp2_square,jc0,&obj);

    obj.acoustic_p=obj.acoustic_p_pdz_t1+obj.acoustic_p_pdx_t1\
        +obj.acoustic_p_pdy_t1;
    return 0;
}
int TimeSliceCal_acoustic2D_MultiThread(class elastic3D_ARMA & obj)
{
    int k,jc0(0),jc1(1);
    thread *useThread;
    useThread=new thread[4];
    obj.ompThreadNum=round(obj.maxThreadNum/2.0);
    obj.ompThreadNum=max(obj.ompThreadNum,1);
    useThread[0]=thread(calx_3d_pseudo2d,&obj.vx_pdx_t2,&obj.vx_pdx_t1,\
        &obj.acoustic_p,&obj.mpar_1,jc1,&obj);
    useThread[1]=thread(calz_3d_pseudo2d,&obj.vz_pdz_t2,&obj.vz_pdz_t1,\
        &obj.acoustic_p,&obj.mpar_1,jc1,&obj);
    for(k=0;k<2;k++){
        if(useThread[k].joinable())
            useThread[k].join();
    }

    useThread[2]=thread(calz_3d_pseudo2d,&obj.acoustic_p_pdz_t2,\
        &obj.acoustic_p_pdz_t1,&obj.vz_pdz_t1,&obj.mpar_vp2_square,jc0,&obj);
    useThread[3]=thread(calx_3d_pseudo2d,&obj.acoustic_p_pdx_t2,\
        &obj.acoustic_p_pdx_t1,&obj.vx_pdx_t1,&obj.mpar_vp2_square,jc0,&obj);
    for(k=2;k<4;k++){
        if(useThread[k].joinable())
            useThread[k].join();
    }
    obj.acoustic_p=obj.acoustic_p_pdz_t1+obj.acoustic_p_pdx_t1;
    delete [] useThread;
    return 0;
}
int TimeSliceCal_acoustic2D_OneThread(class elastic3D_ARMA & obj)
{
    int k,jc0(0),jc1(1);
    obj.ompThreadNum=1;
    OneThread_calx_p_pseudo2d(obj,obj.vx_pdx_t2,obj.vx_pdx_t1,\
        obj.acoustic_p,obj.mpar_1,jc1);
    OneThread_calz_p_pseudo2d(obj,obj.vz_pdz_t2,obj.vz_pdz_t1,\
        obj.acoustic_p,obj.mpar_1,jc1);

    OneThread_calz_p_pseudo2d(obj,obj.acoustic_p_pdz_t2,\
        obj.acoustic_p_pdz_t1,obj.vz_pdz_t1,obj.mpar_vp2_square,jc0);
    OneThread_calx_p_pseudo2d(obj,obj.acoustic_p_pdx_t2,\
        obj.acoustic_p_pdx_t1,obj.vx_pdx_t1,obj.mpar_vp2_square,jc0);

    obj.acoustic_p=obj.acoustic_p_pdz_t1+obj.acoustic_p_pdx_t1;
    return 0;
}
fvec getCoe1st(int nCoe)
{
    nCoe=max(nCoe,2);
    nCoe=min(nCoe,8);
    fvec coe1st(nCoe);
    if(nCoe==2){
        coe1st(0)=1.125;
        coe1st(1)=-0.0416666666666667;
    }else if(nCoe==3){
        coe1st(0)=1.171875;
        coe1st(1)=-0.0651041666666667;
        coe1st(2)=0.0046875;
    }else if(nCoe==4){
        coe1st(0)=1.1962890625;
        coe1st(1)=-0.0797526041666667;
        coe1st(2)=0.0095703125;
        coe1st(3)=-0.000697544642857143;
    }else if(nCoe==5){
        coe1st(0)=1.21124267578125;
        coe1st(1)=-0.0897216796875;
        coe1st(2)=0.0138427734374998;
        coe1st(3)=-0.00176565987723212;
        coe1st(4)=0.00011867947048611;
    }else if(nCoe==6){
        coe1st(0)=1.22133636474608;
        coe1st(1)=-0.096931457519527;
        coe1st(2)=0.0174476623535166;
        coe1st(3)=-0.00296728951590422;
        coe1st(4)=0.000359005398220514;
        coe1st(5)=-2.18478116122177e-05;
    }else if(nCoe==7){
        coe1st(0)=1.22860622406092;
        coe1st(1)=-0.102383852005324;
        coe1st(2)=0.0204767704010242;
        coe1st(3)=-0.00417893273490091;
        coe1st(4)=0.000689453548855505;
        coe1st(5)=-7.69225033846921e-05;
        coe1st(6)=4.23651475172816e-06;
    }else if(nCoe==8){
        coe1st(0)=1.2340911;
        coe1st(1)=-1.0664985e-1;
        coe1st(2)=2.3036367e-2;
        coe1st(3)=-5.3423856e-3;
        coe1st(4)=1.0772712e-3;
        coe1st(5)=-1.6641888e-4;
        coe1st(6)=1.7021711e-5;
        coe1st(7)=-8.5234642e-7;
    }else{cout<<"Error: The coefficient_1st don't exist!!!"<<endl;}
    return coe1st;
}
void elastic3dParModify(elastic3D_ARMA& obj, \
    fcube modelvp, fcube modelvs, fcube modelrho, \
    int nx, int ny, int nz, int nt, \
    float dx, float dy, float dz, float dt, \
    int izFreeSurface, int isPMLSurface, int nthread,
    int nCor=8, int pmlWide=50, float R=9.0)
{
//initialize, do first
    obj.initialize(nx,ny,nz);
//maxThreadNum: Number of Thread for each source modeling;
    obj.maxThreadNum=nthread;
    obj.dz=dz;obj.dx=dx;
    obj.dy=dy;obj.dt=dt;
    obj.nt=nt;
    obj.isPMLSurface=isPMLSurface;
    obj.nzSampleOfFreeSurface=izFreeSurface;
    obj.mpar_vp=modelvp;
    //density-model
    obj.mpar_ro=modelrho;
    //model-vs
    obj.mpar_vs=modelvs;

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
    obj.PML_wide=pmlWide;
    obj.R=R;
    obj.nCoe=nCor;
    //update class obj, do before modeling
    obj.updatepar();
}
fmat acoustic2dTowforward(elastic3D_ARMA& obj, \
    int sx,  float f0=30.0, float waveletType=2.0)
{
    char fileMovie[1024];
    fileMovie[0]='\0';
    strcat(fileMovie,"./movie/movie.dat");
    fmat swap2d=obj.mpar_vp.col(0);
    swap2d=swap2d.st();
    datawrite(swap2d,fileMovie);
    fstream file;
    file.open(fileMovie,ios::in);
    int nx(obj.nx),ny(obj.ny),nz(obj.nz),nCoe(obj.nCoe);
    int sy(0),sz=obj.nzSampleOfFreeSurface;
    int (*fptr)(class elastic3D_ARMA&);
    float (*fptrWavelet)(int k, float DT, float hz,float det2);
    if(waveletType<1.5 || obj.isPMLSurface){
        fptrWavelet=wavelet01;
    }else if(waveletType>=1.5 && waveletType<2.5){
        fptrWavelet=wavelet02;
    }
    if(obj.maxThreadNum>=2){
        fptr=TimeSliceCal_elastic2D_MultiThread;
    }else{
        fptr=TimeSliceCal_elastic2D_OneThread;
    }
    fmat suftzz2d(obj.nt,obj.nx);
    for(int k=0;k<(obj.nt);k++)
    {
        obj.tzz(sx,sy,sz)+=1e6*(*fptrWavelet)(k,obj.dt,f0,0.0);
        if(obj.isPMLSurface>0.5){
            obj.tzz(sx,sy,sz-1)-=1e6*(*fptrWavelet)(k,obj.dt,f0,0.0);
        }
        //Using internal parallelism
        (*fptr)(obj);
        for(int i=0;i<nx;i++){
            suftzz2d(k,i)=obj.tzz(i,0,sz);
        }
        if(k%1000==0){
            cout<<"now is running : "<<k<<"; isMovie: "<<!(!file)<<endl;
        }
        if(!(!file) && k%100==0){
            swap2d=obj.tzz.col(0);
            swap2d+=obj.mpar_vp.col(0);
            swap2d=swap2d.st();
            swap2d=swap2d(span(sz,swap2d.n_rows-1),span::all);
            char str[1024];
            str[0]='\0';
            strcat(str,fileMovie);
            strcat(str,numtostr(k,8));
            datawrite(swap2d,str);
        }
    }
    cout<<"Completed: "<<sx<<endl;
    return suftzz2d;
}
fmat acoustic2dTowSimulation(elastic3D_ARMA& obj, \
    int sx, float f0=30.0, float waveletType=2.0)
{
    int nx(obj.nx),ny(obj.ny),nz(obj.nz),nCoe(1);
    int sy(0),sz=obj.nzSampleOfFreeSurface;
    obj.cleardata();
    int (*fptr)(class elastic3D_ARMA&);
    float (*fptrWavelet)(int k, float DT, float hz,float det2);
    //fptrWavelet=wavelet02;
    fptrWavelet=wavelet01;
    if(obj.maxThreadNum<=1){
        fptr=TimeSliceCal_acoustic2D_OneThread;
        //fptr=TimeSliceCal_elastic2D_OneThread;
    }else{
        fptr=TimeSliceCal_acoustic2D_MultiThread;
        //fptr=TimeSliceCal_elastic2D_MultiThread;

    }


    obj.dataSuface2d.zeros(obj.nx,obj.ny,obj.nt);
    ofstream outf;
    //outf.open("./movie/movie.orig.dat");
    for(int k=0;k<obj.nt;k++)
    {
        if(sx>=0){
            obj.tzz(sx,sy,sz)+=1e6*(*fptrWavelet)(k,obj.dt,f0,0.0);
            if(waveletType<1.5){
                obj.tzz(sx,sy,sz-2)-=1e6*(*fptrWavelet)(k,obj.dt,f0,0.0);
            }
        }else{
            for(int i=0;i<nx;i++){
                obj.tzz(i,sy,sz)+=1e6*(*fptrWavelet)(k,obj.dt,f0,0.0);
                if(waveletType<1.5){
                    obj.tzz(i,sy,sz-2)-=1e6*(*fptrWavelet)(k,obj.dt,f0,0.0);
                }
            }
        }
        //Using internal parallelism
        (*fptr)(obj);
        obj.dataSuface2d.slice(k)=obj.acoustic_p.slice(sz);

        if(k%1000==0)
        {cout<<"now is running : "<<k<<endl;}
        /*
        if(outf.is_open() && k%50==0)
        {
            fmat swap2d=obj.acoustic_p.col(0);
            swap2d=swap2d.st();
            datawrite(swap2d,outf);
        }*/
    }
    //outf.close();

    fmat suf2d;
    suf2d=obj.dataSuface2d.col(0);
    suf2d=suf2d.st();
    return suf2d;
}
fmat acoustic2dTowSimulation(elastic3D_ARMA& obj, elastic3D_ARMA& objdir,\
    int sx, float f0=30.0, bool doRemoveDirect=false)
{
    int nx(obj.nx),ny(obj.ny),nz(obj.nz),nCoe(obj.nCoe);
    int sy(0),sz=obj.nzSampleOfFreeSurface;
    obj.cleardata();
    int (*fptr)(class elastic3D_ARMA&);
    if(obj.maxThreadNum<=1){
        fptr=TimeSliceCal_acoustic2D_OneThread;
    }else{
        fptr=TimeSliceCal_acoustic2D_MultiThread;
    }

    obj.dataSaveUp2d.zeros(obj.nx,obj.nCoe,obj.nt);
    obj.dataSuface2d.zeros(obj.nx,obj.ny,obj.nt);
    ofstream outf;
    char file[1024];
    file[0]='\0';
    strcat(file,"./movie/movie.orig");
    for(int k=0;k<obj.nt;k++)
    {
        obj.tzz(sx,sy,sz)+=wavelet01(k,obj.dt,f0,0.0);
        obj.tzz(sx,sy,sz-2)-=wavelet01(k,obj.dt,f0,0.0);
        //Using internal parallelism
        (*fptr)(obj);
        obj.dataSuface2d.slice(k)=obj.acoustic_p.slice(sz);
        for(int k2=0;k2<nCoe;k2++){
        for(int k1=0;k1<nx;k1++){
            obj.dataSaveUp2d(k1,k2,k)=obj.acoustic_p(k1,0,sz+k2);
        }}
        if(k%1000==0)
        {cout<<"now is running: "<<k<<endl;}
        if(file[0]!='\0' && k%200==0)
        {
            char str[1024];
            str[0]='\0';
            strcat(str,file);
            strcat(str,numtostr(k,8));
            fmat swap2d=obj.acoustic_p.col(0);
            swap2d=swap2d.st();
            datawrite(swap2d,str);
        }
    }
    //outf.close();

    fmat suf2d;
if(doRemoveDirect){
    for(int k=obj.nzSampleOfFreeSurface+1;k<obj.mpar_vp.n_slices;k++){
        objdir.mpar_vp.slice(k)=objdir.mpar_vp.slice(obj.nzSampleOfFreeSurface);
        objdir.mpar_vs.slice(k)=objdir.mpar_vs.slice(obj.nzSampleOfFreeSurface);
        objdir.mpar_ro.slice(k)=objdir.mpar_ro.slice(obj.nzSampleOfFreeSurface);
    }   
    objdir.updatepar();
    objdir.cleardata();
    objdir.dataSuface2d.zeros(objdir.nx,objdir.ny,objdir.nt);
    objdir.dataSaveUp2d.zeros(obj.nx,obj.nCoe,obj.nt);
    for(int k=0;k<objdir.nt;k++)
    {
        objdir.tzz(sx,sy,sz)+=wavelet01(k,obj.dt,f0,0.0);
        objdir.tzz(sx,sy,sz-2)-=wavelet01(k,obj.dt,f0,0.0);
        //Using internal parallelism
        (*fptr)(objdir);
        objdir.dataSuface2d.slice(k)=objdir.acoustic_p.slice(sz);
        for(int k2=0;k2<nCoe;k2++){
        for(int k1=0;k1<nx;k1++){
            objdir.dataSaveUp2d(k1,k2,k)=objdir.acoustic_p(k1,0,sz+k2);
        }}
        if(k%1000==0)
        {cout<<"now is running : "<<k<<endl;}
    }
    cout<<"Completed task."<<endl;

    suf2d=obj.dataSuface2d.col(0)-objdir.dataSuface2d.col(0);
    obj.dataSaveUp2d-=objdir.dataSaveUp2d;
}else{
    suf2d=obj.dataSuface2d.col(0);
}
    suf2d=suf2d.st();
    return suf2d;
}
fmat acoustic2dSRMTowSimulation(elastic3D_ARMA& obj, elastic3D_ARMA& objdir, fcube suf3d)
{
/*
*/
for(int k=0;k<obj.nCoe;k++){
    fmat suf2d=suf3d.col(k);
    suf2d=suf2d.st();
    fmat suf2ddir=suf2d;
    suf2d.fill(0.0);
    suf2d(span(12,suf2d.n_rows-1),span::all)\
        =suf2ddir(span(0,suf2d.n_rows-13),span::all);
    //suf2d(span(0,suf2d.n_rows-1),span::all)\
        =suf2ddir(span(0,suf2d.n_rows-1),span::all);
    suf3d.col(k)=suf2d.st();

    cx_fmat suf2dcx(suf2d.n_rows,suf2d.n_cols);
    cx_fmat suf2dcx2(suf2d.n_rows,suf2d.n_cols);
    suf2dcx=fft2(suf2d);
    suf2dcx2.fill(0.0);
    float df=1/obj.dt/obj.nt;
    float pi=3.1415926;
    float dkx=1.0/obj.dx/obj.nx;
    cx_float a,b,c;
    a.real(0.0);
    for(int jf=0;jf<obj.nt/2;jf++){
        float wf=2.0*pi*df*jf;
    for(int jk=0;jk<obj.nx/2;jk++){
        float kx=2.0*pi*dkx*jk;
        float kz=wf*wf/1500/1500-kx*kx;
        if(kz>0.0){
            kz=sqrt(kz);
            a.imag(kz);
            suf2dcx2(jf,jk)=a*suf2dcx(jf,jk);
            suf2dcx2(jf,obj.nx-jk-1)=a*suf2dcx(jf,obj.nx-jk-1);
        }
    }
    }
    suf2d=real(ifft2(suf2dcx2))*2.0;
    suf3d.col(k)=suf2d.st();
}
    bool useMiddlePML=false;    
    int begMiddlePML=0;
    int nx(obj.nx),ny(obj.ny),nz(obj.nz),nCoe(obj.nCoe);
    int sy(0),sz=obj.nzSampleOfFreeSurface;
    obj.dataSuface2d.zeros(obj.nx,obj.ny,obj.nt);
    //obj.dataSuface2d.col(0)=suf2d.st();
    obj.cleardata();
    int (*fptr)(class elastic3D_ARMA&);
    if(obj.maxThreadNum<=1){
        fptr=TimeSliceCal_acoustic2D_OneThread;
    }else{
        fptr=TimeSliceCal_acoustic2D_MultiThread;
    }
    ofstream outf;
    char file[1024];
    file[0]='\0';
    strcat(file,"./movie/movie.srm");
    obj.dataSaveUp2d=suf3d;
    for(int k=0;k<obj.nt;k++)
    {
        //obj.tzz(sx,sy,sz)+=1e6*wavelet02(k,obj.dt,f0,0.0);
        //obj.acoustic_p.slice(sz)=obj.dataSuface2d.slice(k);
        for(int k2=0;k2<nCoe;k2++){
        for(int k1=0;k1<nx;k1++){
            obj.acoustic_p(k1,0,sz-k2)=obj.dataSaveUp2d(k1,k2,k);
        }}
        //Using internal parallelism
        (*fptr)(obj);
        obj.dataSuface2d.slice(k)=obj.acoustic_p.slice(sz);
        //obj.dataSuface2d.slice(k)-=obj.acoustic_p.slice(sz-1);
        if(k%1000==0)
        {cout<<"now is running : "<<k<<endl;}
        if(file[0]!='\0' && k%200==0)
        {
            char str[1024];
            str[0]='\0';
            strcat(str,file);
            strcat(str,numtostr(k,8));
            fmat swap2d=obj.acoustic_p.col(0);
            swap2d=swap2d.st();
            datawrite(swap2d,str);
        }
        
    }
    outf.close();
    cout<<"Completed task."<<endl;

    objdir.dataSuface2d.zeros(objdir.nx,objdir.ny,objdir.nt);
    //objdir.dataSuface2d.col(0)=suf2d.st();
    for(int k=obj.nzSampleOfFreeSurface+1;k<obj.mpar_vp.n_slices;k++){
        objdir.mpar_vp.slice(k)=objdir.mpar_vp.slice(obj.nzSampleOfFreeSurface);
        objdir.mpar_vs.slice(k)=objdir.mpar_vs.slice(obj.nzSampleOfFreeSurface);
        objdir.mpar_ro.slice(k)=objdir.mpar_ro.slice(obj.nzSampleOfFreeSurface);
    }   
    objdir.updatepar();
    objdir.cleardata();
    objdir.dataSaveUp2d=suf3d;
    for(int k=0;k<objdir.nt;k++)
    {
        //obj.tzz(sx,sy,sz)+=1e6*wavelet02(k,obj.dt,f0,0.0);
        //objdir.acoustic_p.slice(sz)=objdir.dataSuface2d.slice(k);
        for(int k2=0;k2<nCoe;k2++){
        for(int k1=0;k1<nx;k1++){
            objdir.acoustic_p(k1,0,sz-k2)=objdir.dataSaveUp2d(k1,k2,k);
        }}
        //Using internal parallelism
        (*fptr)(objdir);
        objdir.dataSuface2d.slice(k)=objdir.acoustic_p.slice(sz);
        //objdir.dataSuface2d.slice(k)-=objdir.acoustic_p.slice(sz-1);
        if(k%1000==0)
        {cout<<"now is running : "<<k<<endl;}
    }
    cout<<"Completed task."<<endl;
    
    fmat suf2d=obj.dataSuface2d.col(0)-objdir.dataSuface2d.col(0);
    suf2d=suf2d.st();
    //suf2d=suf2d/objdir.mpar_ro(0,sy,sz);
/*
    cx_fmat suf2dcx2(suf2d.n_rows,suf2d.n_cols);
    cx_fmat suf2dcx=fft2(suf2d);
    suf2dcx2.fill(0.0);
    float df=1/obj.dt/obj.nt;
    float pi=3.1415926;
    float dkx=1.0/obj.dx/obj.nx;
    cx_float a,b,c;
    a.real(0.0);
    for(int jf=0;jf<obj.nt/2;jf++){
        float wf=2.0*pi*df*jf;
    for(int jk=0;jk<obj.nx/2;jk++){
        float kx=2.0*pi*dkx*jk;
        float kz=wf*wf/1500/1500-kx*kx;
        if(kz>0.0){
            kz=sqrt(kz);
            a.imag(kz);
            suf2dcx2(jf,jk)=a*suf2dcx(jf,jk);
            suf2dcx2(jf,obj.nx-jk-1)=a*suf2dcx(jf,obj.nx-jk-1);
        }
    }
    }
    suf2d=real(ifft2(suf2dcx2))*2.0;
*/
    return suf2d;
}
void acousticRTM2dTowforward(elastic3D_ARMA& obj, \
    int sx,  float f0=30.0, float waveletType=2.0)
{
    int nfield=ceil((obj.nt-1)/1000.0);
    obj.fieldSave01.zeros(obj.nx,nfield,obj.nz);
    int nx(obj.nx),ny(obj.ny),nz(obj.nz),nCoe(obj.nCoe);
    int sy(0),sz=obj.nzSampleOfFreeSurface;
    obj.dataSaveUp2d.zeros(obj.nx,obj.nCoe,obj.nt);
    obj.dataSaveDown2d.zeros(obj.nx,obj.nCoe,obj.nt);
    obj.dataSaveLeft2d.zeros(obj.nt,obj.nCoe,obj.nz);
    obj.dataSaveRight2d.zeros(obj.nt,obj.nCoe,obj.nz);
    obj.cleardata();
    int endz=obj.nz-obj.nCoe-obj.PML_wide-2;
    int endx=obj.nx-obj.nCoe-obj.PML_wide-2;
    int begx=obj.nCoe+obj.PML_wide+1;
    int begz=obj.nCoe+obj.PML_wide+1;
    int (*fptr)(class elastic3D_ARMA&);
    float (*fptrWavelet)(int k, float DT, float hz,float det2);
    if(obj.maxThreadNum<=1){
        fptr=TimeSliceCal_acoustic2D_OneThread;
    }else{
        fptr=TimeSliceCal_acoustic2D_MultiThread;
    }
    if(waveletType<1.5){
        fptrWavelet=wavelet01;
    }else if(waveletType>=1.5 && waveletType<2.5){
        fptrWavelet=wavelet02;
    }
    fmat suf2d(obj.nx,obj.nt);
    for(int k=0;k<(obj.nt);k++)
    {
        obj.tzz(sx,sy,sz)+=1e6*(*fptrWavelet)(k,obj.dt,f0,0.0);
        //Using internal parallelism
        (*fptr)(obj);
        for(int k2=0;k2<nCoe;k2++){
            for(int k1=0;k1<nx;k1++){
                obj.dataSaveUp2d(k1,k2,k)=obj.tzz(k1,0,begz+k2);
                obj.dataSaveDown2d(k1,k2,k)=obj.tzz(k1,0,endz-nCoe+k2+1);
            }
            for(int k3=0;k3<nz;k3++){
                obj.dataSaveLeft2d(k,k2,k3)=obj.tzz(begx+k2,0,k3);
                obj.dataSaveRight2d(k,k2,k3)=obj.tzz(endx-nCoe+1+k2,0,k3);
            }
        }
        if(k%1000==0)
        {
            obj.fieldSave01.col(floor(k/1000.0))=obj.tzz.col(0);
            cout<<"now is running : "<<k<<endl;
        }
    }
    
    for(int k1=0;k1<begx;k1++){
        obj.dataSaveUp2d.row(k1).fill(0.0);
        obj.dataSaveDown2d.row(k1).fill(0.0);
    }
    for(int k1=endx+1;k1<obj.nx;k1++){
        obj.dataSaveUp2d.row(k1).fill(0.0);
        obj.dataSaveDown2d.row(k1).fill(0.0);
    }
    for(int k1=0;k1<=begz;k1++){
        obj.dataSaveLeft2d.slice(k1).fill(0.0);
        obj.dataSaveRight2d.slice(k1).fill(0.0);
    }
    for(int k1=endz;k1<obj.nz;k1++){
        obj.dataSaveLeft2d.slice(k1).fill(0.0);
        obj.dataSaveRight2d.slice(k1).fill(0.0);
    }
    cout<<"Completed: "<<sx<<endl;
}
fmat acousticRTM2dTowforward(elastic3D_ARMA& obj, \
    fmat& dataSuface2d, int sx)
{
    int nfield=ceil((obj.nt-1)/400.0);
    obj.fieldSave01.zeros(obj.nx,nfield,obj.nz);
    int nx(obj.nx),ny(obj.ny),nz(obj.nz),nCoe(obj.nCoe);
    int sy(0),sz=obj.nzSampleOfFreeSurface;
    fcube suf3d(nx,1,obj.nt);
    for(int i=0;i<nx;i++){
    for(int j=0;j<min(int(obj.nt),int(dataSuface2d.n_rows));j++){
        suf3d(i,0,j)=dataSuface2d(j,i);
    }}
    obj.dataSaveUp2d.zeros(obj.nx,obj.nCoe,obj.nt);
    obj.dataSaveDown2d.zeros(obj.nx,obj.nCoe,obj.nt);
    obj.dataSaveLeft2d.zeros(obj.nt,obj.nCoe,obj.nz);
    obj.dataSaveRight2d.zeros(obj.nt,obj.nCoe,obj.nz);
    obj.cleardata();
    int endz=obj.nz-obj.nCoe-obj.PML_wide-2;
    int endx=obj.nx-obj.nCoe-obj.PML_wide-2;
    int begx=obj.nCoe+obj.PML_wide+1;
    int begz=obj.nCoe+obj.PML_wide+1;
    int (*fptr)(class elastic3D_ARMA&);
    if(obj.maxThreadNum<=1){
        fptr=TimeSliceCal_acoustic2D_OneThread;
    }else{
        fptr=TimeSliceCal_acoustic2D_MultiThread;
    }
    fmat Lighting2d(obj.nx,obj.nz);
    fmat swap2d(obj.nx,obj.nz);
    Lighting2d.fill(0.0);
    for(int k=0;k<(obj.nt);k++)
    {
        obj.tzz.slice(sz)+=suf3d.slice(k);
        //Using internal parallelism
        (*fptr)(obj);
        for(int k2=0;k2<nCoe;k2++){
            for(int k1=0;k1<nx;k1++){
                obj.dataSaveUp2d(k1,k2,k)=obj.tzz(k1,0,begz+k2);
                obj.dataSaveDown2d(k1,k2,k)=obj.tzz(k1,0,endz-nCoe+k2+1);
            }
            for(int k3=0;k3<nz;k3++){
                obj.dataSaveLeft2d(k,k2,k3)=obj.tzz(begx+k2,0,k3);
                obj.dataSaveRight2d(k,k2,k3)=obj.tzz(endx-nCoe+1+k2,0,k3);
            }
        }
        if(k%400==0)
        {
            obj.fieldSave01.col(floor(k/400.0))=obj.tzz.col(0);
            cout<<"now is running : "<<k<<endl;
        }
    }
    
    for(int k1=0;k1<begx;k1++){
        obj.dataSaveUp2d.row(k1).fill(0.0);
        obj.dataSaveDown2d.row(k1).fill(0.0);
    }
    for(int k1=endx+1;k1<obj.nx;k1++){
        obj.dataSaveUp2d.row(k1).fill(0.0);
        obj.dataSaveDown2d.row(k1).fill(0.0);
    }
    for(int k1=0;k1<=begz;k1++){
        obj.dataSaveLeft2d.slice(k1).fill(0.0);
        obj.dataSaveRight2d.slice(k1).fill(0.0);
    }
    for(int k1=endz;k1<obj.nz;k1++){
        obj.dataSaveLeft2d.slice(k1).fill(0.0);
        obj.dataSaveRight2d.slice(k1).fill(0.0);
    }
    cout<<"Completed: "<<sx<<endl;
    return Lighting2d;
}
int acousticRTM2dTowforward(elastic3D_ARMA& obj, \
    fmat& suf2dout, int sx, int rx, int rz, float f0=30.0)
{
    int nfield=ceil((obj.nt-1)/400.0);
    obj.fieldSave01.zeros(obj.nx,nfield,obj.nz);
    int nx(obj.nx),ny(obj.ny),nz(obj.nz),nCoe(obj.nCoe);
    int sy(0),sz=obj.nzSampleOfFreeSurface;
    fcube suf3d(nx,1,obj.nt);

    obj.dataSaveUp2d.zeros(obj.nx,obj.nCoe,obj.nt);
    obj.dataSaveDown2d.zeros(obj.nx,obj.nCoe,obj.nt);
    obj.dataSaveLeft2d.zeros(obj.nt,obj.nCoe,obj.nz);
    obj.dataSaveRight2d.zeros(obj.nt,obj.nCoe,obj.nz);
    obj.cleardata();
    int endz=obj.nz-obj.nCoe-obj.PML_wide-2;
    int endx=obj.nx-obj.nCoe-obj.PML_wide-2;
    int begx=obj.nCoe+obj.PML_wide+1;
    int begz=obj.nCoe+obj.PML_wide+1;
    int (*fptr)(class elastic3D_ARMA&);
    if(obj.maxThreadNum<=1){
        fptr=TimeSliceCal_acoustic2D_OneThread;
    }else{
        fptr=TimeSliceCal_acoustic2D_MultiThread;
    }
    fmat Lighting2d(obj.nx,obj.nz);
    fmat swap2d(obj.nx,obj.nz);
    Lighting2d.fill(0.0);
    suf3d.fill(0.0);
    for(int k=0;k<(obj.nt);k++)
    {
        obj.tzz(sx,sy,sz)+=1e6*wavelet02(k,obj.dt,f0,0.0);
        //obj.tzz.slice(sz)+=suf3d.slice(k);
        //Using internal parallelism
        (*fptr)(obj);
        for(int k2=0;k2<nCoe;k2++){
            for(int k1=0;k1<nx;k1++){
                obj.dataSaveUp2d(k1,k2,k)=obj.tzz(k1,0,begz+k2);
                obj.dataSaveDown2d(k1,k2,k)=obj.tzz(k1,0,endz-nCoe+k2+1);
            }
            for(int k3=0;k3<nz;k3++){
                obj.dataSaveLeft2d(k,k2,k3)=obj.tzz(begx+k2,0,k3);
                obj.dataSaveRight2d(k,k2,k3)=obj.tzz(endx-nCoe+1+k2,0,k3);
            }
        }
        suf3d(rx,0,k)=obj.tzz(rx,0,rz);
        if(k%400==0)
        {
            obj.fieldSave01.col(floor(k/400.0))=obj.tzz.col(0);
            cout<<"now is running : "<<k<<endl;
        }
    }
    
    for(int k1=0;k1<begx;k1++){
        obj.dataSaveUp2d.row(k1).fill(0.0);
        obj.dataSaveDown2d.row(k1).fill(0.0);
    }
    for(int k1=endx+1;k1<obj.nx;k1++){
        obj.dataSaveUp2d.row(k1).fill(0.0);
        obj.dataSaveDown2d.row(k1).fill(0.0);
    }
    for(int k1=0;k1<=begz;k1++){
        obj.dataSaveLeft2d.slice(k1).fill(0.0);
        obj.dataSaveRight2d.slice(k1).fill(0.0);
    }
    for(int k1=endz;k1<obj.nz;k1++){
        obj.dataSaveLeft2d.slice(k1).fill(0.0);
        obj.dataSaveRight2d.slice(k1).fill(0.0);
    }
    cout<<"Completed: "<<sx<<endl;
    for(int i=0;i<nx;i++){
    for(int j=0;j<min(int(obj.nt),int(suf2dout.n_rows));j++){
        suf2dout(j,i)=suf3d(i,0,j);
    }}
    return 0;
}
fmat acousticRTM2dTowInverse(fmat& Image2d, elastic3D_ARMA& objSave, \
    elastic3D_ARMA& objBlank, fmat dataSuface2d, int sx, \
    float f0=30.0, float waveletType=2.0)
{
    int nfield=ceil((objSave.nt-1)/400.0);
    objSave.fieldSave02.zeros(objSave.nx,nfield,objSave.nz);
    int sy(0),sz=objBlank.nzSampleOfFreeSurface;
    int nCoe=objSave.nCoe;
    float nWin=objSave.nt*0.1;
    objSave.cleardata();
    objBlank.cleardata();
    int endz=objSave.nz-objSave.nCoe-objSave.PML_wide-2;
    int begz=objSave.nCoe+objSave.PML_wide+1;
    int endx=objSave.nx-objSave.nCoe-objSave.PML_wide-2;
    int begx=objSave.nCoe+objSave.PML_wide+1;
    for(int k=0;k<begx;k++){
        //objSave.mpar_vp.row(k).fill(0.0);
    }
    for(int k=endx+1;k<objSave.nx;k++){
        //objSave.mpar_vp.row(k).fill(0.0);
    }
    for(int k=0;k<=begz;k++){
        //objSave.mpar_vp.slice(k).fill(0.0);
    }
    for(int k=endz;k<objSave.nz;k++){
        //objSave.mpar_vp.slice(k).fill(0.0);
    }
    objSave.updatepar();
    
    fmat rtmImage2d(objSave.nx,objSave.nz);
    fmat swap2d(objSave.nx,objSave.nz);
    rtmImage2d.fill(0.0);
    int (*fptr)(class elastic3D_ARMA&);
    float (*fptrWavelet)(int k, float DT, float hz,float det2);
    if(waveletType<1.5){
        fptrWavelet=wavelet01;
    }else if(waveletType>=1.5 && waveletType<2.5){
        fptrWavelet=wavelet02;
    }
    if(objSave.maxThreadNum<=1){
        fptr=TimeSliceCal_acoustic2D_OneThread;
    }else{
        fptr=TimeSliceCal_acoustic2D_MultiThread;
    }
    //ofstream outf;
    //outf.open("movie.dat");
    for(int k=(objSave.nt-1);k>=objBlank.nt;k--)
    {
        for(int k2=0;k2<nCoe;k2++){
            for(int kx=(begx+1);kx<endx;kx++){
                objSave.tzz(kx,0,begz+k2)=objSave.dataSaveUp2d(kx,k2,k);
                objSave.tzz(kx,0,endz-nCoe+k2+1)\
                    =objSave.dataSaveDown2d(kx,k2,k);
            }
            for(int kz=(begz+1);kz<endz;kz++){
                objSave.tzz(begx+k2,0,kz)=objSave.dataSaveLeft2d(k,k2,kz);
                objSave.tzz(endx-nCoe+1+k2,0,kz)\
                    =objSave.dataSaveRight2d(k,k2,kz);
            }
        }
        //Using internal parallelism
        (*fptr)(objSave);
        //swap2d=objSave.tzz.col(0);
        //if(k%5==0)datawrite(swap2d,outf);

        if(k%1000==1)
        {
            //objSave.fieldSave02.col(round(k/400.0))=objSave.tzz.col(0);
            cout<<"now is running : "<<k<<endl;
        }
    }
    //int endt=round(0.05/objSave.dt);
    fmat Lighting2d(objSave.nx,objSave.nz);
    fmat swap2d2(objSave.nx,objSave.nz);
    Lighting2d.fill(0.0);
    fcube dataSuface3d(objBlank.nx,1,objBlank.nt);
    fcube source3d(objBlank.nx,1,objBlank.nt);
    dataSuface3d.col(0)=dataSuface2d.st();
    //source3d.col(0)=source2d.st();
    for(int k=(objBlank.nt-1);k>=0;k--)
    {
        objBlank.tzz.slice(sz)+=dataSuface3d.slice(k);
        for(int k2=0;k2<nCoe;k2++){
            for(int kx=(begx+1);kx<endx;kx++){
                objSave.tzz(kx,0,begz+k2)=objSave.dataSaveUp2d(kx,k2,k);
                objSave.tzz(kx,0,endz-nCoe+k2+1)\
                    =objSave.dataSaveDown2d(kx,k2,k);
            }
            for(int kz=(begz+1);kz<endz;kz++){
                objSave.tzz(begx+k2,0,kz)=objSave.dataSaveLeft2d(k,k2,kz);
                objSave.tzz(endx-nCoe+1+k2,0,kz)\
                    =objSave.dataSaveRight2d(k,k2,kz);
            }
        }
        //Using internal parallelism
        (*fptr)(objSave);
        (*fptr)(objBlank);
        objSave.tzz(sx,sy,sz)+=1e6*(*fptrWavelet)(k,objSave.dt,f0,0.0);
        swap2d=objSave.tzz.col(0);
        //if(k%5==0)datawrite(swap2d,outf);
        for(int k1=0;k1<objSave.nx;k1++){
        for(int k2=0;k2<objSave.nz;k2++){
            swap2d(k1,k2)=swap2d(k1,k2)*objBlank.tzz(k1,0,k2);
            //swap2d2(k1,k2)=abs(swap2d(k1,k2))*abs(objBlank.tzz(k1,0,k2));
        }}
        rtmImage2d+=swap2d;
        //Lighting2d+=swap2d2;

        if(k%1000==1)
        {
            //objSave.fieldSave02.col(round(k/1000.0))=objSave.tzz.col(0);
            cout<<"now is running : "<<k<<endl;
        }
    }
    //outf.close();
    cout<<"Completed: "<<sx<<endl;
    Image2d=rtmImage2d.st();
    return Lighting2d.st();
}

fmat acousticRTM2dTowInverse(fmat& Image2d, elastic3D_ARMA& objSave, \
    elastic3D_ARMA& objBlank, fmat source2d, fmat dataSuface2d, int sx)
{
    int nfield=ceil((objSave.nt-1)/400.0);
    objSave.fieldSave02.zeros(objSave.nx,nfield,objSave.nz);
    int sy(0),sz=objBlank.nzSampleOfFreeSurface;
    int nCoe=objSave.nCoe;
    float nWin=objSave.nt*0.1;
    objSave.cleardata();
    objBlank.cleardata();
    int endz=objSave.nz-objSave.nCoe-objSave.PML_wide-2;
    int begz=objSave.nCoe+objSave.PML_wide+1;
    int endx=objSave.nx-objSave.nCoe-objSave.PML_wide-2;
    int begx=objSave.nCoe+objSave.PML_wide+1;
    for(int k=0;k<begx;k++){
        //objSave.mpar_vp.row(k).fill(0.0);
    }
    for(int k=endx+1;k<objSave.nx;k++){
        //objSave.mpar_vp.row(k).fill(0.0);
    }
    for(int k=0;k<=begz;k++){
        //objSave.mpar_vp.slice(k).fill(0.0);
    }
    for(int k=endz;k<objSave.nz;k++){
        //objSave.mpar_vp.slice(k).fill(0.0);
    }
    objSave.updatepar();
    
    fmat rtmImage2d(objSave.nx,objSave.nz);
    fmat swap2d(objSave.nx,objSave.nz);
    rtmImage2d.fill(0.0);
    int (*fptr)(class elastic3D_ARMA&);
    if(objSave.maxThreadNum<=1){
        fptr=TimeSliceCal_acoustic2D_OneThread;
    }else{
        fptr=TimeSliceCal_acoustic2D_MultiThread;
    }
    //ofstream outf;
    //outf.open("movie.dat");
    for(int k=(objSave.nt-1);k>=objBlank.nt;k--)
    {
        for(int k2=0;k2<nCoe;k2++){
            for(int kx=(begx+1);kx<endx;kx++){
                objSave.tzz(kx,0,begz+k2)=objSave.dataSaveUp2d(kx,k2,k);
                objSave.tzz(kx,0,endz-nCoe+k2+1)\
                    =objSave.dataSaveDown2d(kx,k2,k);
            }
            for(int kz=(begz+1);kz<endz;kz++){
                objSave.tzz(begx+k2,0,kz)=objSave.dataSaveLeft2d(k,k2,kz);
                objSave.tzz(endx-nCoe+1+k2,0,kz)\
                    =objSave.dataSaveRight2d(k,k2,kz);
            }
        }
        //Using internal parallelism
        (*fptr)(objSave);
        //swap2d=objSave.tzz.col(0);
        //if(k%5==0)datawrite(swap2d,outf);

        if(k%400==1)
        {
            objSave.fieldSave02.col(round(k/400.0))=objSave.tzz.col(0);
            cout<<"now is running : "<<k<<endl;
        }
    }
    //int endt=round(0.05/objSave.dt);
    fmat Lighting2d(objSave.nx,objSave.nz);
    fmat swap2d2(objSave.nx,objSave.nz);
    Lighting2d.fill(0.0);
    fcube dataSuface3d(objBlank.nx,1,objBlank.nt);
    fcube source3d(objBlank.nx,1,objBlank.nt);
    dataSuface3d.col(0)=dataSuface2d.st();
    source3d.col(0)=source2d.st();
    for(int k=(objBlank.nt-1);k>=0;k--)
    {
        objBlank.tzz.slice(sz)+=dataSuface3d.slice(k);
        for(int k2=0;k2<nCoe;k2++){
            for(int kx=(begx+1);kx<endx;kx++){
                objSave.tzz(kx,0,begz+k2)=objSave.dataSaveUp2d(kx,k2,k);
                objSave.tzz(kx,0,endz-nCoe+k2+1)\
                    =objSave.dataSaveDown2d(kx,k2,k);
            }
            for(int kz=(begz+1);kz<endz;kz++){
                objSave.tzz(begx+k2,0,kz)=objSave.dataSaveLeft2d(k,k2,kz);
                objSave.tzz(endx-nCoe+1+k2,0,kz)\
                    =objSave.dataSaveRight2d(k,k2,kz);
            }
        }
        //Using internal parallelism
        (*fptr)(objSave);
        (*fptr)(objBlank);
        objSave.tzz.slice(sz)+=source3d.slice(k);
        swap2d=objSave.tzz.col(0);
        //if(k%5==0)datawrite(swap2d,outf);
        for(int k1=0;k1<objSave.nx;k1++){
        for(int k2=0;k2<objSave.nz;k2++){
            swap2d(k1,k2)=swap2d(k1,k2)*objBlank.tzz(k1,0,k2);
            //swap2d2(k1,k2)=abs(swap2d(k1,k2))*abs(objBlank.tzz(k1,0,k2));
        }}
        rtmImage2d+=swap2d;
        //Lighting2d+=swap2d2;

        if(k%400==1)
        {
            objSave.fieldSave02.col(round(k/400.0))=objSave.tzz.col(0);
            cout<<"now is running : "<<k<<endl;
        }
    }
    //outf.close();
    cout<<"Completed: "<<sx<<endl;
    Image2d=rtmImage2d.st();
    return Lighting2d.st();
}
fcube model3dExtendSliceUp(fcube model, int nSlice)
{
    int n1(model.n_rows),n2(model.n_cols),\
        n3(model.n_slices);
    fcube modelEx(n1,n2,n3+nSlice);
    modelEx(span::all,span::all,span(nSlice,n3+nSlice-1))\
        =model(span::all,span::all,span::all);
    for(int k=0;k<nSlice;k++){
        modelEx.slice(k)=modelEx.slice(nSlice);
    }
    return modelEx;
}
fcube model3dExtendSliceDown(fcube model, int nSlice)
{
    int n1(model.n_rows),n2(model.n_cols),\
        n3(model.n_slices);
    fcube modelEx(n1,n2,n3+nSlice);
    modelEx(span::all,span::all,span(0,n3-1))\
        =model(span::all,span::all,span::all);
    for(int k=n3;k<(n3+nSlice);k++){
        modelEx.slice(k)=modelEx.slice(n3-1);
    }
    return modelEx;
}
fmat acoustic2dOBNforward(elastic3D_ARMA& obj, \
    int sx, int sz, int rz, float f0=30.0, float waveletType=1.0)
{
    int nx(obj.nx),ny(obj.ny),nz(obj.nz),nCoe(obj.nCoe);
    int sy(0);
    int (*fptr)(class elastic3D_ARMA&);
    float (*fptrWavelet)(int k, float DT, float hz,float det2);
    if(waveletType<1.5){
        fptrWavelet=wavelet01;
    }else if(waveletType>=1.5 && waveletType<2.5){
        fptrWavelet=wavelet02;
    }
    if(true){
        fptr=TimeSliceCal_elastic2D_MultiThread;
    }
    fmat suftzz2d(obj.nt,obj.nx);
    for(int k=0;k<(obj.nt);k++)
    {
        obj.tzz(sx,sy,sz)+=1e6*(*fptrWavelet)(k,obj.dt,f0,0.0);
        //Using internal parallelism
        (*fptr)(obj);
        for(int i=0;i<nx;i++){
            suftzz2d(k,i)=obj.tzz(i,0,rz);
        }
        if(k%1000==0)
        {
            cout<<"now is running : "<<k<<endl;
        }
    }
    cout<<"Completed: "<<sx<<endl;
    return suftzz2d;
}
fmat acoustic2dInterbedforward(elastic3D_ARMA& obj, \
    int sx, fvec recv, float f0=30.0, float waveletType=2.0)
{
    int nx(obj.nx),ny(obj.ny),nz(obj.nz),nCoe(obj.nCoe);
    int sy(0),sz=obj.nzSampleOfFreeSurface;
    int (*fptr)(class elastic3D_ARMA&);
    float (*fptrWavelet)(int k, float DT, float hz,float det2);
    if(waveletType<1.5 || obj.isPMLSurface){
        fptrWavelet=wavelet01;
    }else if(waveletType>=1.5 && waveletType<2.5){
        fptrWavelet=wavelet02;
    }
    if(obj.maxThreadNum>=2){
        fptr=TimeSliceCal_acoustic2D_MultiThread;
    }else{
        fptr=TimeSliceCal_acoustic2D_OneThread;
    }
    fmat suftzz2d(obj.nt,obj.nx);
    for(int k=0;k<(obj.nt);k++)
    {
        obj.tzz(sx,sy,sz)+=1e6*(*fptrWavelet)(k,obj.dt,f0,0.0);
        //Using internal parallelism
        (*fptr)(obj);
        for(int i=0;i<nx;i++){
            int kz=sz+round(recv(i)/obj.dz);
            suftzz2d(k,i)=obj.tzz(i,0,kz);
            for(int k2=0;k2<nCoe;k2++){
                obj.dataSaveDown2d(k,i,k2)=obj.tzz(i,0,kz+k2);
            }
        }
        if(k%1000==0)
        {
            cout<<"now is running : "<<k<<endl;
        }
    }
    cout<<"Completed: "<<sx<<endl;
    return suftzz2d;
}
fmat getBorn2dScatter(fmat vp2d, fmat rho2d, 
    fmat& backvp2d, fmat& backrho2d, \
    int sufnz, int nSmooth=25)
{
    int nz(vp2d.n_rows),nx(vp2d.n_cols);
    for(int k1=0;k1<sufnz;k1++){
        vp2d.row(k1).fill(vp2d(sufnz,0));
        rho2d.row(k1).fill(rho2d(sufnz,0));
    }
    backrho2d=fmatsmooth(rho2d,nz,nx,nSmooth);
    backvp2d=fmatsmooth(vp2d,nz,nx,nSmooth);
    fmat scatter(nz,nx);
    for(int k1=0;k1<nz;k1++){
    for(int k2=0;k2<nx;k2++){
        scatter(k1,k2)=(1.0/vp2d(k1,k2)/vp2d(k1,k2)\
            -1.0/backvp2d(k1,k2)/backvp2d(k1,k2))\
            /(1.0/backvp2d(k1,k2)/backvp2d(k1,k2))\
            +(1.0/rho2d(k1,k2)/rho2d(k1,k2)\
            -1.0/backrho2d(k1,k2)/backrho2d(k1,k2))\
            /(1.0/backrho2d(k1,k2)/backrho2d(k1,k2));
    }}
    return scatter;
}
fmat getLayerDepth(fmat vp2d, float vel, \
    float width1, float width2)
{
    int n1(vp2d.n_rows), n2(vp2d.n_cols);
    fmat win2d(n1,n2);
    fvec depth(n2);
    depth.fill(0.0);
    for(int j=0;j<n2;j++){
    for(int i=0;i<n1;i++){
        if(vp2d(i,j)>vel){
            depth(j)=i;
            break;
        }
    }}
    win2d.fill(0.0);
    for(int j=0;j<n2;j++){
    for(int i=0;i<n1;i++){
        if(i>=(depth(j)-width1-width2) && i<(depth(j)-width1)){
            win2d(i,j)=Blackman(i-depth(j)+width1+width2,width2);
        }else if(i>=(depth(j)-width1) && i<=(depth(j)+width1)){
            win2d(i,j)=1.0;
        }else if(i>(depth(j)+width1) && i<=(depth(j)+width1+width2)){
            win2d(i,j)=Blackman(depth(j)+width1+width2-i,width2);
        }
    }}
    return win2d;
}
fmat born2dTowSRMSimulation(
    elastic3D_ARMA& backfield, \
    elastic3D_ARMA& scatterfield, \
    fmat& suf2d, fmat scatter, int sx)
{
    int nx(backfield.nx),nt(backfield.nt);
    int sz(backfield.nzSampleOfFreeSurface);
    int nthread=backfield.maxThreadNum;
    int rz=sz;
    fcube suf3d(nx,1,nt);
    suf3d.col(0)=suf2d.st();

    int (*fptr)(class elastic3D_ARMA&);
    if(nthread<=1){
        fptr=TimeSliceCal_acoustic2D_OneThread;
    }else{
        fptr=TimeSliceCal_acoustic2D_MultiThread;
    }
    scatter=scatter.st();
    for(int k=0;k<nt;k++)
    {
        backfield.tzz.slice(sz)+=suf3d.slice(k);
        scatterfield.tzz.col(0)+=fmatmul\
                (scatter,backfield.tzz.col(0),nthread);
        //Using internal parallelism
        (*fptr)(backfield);
        (*fptr)(scatterfield);
        suf3d.slice(k)=scatterfield.tzz.slice(sz);
        if(k%1000==0){
            cout<<"now is running : "<<k<<endl;
        }
    }
    fmat outSuf2d(nx,nt);
    outSuf2d=suf3d.col(0);
    outSuf2d=outSuf2d.st();
    cout<<"Completed: "<<sx<<endl;
    return outSuf2d;
}

void acousticRTM2dOBNforward(elastic3D_ARMA& obj, \
    int sx, int sz, float f0=30.0, float waveletType=2.0)
{
    int nfield=ceil((obj.nt-1)/1000.0);
    obj.fieldSave01.zeros(obj.nx,nfield,obj.nz);
    int nx(obj.nx),ny(obj.ny),nz(obj.nz),nCoe(obj.nCoe);
    int sy(0);
    obj.dataSaveUp2d.zeros(obj.nx,obj.nCoe,obj.nt);
    obj.dataSaveDown2d.zeros(obj.nx,obj.nCoe,obj.nt);
    obj.dataSaveLeft2d.zeros(obj.nt,obj.nCoe,obj.nz);
    obj.dataSaveRight2d.zeros(obj.nt,obj.nCoe,obj.nz);
    obj.cleardata();
    int endz=obj.nz-obj.nCoe-obj.PML_wide-2;
    int endx=obj.nx-obj.nCoe-obj.PML_wide-2;
    int begx=obj.nCoe+obj.PML_wide+1;
    int begz=obj.nCoe+obj.PML_wide+1;
    int (*fptr)(class elastic3D_ARMA&);
    float (*fptrWavelet)(int k, float DT, float hz,float det2);
    if(obj.maxThreadNum<=1){
        fptr=TimeSliceCal_acoustic2D_OneThread;
    }else{
        fptr=TimeSliceCal_acoustic2D_MultiThread;
    }
    if(waveletType<1.5){
        fptrWavelet=wavelet01;
    }else if(waveletType>=1.5 && waveletType<2.5){
        fptrWavelet=wavelet02;
    }
    fmat suf2d(obj.nx,obj.nt);
    for(int k=0;k<(obj.nt);k++)
    {
        obj.tzz(sx,sy,sz)+=1e6*(*fptrWavelet)(k,obj.dt,f0,0.0);
        //Using internal parallelism
        (*fptr)(obj);
        for(int k2=0;k2<nCoe;k2++){
            for(int k1=0;k1<nx;k1++){
                obj.dataSaveUp2d(k1,k2,k)=obj.tzz(k1,0,begz+k2);
                obj.dataSaveDown2d(k1,k2,k)=obj.tzz(k1,0,endz-nCoe+k2+1);
            }
            for(int k3=0;k3<nz;k3++){
                obj.dataSaveLeft2d(k,k2,k3)=obj.tzz(begx+k2,0,k3);
                obj.dataSaveRight2d(k,k2,k3)=obj.tzz(endx-nCoe+1+k2,0,k3);
            }
        }
        if(k%1000==0)
        {
            obj.fieldSave01.col(floor(k/1000.0))=obj.tzz.col(0);
            cout<<"now is running : "<<k<<endl;
        }
    }
    
    for(int k1=0;k1<begx;k1++){
        obj.dataSaveUp2d.row(k1).fill(0.0);
        obj.dataSaveDown2d.row(k1).fill(0.0);
    }
    for(int k1=endx+1;k1<obj.nx;k1++){
        obj.dataSaveUp2d.row(k1).fill(0.0);
        obj.dataSaveDown2d.row(k1).fill(0.0);
    }
    for(int k1=0;k1<=begz;k1++){
        obj.dataSaveLeft2d.slice(k1).fill(0.0);
        obj.dataSaveRight2d.slice(k1).fill(0.0);
    }
    for(int k1=endz;k1<obj.nz;k1++){
        obj.dataSaveLeft2d.slice(k1).fill(0.0);
        obj.dataSaveRight2d.slice(k1).fill(0.0);
    }
    cout<<"Completed: "<<sx<<endl;
}
fmat acousticRTM2dOBNInverse(fmat& Image2d, elastic3D_ARMA& objSave, \
    elastic3D_ARMA& objBlank, fmat dataSuface2d, int sx, int sz, int rz, \
    float f0=30.0, float waveletType=2.0)
{
    int nfield=ceil((objSave.nt-1)/1000.0);
    objSave.fieldSave02.zeros(objSave.nx,nfield,objSave.nz);
    int sy(0);
    int nCoe=objSave.nCoe;
    float nWin=objSave.nt*0.1;
    objSave.cleardata();
    objBlank.cleardata();
    int endz=objSave.nz-objSave.nCoe-objSave.PML_wide-2;
    int begz=objSave.nCoe+objSave.PML_wide+1;
    int endx=objSave.nx-objSave.nCoe-objSave.PML_wide-2;
    int begx=objSave.nCoe+objSave.PML_wide+1;
    for(int k=0;k<begx;k++){
        //objSave.mpar_vp.row(k).fill(0.0);
    }
    for(int k=endx+1;k<objSave.nx;k++){
        //objSave.mpar_vp.row(k).fill(0.0);
    }
    for(int k=0;k<=begz;k++){
        //objSave.mpar_vp.slice(k).fill(0.0);
    }
    for(int k=endz;k<objSave.nz;k++){
        //objSave.mpar_vp.slice(k).fill(0.0);
    }
    objSave.updatepar();
    
    fmat rtmImage2d(objSave.nx,objSave.nz);
    fmat swap2d(objSave.nx,objSave.nz);
    rtmImage2d.fill(0.0);
    int (*fptr)(class elastic3D_ARMA&);
    float (*fptrWavelet)(int k, float DT, float hz,float det2);
    if(waveletType<1.5){
        fptrWavelet=wavelet01;
    }else if(waveletType>=1.5 && waveletType<2.5){
        fptrWavelet=wavelet02;
    }
    if(objSave.maxThreadNum<=1){
        fptr=TimeSliceCal_acoustic2D_OneThread;
    }else{
        fptr=TimeSliceCal_acoustic2D_MultiThread;
    }
    //ofstream outf;
    //outf.open("movie.dat");
    for(int k=(objSave.nt-1);k>=objBlank.nt;k--)
    {
        for(int k2=0;k2<nCoe;k2++){
            for(int kx=(begx+1);kx<endx;kx++){
                objSave.tzz(kx,0,begz+k2)=objSave.dataSaveUp2d(kx,k2,k);
                objSave.tzz(kx,0,endz-nCoe+k2+1)\
                    =objSave.dataSaveDown2d(kx,k2,k);
            }
            for(int kz=(begz+1);kz<endz;kz++){
                objSave.tzz(begx+k2,0,kz)=objSave.dataSaveLeft2d(k,k2,kz);
                objSave.tzz(endx-nCoe+1+k2,0,kz)\
                    =objSave.dataSaveRight2d(k,k2,kz);
            }
        }
        //Using internal parallelism
        (*fptr)(objSave);
        //swap2d=objSave.tzz.col(0);
        //if(k%5==0)datawrite(swap2d,outf);

        if(k%1000==1)
        {
            //objSave.fieldSave02.col(round(k/1000.0))=objSave.tzz.col(0);
            cout<<"now is running : "<<k<<endl;
        }
    }
    //int endt=round(0.05/objSave.dt);
    fmat Lighting2d(objSave.nx,objSave.nz);
    fmat swap2d2(objSave.nx,objSave.nz);
    Lighting2d.fill(0.0);
    fcube dataSuface3d(objBlank.nx,1,objBlank.nt);
    fcube source3d(objBlank.nx,1,objBlank.nt);
    dataSuface3d.col(0)=dataSuface2d.st();
    //source3d.col(0)=source2d.st();
    for(int k=(objBlank.nt-1);k>=0;k--)
    {
        objBlank.tzz.slice(rz)+=dataSuface3d.slice(k);
        for(int k2=0;k2<nCoe;k2++){
            for(int kx=(begx+1);kx<endx;kx++){
                objSave.tzz(kx,0,begz+k2)=objSave.dataSaveUp2d(kx,k2,k);
                objSave.tzz(kx,0,endz-nCoe+k2+1)\
                    =objSave.dataSaveDown2d(kx,k2,k);
            }
            for(int kz=(begz+1);kz<endz;kz++){
                objSave.tzz(begx+k2,0,kz)=objSave.dataSaveLeft2d(k,k2,kz);
                objSave.tzz(endx-nCoe+1+k2,0,kz)\
                    =objSave.dataSaveRight2d(k,k2,kz);
            }
        }
        //Using internal parallelism
        (*fptr)(objSave);
        (*fptr)(objBlank);
        objSave.tzz(sx,sy,sz)+=1e6*(*fptrWavelet)(k,objSave.dt,f0,0.0);
        swap2d=objSave.tzz.col(0);
        //if(k%5==0)datawrite(swap2d,outf);
        for(int k1=0;k1<objSave.nx;k1++){
        for(int k2=0;k2<objSave.nz;k2++){
            swap2d(k1,k2)=swap2d(k1,k2)*objBlank.tzz(k1,0,k2);
            swap2d2(k1,k2)=abs(swap2d(k1,k2))*abs(objBlank.tzz(k1,0,k2));
        }}
        rtmImage2d+=swap2d;
        Lighting2d+=swap2d2;

        if(k%1000==1)
        {
            //objSave.fieldSave02.col(round(k/1000.0))=objSave.tzz.col(0);
            cout<<"now is running : "<<k<<endl;
        }
    }
    //outf.close();
    cout<<"Completed: "<<sx<<endl;
    Image2d=rtmImage2d.st();
    return Lighting2d.st();
}


#endif
