//Ԥ����
#ifndef XJC_H
#define XJC_H
#define ARMA_DONT_USE_WRAPPER
#define ARMA_DONT_USE_BLAS
#define ARMA_DONT_USE_OPENMP
#define ARMA_OPENMP_THREADS 1
/*@Xiang Jian, Used for Geophycise Research

*/
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <omp.h>
#include "./armadillo-9.800.2/include/armadillo"
//#include <armadillo>
using namespace arma;
using namespace std;

#include "./xj.include/mat.hpp"
#include "./xj.include/segyhead.hpp"
#include "./xj.include/my_armadillo.hpp"
#include "./xj.include/function.hpp"
#include "./xj.include/threadpool.hpp"
#include "./xj.include/readsegy.hpp"
#include "./xj.include/wave2D.hpp"
#include "./xj.include/elastic2D.hpp"
#include "./xj.include/elastic3D.arma.hpp"
#include "./xj.include/radon3d.hpp"
//#include "./xj.include/radon3dmixft.hpp"
#include "./xj.include/radon.all.hpp"

#include "./xj.include/ssf.born2d.hpp"
#include "./xj.include/datamatch.hpp"
#include "./xj.include/CRMD.new.hpp"
#include "./xj.include/CRMD2d.hpp"
#include "./xj.include/TravelTime3d.hpp"

#endif






