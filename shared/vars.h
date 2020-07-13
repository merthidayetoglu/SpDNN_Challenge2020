#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include "mma.h"

using namespace std;


void readweights();
void preproc();
void readinput();

void setup_gpu();
void final_gpu();
double infer_gpu(int);

#define WARPSIZE 32
#define MINIBATCH 16
#define INDPREC unsigned short
#define VALPREC half
#define FEATPREC float
