#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <algorithm>
//#include "mma.h"

//using namespace std;

void readweights();
void readinput();

void setup_gpu();
void final_gpu();
void infer_gpu(int);

//#define BALANCE 30 //BALANCE LAYER 0 FOR EVERY LAYER COMMENT OUT FOR TURN OFF
#define OUTOFCORE //COMMENT THIS OUT IF YOU HAVE ENOUGH MEMORY
#define OVERLAP //WORKS ONLY WHEN OUTOFCORE IS ENABLED
#define INDPREC int
#define VALPREC float
#define FEATPREC float

