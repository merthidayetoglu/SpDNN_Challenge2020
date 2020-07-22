#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <chrono>

using namespace std;
typedef std::chrono::duration<double> Duration;
typedef std::chrono::system_clock sc;

void readweights();
void preproc();
void readinput();

void setup_gpu();
void final_gpu();
void infer_gpu(int);

#define OUTOFCORE //COMMENT THIS OUT IF YOU HAVE ENOUGH MEMORY
#define OVERLAP //WORKS ONLY WHEN OUTOFCORE IS ENABLED
#define WARPSIZE 32
#define MINIBATCH 12
#define MAPPREC unsigned short
#define INDPREC unsigned short
#define VALPREC float
#define FEATPREC float

