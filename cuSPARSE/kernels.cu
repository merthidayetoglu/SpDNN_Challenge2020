#include "vars.h"
#include "cuda_runtime.hpp"

#include <cuda.h>
#include <cusparse.h>

extern int neuron;
extern int layer;
extern int batch;
extern int input;
extern float bias;

extern int **csrdispl;
extern INDPREC **csrindex;
extern VALPREC **csrvalue;

extern FEATPREC *currfeat;
extern FEATPREC *nextfeat;
extern int *active;
extern int *categories;
extern int *globalcategories;

extern int myid;
extern int numproc;
extern int numthreads;

extern int *numbatch;
extern int *batchdispl;
extern int mybatch;
extern int extbatch;

extern Duration timekernel;
extern Duration timecopy;


int **csrdispl_d;
INDPREC *indbuff_d;
VALPREC *valbuff_d;;
#ifdef OUTOFCORE
int  weightsizemax;
#ifdef OVERLAP
INDPREC *indstream_d;
VALPREC *valstream_d;
#endif
#else
INDPREC **csrindex_d;
VALPREC **csrvalue_d;
#endif

FEATPREC *currfeat_d;
FEATPREC *nextfeat_d;
int *active_d;
int *categories_d;

int blocksize;
int numblocks;
int numwarp;
int buffsize;

//CUSPARSE
cusparseHandle_t handle;

cudaEvent_t copystart, copystop;
cudaEvent_t kernelstart, kernelstop;
cudaStream_t copystream;
cudaStream_t kernelstream;
float elapsedTime;

__device__ float __ReLU(float x){
   return x<0.0?0.0:x>32.0?32.0:x;
};
__global__ void __launch_bounds__(1024,1) csr_kernel(FEATPREC *nextfeat, FEATPREC *currfeat, int *displ, INDPREC *index, VALPREC *value, float bias , int neuron, int *categories, int *active){
  float reduce[MINIBATCH] = {0.0};
  int m = blockIdx.x*blockDim.x+threadIdx.x;
  categories += blockIdx.y*MINIBATCH;
  nextfeat += blockIdx.y*MINIBATCH*neuron;
  active += blockIdx.y*MINIBATCH;
  for(int n = displ[m]; n < displ[m+1]; n++){
    int ind = index[n];
    float val = value[n];
    for(int k = 0; k < MINIBATCH; k++)
      reduce[k] += currfeat[categories[k]*neuron+ind]*val;
  }
  for(int k = 0; k < MINIBATCH; k++)
    if(nextfeat[k*neuron+m]=__ReLU(reduce[k]+bias))
      atomicAdd(active+k,1);
};
__global__ void __launch_bounds__(1024,2) relu_kernel(FEATPREC *nextfeat, float bias , int neuron, int *active){
  int row = blockIdx.x*blockDim.x+threadIdx.x;
  int col = blockIdx.y*blockDim.y+threadIdx.y;
  int ind = col*neuron+row;
  if(nextfeat[ind]=__ReLU(nextfeat[ind]+bias))
    atomicAdd(active+col,1);
};

void setup_gpu(){

  OR_FATAL(cudaSetDevice(1));
  printf("myid %d mydevice %d\n",myid,myid%4);
  //OR_FATAL(cudaFuncSetAttribute(dummy_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,98304));
  if(myid==0){
    int deviceCount;
    OR_FATAL(cudaGetDeviceCount(&deviceCount));
    printf("\n");
    printf("Device Count: %d\n",deviceCount);
    int dev = 0;
    cudaDeviceProp deviceProp;
    OR_FATAL(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Device %d name: %s\n",dev,deviceProp.name);
    printf("Computational Capabilities: %d, %d\n",deviceProp.major,deviceProp.minor);
    printf("Maximum global memory size: %lu\n",deviceProp.totalGlobalMem);
    printf("Maximum constant memory size: %lu\n",deviceProp.totalConstMem);
    printf("Maximum shared memory size per block: %lu\n",deviceProp.sharedMemPerBlock);
    printf("Maximum block dimensions: %dx%dx%d\n",deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);
    printf("Maximum grid dimensions: %dx%dx%d\n",deviceProp.maxGridSize[0],deviceProp.maxGridSize[1],deviceProp.maxGridSize[2]);
    printf("Maximum threads per block: %d\n",deviceProp.maxThreadsPerBlock);
    printf("Warp size: %d\n",deviceProp.warpSize);
    printf("\n");
  }
  OR_FATAL(cudaEventCreate(&kernelstart));
  OR_FATAL(cudaEventCreate(&kernelstop));
  OR_FATAL(cudaEventCreate(&copystart));
  OR_FATAL(cudaEventCreate(&copystop));
  OR_FATAL(cudaStreamCreate(&copystream));
  OR_FATAL(cudaStreamCreate(&kernelstream));

  char *chartemp;
  chartemp = getenv("BLOCKSIZE");
  blocksize = atoi(chartemp);
  chartemp = getenv("BUFFER");
  buffsize = atoi(chartemp)*1024/sizeof(float)/MINIBATCH;
  numblocks = neuron/blocksize;
  numwarp = blocksize/WARPSIZE;
  if(myid==0){
    printf("MINIBATCH SIZE: %d\n",MINIBATCH);
    printf("BLOCK SIZE: %d\n",blocksize);
    printf("WARP SIZE: %d\n",WARPSIZE);
    printf("NUM BLOCKS: %d\n",numblocks);
    printf("NUMWARPS: %d\n",numwarp);
    printf("BUFFER SIZE: %d (%f KB) PER FEATURE: %d (%f KB)\n",buffsize*MINIBATCH,buffsize*sizeof(float)/1024.0*MINIBATCH,buffsize,buffsize*sizeof(float)/1024.0);
    printf("\n");
  }

  double memother = 0.0;
  OR_FATAL(cudaMallocHost((void**)&globalcategories,sizeof(int)*mybatch));
  OR_FATAL(cudaMallocHost((void**)&categories,sizeof(int)*mybatch));
  OR_FATAL(cudaMallocHost((void**)&active,sizeof(int)*mybatch));
  OR_FATAL(cudaMalloc((void**)&active_d,sizeof(int)*extbatch));
  OR_FATAL(cudaMalloc((void**)&categories_d,sizeof(int)*extbatch));
  memother += sizeof(int)*extbatch/1.0e9;
  memother += sizeof(int)*extbatch/1.0e9;
  for(int k = 0; k < mybatch; k++){
    active[k] = neuron;
    categories[k] = k;
    globalcategories[k] = batchdispl[myid]+k;
  }
  OR_FATAL(cudaMemset(active_d,0,sizeof(int)*extbatch));
  OR_FATAL(cudaMemset(categories_d,0,sizeof(int)*extbatch));
  OR_FATAL(cudaMemcpy(active_d,active,sizeof(int)*mybatch,cudaMemcpyHostToDevice));
  OR_FATAL(cudaMemcpy(categories_d,categories,sizeof(int)*mybatch,cudaMemcpyHostToDevice));

  #ifdef OUTOFCORE
  if(myid==0)printf("OUT OF CORE IS ENABLED\n");
  #ifdef OVERLAP
  if(myid==0)printf("OVERLAPPING IS ENABLED\n");
  #else
  if(myid==0)printf("OVERLAPPING IS DISABLED\n");
  #endif
  #else
  if(myid==0)printf("OUT OF CORE IS DISABLED\n");
  #endif

  double memweight = 0.0;
  double memdispl = 0.0;
  double memmap = 0.0;
  csrdispl_d = new int*[layer];
  #ifdef OUTOFCORE
  weightsizemax = 0;
  #else
  csrindex_d = new INDPREC*[layer];
  csrvalue_d = new VALPREC*[layer];
  #endif
  for(int l = 0; l < layer; l++){
    OR_FATAL(cudaMalloc((void**)&csrdispl_d[l],sizeof(int)*(neuron+1)));
    memdispl += sizeof(int)*(neuron+1)/1.0e9;
    OR_FATAL(cudaMemcpy(csrdispl_d[l],csrdispl[l],sizeof(int)*(neuron+1),cudaMemcpyHostToDevice));
    #ifdef OUTOFCORE
    int weightsize = csrdispl[l][neuron];
    if(weightsize > weightsizemax)
      weightsizemax = weightsize; 
    #else
    OR_FATAL(cudaMalloc((void**)&csrindex_d[l],sizeof(INDPREC)*csrdispl[l][neuron]));
    OR_FATAL(cudaMalloc((void**)&csrvalue_d[l],sizeof(VALPREC)*csrdispl[l][neuron]));
    memweight += sizeof(INDPREC)*csrdispl[l][neuron]/1.0e9;
    memweight += sizeof(VALPREC)*csrdispl[l][neuron]/1.0e9;
    OR_FATAL(cudaMemcpy(csrindex_d[l],csrindex[l],sizeof(INDPREC)*csrdispl[l][neuron],cudaMemcpyHostToDevice));
    OR_FATAL(cudaMemcpy(csrvalue_d[l],csrvalue[l],sizeof(VALPREC)*csrdispl[l][neuron],cudaMemcpyHostToDevice));
    #endif
  }
  #ifdef OUTOFCORE
  if(myid==0)printf("\n");
  if(myid==0)printf("weightsizemax: %d (%f KB)\n",weightsizemax,(sizeof(INDPREC)+sizeof(VALPREC))*weightsizemax/1.0e6);
  #ifdef OVERLAP
  OR_FATAL(cudaMalloc((void**)&indstream_d,sizeof(INDPREC)*weightsizemax*2));
  OR_FATAL(cudaMalloc((void**)&valstream_d,sizeof(VALPREC)*weightsizemax*2));
  memweight += 2*sizeof(INDPREC)*weightsizemax/1.0e9;
  memweight += 2*sizeof(VALPREC)*weightsizemax/1.0e9;
  OR_FATAL(cudaMemcpy(indstream_d,csrindex[0],sizeof(INDPREC)*csrdispl[0][neuron],cudaMemcpyHostToDevice));
  OR_FATAL(cudaMemcpy(valstream_d,csrvalue[0],sizeof(VALPREC)*csrdispl[0][neuron],cudaMemcpyHostToDevice));
  #else
  OR_FATAL(cudaMalloc((void**)&indbuff_d,sizeof(INDPREC)*weightsizemax));
  OR_FATAL(cudaMalloc((void**)&valbuff_d,sizeof(VALPREC)*weightsizemax));
  memweight += sizeof(INDPREC)*weightsizemax/1.0e9;
  memweight += sizeof(VALPREC)*weightsizemax/1.0e9;
  #endif
  #endif

  double memfeat = 0.0;
  fprintf(stderr, "extbatch=%d, neuron=%d\n", extbatch, neuron);
  {
    const size_t bytes = sizeof(FEATPREC) * size_t(extbatch) * size_t(neuron);
    fflush(stdout);
    fprintf(stderr, "cudaMalloc %lu MB\n", bytes/1024/1024);
    if (cudaSuccess != cudaMalloc((void**)&currfeat_d,bytes)) {
      fprintf(stderr, "ERROR: need more GPU memory\n");
      exit(EXIT_FAILURE);
    }
    fprintf(stderr, "cudaMalloc %lu MB\n", bytes/1024/1024);
    if (cudaSuccess != cudaMalloc((void**)&nextfeat_d,bytes)) {
      fprintf(stderr, "ERROR: need more GPU memory\n");
      exit(EXIT_FAILURE);
    }
    memfeat += bytes/1.0e9;
    memfeat += bytes/1.0e9;
    OR_FATAL(cudaMemset(currfeat_d,0,bytes));
    OR_FATAL(cudaMemset(nextfeat_d,0,bytes));
    OR_FATAL(cudaMemcpy(currfeat_d,currfeat,sizeof(FEATPREC)*mybatch*neuron,cudaMemcpyHostToDevice));
  }

  double memothers[numproc];
  double memweights[numproc];
  double memdispls[numproc];
  double memmaps[numproc];
  double memfeats[numproc];
  // MPI_Allgather(&memother,1,MPI_DOUBLE,memothers,1,MPI_DOUBLE,MPI_COMM_WORLD);
  // MPI_Allgather(&memweight,1,MPI_DOUBLE,memweights,1,MPI_DOUBLE,MPI_COMM_WORLD);
  // MPI_Allgather(&memdispl,1,MPI_DOUBLE,memdispls,1,MPI_DOUBLE,MPI_COMM_WORLD);
  // MPI_Allgather(&memmap,1,MPI_DOUBLE,memmaps,1,MPI_DOUBLE,MPI_COMM_WORLD);
  // MPI_Allgather(&memfeat,1,MPI_DOUBLE,memfeats,1,MPI_DOUBLE,MPI_COMM_WORLD);
memothers[0] = memother;
memweights[0] = memweight;
memdispls[0] = memdispl;
memmaps[0] = memmap;
memfeats[0] = memfeat;
  if(myid==0){
    double memmax = 0.0;
    printf("\n");
    for(int p = 0; p < numproc; p++){
      double memtot = memdispls[p]+memmaps[p]+memweights[p]+memfeats[p];
      printf("GPU %d: OTHERS: %f DISPLS: %f MAPS: %f WEIGHTS: %f FEATURES: %f TOTAL: %f GB\n",p,memothers[p],memdispls[p],memmaps[p],memweights[p],memfeats[p],memtot);
      if(memtot>memmax)memmax=memtot;
    }
    printf("MAX GPU MEM: %f GB\n",memmax);
  }

  cusparseCreate(&handle);
}


/* 
Simultaneously launch the kernel and copy weights for the next layer.

Two streams: kernelStream and copyStream.
kernelStream contains the kernel, as well as the associated memset, and bookkeeping operations
copyStream just has the copy operations for the next layer

use copyStart / copyStop events to time the stream, and start/stop events to time the kernel

*/
void infer_gpu(int l){

/* if OUTOFCORE and OVERLAP, point at the right part of the double-buffer to get the weights from the previous iteration
  if OUTOFCORE and !OVERLAP, copy arguments into the kernel
  otherwise, just get the right layer pointers
*/
  #ifdef OUTOFCORE
  #ifdef OVERLAP
  indbuff_d = indstream_d+(l%2)*weightsizemax;
  valbuff_d = valstream_d+(l%2)*weightsizemax;
  OR_FATAL(cudaStreamSynchronize(copystream));
  #else
  OR_FATAL(cudaEventRecord(copystart,kernelstream));
  int weightsize = csrdispl[l][neuron];
  OR_FATAL(cudaMemcpyAsync(indbuff_d,csrindex[l],sizeof(INDPREC)*weightsize,cudaMemcpyHostToDevice,kernelstream));
  OR_FATAL(cudaMemcpyAsync(valbuff_d,csrvalue[l],sizeof(VALPREC)*weightsize,cudaMemcpyHostToDevice,kernelstream));
  OR_FATAL(cudaEventRecord(copystop,kernelstream));
  #endif
  #else
  indbuff_d = csrindex_d[l];
  valbuff_d = csrvalue_d[l];
  #endif

  dim3 block(blocksize);
  dim3 grid(numblocks,(mybatch+MINIBATCH-1)/MINIBATCH);
  // initialize active features in the batch
  OR_FATAL(cudaMemsetAsync(active_d,0,sizeof(int)*mybatch,kernelstream));

  OR_FATAL(cudaEventRecord(kernelstart,kernelstream));
  cusparseSetStream(handle,kernelstream);
  cusparseSpMatDescr_t Adescr;
  cusparseDnMatDescr_t Bdescr;
  cusparseDnMatDescr_t Cdescr;
  cusparseCreateDnMat(&Bdescr,
                     (int64_t)neuron,
                     (int64_t)mybatch,
                     (int64_t)neuron,
                     (void*)currfeat_d,
                     CUDA_R_32F,
                     CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&Cdescr,
                     (int64_t)neuron,
                     (int64_t)mybatch,
                     (int64_t)neuron,
                     (void*)nextfeat_d,
                     CUDA_R_32F,
                     CUSPARSE_ORDER_COL);

  float alpha = 1;
  float beta = 0;
  size_t bufferSize;
  cusparseCreateCsr(&Adescr,
                   (int64_t)neuron,
                   (int64_t)neuron,
                   (int64_t)csrdispl[l][neuron],
                   (void*)csrdispl_d[l],
                   (void*)indbuff_d,
                   (void*)valbuff_d,
                   CUSPARSE_INDEX_32I,
                   CUSPARSE_INDEX_32I,
                   CUSPARSE_INDEX_BASE_ZERO,
                   CUDA_R_32F);
  cusparseSpMM_bufferSize(handle,
                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                          (void*)&alpha,
                          Adescr,
                          Bdescr,
                          (void*)&beta,
                          Cdescr,
                          CUDA_R_32F,
                          CUSPARSE_SPMM_CSR_ALG1,
                          &bufferSize);
  cusparseSpMM(handle,
               CUSPARSE_OPERATION_NON_TRANSPOSE,
               CUSPARSE_OPERATION_NON_TRANSPOSE,
               (void*)&alpha,
               Adescr,
               Bdescr,
               (void*)&beta,
               Cdescr,
               CUDA_R_32F,
               CUSPARSE_SPMM_CSR_ALG1,
               &bufferSize);
  //cusparseDestroySpMat(Adescr);
  //cusparseDestroyDnMat(Bdescr);
  //cusparseDestroyDnMat(Cdescr);
  grid.y = mybatch;
  relu_kernel<<<grid,block,0,kernelstream>>>(nextfeat_d,bias,neuron,active_d);
  OR_FATAL(cudaEventRecord(kernelstop,kernelstream));

  OR_FATAL(cudaMemcpyAsync(active,active_d,sizeof(int)*mybatch,cudaMemcpyDeviceToHost,kernelstream));

  #ifdef OUTOFCORE
  #ifdef OVERLAP
  if(l+1 < layer){
    OR_FATAL(cudaMemcpyAsync(indstream_d+((l+1)%2)*weightsizemax,csrindex[l+1],sizeof(INDPREC)*csrdispl[l+1][neuron],cudaMemcpyHostToDevice,copystream));
    OR_FATAL(cudaMemcpyAsync(valstream_d+((l+1)%2)*weightsizemax,csrvalue[l+1],sizeof(VALPREC)*csrdispl[l+1][neuron],cudaMemcpyHostToDevice,copystream));
  }
  #else
  OR_FATAL(cudaEventElapsedTime(&elapsedTime,copystart,copystop));
  timecopy += (Duration)elapsedTime/1.0e3;
  #endif
  #endif

  OR_FATAL(cudaStreamSynchronize(kernelstream));

  int feature = 0;
  for(int k = 0; k < mybatch; k++)
    if(active[k]){
      cudaMemcpyAsync(currfeat_d+feature*neuron,nextfeat_d+k*neuron,sizeof(FEATPREC)*neuron,cudaMemcpyDeviceToDevice,kernelstream);
      globalcategories[feature] = globalcategories[k];
      feature++;
    }
  mybatch = feature;

  OR_FATAL(cudaEventElapsedTime(&elapsedTime,kernelstart,kernelstop));
  timekernel += std::chrono::duration<float, std::milli>(elapsedTime);

  //int allfeature = 0;
  //MPI_Allreduce(&feature,&allfeature,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  //if(myid==0)printf("layer %d features %d\n",l,allfeature);
  printf("layer %d features %d\n",l,feature);

};
