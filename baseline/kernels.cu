#include "vars.h"
#include <cuda.h>

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

extern double timebalance;
extern double timekernel;
extern double timecopy;

int **csrdispl_d;
INDPREC *indbuff_d;
VALPREC *valbuff_d;
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

#ifdef BALANCE
int numfeature;
FEATPREC *sendbuff;
FEATPREC *recvbuff;
MPI_Request *catrecvrequests;
MPI_Request *catsendrequests;
MPI_Request *featrecvrequests;
MPI_Request *featsendrequests;
#endif

cudaEvent_t copystart, copystop;
cudaEvent_t kernelstart, kernelstop;
cudaStream_t copystream;
cudaStream_t kernelstream;
float elapsedTime;

__device__ float __ReLU(float x){
   return x<0.0?0.0:x>32.0?32.0:x;
};

__global__ void __launch_bounds__(256,4) dummy_kernel(FEATPREC *nextfeat, FEATPREC *currfeat, int *wdispl, INDPREC *windex, VALPREC *wvalue, float bias, int  *categories, int *active){
  int neuron = blockDim.x*gridDim.x;
  int xoff = blockIdx.x*blockDim.x+threadIdx.x;
  int yoff = categories[blockIdx.y]*neuron;
  float acc = 0;
  for(int n = wdispl[xoff]; n < wdispl[xoff+1]; n++)
    acc += currfeat[yoff+windex[n]]*wvalue[n];
  if(nextfeat[blockIdx.y*neuron+xoff]=__ReLU(acc+bias))
    atomicAdd(active+blockIdx.y,1);
};

void setup_gpu(){

  cudaSetDevice(myid%6);
  //printf("myid %d mydevice %d\n",myid,myid%4);
  cudaFuncSetAttribute(dummy_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,98304);
  if(myid==0){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("\n");
    printf("Device Count: %d\n",deviceCount);
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
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
  cudaEventCreate(&kernelstart);
  cudaEventCreate(&kernelstop);
  cudaEventCreate(&copystart);
  cudaEventCreate(&copystop);
  cudaStreamCreate(&copystream);
  cudaStreamCreate(&kernelstream);

  char *chartemp;
  chartemp = getenv("BLOCKSIZE");
  blocksize = atoi(chartemp);
  numblocks = neuron/blocksize;
  if(myid==0){
    printf("BLOCK SIZE: %d\n",blocksize);
    printf("NUM BLOCKS: %d\n",numblocks);
    printf("\n");
  }

  double memother = 0.0;
  cudaMallocHost((void**)&globalcategories,sizeof(int)*mybatch);
  cudaMallocHost((void**)&categories,sizeof(int)*mybatch);
  cudaMallocHost((void**)&active,sizeof(int)*mybatch);
  cudaMalloc((void**)&active_d,sizeof(int)*mybatch);
  cudaMalloc((void**)&categories_d,sizeof(int)*mybatch);
  memother += sizeof(int)*mybatch/1.0e9;
  memother += sizeof(int)*mybatch/1.0e9;
  for(int k = 0; k < mybatch; k++){
    active[k] = neuron;
    categories[k] = k;
    globalcategories[k] = batchdispl[myid]+k;
  }
  cudaMemset(active_d,0,sizeof(int)*mybatch);
  cudaMemset(categories_d,0,sizeof(int)*mybatch);
  cudaMemcpy(active_d,active,sizeof(int)*mybatch,cudaMemcpyHostToDevice);
  cudaMemcpy(categories_d,categories,sizeof(int)*mybatch,cudaMemcpyHostToDevice);

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
  csrdispl_d = new int*[layer];
  #ifdef OUTOFCORE
  weightsizemax = 0;
  #else
  csrindex_d = new INDPREC*[layer];
  csrvalue_d = new VALPREC*[layer];
  #endif
  for(int l = 0; l < layer; l++){
    cudaMalloc((void**)&csrdispl_d[l],sizeof(int)*(neuron+1));
    memdispl += sizeof(int)*(neuron+1)/1.0e9;
    cudaMemcpy(csrdispl_d[l],csrdispl[l],sizeof(int)*(neuron+1),cudaMemcpyHostToDevice);
    #ifdef OUTOFCORE
    int weightsize = csrdispl[l][neuron];
    if(weightsize > weightsizemax)
      weightsizemax = weightsize; 
    #else
    cudaMalloc((void**)&csrindex_d[l],sizeof(INDPREC)*csrdispl[l][neuron]);
    cudaMalloc((void**)&csrvalue_d[l],sizeof(VALPREC)*csrdispl[l][neuron]);
    memweight += sizeof(INDPREC)*csrdispl[l][neuron]/1.0e9;
    memweight += sizeof(VALPREC)*csrdispl[l][neuron]/1.0e9;
    cudaMemcpy(csrindex_d[l],csrindex[l],sizeof(INDPREC)*csrdispl[l][neuron],cudaMemcpyHostToDevice);
    cudaMemcpy(csrvalue_d[l],csrvalue[l],sizeof(VALPREC)*csrdispl[l][neuron],cudaMemcpyHostToDevice);
    #endif
  }
  #ifdef OUTOFCORE
  if(myid==0)printf("\n");
  if(myid==0)printf("weightsizemax: %d (%f KB)\n",weightsizemax,(sizeof(INDPREC)+sizeof(VALPREC))*weightsizemax/1.0e6);
  #ifdef OVERLAP
  cudaMalloc((void**)&indstream_d,sizeof(INDPREC)*weightsizemax*2);
  cudaMalloc((void**)&valstream_d,sizeof(VALPREC)*weightsizemax*2);
  memweight += 2*sizeof(INDPREC)*weightsizemax/1.0e9;
  memweight += 2*sizeof(VALPREC)*weightsizemax/1.0e9;
  cudaMemcpy(indstream_d,csrindex[0],sizeof(INDPREC)*csrdispl[0][neuron],cudaMemcpyHostToDevice);
  cudaMemcpy(valstream_d,csrvalue[0],sizeof(VALPREC)*csrdispl[0][neuron],cudaMemcpyHostToDevice);
  #else
  cudaMalloc((void**)&indbuff_d,sizeof(INDPREC)*weightsizemax);
  cudaMalloc((void**)&valbuff_d,sizeof(VALPREC)*weightsizemax);
  memweight += sizeof(INDPREC)*weightsizemax/1.0e9;
  memweight += sizeof(VALPREC)*weightsizemax/1.0e9;
  #endif
  #endif

  double memfeat = 0.0;
  cudaMalloc((void**)&currfeat_d,sizeof(FEATPREC)*mybatch*neuron);
  cudaMalloc((void**)&nextfeat_d,sizeof(FEATPREC)*mybatch*neuron);
  memfeat += sizeof(FEATPREC)*mybatch*neuron/1.0e9;
  memfeat += sizeof(FEATPREC)*mybatch*neuron/1.0e9;
  cudaMemset(currfeat_d,0,sizeof(FEATPREC)*mybatch*neuron);
  cudaMemset(nextfeat_d,0,sizeof(FEATPREC)*mybatch*neuron);
  cudaMemcpy(currfeat_d,currfeat,sizeof(FEATPREC)*mybatch*neuron,cudaMemcpyHostToDevice);

  double memweights[numproc];
  double memdispls[numproc];
  double memfeats[numproc];
  MPI_Allgather(&memweight,1,MPI_DOUBLE,memweights,1,MPI_DOUBLE,MPI_COMM_WORLD);
  MPI_Allgather(&memdispl,1,MPI_DOUBLE,memdispls,1,MPI_DOUBLE,MPI_COMM_WORLD);
  MPI_Allgather(&memfeat,1,MPI_DOUBLE,memfeats,1,MPI_DOUBLE,MPI_COMM_WORLD);
  if(myid==0){
    double memmax = 0.0;
    printf("\n");
    for(int p = 0; p < numproc; p++){
      double memtot = memdispls[p]+memweights[p]+memfeats[p];
      printf("GPU %d: DISPLS: %f WEIGHTS: %f FEATURES: %f TOTAL: %f GB\n",p,memdispls[p],memweights[p],memfeats[p],memtot);
      if(memtot>memmax)memmax=memtot;
    }
    printf("MAX GPU MEM: %f GB\n",memmax);
  }
  #ifdef BALANCE
  catrecvrequests = new MPI_Request[numproc];
  catsendrequests = new MPI_Request[numproc];
  featrecvrequests = new MPI_Request[numproc];
  featsendrequests = new MPI_Request[numproc];
  cudaMallocHost((void**)&recvbuff,sizeof(FEATPREC)*mybatch*neuron);
  cudaMallocHost((void**)&sendbuff,sizeof(FEATPREC)*mybatch*neuron);
  numfeature = batch;
  #endif
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
  cudaStreamSynchronize(copystream);
  #else
  cudaEventRecord(copystart,kernelstream);
  int weightsize = csrdispl[l][neuron];
  cudaMemcpyAsync(indbuff_d,csrindex[l],sizeof(INDPREC)*weightsize,cudaMemcpyHostToDevice,kernelstream);
  cudaMemcpyAsync(valbuff_d,csrvalue[l],sizeof(VALPREC)*weightsize,cudaMemcpyHostToDevice,kernelstream);
  cudaEventRecord(copystop,kernelstream);
  #endif
  #else
  indbuff_d = csrindex_d[l];
  valbuff_d = csrvalue_d[l];
  #endif

  dim3 block(blocksize);
  dim3 grid(numblocks,mybatch);
  // initialize active features in the batch
  cudaMemsetAsync(active_d,0,sizeof(int)*mybatch,kernelstream);

  cudaEventRecord(kernelstart,kernelstream);
  dummy_kernel<<<grid,block,0,kernelstream>>>(nextfeat_d,currfeat_d,csrdispl_d[l],indbuff_d,valbuff_d,bias,categories_d,active_d);
  cudaEventRecord(kernelstop,kernelstream);

  cudaMemcpyAsync(active,active_d,sizeof(int)*mybatch,cudaMemcpyDeviceToHost,kernelstream);

  #ifdef OUTOFCORE
  #ifdef OVERLAP
  if(l+1 < layer){
    cudaMemcpyAsync(indstream_d+((l+1)%2)*weightsizemax,csrindex[l+1],sizeof(INDPREC)*csrdispl[l+1][neuron],cudaMemcpyHostToDevice,copystream);
    cudaMemcpyAsync(valstream_d+((l+1)%2)*weightsizemax,csrvalue[l+1],sizeof(VALPREC)*csrdispl[l+1][neuron],cudaMemcpyHostToDevice,copystream);
  }
  #else
  cudaEventElapsedTime(&elapsedTime,copystart,copystop);
  timecopy += elapsedTime/1.0e3;
  #endif
  #endif

  cudaStreamSynchronize(kernelstream);

  int feature = 0;
  for(int k = 0; k < mybatch; k++)
    if(active[k]){
      globalcategories[feature] = globalcategories[k];
      categories[feature] = k;
      feature++;
    }

  #ifdef BALANCE
  double time = MPI_Wtime();
  int allfeature = 0;
  MPI_Allreduce(&feature,&allfeature,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  #if BALANCE==0
  if(allfeature!=numfeature){
  #else
  if(l==BALANCE){
  #endif
    if(myid==0)printf("layer %d oldfeature %d newfeature %d load balancing...\n",l,numfeature,allfeature);
    int features[numproc];
    MPI_Allgather(&feature,1,MPI_INT,features,1,MPI_INT,MPI_COMM_WORLD);
    if(myid==0){
      for(int p = 0; p < numproc; p++)
        printf("%d ",features[p]);
      printf("\n");
    }
    int newfeatures[numproc];
    int tempfeature1 = allfeature/numproc;
    int tempfeature2 = tempfeature1*numproc;
    for(int m = 0; m < numproc; m++)
      if(tempfeature2 < allfeature){
        newfeatures[m] = tempfeature1+1;
        tempfeature2++;
      }
      else
        newfeatures[m] = tempfeature1;
    if(myid==0){
      for(int p = 0; p < numproc; p++)
        printf("%d ",newfeatures[p]);
      printf("\n");
    }
    int commap[numproc][numproc];
    for(int m = 0; m < numproc; m++)
      for(int n = 0; n < numproc; n++){
        commap[m][n] = 0;
        int excess = features[m]-newfeatures[m];
        int need = newfeatures[n]-features[n];
        if(excess > 0 && need > 0){
          int carry = 0;
          if(excess >= need)
            carry = need;
          else
            carry = excess;
          features[m] -= carry;
          features[n] += carry;
          commap[m][n] = carry;
        }
      }
    /*if(myid==0)
      for(int m = 0; m < numproc; m++){
        for(int n = 0; n < numproc; n++)
          printf("%d ",commap[m][n]);
        printf("\n");
      }*/
    //IF A RECEIVER
    if(newfeatures[myid] > feature){
      int irecv = 0;
      int recvamount = 0;
      for(int m = 0; m < numproc; m++)
        if(int amount = commap[m][myid]){
          MPI_Irecv(globalcategories+feature+recvamount,amount,MPI_INT,m,0,MPI_COMM_WORLD,catrecvrequests+irecv);
          MPI_Irecv(recvbuff+recvamount*neuron,sizeof(FEATPREC)*amount*neuron,MPI_BYTE,m,0,MPI_COMM_WORLD,featrecvrequests+irecv);
          recvamount += amount;
          irecv++;
        }
      MPI_Waitall(irecv,featrecvrequests,MPI_STATUSES_IGNORE);
      int counter = 0;
      for(int n = 0; n < batch; n++)
        if(!active[n]){
          cudaMemcpyAsync(nextfeat_d+n*neuron,recvbuff+counter*neuron,sizeof(FEATPREC)*neuron,cudaMemcpyHostToDevice,kernelstream);
          categories[feature] = n;
          feature++;
          counter++;
          if(counter==recvamount)
            break;
        }
      MPI_Waitall(irecv,catrecvrequests,MPI_STATUS_IGNORE);
    }
    //IF A SENDER
    if(newfeatures[myid] < feature){
      int isend = 0;
      for(int n = 0; n < numproc; n++)
        if(int amount = commap[myid][n]){
          for(int m = 0; m < amount; m++){
            feature--;
            cudaMemcpyAsync(sendbuff+feature*neuron,nextfeat_d+categories[feature]*neuron,sizeof(FEATPREC)*neuron,cudaMemcpyDeviceToHost,kernelstream);
          }
          MPI_Issend(globalcategories+feature,amount,MPI_INT,n,0,MPI_COMM_WORLD,catsendrequests+isend);
          cudaStreamSynchronize(kernelstream);
          MPI_Issend(sendbuff+feature*neuron,sizeof(FEATPREC)*amount*neuron,MPI_BYTE,n,0,MPI_COMM_WORLD,featsendrequests+isend);
          isend++;
        }
      MPI_Waitall(isend,catsendrequests,MPI_STATUSES_IGNORE);
      MPI_Waitall(isend,featsendrequests,MPI_STATUSES_IGNORE);
    }
    /*MPI_Allgather(&feature,1,MPI_INT,features,1,MPI_INT,MPI_COMM_WORLD);
    if(myid==0){
      for(int p = 0; p < numproc; p++)
        printf("%d ",features[p]);
      printf("\n");
    }*/
  }
  numfeature = allfeature;
  timebalance += MPI_Wtime()-time;
  #endif

  cudaMemsetAsync(categories_d,0,sizeof(int)*feature,kernelstream);
  cudaMemcpyAsync(categories_d,categories,sizeof(int)*feature,cudaMemcpyHostToDevice,kernelstream);

  cudaEventElapsedTime(&elapsedTime,kernelstart,kernelstop);
  timekernel += elapsedTime/1.0e3;

  mybatch = feature;
  FEATPREC *tempfeat_d = currfeat_d;
  currfeat_d = nextfeat_d;
  nextfeat_d = tempfeat_d;
};
