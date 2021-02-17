#include "vars.h"


#include <cstring>
#include <cassert>

char *dataset;

int neuron;
int layer;
int batch;
int input;
float bias;

long totnz;
int **csrdispl;
INDPREC **csrindex;
VALPREC **csrvalue;

FEATPREC *currfeat;
FEATPREC *nextfeat;
int *active;
int *categories;
int *globalcategories;

Duration timeio;
Duration timetot;
Duration timeinfer;
Duration timekernel;
Duration timecopy;

int myid;
int numproc;
int numthreads;

int *numbatch;
int *batchdispl;
int mybatch;
int extbatch;



int main(int argc, char** argv) {

  auto startTot = sc::now();

  // timetot = MPI_Wtime();
  // MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  myid = 0;
  // MPI_Comm_size(MPI_COMM_WORLD,&numproc);
  numproc = 1;

  if(myid==0)printf("\n");
  if(myid==0)printf("NUMBER OF PROCESS: %d\n",numproc);

  #pragma omp parallel
  {
    #pragma omp single
    numthreads = omp_get_num_threads();
  }
  if(myid==0)printf("NUMBER OF THREADS: %d\n",numthreads); 

  dataset = getenv("DATASET");
  char *chartemp;
  chartemp = getenv("NEURON");
  neuron = atoi(chartemp);
  chartemp = getenv("LAYER");
  layer = atoi(chartemp);
  chartemp = getenv("BATCH");
  batch = atoi(chartemp);
  chartemp = getenv("INPUT");
  input = atoi(chartemp);
  chartemp = getenv("BIAS");
  bias = atof(chartemp);

  if(myid==0)printf("\n");
  if(myid==0)printf("DATASET: %s\n",dataset);
  if(myid==0)printf("NUMBER OF NEURONS: %d\n",neuron);
  if(myid==0)printf("NUMBER OF LAYERS: %d\n",layer);
  if(myid==0)printf("BATCH SIZE: %d\n",batch);
  if(myid==0)printf("INPUT NZ: %d\n",input);
  if(myid==0)printf("BIAS: %f\n",bias);

  if(myid==0)printf("\n");
  if(myid==0)printf("PARTITIONING\n");
  numbatch = new int[numproc];
  batchdispl = new int[numproc+1];
  int totbatch = batch/numproc*numproc;
  batchdispl[0] = 0;
  for(int p = 0; p < numproc; p++){
    numbatch[p] = batch/numproc;
    if(totbatch < batch){
      totbatch++;
      numbatch[p]++;
    }
    batchdispl[p+1] = batchdispl[p] + numbatch[p];
  }
  mybatch = numbatch[myid];
  extbatch = (mybatch+MINIBATCH-1)/MINIBATCH*MINIBATCH;
 
  if(myid==0) 
    for(int p = 0; p < numproc; p++)
      printf("proc %d batch: %d/%d extbatch %d\n",p,numbatch[p],batch,extbatch);

  csrdispl = new int*[layer];
  csrindex = new INDPREC*[layer];
  csrvalue = new VALPREC*[layer];
  currfeat = new FEATPREC[neuron*(long)mybatch];
  nextfeat = new FEATPREC[neuron*(long)mybatch];

  // MPI_Barrier(MPI_COMM_WORLD);
  // timeio = MPI_Wtime();
  auto startio = sc::now();
  if(myid==0)printf("\n");
  if(myid==0)printf("READING WEIGHTS\n");
  fflush(0);
  readweights();
  fflush(0);
  if(myid==0)printf("READING INPUT\n");
  fflush(0);
  readinput();
  fflush(0);
  // MPI_Barrier(MPI_COMM_WORLD);
  // timeio = MPI_Wtime()-timeio;
  Duration timeio = sc::now() - startio;

  setup_gpu();

  if(myid==0)printf("\n");
  if(myid==0)printf("START INFERENCE\n");
  // MPI_Barrier(MPI_COMM_WORLD);
  // timeinfer = MPI_Wtime();
  auto startInfer = sc::now();
  for(int l = 0; l < layer; l++)
    infer_gpu(l);
  // MPI_Barrier(MPI_COMM_WORLD);
  Duration timeinfer = sc::now() - startInfer;
  if(myid==0)printf("END INFERENCE\n");

  if(myid==0)printf("\n");
  if(myid==0)printf("CHECK CATEGORIES\n");
  int batches[numproc];
  // MPI_Allgather(&mybatch,1,MPI_INT,batches,1,MPI_INT,MPI_COMM_WORLD);
  batches[myid] = mybatch;
  if(myid==0)
    for(int p = 0; p < numproc; p++)
      printf("proc %d categories: %d\n",p,batches[p]);
  int batchesdispl[numproc+1];
  batchesdispl[0] = 0;
  for(int p = 1; p < numproc+1; p++)
    batchesdispl[p] = batchesdispl[p-1] + batches[p-1];
  if(myid==0)
     printf("all categories: %d\n",batchesdispl[numproc]);
  int *allcategories = new int[batchesdispl[numproc]];
  // MPI_Allgatherv(globalcategories,mybatch,MPI_INT,allcategories,batches,batchesdispl,MPI_INT,MPI_COMM_WORLD);
  {
    std::memcpy(allcategories + batchesdispl[0], globalcategories, mybatch * sizeof(*allcategories));
  }

  if(myid==0){
    char chartemp[500];
    sprintf(chartemp,"%s/neuron%d-l%d-categories.tsv",dataset,neuron,layer);
    //sprintf(chartemp,"%s/neuron16384-l120-categories.tsv",dataset);
    FILE *catf = fopen(chartemp,"r");
    bool pass = true;
    for(int k = 0; k < batchesdispl[numproc]; k++){
      int cat;
      if (1 != fscanf(catf,"%d\n",&cat)) {
        fprintf(stderr, "unexpected characters read\n");
        exit(EXIT_FAILURE);
      }
      //printf("cat %d %d\n",cat-1,allcategories[k]);
      if(cat-1!=allcategories[k])
        pass = false;
    }
    fclose(catf);
    if(pass)
      printf("CHALLENGE PASSED!\n");
  }

  if(myid==0){
    printf("\n");
    printf("      I/O TIME: %f s\n",timeio.count());
    printf("INFERENCE TIME: %f s\n",timeinfer.count());
    printf("INFERENCE THRP: %e EDGES/s (%f TFLOPS)\n",totnz/timeinfer.count()*batch,totnz/timeinfer.count()*batch*2/1e12);
    printf("--------------------------------------\n");
  }
  // MPI_Allreduce(MPI_IN_PLACE,&timekernel,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  // MPI_Allreduce(MPI_IN_PLACE,&timecopy,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  if(myid==0){
    printf("KERNEL TIME: %e s\n",timekernel.count()/numproc);
    printf("  COPY TIME: %e s\n",timecopy.count()/numproc);
    printf(" OTHER TIME: %e s\n",(timeinfer-timekernel-timecopy).count()/numproc);
    printf(" TOTAL TIME: %e s\n",timeinfer.count());
  }
  // MPI_Barrier(MPI_COMM_WORLD);
  if(myid==0)printf("\n");
  timetot = sc::now() - startTot;
  if(myid==0)printf("EXECUTION TIME: %f\n",timetot.count());
  // MPI_Finalize();
}

void readweights(){
  totnz = 0;
  for(int l = 0; l < layer; l++){
    int rownz[neuron];
    for(int n = 0; n < neuron; n++)
      rownz[n] = 32;
    csrdispl[l] = new int[neuron+1];
    csrdispl[l][0] = 0;
    for(int n = 1; n < neuron+1; n++)
      csrdispl[l][n] = csrdispl[l][n-1]+rownz[n-1];
    totnz += csrdispl[l][neuron];
    csrindex[l] = new INDPREC[csrdispl[l][neuron]];
    csrvalue[l] = new VALPREC[csrdispl[l][neuron]];
  }
  if(myid==0)printf("weights: %ld (%f GB)\n",totnz,totnz*(sizeof(INDPREC)+sizeof(VALPREC))/1.0e9);
  char chartemp[500];
  //assert(sizeof(chartemp) < sprintf(chartemp,"%s/neuron%d.bin",dataset,neuron));
  sprintf(chartemp,"%s/neuron%d.bin",dataset,neuron);
  printf("open %s", chartemp);
  fflush(0);
  FILE *weightf = fopen(chartemp,"rb");
  if (!weightf) {
    fprintf(stderr, "failed to open %s", chartemp);
    exit(EXIT_FAILURE);
  }
  for(int l = 0; l < layer; l++){
    int *row = new int[csrdispl[l][neuron]];
    int *col = new int[csrdispl[l][neuron]];
    float *val = new float[csrdispl[l][neuron]];
    assert(csrdispl[l][neuron] == fread(row,sizeof(int),csrdispl[l][neuron],weightf));
    assert(csrdispl[l][neuron] == fread(col,sizeof(int),csrdispl[l][neuron],weightf));
    assert(csrdispl[l][neuron] == fread(val,sizeof(int),csrdispl[l][neuron],weightf));
    int rownz[neuron];
    #pragma omp parallel for
    for(int n = 0; n < neuron; n++)
      rownz[n] = 0;
    for(int n = 0; n < csrdispl[l][neuron]; n++){
      csrindex[l][csrdispl[l][row[n]-1]+rownz[row[n]-1]] = col[n]-1;
      csrvalue[l][csrdispl[l][row[n]-1]+rownz[row[n]-1]] = val[n];
      rownz[row[n]-1]++;
    }
    delete[] row;
    delete[] col;
    delete[] val;
  }
  fclose(weightf);
};

void readinput(){
  char chartemp[500];
  FEATPREC *tempfeat;
  if(myid==0){
    printf("features: %ld (%f GB)\n",neuron*(long)batch*2,neuron*(long)batch*2*sizeof(FEATPREC)/1.0e9);
    sprintf(chartemp,"%s/sparse-images-%d.bin",dataset,neuron);
    FILE *inputf = fopen(chartemp,"rb");
    if (!inputf) {
      fprintf(stderr, "missing %s\n", chartemp);
      exit(1);
    }
    int *row = new int[input];
    int *col = new int[input];
    float *val = new float[input];
    assert(input == fread(row,sizeof(int),input,inputf));
    assert(input == fread(col,sizeof(int),input,inputf));
    assert(input == fread(val,sizeof(float),input,inputf));
    if(myid==0){
      tempfeat = new FEATPREC[neuron*(long)batch];
      #pragma omp parallel for
      for(long n = 0; n < neuron*(long)batch; n++)
        tempfeat[n] = 0.0;
      #pragma omp parallel for
      for(int n = 0; n < input; n++)
        if(col[n]-1 < batch)
          tempfeat[(col[n]-1)*(long)neuron+row[n]-1] = val[n];
    }
    fclose(inputf);
    delete[] row;
    delete[] col;
    delete[] val;
  }

  {
    size_t numBytes = size_t(neuron) * size_t(mybatch) * sizeof(FEATPREC);
    fprintf(stderr, "about to memcpy %lu MB\n", numBytes / 1024 / 1024);
    std::memcpy(currfeat, tempfeat, numBytes);
    fprintf(stderr, "done memcpy\n");
  }
  #if 0
  int packetsize = 1000;
  MPI_Request *request;
  //MPI_Request request;
  {
    int numpacket = (mybatch+packetsize-1)/packetsize;
    request = new MPI_Request[numpacket];
    for(int packet = 0; packet < numpacket; packet++){
      int size = packetsize;
      if((packet+1)*packetsize>mybatch)
        size = mybatch%size;
      MPI_Irecv(currfeat+packet*packetsize*(long)neuron,sizeof(FEATPREC)*size*neuron,MPI_BYTE,0,0,MPI_COMM_WORLD,request+packet);
    }
    //MPI_Irecv(currfeat,mybatch*neuron,MPI_FLOAT,0,0,MPI_COMM_WORLD,&request);
  }
  if(myid==0){
    long displ = 0;
    for(int p = 0; p < numproc; p++){
      int numpacket = (numbatch[p]+packetsize-1)/packetsize;
      for(int packet = 0; packet < numpacket; packet++){
        int size = packetsize;
        if((packet+1)*packetsize>numbatch[p])
          size = numbatch[p]%size;
        MPI_Ssend(tempfeat+displ+packet*packetsize*(long)neuron,sizeof(FEATPREC)*size*neuron,MPI_BYTE,p,0,MPI_COMM_WORLD);
      }
      //MPI_Ssend(tempfeat+displ,numbatch[p]*neuron,MPI_FLOAT,p,0,MPI_COMM_WORLD);
      displ += numbatch[p]*(long)neuron;
    }
  }

  {
    int numpacket = (mybatch+packetsize-1)/packetsize;
    MPI_Waitall(numpacket,request,MPI_STATUS_IGNORE);
    delete[] request;
  }
  //MPI_Wait(&request,MPI_STATUS_IGNORE);
  #endif
  if(myid==0){
    delete[] tempfeat;
  }
}
