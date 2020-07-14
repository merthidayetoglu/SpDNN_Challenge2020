#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {

  int layer = 1920;
  if(argc != 2){
	  printf("./prog <neuronvalue>\n");
	  exit(0);
  }
  int neuron = atoi(argv[1]);
  printf("Neuron: %d \n", neuron);

  int *row = new int[32*neuron];
  int *col = new int[32*neuron];
  float *val = new float[32*neuron];

  char outfile[20];
  sprintf(outfile, "neuron%d.bin", neuron);
  printf("Outfile: %s\n", outfile);

  FILE *outputf = fopen(outfile,"wb");
  for(int l = 0; l < layer; l++){
    char strtemp[80];
    sprintf(strtemp,"neuron%d/n%d-l%d.tsv",neuron, neuron, l+1);
    printf("file %s\n",strtemp);
    FILE *inputf = fopen(strtemp,"r");
    for(int n = 0; n < 32*neuron; n++)
      fscanf(inputf,"%d %d %f\n",col+n,row+n,val+n);
    fclose(inputf);
    fwrite(row,sizeof(int),32*neuron,outputf);
    fwrite(col,sizeof(int),32*neuron,outputf);
    fwrite(val,sizeof(float),32*neuron,outputf);
  }
  fclose(outputf);
}
