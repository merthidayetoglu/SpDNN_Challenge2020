#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
 //1024 4096 16384 65536
 if(argc != 2){
	  printf("./prog <neuronvalue>\n");
	  exit(0);
  }
 int neuron = atoi(argv[1]);

 //6374505 25019051 98858913 392191985
 int input = 6374505;
 if(neuron == 4096)
     input = 25019051;
 if(neuron == 16384)
     input = 98858913;
 if(neuron == 65536) 
     input = 392191985;
 
 int *row = new int[input];
 int *col = new int[input];
 float *val = new float[input];
 char in[80];
 sprintf(in, "sparse-images-%d.tsv", neuron);
 printf("Converting %s to ", in); 
 FILE *inputf = fopen(in,"r");
 for(int n = 0; n < input; n++)
  fscanf(inputf,"%d %d %f\n",col+n,row+n,val+n);
 fclose(inputf);
 char outfile[40];
 sprintf(outfile, "sparse-images-%d.bin", neuron);
 printf("%s\n", outfile); 
 FILE *outputf = fopen(outfile,"wb");
 fwrite(row,sizeof(int),input,outputf);
 fwrite(col,sizeof(int),input,outputf);
 fwrite(val,sizeof(float),input,outputf);
 fclose(outputf);
}
