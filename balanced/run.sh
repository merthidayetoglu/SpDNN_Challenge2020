#!/bin/bash

#BSUB -P CSC362
#BSUB -q debug
#BSUB -W 01:20
#BSUB -nnodes 128
#BSUB -alloc_flags "gpudefault"
#BUSB -env "all,LSF_CPU_ISOLATION=on"
#BSUB -J outt_128nodes
#BSUB -o outt_128nodes.%J
#BSUB -e outt_128nodes.%J

date

export DATASET=/gpfs/alpine/csc362/scratch/merth/spDNN_data/
#1024 4096 16384 65536
export NEURON=16384
#-0.3 -0.35 -0.4 -0.45
export BIAS=-0.4
#6374505 25019051 98858913 392191985
export INPUT=98858913

#120 480 1920
export LAYER=120
export BATCH=60000
#1812 1801 1918 1994

export BLOCKSIZE=256
export BUFFER=24 #KB

#jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info nvprof -o /gpfs/alpine/scratch/merth/csc362/profile/timeline_%p.nvvp -f ./inference
#mv /gpfs/alpine/scratch/merth/csc362/profile/timeline_*.nvvp .
#jsrun -n1 -a1 -g1 -c7 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info nvprof --analysis-metrics -o /gpfs/alpine/scratch/merth/csc362/profile/analysis_%p.nvvp -f ./inference
#mv /gpfs/alpine/scratch/merth/csc362/profile/analysis_*.nvvp .

#jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference

#exit

#1024 4096 16384 65536
export NEURON=1024
#-0.3 -0.35 -0.4 -0.45
export BIAS=-0.3
#6374505 25019051 98858913 392191985
export INPUT=6374505

for l in 120 480 1920
do
  export LAYER=$l
  jsrun -n1 -a1 -g1 -c7 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n1 -a3 -g3 -c21 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n1 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n2 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n8 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n16 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n32 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n64 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n128 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
done

#1024 4096 16384 65536
export NEURON=4096
#-0.3 -0.35 -0.4 -0.45
export BIAS=-0.35
#6374505 25019051 98858913 392191985
export INPUT=25019051

for l in 120 480 1920
do
  export LAYER=$l
  jsrun -n1 -a1 -g1 -c7 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n1 -a3 -g3 -c21 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n1 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n2 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n8 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n16 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n32 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n64 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n128 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
done

#1024 4096 16384 65536
export NEURON=16384
#-0.3 -0.35 -0.4 -0.45
export BIAS=-0.4
#6374505 25019051 98858913 392191985
export INPUT=98858913

for l in 120 480 1920
do
  export LAYER=$l
  jsrun -n1 -a1 -g1 -c7 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n1 -a3 -g3 -c21 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n1 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n2 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n8 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n16 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n32 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n64 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n128 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
done

#1024 4096 16384 65536
export NEURON=65536
#-0.3 -0.35 -0.4 -0.45
export BIAS=-0.45
#6374505 25019051 98858913 392191985
export INPUT=392191985

for l in 120 480 1920
do
  export LAYER=$l
  export BATCH=30000
  jsrun -n1 -a1 -g1 -c7 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  export BATCH=60000
  jsrun -n1 -a3 -g3 -c21 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n1 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n2 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n8 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n16 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n32 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n64 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  jsrun -n128 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
done

date
