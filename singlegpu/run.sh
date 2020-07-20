#!/bin/bash
#SBATCH --partition=gpu 
#SBATCH --time=1:00:00 
#SBATCH --reservation=root_34 
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4 
#SBATCH --sockets-per-node=2 
#SBATCH --cores-per-socket=20 
#SBATCH --threads-per-core=4 
#SBATCH --mem-per-cpu=1200
#SBATCH --gres=gpu:v100:4


module load cuda

set -e -x -ou pipefail

date

export DATASET=/gpfs/alpine/csc362/scratch/cpearson/spDNN_data/
export PROFILE_PREFIX=/gpfs/alpine/csc362/scratch/cpearson
#1024 4096 16384 65536
export NEURON=16384
#-0.3 -0.35 -0.4 -0.45
export BIAS=-0.4
#6374505 25019051 98858913 392191985
export INPUT=98858913

export BATCH=60000

export BLOCKSIZE=256
export BUFFER=24

for l in 120 #480 1920
do
  export LAYER=$l
  jsrun -n1 -a1 -g1 -c7 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
  #jsrun -n1 -a1 -g1 -c7 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info cuda-memcheck ./inference
done

date
