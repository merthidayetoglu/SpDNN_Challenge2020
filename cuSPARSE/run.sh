#!/bin/bash

set -e -x -ou pipefail

date

export DATASET=../dataset
export PROFILE_PREFIX=/gpfs/alpine/csc362/scratch/cpearson
#1024 4096 16384 65536
export NEURON=65536
#-0.3 -0.35 -0.4 -0.45
export BIAS=-0.45
#6374505 25019051 98858913 392191985
export INPUT=392191985

export BATCH=30000

export BLOCKSIZE=1024
export BUFFER=48

for l in 120
#for l in 120 480 1920
do
  export LAYER=$l
  for reg in {1..1}
  do
    sed -i '21s/.*/#define MINIBATCH '"$reg"'/' vars.h
    make clean
    make -j

    mpirun -np 1 ./inference
  done
done

date
