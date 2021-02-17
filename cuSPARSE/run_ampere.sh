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

if [[ -z "${PROJREPO}" ]]; then 
       echo "ERROR: Set PROJREPO enviornment variable"	
       exit 0
fi


echo "Cleaning binaries and compiling again"
cp -f Makefile.ampere Makefile
make clean;make -j 

echo "Starting Benchmark"
date

#export DATASET=/home/vsm2/SpDNN_Challenge2020/iostream/dataset
export DATASET=$PROJREPO/dataset
#export DATASET=/home/vsm2/dataset

#1024 4096 16384 65536
#export NEURON=65536
#-0.3 -0.35 -0.4 -0.45
#export BIAS=-0.45
#6374505 25019051 98858913 392191985
#export INPUT=392191985

#120 480 1920
#export LAYER=1920
export BATCH=60000

export BLOCKSIZE=256
export BUFFER=24

export OMP_NUM_THREADS=16


for layer in 120 480 1920 
do 
	for neuron in 1024 4096 16384 65536 
	do 
		if [[ $neuron -eq 1024 ]]
		then 
			export BIAS=-0.3
			export INPUT=6374505
		fi
		if [[ $neuron -eq 4096 ]]
		then 
			export BIAS=-0.35
			export INPUT=25019051
		fi
		if [[ $neuron -eq 16384 ]]
		then 
			export BIAS=-0.4
			export INPUT=98858913
		fi
		if [[ $neuron -eq 65536 ]]
		then 
			export BIAS=-0.45
			export INPUT=392191985
		fi

		export NEURON=$neuron
		export LAYER=$layer

		echo $LAYER
		echo $NEURON
		echo $BIAS
		echo $INPUT
		echo $DATASET
		./inference

		echo "****************************************************"

	done 

done


#for l in 120 480 1920
#do
#  export LAYER=$l
#  jsrun -n1 -a1 -g1 -c7 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
#  jsrun -n1 -a3 -g3 -c21 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
#  jsrun -n1 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
#  jsrun -n2 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
#  jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
#done
#
date
