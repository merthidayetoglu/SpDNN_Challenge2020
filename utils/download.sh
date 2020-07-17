# This script basically downloads the entire dataset and converts them to required binary format used by the program.
#!/bin/bash 

if [[ -z "${PROJREPO}" ]]; then 
       echo "ERROR: Set PROJREPO enviornment variable"	
       exit 0
fi

echo "Downloading Categories"
wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron1024-l120-categories.tsv 
wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron1024-l480-categories.tsv 
wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron1024-l1920-categories.tsv 

wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron4096-l120-categories.tsv 
wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron4096-l480-categories.tsv 
wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron4096-l1920-categories.tsv 

wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron16384-l120-categories.tsv 
wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron16384-l480-categories.tsv 
wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron16384-l1920-categories.tsv 

wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron65536-l120-categories.tsv 
wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron65536-l480-categories.tsv 
wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron65536-l1920-categories.tsv 


echo "Downloading Spase Images"

wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/mnist/sparse-images-1024.tsv.gz
wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/mnist/sparse-images-4096.tsv.gz
wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/mnist/sparse-images-16384.tsv.gz
wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/mnist/sparse-images-65536.tsv.gz


echo "Downloading Weights"
wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron1024.tar.gz
wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron4096.tar.gz
wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron16384.tar.gz
wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron65536.tar.gz


echo "Uncompressing Sparse Images" 
gzip -d sparse-images-1024.tsv.gz
gzip -d sparse-images-4096.tsv.gz
gzip -d sparse-images-16384.tsv.gz
gzip -d sparse-images-65536.tsv.gz

echo "Converting Sparse Images to Binary" 
g++ $PROJREPO/utils/convert_sparseimages.cpp  -o convertimages
./convertimages 1024
./convertimages 4096
./convertimages 16384
./convertimages 65536

echo "Uncompressing Weights"
tar vxfz neuron1024.tar.gz
tar vxfz neuron4096.tar.gz
tar vxfz neuron16384.tar.gz
tar vxfz neuron65536.tar.gz


echo "Converting Weights to Binary"
g++ $PROJREPO/utils/convert_weights.cpp -o convertneuron 
./convertneuron 1024
./convertneuron 4096
./convertneuron 16384
./convertneuron 65536


echo "Is the dataset downloaded successfully and uncompressed? Enter [Y/N]:"
read success

if [[ "$success" == "Y" ]]; then 
	echo "Cleanup not needed files."
	rm -rf sparse-image*.tsv*
	rm -rf neuron1024 neuron4096 neuron16384 neuron65536
	rm -rf neuron1024.tar.gz neuron4096.tar.gz neuron16384.tar.gz neuron65536.tar.gz
else 
	echo "Skipping cleanup!"
fi

echo "Done"
