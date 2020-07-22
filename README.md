# Dataset 
There are two ways to get the dataset. 

## Option 1
We provide converted files that are dervied from the Graph Challenge website over a Box. This can be downloaded from (Total of ~90GB): 
[BoxFolder](https://uofi.box.com/s/gseet60dz0f939r6n69veggn80i9twwh)

After download is complete, untar all the files present inside the dataset. 

## Option 2
Automatic download and compilation (requires gzip and tar support).
Space Required: ~200GB. Post processing ~90GB. 

We assume you have set PROJREPO environment variable to the repo home. 

```
git clone https://github.com/merthidayetoglu/SpDNN_Challenge2020.git
cd SpDNN_Challenge2020
export PROJREPO=$PWD
mkdir dataset
cd dataset
bash $PROJREPO/utils/download.sh
```
# Dependencies

1. Latest version of CUDA. 
2. g++ compiler 

## Installing mpicxx compiler - Ignore if single GPU.
```
# For CentOS/RedHat system
sudo dnf install mpich mpich-devel

# For Ubuntu system
sudo apt-get install -y mpich
```

`export` the installed mpich binary path and lib paths to `$PATH` and `$LD_LIBRARY_PATH` variables. 

# Run 
After clearing dependencies and setting PROJREPO environment variable, run the following. 

```
cd singlegpu 
bash run_ampere.sh > output.log 
```

Do let us know if you get any errors in output.log. Ideally it should work without any issues. 
