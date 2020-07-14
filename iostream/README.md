# Dataset:

Download the dataset from: https://uofi.box.com/s/gseet60dz0f939r6n69veggn80i9twwh 

Untar all the files present inside the dataset. 

# Dependencies

1. Latest version of CUDA. 
2. mpicxx compiler 

## Installing mpicxx compiler
```
# For CentOS/RedHat system
sudo dnf install mpich mpich-devel

# For Ubuntu system
sudo apt-get install -y mpich
```

`export` the installed mpich binary path and lib paths to `$PATH` and `$LD_LIBRARY_PATH` variables. 

# Run 
After clearing dependencies, run the following. 

```
cd iostream
make -j 
```

Change DATASET environment variable to reflect your downloaded path in `run_local.sh`

```
bash run_local.sh > output.log 
```

Do let us know if you get any errors in output.log. Ideally it should work without any issues. 
