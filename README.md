# SOLAR: A High-Performance Data Loading Framework for Distributed DNN Training with Large Datasets

SOLAR is a data-loading framework designed for distributed Deep Neural Networks (DNN) training. It enhances the data loading time by making efficient use of the in-memory buffer. SOLAR is integrated with the PyTorch framework, leveraging parallel HDF5 Python APIs.

While preparing the artifacts, we ran them on a single node from a cluster equipped with 1TB of disk storage, 382GiB of memory, two Intel(R) Xeon(R) Gold 6238 CPUs, and two Tesla A100 GPUs. We recommend that reviewers use a similar system configuration or meet the minimum system requirements.


## Minimum system requirements
OS: Ubuntu (20.04 is recommended)

Storage System: >= 512GB

Memory: >= 64GB RAM

## Step 1: Install Singularity
Install [Singularity](https://singularity-tutorial.github.io/01-installation/)

## Step 2: Download the pre-built Singularity image file
### Method 1: via gdown
```
pip3 install gdown
gdown https://drive.google.com/uc?id=xxx=download
```
### Method 2: via GitHub
```
git clone https://github.com/hipdac-lab/PPoPP24-SOLAR-Image.git
cat PPoPP24-SOLAR-Image/img/solar.sif-* > solar.sif
```

## Step 3: Build and run the image file
```
singularity build --sandbox solar_img/ solar.sif
singularity exec --nv -B /path/to/storage/:`pwd`/solar_img/home/data solar_img/ bash
```

## Step 4: Download and preprocess a sample dataset
```
cd solar_img/home/solar/Cosmoflow/utils
```
set the desired dataset size (in GB)
```
export MY_SIZE=16
```
```
chmod 777 *
./download_and_preprocess.sh
```
```
cd ../
```

## Step 5: I/O Performance Evaluation
### Setup the number of MPI processes
```
export NPROCS=4
```
### Execute I/O evaluation
```
chmod 777 *
./run_io.sh 2>&1 | tee io_results.txt
```

### Execute end-to-end evaluation
```
chmod 777 *
./run_end2end.sh 2>&1 | tee end2end_results.txt
```

## Expected Evaluation Results
### The expected results for the baseline PyTorch Data Loader and SOLAR I/O performances are:
```
Running Baseline IO
This is GPU 0 from node: node0
number of training:1024
Will have 16 steps.
16it [00:18,  1.17s/it]
16it [00:18,  1.13s/it]
16it [00:17,  1.09s/it]
*******************************************
Number of Processes used: 4
Number of Epochs: 3
Batch Size: 16
DataLoading time baseline: 32.829884386388585
DataLoading time baseline each epoch: 
[11.305620059138164, 10.927908385172486, 10.596355942077935]
*******************************************

Running SOLAR shuffle
Cost matrix done! Time: 0.00 s
PSO done! Time: 0.00 s
scheduling done!, Time: 0.01 s
Running SOLAR IO
This is GPU 0 from node: node0
number of training:1024
Will have 16 steps.
16it [00:42,  2.69s/it]
16it [00:18,  1.15s/it]
16it [00:16,  1.06s/it]
*******************************************
Number of Processes used: 4
Number of Epochs: 3
Batch Size: 16
DataLoading time SOLAR: 26.860064087202772
DataLoading time SOLAR each epoch: 

[10.869527072412893, 8.379857301479205, 7.610679713310674]
*******************************************
```
**Note that the data loading time are in seconds.**

### The expected results for the baseline and SOLAR end-to-end performances are:

```
Running Baseline Training
This is GPU 0 from node: node0
number of training:32
Will have 2 steps.
************Baseline***************
Number of Processes used: 1
Number of Epochs: 3
Batch Size: 16
DataLoading time: ['1.091', '1.234', '1.311'] s
Epoch time: ['1.146', '1.240', '1.318'] s
Training Loss: ['0.28433', '0.27858', '0.27614']
Validation Loss: ['0.41840', '0.41763', '0.41663']
*******************************************
Running SOLAR shuffle
Cost matrix done! Time: 0.00 s
PSO done! Time: 0.01 s
scheduling done!, Time: 0.00 s
Running SOLAR Training
This is GPU 0 from node: node0
number of training:32
Will have 2 steps.
Shuffle list loading not yet supported
************SOLAR***************
Number of Processes used: 1
Number of Epochs: 3
Batch Size: 16
DataLoading time: ['1.195', '1.221', '1.179'] s
Epoch time: ['1.256', '1.229', '1.186'] s
Training Loss: ['0.20079', '0.19213', '0.19029']
Validation Loss: ['0.32161', '0.32487', '0.32920']
*******************************************
```
