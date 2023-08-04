# SOLAR: A High-Performance Data Loading Framework for Distributed DNN Training with Large Datasets

SOLAR is a data-loading framework designed for distributed Deep Neural Networks (DNN) training. It enhances the data loading time by making efficient use of the in-memory buffer. SOLAR is integrated with the PyTorch framework, leveraging parallel HDF5 Python APIs.

While preparing the artifacts, we ran them on a single node from a cluster equipped with 1TB of disk storage, 382GiB of memory, two Intel(R) Xeon(R) Gold 6238 CPUs, and two Tesla A100 GPUs. We recommend that reviewers use a similar system configuration or meet the minimum system requirements.


## Minimum System Requirements
OS: Ubuntu (20.04 is recommended)

Storage System: >= 512GB

Memory: >= 64GB RAM

Number of GPU >= 2

GPU Memory >= 16 GB

Python >= 3.8



## Step 1: Install Singularity Container Platform
Install [Singularity](https://singularity-tutorial.github.io/01-installation/)

## Step 2: Download Pre-built Singularity Image File
### Method 1: via gdown
```
pip3 install gdown
gdown https://drive.google.com/u/5/uc?id=1phLdMSgpniiZW0S0qnRoHt_rXhVA74gI&export=download
```

### Method 2: via GitHub
```
git clone https://github.com/hipdac-lab/PPoPP24-SOLAR-Image.git
cat PPoPP24-SOLAR-Image/img/solar.sif-* > solar.sif
```

## Step 3: Build and Run Our Singularity Image
```
singularity build --sandbox solar_img/ solar.sif
singularity exec --nv -B /path/to/storage/:`pwd`/solar_img/home/data solar_img/ bash
```
Please note that you should change **/path/to/storage/** to a path that points to an external storage system with hard disks or SSDs.

## Step 4: Download and Preprocess Sample Dataset
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
### Execute I/O evaluation script
```
./run_io.sh 2>&1 | tee io_results.txt
```

### Execute end-to-end evaluation script
```
./run_end2end.sh 2>&1 | tee end2end_results.txt
```

## Expected Evaluation Results
### The expected results for the baseline and SOLAR I/O performances are:
```
Running Baseline IO
This is GPU 0 from node: node0
number of training:800
Will have 12 steps.
13it [00:08,  1.57it/s]
13it [00:08,  1.58it/s]
13it [00:08,  1.61it/s]
*******************************************
Number of Processes used: 4
Number of Epochs: 3
Batch Size: 16
DataLoading time baseline: 15.191988468635827
DataLoading time baseline each epoch: [5.0673624468036, 5.1143175065517426, 5.010308515280485]
*******************************************

Running SOLAR shuffle
Cost matrix done! Time: 0.00 s
PSO done! Time: 0.01 s
scheduling done!, Time: 0.01 s
Running SOLAR IO
This is GPU 0 from node: node0
number of training:800
Will have 12 steps.
13it [00:24,  1.87s/it]
13it [00:07,  1.68it/s]
13it [00:08,  1.62it/s]
*******************************************
Number of Processes used: 4
Number of Epochs: 3
Batch Size: 16
DataLoading time SOLAR: 10.664569426560774
DataLoading time SOLAR each epoch: [5.050071187783033, 2.7492021687794477, 2.865296069998294]
*******************************************
```
**Note that all data loading time are in seconds.**

### The expected results for the baseline and SOLAR end-to-end performances are:

```
Running Baseline Training
This is GPU 0 from node: hipdac
number of training:32
Will have 2 steps.
100%|██████████| 3/3 [00:10<00:00,  3.49s/it]
************Baseline***************
Number of Processes used: 2
Number of Epochs: 3
Batch Size: 8
DataLoading time: ['0.317', '0.317', '0.314'] s
Epoch time: ['0.348', '0.322', '0.319'] s
Training Loss: ['0.22954', '0.22925', '0.22760']
Validation Loss: ['0.52125', '0.52459', '0.52947']
*******************************************

Running SOLAR shuffle
Cost matrix done! Time: 0.00 s
PSO done! Time: 0.01 s
scheduling done!, Time: 0.00 s
Running SOLAR Training
This is GPU 0 from node: hipdac
number of training:32
Will have 2 steps.
Loading Shuffle List
Loading Shuffle List
100%|██████████| 3/3 [00:10<00:00,  3.45s/it]
************SOLAR***************
Number of Processes used: 2
Number of Epochs: 3
Batch Size: 8
DataLoading time: ['0.301', '0.312', '0.311'] s
Epoch time: ['0.328', '0.316', '0.316'] s
Training Loss: ['0.24233', '0.24393', '0.22934']
Validation Loss: ['0.48375', '0.48202', '0.47959']
*******************************************
```
