# SOLAR: A High-Performance Data Loading Framework for Distributed DNN Training with Large Datasets

SOLAR is a data loading framework designed for distributed Deep Neural Networks (DNN) training.

```
Version 1.0
```

## Minimum system requirements
Parallel File System (e.g., Lustre): >= 1TB

Memory: >= 40GB RAM each node

Number of Compute Nodes: >= 2 (Recommended 16 nodes for the example)


## Step 1: Download This Repo
```
git clone https://github.com/hipdac-lab/PPoPP24-SOLAR.git
```

## Step 2: Build Environment

Requirements:
    
(1) spack
        
We use spack to help build this environemnt and manage the dependecies

```        
git clone -c feature.manyFiles=true https://github.com/spack/spack.git
```
```
. spack/share/spack/setup-env.sh
```
```
spack env create solar
```
```
spack env activate solar
```

(2) One of the MPI library that supports MPI-IO: openmpi/4.0.5, mpich/8.1.23
    
(3) Parallel HDF5 (PHDF5)

```
spack install --add hdf5@1.12.1
```
Spack will handle the mpi dependicies

## Step 3: Install anaconda
```
wget https://repo.anaconda.com/archive/Anaconda3-2023.07-1-Linux-x86_64.sh
```
Hit enter if agree the license
```
cd ~/anaconda3/bin
```
```
./conda init
```
```
source ~/.bashrc
```

##Step 4: Build anaconda environment
```
conda create -n solar python=3.8
```
```
conda activate solar
```
```
pip install mpi4py
```
```
export HDF5_ROOT=$(h5cc --version | sed -n '2 p' | awk '{print substr($0, 8 )}')
```
```
HDF5_MPI="ON" HDF5_DIR=$HDF5_ROOT pip install --user --no-binary=h5py h5py
```

## Step 3: Download Dataset
```
cd PPoPP24-SOLAR/cosmoflow/data
```

