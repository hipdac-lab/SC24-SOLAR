from __future__ import print_function
from typing import Text, TextIO
import json
import numpy as np
import os
import torch
import random
import argparse
import time
import socket
import math
import itertools
from tqdm import tqdm 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data.distributed
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import multiprocessing as mp
from ctypes import *
import h5py
import pickle
import functools
import operator
import argparse
from utils import (Logger, AverageTracker, AverageTrackerDevice)

parser = argparse.ArgumentParser(
    description='Run PyTroch Baseline to load Cosmoflow Dataset and preprocess')
parser.add_argument('--data_path', type=str,
                    help='Directory to load CosmoFlow dataset')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Local batch size')
parser.add_argument('--nepochs', type=int, default=3,
                    help='Number of Epochs')
parser.add_argument('--nsamples', type=int, default=64,
                    help='Number of Samples')
parser.add_argument('--buffer_size', type=int, default=4096,
                    help='Number of Samples in buffer')
parser.add_argument('--lists', type=str,help='shuffle lists')
args = parser.parse_args()

#MPI setting

def get_local_rank(required=False):
    """Get local rank from environment."""
    if 'MV2_COMM_WORLD_LOCAL_RANK' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_LOCAL_RANK'])
    if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    if 'SLURM_LOCALID' in os.environ:
        return int(os.environ['SLURM_LOCALID'])
    if required:
        raise RuntimeError('Could not get local rank')
    return 0


def get_local_size(required=False):
    """Get local size from environment."""
    if 'MV2_COMM_WORLD_LOCAL_SIZE' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_LOCAL_SIZE'])
    if 'OMPI_COMM_WORLD_LOCAL_SIZE' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
    if 'SLURM_NTASKS_PER_NODE' in os.environ:
        return int(os.environ['SLURM_NTASKS_PER_NODE'])
    if required:
        raise RuntimeError('Could not get local size')
    return 1


def get_world_rank(required=False):
    """Get rank in world from environment."""
    if 'MV2_COMM_WORLD_RANK' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_RANK'])
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_RANK'])
    if 'SLURM_PROCID' in os.environ:
        return int(os.environ['SLURM_PROCID'])
    if required:
        raise RuntimeError('Could not get world rank')
    return 0


def get_world_size(required=False):
    """Get world size from environment."""
    if 'MV2_COMM_WORLD_SIZE' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_SIZE'])
    if 'OMPI_COMM_WORLD_SIZE' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_SIZE'])
    if 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS'])
    if required:
        raise RuntimeError('Could not get world size')
    return 1


# Set global variables for rank, local_rank, world size
try:
    from mpi4py import MPI

    with_ddp=True
    local_rank=get_local_rank()
    rank=get_world_rank()
    size=get_world_size()

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(size)

    # It will want the master address too, which we'll broadcast:
    if rank == 0:
        master_addr = socket.gethostname()
    else:
        master_addr = None

    master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(2345)

    if local_rank == 0:
        print("This is GPU 0 from node: %s" %(socket.gethostname()))

except Exception as e:
    with_ddp=False
    local_rank = 0
    size = 1
    rank = 0
    print("MPI initialization failed!")
    print(e)

class CosmoFlowConvBlock(torch.nn.Module):
    """Convolution block for CosmoFlow."""

    def __init__(self, input_channels, output_channels, kernel_size,
                 act, pool, padding='valid'):
        """Set up a CosmoFlow convolutional block.

        input_channels: Number of input channels.
        output_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        act: Activation function.
        pool: Pooling function.
        padding: Type of padding to apply, either 'valid' or 'same'.

        """
        super().__init__()
        # Compute padding.
        self.pad_layer = None
        if padding == 'valid':
            pad_size = 0
        elif padding == 'same':
            # Same padding is easy for odd kernel sizes.
            if kernel_size % 2 == 1:
                pad_size = (kernel_size - 1) // 2
            else:
                # For even kernel sizes, we have to manually pad because we
                # need different padding on each side.
                # We follow the TF/Keras convention of putting extra padding on
                # the right and bottom.
                pad_size = 0
                kernel_sizes = [kernel_size]*3
                tf_pad = functools.reduce(
                    operator.__add__,
                    [(k // 2 + (k - 2*(k//2)) - 1, k // 2) for k
                     in kernel_sizes[::-1]])
                self.pad_layer = torch.nn.ConstantPad3d(tf_pad, value=0.0)
        else:
            raise ValueError(f'Unknown padding type {padding}')
        self.conv = torch.nn.Conv3d(input_channels, output_channels,
                                    kernel_size, padding=pad_size)
        self.act = act()
        self.pool = pool(kernel_size=2, stride=2)

    def forward(self, x):
        if self.pad_layer is not None:
            x = self.pad_layer(x)
        return self.pool(self.act(self.conv(x)))


class CosmoFlowModel(torch.nn.Module):
    """Main CosmoFlow model."""

    def __init__(self, input_shape, output_shape,
                 conv_channels=16, kernel_size=2, n_conv_layers=5,
                 fc1_size=128, fc2_size=64,
                 act=torch.nn.LeakyReLU,
                 pool=torch.nn.MaxPool3d,
                 dropout=0.0):
        """Set up the CosmoFlow model.

        input_channels: Dimensions of the input (excluding batch).
            This should be (channels, height, width, depth).
        output_shape: Dimensions of the output (excluding batch).
            This should be the number of regression targets.
        conv_channels: Number of channels in the first conv layer.
            This will increase by a factor of 2 for enach layer.
        kernel_size: Convolution kernel size.
        n_conv_layers: Number of convolutional blocks.
        fc1_size: Number of neurons in the first fully-connected layer.
        fc2_size: Number of neurons in the second fully-connected layer.
        act: Activation function.
        pool: Pooling function.
        dropout: Dropout rate.

        """
        super().__init__()
        # Build the convolutional stack.
        conv_layers = []
        conv_layers.append(CosmoFlowConvBlock(
            input_shape[0], conv_channels, kernel_size, act, pool,
            padding='same'))
        for i in range(1, n_conv_layers):
            conv_layers.append(CosmoFlowConvBlock(
                conv_channels * 2**(i-1), conv_channels * 2**i, kernel_size,
                act, pool, padding='same'))

        # Compute output height/width/depth/channels for first FC layer.
        out_channels = conv_channels * 2**(n_conv_layers-1)
        out_height = input_shape[1] // 2**n_conv_layers
        out_width = input_shape[2] // 2**n_conv_layers
        out_depth = input_shape[3] // 2**n_conv_layers

        # Build the FC stack.
        fc_layers = [
            torch.nn.Flatten(),
            torch.nn.Linear(
                out_channels*out_height*out_depth*out_width, fc1_size),
            act(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(fc1_size, fc2_size),
            act(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(fc2_size, output_shape),
            torch.nn.Tanh()
        ]

        self.layers = torch.nn.Sequential(*conv_layers, *fc_layers)

    def forward(self, x):
        # Output is scaled by 1.2.
        return self.layers(x) * 1.2

class CosmoFlowTransform:
    """Standard transformations for a single CosmoFlow sample."""

    def __init__(self, apply_log):
        """Set up the transform.

        apply_log: If True, log-transform the data, otherwise use
        mean normalization.

        """
        self.apply_log = apply_log

    def __call__(self, x):
        x = x.float()
        if self.apply_log:
            x.log1p_()
        else:
            x /= x.mean() / functools.reduce(operator.__mul__, x.size())
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'

class CosDataset(torch.utils.data.Dataset):
    """Cosmoflow data."""

    SUBDIR_FORMAT = '{:03d}'

    def __init__(self, indices,rank,size, data_dir, dataset_size,cache_size, to_load, local_batch_size,transform=None, transform_y=None):
        """Set up the CosmoFlow HDF5 dataset.

        This expects pre-split universes per split_hdf5_cosmoflow.py.

        You may need to transpose the universes to make the channel
        dimension be first. It is up to you to do this in the
        transforms or preprocessing.

        The sample will be provided to transforms in int16 format.
        The target will be provided to transforms in float format.

        """
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.transform_y = transform_y
        base_universe_size=512
        if h5py is None:
            raise ImportError('HDF5 dataset requires h5py')
        # Load info from cached index.
        idx_filename = os.path.join(data_dir, 'idx')
        with open(idx_filename, 'rb') as f:
            idx_data = pickle.load(f)
        self.sample_base_filenames = idx_data['filenames']
        self.num_subdirs = idx_data['num_subdirs']
        self.num_splits = (base_universe_size // idx_data['split_size'])**3

        self.num_samples = len(self.sample_base_filenames) * self.num_splits
        if dataset_size is not None:
            self.num_samples = min(dataset_size, self.num_samples)
        self.rank = rank
        self.size = size
        self.load_numbers = 0
        self.cache_load = 0
        self.indices = indices
        self.epoch = 0
        self.load_time = 0
        self.cache_time = 0
        self.bench_load_step=set()
        self.cached_data_idx = dict()
        self.prefetch_buffer = dict()
        self.cache_size = cache_size
        self.loc_batch_size=local_batch_size
        self.step = 0
        self.idx_to_load=to_load
        self.idx_to_load_total=to_load
        self.idx_extra_load=set()
        self.loaded=dict()
        self.to_load_per_call = 0
        self.no_load_after_call = 0
        self.not_using=set()
        self.idx_extra_load_total = set()
        self.num_call = 0
        self.loaded_curr_step = set()
        self.num_splits = 64

    def __len__(self):
        'Denotes the total number of samples'
        return self.num_samples

    def set_epoch(self,epoch):
        self.epoch = epoch
        self.num_call = 0
        self.load_numbers = 0
        self.cache_load = 0
        self.load_time = 0
        self.cache_time = 0

    def set_step(self,step):
        self.step=step
        self.load_time = 0
        self.cache_time = 0
        self.load_numbers = 0
        self.cache_load = 0
        self.to_load_per_call = 0
        self.num_call = 0
        if self.epoch > 0 and self.step < len(self.idx_to_load[self.epoch-1]):
            self.not_using = self.idx_to_load[self.epoch-1][self.step]
        if self.epoch > 0:
            if len(self.idx_to_load[self.epoch-1]) > self.step:
                self.idx_extra_load = set(list(self.idx_to_load[self.epoch-1][self.step])[self.rank::self.size])
                self.idx_extra_load_total = set(list(self.idx_to_load[self.epoch-1][self.step]))
                self.to_load_per_call = math.ceil(len(self.idx_extra_load_total)/self.loc_batch_size)
                self.no_load_after_call = len(self.idx_extra_load_total)
            else: 
                self.idx_extra_load = set()
        self.loaded_curr_step = set()
        self.bench_load_step = set()

    def getLoadNumber(self):
        return self.load_numbers

    def getCacheLoad(self):
        return self.cache_load

    def getItemBalancing(self,idx,flag):
        self.loaded_curr_step.add(idx)
        cached=False
        prefetched=False
        if idx in self.cached_data_idx.keys():
            cached = True
        if idx in self.prefetch_buffer.keys():
            prefetched = True
        if not cached and not prefetched:
            load_time_start=time.perf_counter()
            self.load_numbers += 1
            base_index = idx // self.num_splits
            split_index = idx % self.num_splits
            load_time_start=time.perf_counter()
            if self.num_subdirs:
                subdir = CosDataset.SUBDIR_FORMAT.format(
                    base_index // self.num_subdirs)
                filename = os.path.join(
                    self.data_dir,
                    subdir,
                    self.sample_base_filenames[base_index]
                    + f'_{split_index:03d}.hdf5')
                x_idx = 'split'
            else:
                filename = os.path.join(
                    self.data_dir,
                    self.sample_base_filenames[base_index]
                    + f'_{split_index:03d}.hdf5')
                x_idx = 'full'
            with h5py.File(filename, 'r') as f:
                x, y = f[x_idx][:], f['unitPar'][:]
            # Convert to Tensors.
            x = torch.from_numpy(x)
            y = torch.from_numpy(y)
            if self.transform is not None:
                x = self.transform(x)
            if self.transform_y is not None:
                y = self.transform_y(y)
            self.load_time +=time.perf_counter()-load_time_start
            if flag:
                self.cached_data_idx[idx]=[x,y]
                if len(self.cached_data_idx) > self.cache_size:
                    if 0 == self.epoch:
                        idx_to_replace=list(self.cached_data_idx.keys())[0]
                    else:
                        idx_to_replace=list(self.cached_data_idx.keys())[0]
                        if self.step < len(self.idx_to_load[self.epoch-1]):
                            for k in self.cached_data_idx.keys():
                                if k in self.not_using and k!=idx:
                                    idx_to_replace = k
                                    #self.idx_to_load[self.epoch-1][self.step].remove(k)
                                    break
                    self.cached_data_idx.pop(idx_to_replace)
            else:
                self.prefetch_buffer[idx]=[x,y]
            
        elif cached and not prefetched:
            self.cache_load += 1
            cache_time_start=time.perf_counter()
            x=self.cached_data_idx[idx][0]
            y=self.cached_data_idx[idx][1]
            self.cache_time +=time.perf_counter()-cache_time_start
        elif prefetched and not cached:
            self.cache_load += 1
            cache_time_start=time.perf_counter()
            x=self.prefetch_buffer[idx][0]
            y=self.prefetch_buffer[idx][1]
            self.cache_time +=time.perf_counter()-cache_time_start
        return x,y

    def get_time(self):
        return self.load_time,self.cache_time

    def __getitem__(self, index):
        
        self.num_call += 1
        idx = int(self.indices[self.epoch][index])
        x_list=[]
        y_list=[]
        if self.epoch == 0:
            return self.getItemBalancing(idx,True)
       
        if self.to_load_per_call > 0 and self.num_call <= self.no_load_after_call and self.epoch > 0:
        
            for tt in range(self.to_load_per_call):
                if len(self.idx_extra_load) > 0:
                    tt_idx = int(self.idx_extra_load.pop())
                    x_temp,y_temp = self.getItemBalancing(tt_idx,False)
                    x_list.append(x_temp)
                    y_list.append(y_temp)
        if idx not in self.idx_extra_load_total:  
            x,y = self.getItemBalancing(idx,True)
            x_list.append(x)
            y_list.append(y)
        return x_list,y_list

collate_times=[]
def swift_collate(batch):
    collate_time_start=time.perf_counter()
    numel = 0
    for xx in batch:
        if xx[0] is not None:
            numel += len(xx[0])
    tensor_x = torch.zeros(size=(numel,4,128,128,128))
    tensor_y = torch.zeros(size=(numel,4))
    pointer=0
    for i, ele in enumerate(batch):
        if len(ele[0])==1: #First epoch
            tensor_x[pointer] += ele[0][0]
            tensor_y[pointer] += ele[1][0]
            pointer += 1
        else:
            for k in range(len(ele[0])):
                tensor_x[pointer] += ele[0][k]
                tensor_y[pointer] += ele[1][k]
                pointer += 1
    collate_time=time.perf_counter()-collate_time_start
    collate_times.append(collate_time)
    return tensor_x,tensor_y

class CosDataset_val(torch.utils.data.Dataset):
    """Cosmoflow data."""

    SUBDIR_FORMAT = '{:03d}'

    def __init__(self, indices, data_dir,dataset_size,
                 transform=None, transform_y=None):
        """Set up the CosmoFlow HDF5 dataset.

        This expects pre-split universes per split_hdf5_cosmoflow.py.

        You may need to transpose the universes to make the channel
        dimension be first. It is up to you to do this in the
        transforms or preprocessing.

        The sample will be provided to transforms in int16 format.
        The target will be provided to transforms in float format.

        """
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.transform_y = transform_y
        base_universe_size=512
        if h5py is None:
            raise ImportError('HDF5 dataset requires h5py')
        # Load info from cached index.
        idx_filename = os.path.join(data_dir, 'idx')
        with open(idx_filename, 'rb') as f:
            idx_data = pickle.load(f)
        self.sample_base_filenames = idx_data['filenames']
        self.num_subdirs = idx_data['num_subdirs']
        self.num_splits = (base_universe_size // idx_data['split_size'])**3

        self.num_samples = len(self.sample_base_filenames) * self.num_splits
        if dataset_size is not None:
            self.num_samples = min(dataset_size, self.num_samples)
        self.rank = rank
        self.load_numbers = 0
        self.cache_load = 0
        self.indices = indices
        self.epoch = 0
        self.load_time = 0
        self.cache_time = 0
        self.num_splits = 64
        

    def __len__(self):
        'Denotes the total number of samples'
        return self.num_samples

    def set_epoch(self,epoch):
        self.epoch = epoch
        self.load_numbers = 0
        self.cache_load = 0
        self.load_time = 0
        self.cache_time = 0

    def set_step(self):
        self.load_numbers = 0
        self.cache_load = 0
        self.load_time = 0
        self.cache_time = 0

    def getLoadNumber(self):
        return self.load_numbers

    def getCacheLoad(self):
        return self.cache_load

    
    def get_time(self):
        return self.load_time,self.cache_time

    def __getitem__(self, index):
        idx = int(self.indices[self.epoch][index])
        self.load_numbers += 1
        base_index = idx // self.num_splits
        split_index = idx % self.num_splits
        load_time_start=time.perf_counter()
        if self.num_subdirs:
            subdir = CosDataset.SUBDIR_FORMAT.format(
                base_index // self.num_subdirs)
            filename = os.path.join(
                self.data_dir,
                subdir,
                self.sample_base_filenames[base_index]
                + f'_{split_index:03d}.hdf5')
            x_idx = 'split'
        else:
            filename = os.path.join(
                self.data_dir,
                self.sample_base_filenames[base_index]
                + f'_{split_index:03d}.hdf5')
            x_idx = 'full'
        with h5py.File(filename, 'r') as f:
            x, y = f[x_idx][:], f['unitPar'][:]
        # Convert to Tensors.
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        if self.transform is not None:
            x = self.transform(x)
        if self.transform_y is not None:
            y = self.transform_y(y)
        self.load_time +=time.perf_counter()-load_time_start
        return x, y

######################Parameter setup###############################
DATA_PATH = os.path.join(args.data_path,'train')
VAL_PATH = os.path.join(args.data_path,'validation')
device='cpu'
batch_size = args.batch_size
run_time = 2
nepochs=args.nepochs
# load data
filelist = []
cache_size = args.buffer_size
total_train_size=args.nsamples
saving_path=args.lists
total_val_size=128
apply_log = True
BATCH_SIZE=batch_size
nsamples = total_train_size
GLOBAL_BATCH_SIZE=BATCH_SIZE*size
step_size = round(nsamples/GLOBAL_BATCH_SIZE)
if rank == 0:
    print('number of training:%d' % total_train_size)
    print("Will have %s steps." %step_size)
torch.cuda.init()
torch.cuda.set_device(get_local_rank())
torch.distributed.init_process_group('nccl', init_method='env://', rank=get_local_rank(), world_size=get_world_size())
######################END Parameter setup###############################
#shuffle list
shuffle_list=np.zeros([nepochs,nsamples])
shuffle_list_sorted=np.zeros([nepochs,nsamples])



def generate_weight_matrix_cache_fifo_new(arr,local_batch_size,cache_size,size):
    matrix=np.zeros([arr.shape[0],arr.shape[0]])
    for idx in range(arr.shape[0]):
        source_cache = arr[idx,-cache_size:].reshape(cache_size)
        for r in range(arr.shape[0]):
            cost=0
            rank_count=np.zeros(size)
            if not idx==r:
                curr_samples = arr[r,:cache_size].reshape(cache_size)
                d = dict()
                for i in range(source_cache.shape[0]):
                    d[source_cache[i]] = 0
                for s in range(curr_samples.shape[0]):
                    curr_rank = s % size
                    if not curr_samples[s] in d.keys():
                        cost += 1
                matrix[idx][r] = int(cost//size)
            
            else:
                matrix[idx][r] = np.nan
    
    return matrix


def shard(ngpus,array):
    list_result=np.zeros([ngpus,math.ceil(array.shape[0]/ngpus)])
    for g in range(ngpus):
        ii=0
        list_result[g]=array[g::ngpus]
    return list_result  

#https://github.com/marcoscastro/tsp_pso/blob/master/tsp_pso.py
# encoding:utf-8

'''
    Solution for Travelling Salesman Problem using PSO (Particle Swarm Optimization)
    Discrete PSO for TSP
    References: 
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.258.7026&rep=rep1&type=pdf
        http://www.cs.mun.ca/~tinayu/Teaching_files/cs4752/Lecture19_new.pdf
        http://www.swarmintelligence.org/tutorials.php
    References are in the folder "references" of the repository.
'''

from operator import attrgetter
import random, sys, time, copy


# class that represents a graph
class Graph:

    def __init__(self, amount_vertices):
        self.edges = {} # dictionary of edges
        self.vertices = set() # set of vertices
        self.amount_vertices = amount_vertices # amount of vertices


    def addEdge(self, src, dest, cost = 0):
        if not self.existsEdge(src, dest):
            self.edges[(src, dest)] = cost
            self.vertices.add(src)
            self.vertices.add(dest)

    def existsEdge(self, src, dest):
        return (True if (src, dest) in self.edges else False)


    # shows all the links of the graph
    def showGraph(self):
        print('Showing the graph:\n')
        for edge in self.edges:
            print('%d linked in %d with cost %d' % (edge[0], edge[1], self.edges[edge]))

    def getCostPath(self, path):

        total_cost = 0
        for i in range(self.amount_vertices - 1):
            total_cost += self.edges[(path[i], path[i+1])]

        
        return total_cost


    def getRandomPaths(self, max_size):

        random_paths, list_vertices = [], list(self.vertices)

        initial_vertice = random.choice(list_vertices)
        if initial_vertice not in list_vertices:
            print('Error: initial vertice %d not exists!' % initial_vertice)
            sys.exit(1)

        list_vertices.remove(initial_vertice)
        list_vertices.insert(0, initial_vertice)

        for i in range(max_size):
            list_temp = list_vertices[1:]
            random.shuffle(list_temp)
            list_temp.insert(0, initial_vertice)

            if list_temp not in random_paths:
                random_paths.append(list_temp)

        return random_paths


# class that represents a complete graph
class CompleteGraph(Graph):

    # generates a complete graph
    def generates(self):
        for i in range(self.amount_vertices):
            for j in range(self.amount_vertices):
                if i != j:
                    weight = random.randint(1, 10)
                    self.addEdge(i, j, weight)


# class that represents a particle
class Particle:

    def __init__(self, solution, cost):

        # current solution
        self.solution = solution

        # best solution (fitness) it has achieved so far
        self.pbest = solution

        # set costs
        self.cost_current_solution = cost
        self.cost_pbest_solution = cost

        self.velocity = []

    # set pbest
    def setPBest(self, new_pbest):
        self.pbest = new_pbest

    # returns the pbest
    def getPBest(self):
        return self.pbest

    # set the new velocity (sequence of swap operators)
    def setVelocity(self, new_velocity):
        self.velocity = new_velocity

    # returns the velocity (sequence of swap operators)
    def getVelocity(self):
        return self.velocity

    # set solution
    def setCurrentSolution(self, solution):
        self.solution = solution

    # gets solution
    def getCurrentSolution(self):
        return self.solution

    # set cost pbest solution
    def setCostPBest(self, cost):
        self.cost_pbest_solution = cost

    # gets cost pbest solution
    def getCostPBest(self):
        return self.cost_pbest_solution

    # set cost current solution
    def setCostCurrentSolution(self, cost):
        self.cost_current_solution = cost

    # gets cost current solution
    def getCostCurrentSolution(self):
        return self.cost_current_solution

    # removes all elements of the list velocity
    def clearVelocity(self):
        del self.velocity[:]


# PSO algorithm
class PSO:

    def __init__(self, graph, iterations, size_population, beta=1, alfa=1):
        self.graph = graph # the graph
        self.iterations = iterations # max of iterations
        self.size_population = size_population # size population
        self.particles = [] # list of particles
        self.beta = beta # the probability that all swap operators in swap sequence (gbest - x(t-1))
        self.alfa = alfa # the probability that all swap operators in swap sequence (pbest - x(t-1))

        # initialized with a group of random particles (solutions)
        solutions = self.graph.getRandomPaths(self.size_population)

        # checks if exists any solution
        if not solutions:
            print('Initial population empty! Try run the algorithm again...')
            sys.exit(1)

        # creates the particles and initialization of swap sequences in all the particles
        for solution in solutions:
            # creates a new particle
            particle = Particle(solution=solution, cost=graph.getCostPath(solution))
            # add the particle
            self.particles.append(particle)

        # updates "size_population"
        self.size_population = len(self.particles)


    # set gbest (best particle of the population)
    def setGBest(self, new_gbest):
        self.gbest = new_gbest

    # returns gbest (best particle of the population)
    def getGBest(self):
        return self.gbest


    # shows the info of the particles
    def showsParticles(self):

        print('Showing particles...\n')
        for particle in self.particles:
            print('pbest: %s\t|\tcost pbest: %d\t|\tcurrent solution: %s\t|\tcost current solution: %d' \
                % (str(particle.getPBest()), particle.getCostPBest(), str(particle.getCurrentSolution()),
                            particle.getCostCurrentSolution()))
        print('')


    def run(self):

        for t in range(self.iterations):

            self.gbest = min(self.particles, key=attrgetter('cost_pbest_solution'))

            for particle in self.particles:

                particle.clearVelocity() 
                temp_velocity = []
                solution_gbest = copy.copy(self.gbest.getPBest()) 
                solution_pbest = particle.getPBest()[:] 
                solution_particle = particle.getCurrentSolution()[:] 

                for i in range(self.graph.amount_vertices):
                    if solution_particle[i] != solution_pbest[i]:
                        swap_operator = (i, solution_pbest.index(solution_particle[i]), self.alfa)

                        temp_velocity.append(swap_operator)

                        aux = solution_pbest[swap_operator[0]]
                        solution_pbest[swap_operator[0]] = solution_pbest[swap_operator[1]]
                        solution_pbest[swap_operator[1]] = aux

                
                for i in range(self.graph.amount_vertices):
                    if solution_particle[i] != solution_gbest[i]:
                        swap_operator = (i, solution_gbest.index(solution_particle[i]), self.beta)
                        temp_velocity.append(swap_operator)
                        aux = solution_gbest[swap_operator[0]]
                        solution_gbest[swap_operator[0]] = solution_gbest[swap_operator[1]]
                        solution_gbest[swap_operator[1]] = aux
                particle.setVelocity(temp_velocity)

                for swap_operator in temp_velocity:
                    if random.random() <= swap_operator[2]:
                        aux = solution_particle[swap_operator[0]]
                        solution_particle[swap_operator[0]] = solution_particle[swap_operator[1]]
                        solution_particle[swap_operator[1]] = aux


                particle.setCurrentSolution(solution_particle)
                cost_current_solution = self.graph.getCostPath(solution_particle)
                particle.setCostCurrentSolution(cost_current_solution)

                if cost_current_solution < particle.getCostPBest():
                    particle.setPBest(solution_particle)
                    particle.setCostPBest(cost_current_solution)



if run_time == 1:
    for epoch in range(nepochs):
        idx_arr = np.arange(nsamples)
        np.random.shuffle(idx_arr)
        
        shuffle_list[epoch] = idx_arr
    
    np.save(saving_path+'original_shuffle_list.npy', shuffle_list)
    idx_to_load_total=[]
    if 0 == rank:
        mat_start_time = time.perf_counter()
        matrix_result = generate_weight_matrix_cache_fifo_new(shuffle_list,BATCH_SIZE,cache_size,size)
        mat_time = time.perf_counter() - mat_start_time
        print("Cost matrix done! Time: ",mat_time)
        np.save(saving_path+'matrix_result.npy', matrix_result)
        #Print the original cost
        cost_default1=0
        for e in range(nepochs-1):
            cost_default1 +=  matrix_result[e][e+1]
        print("Original cost: %s" %(cost_default1))
        # creates the Graph instance
        graph = Graph(amount_vertices=nepochs)
        for i in range(nepochs):
            for j in range(nepochs):
                if not i == j:
                    weight = matrix_result[i][j]
                    graph.addEdge(i,j,weight)
                    
        # creates a PSO instance
        pso = PSO(graph, iterations=1000, size_population=nepochs, beta=1, alfa=0.9)
        pso_start_time = time.perf_counter()
        pso.run() # runs the PSO algorithm
        pso_time = time.perf_counter() - pso_start_time
        print("PSO done! Time: ",pso_time)
        res_path=pso.getGBest().getPBest()
        pso_cost = pso.getGBest().getCostPBest()
        #Get the scheduled shuffle list
        cost_def = 0
        for i in range(nepochs-1):
            print(matrix_result[i][i+1])
            cost_def += matrix_result[i][i+1]
        print(res_path)
        cost_pso=0
        for j in range(nepochs-1):
            u=int(res_path[j])
            v=int(res_path[j+1])
            cost_pso += matrix_result[u][v]
        print("Default cost: %s, PSO cost: %s" %(cost_def,pso_cost))
        
        for e_i in range(nepochs):
            curr_idx = res_path[e_i]
            shuffle_list_sorted[e_i] = shuffle_list[curr_idx]
        np.save(saving_path+'shuffled_list_sorted.npy', shuffle_list_sorted)
        num_steps_in_cache = int(cache_size//GLOBAL_BATCH_SIZE)
        nswap_list=[]
        scheduling_start = time.perf_counter()
        for i in range(nepochs-1):
            
            source_sample = shuffle_list_sorted[i,-cache_size:]

            idx_to_load_epoch=[]
            for step in range(num_steps_in_cache):
                
                st = step * GLOBAL_BATCH_SIZE
                ed = st + GLOBAL_BATCH_SIZE
                target_sample = shuffle_list_sorted[i+1,st:ed]
                idx_to_load=[]
                idx_to_load_step=set()
                pointer_right=GLOBAL_BATCH_SIZE-1
                next_avail=np.arange(size)
                temp_list=np.zeros(GLOBAL_BATCH_SIZE)
                temp_list[:] = np.nan
                sevisity=0
                for s in range(GLOBAL_BATCH_SIZE):
                    idx_in_source = np.where(source_sample == target_sample[s])
                    temp_var=-1
                    if 1 == np.size(idx_in_source):
                        rank_in_source = idx_in_source[0] % size
                        idx_in_temp = next_avail[rank_in_source]
                        if idx_in_temp < GLOBAL_BATCH_SIZE:
                            temp_list[idx_in_temp] = target_sample[s]
                            next_avail[rank_in_source] += size
                        else:
                            avail_idx = np.argwhere(np.isnan(temp_list))
                            de_idx = avail_idx[0]
                            de_idx_rank = de_idx % size
                            temp_list[de_idx] = target_sample[s]
                            next_avail[de_idx_rank] += size
                            sevisity += 1
                            idx_to_load_step.add(target_sample[s])
                    else:
                        idx_to_load.append(target_sample[s])
                        idx_to_load_step.add(target_sample[s])
                
                idx_to_fit=np.argwhere(np.isnan(temp_list))
                idx_shaped = idx_to_fit.reshape(np.size(idx_to_fit))
                temp_list[idx_shaped] = idx_to_load
                idx_to_load_epoch.append(idx_to_load_step)
                shuffle_list_sorted[i+1,st:ed] = temp_list #Replace original order with the new order     
            idx_to_load_total.append(idx_to_load_epoch)
        scheduling_time = time.perf_counter() - scheduling_start
        print("scheduling done!, time: %s", scheduling_time)
    #Boardcast the shuffle lists before using them
    shuffle_list_sorted = MPI.COMM_WORLD.bcast(shuffle_list_sorted, root=0)
    idx_to_load_total = MPI.COMM_WORLD.bcast(idx_to_load_total, root=0)
    np.save(saving_path+'shuffle_list_sorted_debug.npy', shuffle_list_sorted)
    with open(saving_path+"idx_to_load_total_debug", "wb") as fp:
        pickle.dump(idx_to_load_total,fp)
else:
    shuffle_list = np.load(saving_path+'original_shuffle_list.npy')
    shuffle_list_sorted = np.load(saving_path+'shuffled_list_sorted.npy')
    with open(saving_path+"idx_to_load_total", "rb") as fp:
        idx_to_load_total = pickle.load(fp)

val_shuffle_list=np.zeros([nepochs,total_val_size])
if run_time == 1:
    for epoch in range(nepochs):
        idx_arr = np.arange(total_val_size)
        np.random.shuffle(idx_arr)
        val_shuffle_list[epoch] = idx_arr
    val_shuffle_list = MPI.COMM_WORLD.bcast(val_shuffle_list, root=0)
else:
    print('Loading Shuffle List')
transform = CosmoFlowTransform(apply_log)
train_data2=CosDataset(indices=shuffle_list_sorted,
                        rank=rank,
                        size=size,
                        data_dir=DATA_PATH,
                        dataset_size=total_train_size,
                        cache_size=cache_size/size,
                        to_load=idx_to_load_total,
                        local_batch_size=BATCH_SIZE,
                        transform=transform
                        )

kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'gpu' else {}
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_data2, num_replicas=size, shuffle=False, rank=rank)
train_loader = torch.utils.data.DataLoader(
    train_data2, batch_size=BATCH_SIZE, sampler=train_sampler,  **kwargs)

val_data2=CosDataset_val(indices=val_shuffle_list, data_dir=VAL_PATH, dataset_size=total_val_size,transform=transform)
kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'gpu' else {}
val_sampler = torch.utils.data.distributed.DistributedSampler(
    val_data2, num_replicas=size, shuffle=False, rank=rank)
val_loader = torch.utils.data.DataLoader(
    val_data2, batch_size=BATCH_SIZE, sampler=val_sampler,  **kwargs)


def get_cosmoflow_lr_schedule(optimizer, base_lr, target_lr, warmup_epochs,
                              decay_epochs, decay_factors):
    if warmup_epochs:
        target_warmup_factor = target_lr / base_lr
        warmup_factor = target_warmup_factor / warmup_epochs

    def lr_sched(epoch):
        factor = 1.0
        if warmup_epochs:
            if epoch > 0:
                if epoch < warmup_epochs:
                    factor = warmup_factor * epoch
                else:
                    factor = target_warmup_factor
        for step, decay in zip(decay_epochs, decay_factors):
            if epoch >= step:
                factor *= decay
        return factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

net = CosmoFlowModel(
        [4, 128, 128, 128], 4,
        conv_channels=32,
        kernel_size=2,
        n_conv_layers=5,
        fc1_size=128, fc2_size=64,
        act=torch.nn.LeakyReLU,
        pool=torch.nn.MaxPool3d,
        dropout=0.5)
net = DDP(net.cuda())
scaler = torch.cuda.amp.GradScaler(enabled=False)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(
        net.parameters(), 0.001, momentum=0.9)
scheduler = get_cosmoflow_lr_schedule(
        optimizer, 0.001, 0.0001, 0,
        [32, 64], [0.25, 0.25])
best_loss = float('inf')

epoch_time=[]
io_time_epoch=[]
losses_epoch=[]
mae_epoch=[]
val_loss_epoch=[]
val_mae_epoch=[]
for epoch in tqdm(range(nepochs),disable=(rank==0)):
    start_time = time.perf_counter()
    train_loader.sampler.set_epoch(epoch)
    val_loader.sampler.set_epoch(epoch)
    net.train()
    losses = AverageTrackerDevice(len(train_loader), get_local_rank(),
                                  allreduce=True)
    maes = AverageTrackerDevice(len(train_loader), get_local_rank(),
                                allreduce=True)
    batch_times = AverageTracker()
    data_times = AverageTracker()
    end_time = time.perf_counter()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)


    for batch, (samples, targets) in enumerate(train_loader):
        start_event.record()

        samples = samples.cuda()
        targets = targets.cuda()
        data_times.update(time.perf_counter() - end_time)
        data_time = time.perf_counter() - end_time
        with torch.cuda.amp.autocast(enabled=False):
            output = net(samples)
            loss = criterion(output, targets)
            with torch.no_grad():
                mae = torch.nn.functional.l1_loss(output, targets)

        losses.update(loss, samples.size(0))
        maes.update(mae, samples.size(0))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        batch_times.update(time.perf_counter() - end_time)
        end_time = time.perf_counter()
    
    epoch_time.append(f'{batch*batch_times.mean():.3f}')
    io_time_epoch.append(f'{batch*data_times.mean():.3f}')
    losses_epoch.append(f'{losses.mean():.5f}')
    mae_epoch.append(f'{maes.mean():.5f}')

    net.eval()
    losses = AverageTrackerDevice(len(val_loader), get_local_rank(),
                                  allreduce=True)
    maes = AverageTrackerDevice(len(val_loader), get_local_rank(),
                                allreduce=True)
    with torch.no_grad():
        for samples, targets in val_loader:
            samples = samples.cuda()
            targets = targets.cuda()

            with torch.cuda.amp.autocast(enabled=False):
                output = net(samples)
                loss = criterion(output, targets)
                mae = torch.nn.functional.l1_loss(output, targets)

            losses.update(loss, samples.size(0))
            maes.update(mae, samples.size(0))
    val_loss_epoch.append(f'{losses.mean():.5f}')
    val_mae_epoch.append(f'{maes.mean():.5f}')

if rank==0:
    print("************SOLAR***************")
    print("Number of Processes used: "+str(size))
    print("Number of Epochs: "+str(nepochs))
    print("Batch Size: "+str(BATCH_SIZE))
    print("DataLoading time: %s s" %(io_time_epoch))
    print("Epoch time: %s s" %(epoch_time))
    print("Training Loss: %s" %(losses_epoch))
    print("Validation Loss: %s" %(val_loss_epoch))
    print("*******************************************")
