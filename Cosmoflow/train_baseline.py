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
parser.add_argument('--nsamples', type=int, default=3,
                    help='Number of Samples')
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
run_time = 1
nepochs=args.nepochs
# load data
filelist = []
total_train_size=args.nsamples
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
if run_time == 1:
    for epoch in range(nepochs):
        idx_arr = np.arange(nsamples)
        np.random.shuffle(idx_arr)
        shuffle_list[epoch] = idx_arr
    shuffle_list = MPI.COMM_WORLD.bcast(shuffle_list, root=0)
else:
    print('Shuffle list loading not yet supported')

val_shuffle_list=np.zeros([nepochs,total_val_size])
if run_time == 1:
    for epoch in range(nepochs):
        idx_arr = np.arange(total_val_size)
        np.random.shuffle(idx_arr)
        val_shuffle_list[epoch] = idx_arr
    val_shuffle_list = MPI.COMM_WORLD.bcast(val_shuffle_list, root=0)
else:
    print('Shuffle list loading not yet supported')
transform = CosmoFlowTransform(apply_log)
train_data2=CosDataset(indices=shuffle_list, data_dir=DATA_PATH, dataset_size=total_train_size,transform=transform)

kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'gpu' else {}
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_data2, num_replicas=size, shuffle=False, rank=rank)
train_loader = torch.utils.data.DataLoader(
    train_data2, batch_size=BATCH_SIZE, sampler=train_sampler,  **kwargs)

val_data2=CosDataset(indices=val_shuffle_list, data_dir=VAL_PATH, dataset_size=total_val_size,transform=transform)
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
    print("************Baseline***************")
    print("Number of Processes used: "+str(size))
    print("Number of Epochs: "+str(nepochs))
    print("Batch Size: "+str(BATCH_SIZE))
    print("DataLoading time: %s s" %(io_time_epoch))
    print("Epoch time: %s s" %(epoch_time))
    print("Training Loss: %s" %(losses_epoch))
    print("Validation Loss: %s" %(val_loss_epoch))
    print("*******************************************")