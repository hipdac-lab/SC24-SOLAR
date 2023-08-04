"""Utilities for training."""

import os
import os.path
import statistics
import time

import torch


class Logger:
    """Simple logger that saves to a file and stdout."""

    def __init__(self, out_file, is_primary):
        """Save logging info to out_file."""
        self.is_primary = is_primary
        if is_primary:
            if os.path.exists(out_file):
                raise ValueError(f'Log file {out_file} already exists')
            self.log_file = open(out_file, 'w')
        else:
            self.log_file = None

    def log(self, message):
        """Log message."""
        if self.is_primary:
            # Only the primary writes the log.
            self.log_file.write(message + '\n')
            self.log_file.flush()
            print(message, flush=True)

    def close(self):
        """Close the log."""
        if self.is_primary:
            self.log_file.close()


class AverageTracker:
    """Keeps track of the average of a value."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear the tracker."""
        self.vals = []

    def update(self, val, n=1):
        """Add n copies of val to the tracker."""
        if n == 1:
            self.vals.append(val)
        else:
            self.vals.extend([val]*n)

    def mean(self):
        """Return the mean."""
        if not self.vals:
            return float('nan')
        return statistics.mean(self.vals)

    def latest(self):
        """Return the latest value."""
        if not self.vals:
            return float('nan')
        return self.vals[-1]

    def save(self, filename):
        """Save data to a file."""
        with open(filename, 'a') as fp:
            fp.write(','.join([str(v) for v in self.vals]) + '\n')


@torch.jit.script
def _mean_impl(data, counts):
    """Internal scripted mean implementation."""
    return data.sum() / counts.sum()


class AverageTrackerDevice:
    """Keep track of the average of a value.

    This is optimized for storing the results on device.

    """

    def __init__(self, n, device, allreduce=True):
        """Track n total values on device.

        allreduce: Perform an allreduce over scaled values before
        computing mean.

        """
        self.n = n
        self.device = device
        self.allreduce = allreduce
        self.last_allreduce_count = None
        self.saved_mean = None
        self.reset()

    def reset(self):
        """Clear the tracker."""
        self.data = torch.empty(self.n, device=self.device)
        self.counts = torch.empty(self.n, device='cpu', pin_memory=True)
        self.cur_count = 0
        # For caching results.
        self.last_allreduce_count = None
        self.saved_mean = None

    @torch.no_grad()
    def update(self, val, count=1.0):
        """Add val and associated count to tracker."""
        if self.cur_count == self.n:
            raise RuntimeError('Updating average tracker past end')
        self.data[self.cur_count] = val
        self.counts[self.cur_count] = count
        self.cur_count += 1

    @torch.no_grad()
    def mean(self):
        """Return the mean.

        This will be a device tensor.

        """
        if self.cur_count == 0:
            return float('nan')
        if self.cur_count == self.last_allreduce_count:
            return self.saved_mean
        valid_data = self.data.narrow(0, 0, self.cur_count)
        valid_counts = self.counts.narrow(0, 0, self.cur_count).to(self.device)
        scaled_vals = valid_data * valid_counts
        if self.allreduce:
            scaled_vals = allreduce_tensor(scaled_vals)
        mean = _mean_impl(scaled_vals, valid_counts).item()
        self.last_allreduce_count = self.cur_count
        self.saved_mean = mean
        return mean


def get_cosmoflow_lr_schedule(optimizer, base_lr, target_lr, warmup_epochs,
                              decay_epochs, decay_factors):
    """Return a LambdaLR learning rate schedule for training.

    The optimizer's learning rate should be set to target_lr.

    optimizer: Optimizer to apply the schedule to.
    base_lr: Learning rate to begin training with (before warmup).
    target_lr: Learning rate to reach after linear warmup.
    warmup_epochs: Perform linear warmup for this many epochs.
    decay_epochs: List of epochs at which to decay the learning rate.
    decay_factors: Factors to decay the learning rate by.

    """
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


def get_num_gpus():
    """Number of GPUs on this node."""
    return torch.cuda.device_count()


def get_local_rank(required=False):
    """Get local rank from environment."""
    if 'MV2_COMM_WORLD_LOCAL_RANK' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_LOCAL_RANK'])
    if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    if 'SLURM_LOCALID' in os.environ:
        return int(os.environ['SLURM_LOCALID'])
        # return int(os.environ['LOCAL_RANK'])

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
        # return int(os.environ['LOCAL_RANK'])

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


def find_free_port():
    import socket
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.


def initialize_dist(init_file, rendezvous='file'):
    """Initialize PyTorch distributed backend."""
    torch.cuda.init()
    torch.cuda.set_device(get_local_rank())
    
    init_file = os.path.abspath(init_file)

    if rendezvous == 'env':
        dist_url = 'env://'
        node_id = get_world_rank()
        num_nodes = get_world_size()
        print(f'world_size: {num_nodes}, rank: {node_id}, local_rank: {get_local_rank()}')
        with open(init_file, "w") as f:
            f.write(dist_url)
        torch.distributed.init_process_group('nccl', init_method=dist_url, rank=node_id, world_size=num_nodes)
    elif rendezvous == 'tcp':
        dist_url = None
        node_id = get_world_rank()
        num_nodes = get_world_size()
        if node_id == 0:
            import socket
            ip = socket.gethostbyname(socket.gethostname())
            port = find_free_port()
            dist_url = "tcp://{}:{}".format(ip, port)
            with open(init_file, "w") as f:
                f.write(dist_url)
        else:
            while not os.path.exists(init_file):
                time.sleep(1)
            time.sleep(1)
            with open(init_file, "r") as f:
                dist_url = f.read()
        torch.distributed.init_process_group('nccl', init_method=dist_url,
                                             rank=node_id, world_size=num_nodes)
    elif rendezvous == 'file':
        torch.distributed.init_process_group(
            backend='nccl', init_method=f'file://{init_file}',
            rank=get_world_rank(), world_size=get_world_size())
    else:
        raise NotImplementedError(f'Unrecognized scheme "{rendezvous}"')

    torch.distributed.barrier()
    # Ensure the init file is removed.
    if get_world_rank() == 0 and os.path.exists(init_file):
        os.unlink(init_file)


def get_cuda_device():
    """Get this rank's CUDA device."""
    return torch.device(f'cuda:{get_local_rank()}')


def allreduce_tensor(t):
    """Allreduce and average tensor t."""
    rt = t.clone().detach()
    torch.distributed.all_reduce(rt)
    rt /= get_world_size()
    return rt
