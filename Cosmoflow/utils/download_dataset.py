from mpi4py import MPI
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(
    description='Download cosmoUniverse_21688988 from NERSC')
parser.add_argument('--save_dir', type=str,
                    help='Directory to save CosmoFlow dataset')
parser.add_argument('--data_size', type=str,
                    help='Download dataset size in GB')
args = parser.parse_args()

def download_file(url,args):
    # Use wget or any appropriate method to download the file
    # For example:
    import os
    os.system(f'wget {url} -P {args.save_dir}/cosmoUniverse_21688988/')

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    with open('filenames.txt', 'r') as f:
        urls = f.read().splitlines()
    
    urls=urls[:int(args.data_size)]
    url_each_rank=urls[rank::size]
    if rank==0:
        print('total length:{}'.format(len(urls)))
    print('rank:{},length:{}'.format(rank,len(url_each_rank)))
    for u in tqdm(url_each_rank,total=len(url_each_rank),disable=not rank==0):
        download_file('https://portal.nersc.gov/project/m3363/cosmoUniverse_2019_05_4parE/21688988/{}'.format(u),args)

if __name__ == '__main__':
    main()