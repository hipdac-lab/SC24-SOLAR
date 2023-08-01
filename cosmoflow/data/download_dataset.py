#This code to download cosmoUniverse_21688988 dataset, which is 1TB in total, each .h5 file is 1GB
from mpi4py import MPI
from tqdm import tqdm
import os
#Arguments
import argparse
parser = argparse.ArgumentParser(description='CosmoUniverse Dataset Download')
parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the dataset')
args = parser.parse_args()

def download_file(url,args):
    os.system(f'wget {url} -P {args.save_dir}')

def main():
    #Get MPI info
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #Check save dir
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    #List of Filenames
    with open('filenames.txt', 'r') as f:
        urls = f.read().splitlines()
    
    url_each_rank=urls[rank::size]
    if rank==0:
        print('total length:{}'.format(len(urls)))
    print('rank:{},length:{}'.format(rank,len(url_each_rank)))
    for u in tqdm(url_each_rank,total=len(url_each_rank),disable=not rank==0):
        download_file('https://portal.nersc.gov/project/m3363/cosmoUniverse_2019_05_4parE/21688988/{}'.format(u))

if __name__ == '__main__':
    main()