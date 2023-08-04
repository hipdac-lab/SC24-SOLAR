#!/bin/bash
echo 'Running Baseline Training'
#mpirun -np ${NPROCS} python3 train_baseline.py --data_path ../../data/cosmoUniverse_21688988_128GB_v2 --batch_size 16 --nepochs 3 --nsamples $((MY_SIZE*51))
mpirun -np 2 python3 train_baseline.py --data_path ../../data/cosmoUniverse_21688988_128GB_v2 --batch_size 16 --nepochs 3 --nsamples 32
wait
echo 'Running SOLAR shuffle'
python3 ../utils/solar_shuffle.py --size 4 --gpu_pernode 1 --batch_size 16 --nnodes 4 --epochs 3 --cache_size 16 --ntrain 32 --save_path ./lists/
wait
echo 'Running SOLAR Training'
mpirun -np 2 python3 train_solar.py --data_path ../../data/cosmoUniverse_21688988_128GB_v2 --batch_size 16 --nepochs 3 --nsamples 32 --buffer_size 16 --lists ./lists/