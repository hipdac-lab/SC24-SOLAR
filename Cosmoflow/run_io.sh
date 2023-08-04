#!/bin/bash
echo 'Running Baseline IO'
mpirun -np ${NPROCS} python3 io_baseline.py --data_path ../../data/cosmoUniverse_21688988_128GB_v1 --batch_size 16 --nepochs 3 --nsamples $((MY_SIZE*64))
wait
echo 'Running SOLAR shuffle'
python3 ../utils/solar_shuffle.py --size 4 --gpu_pernode 1 --batch_size 16 --nnodes 4 --epochs 3 --cache_size $((MY_SIZE*32)) --ntrain $((MY_SIZE*51)) --save_path ./lists/
wait
echo 'Running SOLAR IO'
mpirun -np ${NPROCS} python3 io_solar.py --data_path ../../data/cosmoUniverse_21688988_128GB_v1 --batch_size 16 --nepochs 3 --nsamples $((MY_SIZE*64)) --buffer_size $((MY_SIZE*32)) --lists ./lists/
