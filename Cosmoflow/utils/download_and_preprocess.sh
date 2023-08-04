#!/bin/bash

mpirun -np 16 python3 download_dataset.py --save_dir ../../../data/ --data_size ${MY_SIZE}
wait
echo 'Pre-processing ...'
python3 preprocess_dataset.py ../../../data/cosmoUniverse_21688988 ../../../data/cosmoUniverse_21688988_128GB_v1