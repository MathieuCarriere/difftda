#!/bin/bash

#OAR -l /nodes=1/core=8,walltime=30:00:00
module load conda/2020.11-python3.8
eval "$(conda shell.bash hook)"
conda activate myenv

python optim_filters.py $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10}
