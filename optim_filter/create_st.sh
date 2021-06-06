#!/bin/bash

#OAR -l /nodes=1,walltime=30:00:00
module load conda/2020.11-python3.8
eval "$(conda shell.bash hook)"
conda activate myenv

python create_st.py $1
