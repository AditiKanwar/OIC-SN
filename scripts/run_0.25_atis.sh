#!/bin/bash
#SBATCH -N 1 #1 Node
#SBATCH --ntasks-per-node=8
#SBATCH --time=3-00:00:00
#SBATCH --job-name=run_models_gpu_1_16GB
#SBATCH --error=%J.err
#SBATCH --output=%J.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

for s in 0 1 2 3 4 5 6 7 8 9
do 
    python SLMM.py \
        --dataset atis \
        --known_cls_ratio 0.25 \
        --labeled_ratio 1.0 \
        --seed $s \
        --freeze_bert_parameters \
        --beta 0.3 \
        --gamma 0.6 \
        --lr_boundary 2e-5
done
