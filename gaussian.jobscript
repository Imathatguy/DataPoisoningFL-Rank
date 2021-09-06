#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8gb
#SBATCH --time=2:00:00

module load python/3.9.4
module load pytorch/1.8.1-py39-cuda112
module load torchvision/0.9.1-py39

#echo $D $P $M $R
sleep 5s

python gaussian_attack.py --dataset $D --p_workers $P --def_method $M --rep_n $R