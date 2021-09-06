#!/bin/bash -l

# for D in 'FASHION' 'CIFAR10'
# for M in 'mandera_detect' 'multi_krum' 'bulyan'
# for R in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
# for P in 5 10 15 20 25 30

for D in 'FASHION'
do
    for M in 'mandera_detect'
    do
        for R in 0 1 2 3 4
        do 
            for P in 5
            do 
                sbatch --export D=$D,R=$R,P=$P,M=$M --job-name=sign.$P.$M.$R.$D sign_flipping.jobscript
            done
        done
    done
done