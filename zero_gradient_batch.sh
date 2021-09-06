#!/bin/bash -l

# for D in 'FASHION' 'CIFAR10'
# for M in 'mandera_detect' 'multi_krum' 'bulyan'
# for R in 0 1 2 3 4 5 6 7 8 9
# for P in 5 10 15 20 25 30

for D in 'FASHION' 'CIFAR10'
do
    for M in 'median'
    do
        for R in 0 1 2 3 4 5 6 7 8 9
        do 
            for P in 5 10 15 20 25 30
            do 
                sbatch --export D=$D,R=$R,P=$P,M=$M --job-name=ZG.$P.$M.$R.$D zero_gradient.jobscript
            done
        done
    done
done
