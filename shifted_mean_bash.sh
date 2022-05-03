#!/bin/bash -l

# for D in 'FASHION' 'CIFAR10' 'MNIST'
# for M in 'mandera_detect' 'multi_krum' 'bulyan' 'median' 'tr_mean' 'fltrust'
# for R in 0 1 2 3 4 5 6 7 8 9
# for P in 5 10 15 20 25 30

for D in 'FASHION'
do
    for M in 'fltrust'
    do
        for R in 0
        do 
            for P in 5 10 15 20 25 30
            do 
                python shifted_mean_attack.py --dataset $D --p_workers $P --def_method $M --rep_n $R
            done
        done
    done
done
