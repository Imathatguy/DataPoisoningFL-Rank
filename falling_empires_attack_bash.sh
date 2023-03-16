# !/bin/bash -l

# for D in 'FASHION' 'CIFAR10' 'MNIST'
# for M in 'mandera_detect' 'multi_krum' 'bulyan' 'median' 'tr_mean' 'fltrust'
# for R in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
# for P in 5 10 15 20 25 30
# for E in 10 -10 0.1 -0.1

export CUDA_VISIBLE_DEVICES=0

for D in 'QMNIST'
do
    for M in 'mandera_detect'
    do
        for R in 0 1 2 3 4 5 6 7 8 9
        do 
            for P in 5 10 15 20 25 30
            do 
                for E in -0.1
                do 
                    python falling_empires_attack.py --dataset $D --p_workers $P --def_method $M --rep_n $R --epsilon $E
                done
            done
        done
    done
done
