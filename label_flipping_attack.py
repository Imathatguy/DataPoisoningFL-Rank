from federated_learning.utils import replace_0_with_2
from federated_learning.utils import replace_5_with_3
from federated_learning.utils import replace_1_with_9
from federated_learning.utils import replace_4_with_6
from federated_learning.utils import replace_1_with_3
from federated_learning.utils import replace_6_with_0
from federated_learning.utils import no_noise
from federated_learning.utils.defense_methods import mandera_detect
from federated_learning.utils.defense_methods import multi_krum
from federated_learning.utils.defense_methods import bulyan
from federated_learning.utils.defense_methods import median
from federated_learning.utils.defense_methods import tr_mean
from federated_learning.utils.defense_methods import fltrust
from federated_learning.worker_selection import RandomSelectionStrategy
from server import run_exp
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run poisoning attack')
    parser.add_argument('--p_workers', type=int,
                        help='number of poisoned workers')
    parser.add_argument('--rep_n', type=int,
                        help='repetition number')
    parser.add_argument('--dataset', type=str,
                        help='which dataset?')
    parser.add_argument('--def_method', type=str,
                        help='which defense?')
    args = parser.parse_args()

    print(args)

    NUM_POISONED_WORKERS = args.p_workers
    NUM_OFFSET = args.rep_n

    DATASET = args.dataset
    
    if args.def_method == "None":
        START_EXP_IDX = 00000
        DEF_METHOD = None
    elif args.def_method == "mandera_detect":
        START_EXP_IDX = 100000
        DEF_METHOD = mandera_detect
    elif args.def_method == "multi_krum":
        START_EXP_IDX = 200000
        DEF_METHOD = multi_krum
    elif args.def_method == "bulyan":
        START_EXP_IDX = 300000
        DEF_METHOD = bulyan
    elif args.def_method == "median":
        START_EXP_IDX = 400000
        DEF_METHOD = median
    elif args.def_method == "tr_mean":
        START_EXP_IDX = 500000
        DEF_METHOD = tr_mean
    elif args.def_method == "fltrust":
        START_EXP_IDX = 600000
        DEF_METHOD = fltrust
    else:
        assert args.def_method in ["None", "mandera_detect", "multi_krum", "bulyan", "median", "tr_mean", "fltrust"]

    # Using 10000 for baseline
    # the 2-3 digits (X20XX) specifying num of poisioning workers
    # the 4-5 digits (XXX00) specifying run number
    # FASHION
    # 20000 for label flipping
    # 30000 for gaussian noise
    # 40000 for zero grad
    # 50000 for sign flip
    # CIFAR
    # 60000 for label flipping
    # 70000 for gaussian noise
    # 80000 for zero grad
    # 90000 for sign flip
    # Add 100000 for full defense method.
    # START_EXP_IDX = 390000

    # fixed 1 run per call
    NUM_EXP = 1
    KWARGS = {
        "NUM_WORKERS_PER_ROUND" : 100,
    }

    REPLACEMENT_METHOD = replace_1_with_9
    NOISE_METHOD = no_noise

    if DEF_METHOD == None:
        START_EXP_IDX = 00000
    elif DEF_METHOD == mandera_detect:
        START_EXP_IDX = 100000
    elif DEF_METHOD == multi_krum:
        START_EXP_IDX = 200000
    elif DEF_METHOD == bulyan:
        START_EXP_IDX = 300000
    elif DEF_METHOD == median:
        START_EXP_IDX = 400000
    elif DEF_METHOD == tr_mean:
        START_EXP_IDX = 500000
    elif DEF_METHOD == fltrust:
        START_EXP_IDX = 600000
        # Add extra worker as trusted server model
        KWARGS["NUM_WORKERS_PER_ROUND"] = KWARGS["NUM_WORKERS_PER_ROUND"] + 1
    else:
        assert DEF_METHOD in [None, mandera_detect, multi_krum, bulyan, median, tr_mean, fltrust]

    if DATASET == "FASHION":
        START_EXP_IDX = START_EXP_IDX + 20000
    elif DATASET == "CIFAR10":
        START_EXP_IDX = START_EXP_IDX + 60000
    else:
        assert DATASET in ["FASHION", "CIFAR10"]

    # for NUM_POISONED_WORKERS in [5, 10, 15, 20, 25, 30]:
    # for NUM_POISONED_WORKERS in [5, 10, 15]:
    # for NUM_POISONED_WORKERS in [20, 25, 30]:
    # for NUM_POISONED_WORKERS in [10,20,30]:

    START_EXP_IDX = START_EXP_IDX + (NUM_POISONED_WORKERS * 100)

    for experiment_id in range(START_EXP_IDX + NUM_OFFSET, START_EXP_IDX + NUM_EXP + NUM_OFFSET):
        run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, RandomSelectionStrategy(), experiment_id,
                noise_method=NOISE_METHOD, def_method=DEF_METHOD, dataset=DATASET)
