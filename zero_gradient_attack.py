from federated_learning.utils import default_no_change
from federated_learning.utils import zero_gradient
from federated_learning.worker_selection import RandomSelectionStrategy
from server import run_exp
import argparse

if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='Run poisoning attack')
    # parser.add_argument('--pworkers', type=int,
    #                     help='number of poisoned workers')
    # args = parser.parse_args()
    # n_poison = args.pworkers

    # NUM_POISONED_WORKERS = n_poison

    # for NUM_POISONED_WORKERS in [5, 10, 15, 20, 25, 30]:
    # for NUM_POISONED_WORKERS in [5, 10, 15]:
    for NUM_POISONED_WORKERS in [20, 25, 30]:

        # Using 10000 for baseline
        # the 2-3 digits (X20XX) specifying num of poisioning workers
        # the 4-5 digits (XXX00) specifying run number
        # 20000 for label flipping
        # 30000 for gaussian noise
        START_EXP_IDX = 40000
        NUM_OFFSET = 0
        NUM_EXP = 20

        START_EXP_IDX = START_EXP_IDX + (NUM_POISONED_WORKERS * 100)

        REPLACEMENT_METHOD = default_no_change
        NOISE_METHOD = zero_gradient
        KWARGS = {
            "NUM_WORKERS_PER_ROUND" : 100,
        }

        for experiment_id in range(START_EXP_IDX + NUM_OFFSET, START_EXP_IDX + NUM_EXP + NUM_OFFSET):
            run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, RandomSelectionStrategy(), experiment_id, noise_method=NOISE_METHOD)
