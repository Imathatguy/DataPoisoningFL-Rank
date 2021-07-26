from federated_learning.utils import default_no_change
from federated_learning.utils import gaussian_attack
from federated_learning.worker_selection import RandomSelectionStrategy
from server import run_exp

if __name__ == '__main__':
    # Using 1000 for baseline
    # 2000 for default expeirment with the middle 2 digits specifying num of poisioing workers
    START_EXP_IDX = 3000
    NUM_EXP = 1

    NUM_POISONED_WORKERS = 30

    START_EXP_IDX = START_EXP_IDX + (NUM_POISONED_WORKERS * 10)

    REPLACEMENT_METHOD = default_no_change
    NOISE_METHOD = gaussian_attack
    KWARGS = {
        "NUM_WORKERS_PER_ROUND" : 100,
    }

    for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):
        run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, RandomSelectionStrategy(), experiment_id, noise_method=NOISE_METHOD)
