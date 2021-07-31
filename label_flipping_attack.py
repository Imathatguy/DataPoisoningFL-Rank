from federated_learning import utils
from federated_learning.utils import replace_0_with_2
from federated_learning.utils import replace_5_with_3
from federated_learning.utils import replace_1_with_9
from federated_learning.utils import replace_4_with_6
from federated_learning.utils import replace_1_with_3
from federated_learning.utils import replace_6_with_0
from federated_learning.utils import no_noise
from federated_learning.worker_selection import RandomSelectionStrategy
from server import run_exp

# if __name__ == '__main__':
#     # Using 1000 for baseline
#     # 2000 for default expeirment with the middle 2 digits specifying num of poisioing workers
#     START_EXP_IDX = 2000
#     NUM_EXP = 1

#     NUM_POISONED_WORKERS = 30

#     START_EXP_IDX = START_EXP_IDX + (NUM_POISONED_WORKERS * 10)

#     REPLACEMENT_METHOD = replace_1_with_9
#     KWARGS = {
#         "NUM_WORKERS_PER_ROUND" : 100,
#     }

#     for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):
#         run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, RandomSelectionStrategy(), experiment_id)


if __name__ == '__main__':

    for NUM_POISONED_WORKERS in [5]:#, 10, 15, 20, 25, 30]:

        # Using 10000 for baseline
        # the 2-3 digits (X20XX) specifying num of poisioning workers
        # the 4-5 digits (XXX00) specifying run number
        # 20000 for label flipping
        # 30000 for gaussian noise
        START_EXP_IDX = 20000
        NUM_OFFSET = 0
        NUM_EXP = 1

        START_EXP_IDX = START_EXP_IDX + (NUM_POISONED_WORKERS * 100)

        REPLACEMENT_METHOD = replace_1_with_9
        NOISE_METHOD = no_noise
        KWARGS = {
            "NUM_WORKERS_PER_ROUND" : 100,
        }

        for experiment_id in range(START_EXP_IDX + NUM_OFFSET, START_EXP_IDX + NUM_EXP + NUM_OFFSET):
            run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, RandomSelectionStrategy(), experiment_id, noise_method=NOISE_METHOD)
