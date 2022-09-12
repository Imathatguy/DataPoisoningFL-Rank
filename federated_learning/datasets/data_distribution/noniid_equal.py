import torch
from math import floor
import numpy as np

def distribute_batches_noniid(train_data_loader, num_workers, batch_size):
    """
    Gives each worker the same number of batches of training data.

    Stratifies the data by writer to ensure each writers is only in one worker

    :param train_data_loader: Training data loader
    :type train_data_loader: torch.utils.data.DataLoader
    :param num_workers: number of workers
    :type num_workers: int
    """

    # import pickle
    # pickle.dump(train_data_loader, open('debug_loader.pickle', 'wb'))

    distributed_dataset = [[] for _ in range(num_workers)]

    # we build our own dataloaders here
    assert len(train_data_loader) == 1
    batch_idx, (data, target, stratify) = next(enumerate(train_data_loader))
    
    # handle stratify outside of tensor form
    stratify = stratify.numpy().astype(int)

    # build an ordered list of owners from metadata
    ordered_ids = []
    id_set = set([])
    for _id in stratify:
        if _id not in id_set:
            id_set.add(_id)
            ordered_ids.append(_id)
    
    id_per_worker = floor(len(ordered_ids) / num_workers)
    

    validate_holder = []
    for worker_idx in range(num_workers):
        selected_ids = ordered_ids[(worker_idx*id_per_worker):(worker_idx+1)*id_per_worker]

        _many_cond = [stratify == _id for _id in selected_ids]
        _ind_cond = np.array(_many_cond).any(axis=0)
        _ind = np.where(_ind_cond)

        # data[_ind]
        # target[_ind]

        dataset = torch.utils.data.TensorDataset(data[_ind], target[_ind])
        multiple_loaders = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        for _, (_data, _target) in enumerate(multiple_loaders):
            distributed_dataset[worker_idx].append((_data, _target))

        validate_holder.extend(selected_ids)

    # check each id is unique to worker
    assert len(set(validate_holder)) == len(validate_holder)
    # check number of ids is less than total
    assert len(validate_holder) <= len(ordered_ids)

    return distributed_dataset


if __name__ == "__main__":
    import pickle
    import numpy as np
    from torch.utils.data import TensorDataset
    from torch.utils.data import DataLoader

    num_workers = 100
    batch_size = 10
    train_data_loader = pickle.load(open('debug_loader.pickle', 'rb'))

    a = distribute_batches_noniid(train_data_loader, num_workers, batch_size)

    # distributed_dataset = [[] for i in range(num_workers)]

    # # we build our own dataloaders here
    # assert len(train_data_loader) == 1
    # batch_idx, (data, target, stratify) = next(enumerate(train_data_loader))
    
    # # handle stratify outside of tensor form
    # stratify = stratify.numpy().astype(int)

    # # build an ordered list of owners from metadata
    # ordered_ids = []
    # id_set = set([])
    # for _id in stratify:
    #     if _id not in id_set:
    #         id_set.add(_id)
    #         ordered_ids.append(_id)
    
    # id_per_worker = floor(len(ordered_ids) / num_workers)
    

    # validate_holder = []
    # for worker_idx in range(num_workers):
    #     selected_ids = ordered_ids[(worker_idx*id_per_worker):(worker_idx+1)*id_per_worker]

    #     _many_cond = [stratify == _id for _id in selected_ids]
    #     _ind_cond = np.array(_many_cond).any(axis=0)
    #     _ind = np.where(_ind_cond)

    #     # data[_ind]
    #     # target[_ind]

    #     dataset = TensorDataset(data[_ind], target[_ind])
    #     multiple_loaders = DataLoader(dataset, batch_size=batch_size)

    #     for batch_idx, (_data, _target) in enumerate(multiple_loaders):
    #         distributed_dataset[worker_idx].append((_data, _target))

    #     validate_holder.extend(selected_ids)

    # # check each id is unique to worker
    # assert len(set(validate_holder)) == len(validate_holder)
    # # check number of ids is less than total
    # assert len(validate_holder) <= len(ordered_ids)