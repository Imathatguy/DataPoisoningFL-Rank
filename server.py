from federated_learning.utils.noise_injection_methods import gaussian_attack
from federated_learning.utils.noise_injection_methods import zero_gradient
from federated_learning.utils.noise_injection_methods import shifted_mean
from federated_learning.utils.defense_methods import mandera_detect
from federated_learning.utils.defense_methods import multi_krum
from federated_learning.utils.defense_methods import bulyan
from federated_learning.utils.defense_methods import median
from federated_learning.utils.defense_methods import tr_mean
from federated_learning.utils.defense_methods import fltrust
from numpy.lib.npyio import save
from loguru import logger
from federated_learning.arguments import Arguments
from federated_learning.utils import generate_data_loaders_from_distributed_dataset
from federated_learning.datasets.data_distribution import distribute_batches_equally
from federated_learning.utils import average_nn_parameters
from federated_learning.utils import convert_distributed_data_into_numpy
from federated_learning.utils import poison_data
from federated_learning.utils import identify_random_elements
from federated_learning.utils import save_results
from federated_learning.utils import load_train_data_loader
from federated_learning.utils import load_test_data_loader
from federated_learning.utils import generate_experiment_ids
from federated_learning.utils import convert_results_to_csv
from client import Client
import numpy as np
import os
import copy
import time
import pickle
import torch
import pandas as pd

def train_subset_of_clients(epoch, args, clients, poisoned_workers, noise_method, def_method):
    """
    Train a subset of clients per round.

    :param epoch: epoch
    :type epoch: int
    :param args: arguments
    :type args: Arguments
    :param clients: clients
    :type clients: list(Client)
    :param poisoned_workers: indices of poisoned workers
    :type poisoned_workers: list(int)
    :param noise_method: the specified method of applying noise to the parameters
    :type poisoned_workers: class federated_learning.utils.noise_injection_methods
    :param poisoned_workers: indices of poisoned workers
    :type poisoned_workers: list(int)
    """
    kwargs = args.get_round_worker_selection_strategy_kwargs()
    kwargs["current_epoch_number"] = epoch

    random_workers = args.get_round_worker_selection_strategy().select_round_workers(
        list(range(args.get_num_workers())),
        poisoned_workers,
        kwargs)

    existing_parameters = [copy.deepcopy(clients[client_idx].get_nn_parameters()) for client_idx in random_workers]

    for client_idx in random_workers:
        # skip update of poisoned models as we overwrite gradients:
        if noise_method in [gaussian_attack, zero_gradient, shifted_mean]:
            if client_idx in poisoned_workers:
                args.get_logger().info("Skip  Training #{} on client #{}", str(epoch), str(clients[client_idx].get_client_index()))
                continue

        args.get_logger().info("Training epoch #{} on client #{}", str(epoch), str(clients[client_idx].get_client_index()))
        clients[client_idx].train(epoch)

    args.get_logger().info("Averaging client parameters")
    parameters = [clients[client_idx].get_nn_parameters() for client_idx in random_workers]

    # modify gradients of malicious nodes with noise
    if noise_method is not None:
        parameters, gradients = noise_method(existing_parameters, parameters, random_workers, poisoned_workers)
    
    # detection defense occurs just before parameter aggregation
    if def_method in [mandera_detect]:
        bad_indexes = mandera_detect(gradients)
        good_parameters = [param for n, param in enumerate(parameters) if n not in bad_indexes]
        new_nn_params = average_nn_parameters(good_parameters)
    elif def_method in [multi_krum]:
        _, good_indexes = multi_krum(gradients, len(poisoned_workers), multi_k=False)
        good_parameters = [param for n, param in enumerate(parameters) if n in good_indexes]
        new_nn_params = average_nn_parameters(good_parameters)
    elif def_method in [bulyan]:
        _, good_indexes = bulyan(gradients, len(poisoned_workers))
        good_parameters = [param for n, param in enumerate(parameters) if n in good_indexes]
        new_nn_params = average_nn_parameters(good_parameters)
    elif def_method in [median]:
        new_nn_params = median(parameters)
    elif def_method in [tr_mean]:
        new_nn_params = tr_mean(parameters, len(poisoned_workers))
    elif def_method in [fltrust]:
        new_nn_grads = fltrust(gradients)
        new_nn_params = add_grads(existing_parameters[-1], unflatten_grads(new_nn_grads, existing_parameters[-1]))
    else:
        new_nn_params = average_nn_parameters(parameters)

    # Order client gradients
    client_grads = {}
    for n, client_grad in enumerate(gradients):
        client_idx = random_workers[n]
        client_grads[client_idx] = client_grad

    # update client parameters with new global parameters
    for client in clients:
        args.get_logger().info("Updating parameters on client #{}", str(client.get_client_index()))
        client.update_nn_parameters(new_nn_params)

    # import pickle
    # pickle.dump(gradients, open("sf_debug_grads.pickle", "wb"))

    torch.cuda.empty_cache()

    return clients[0].test(), random_workers, client_grads

def create_clients(args, train_data_loaders, test_data_loader):
    """
    Create a set of clients.
    """
    clients = []
    for idx in range(args.get_num_workers()):
        clients.append(Client(args, idx, train_data_loaders[idx], test_data_loader))

    return clients

def run_machine_learning(clients, args, poisoned_workers, noise_method, def_method):
    """
    Complete machine learning over a series of clients.
    """
    epoch_test_set_results = []
    worker_selection = []
    epoch_grads = []
    for epoch in range(1, args.get_num_epochs() + 1):
        results, workers_selected, client_grads = train_subset_of_clients(epoch, args, clients, poisoned_workers, noise_method, def_method)

        epoch_test_set_results.append(results)
        worker_selection.append(workers_selected)
        epoch_grads.append(client_grads)

    return convert_results_to_csv(epoch_test_set_results), worker_selection, epoch_grads

def run_exp(replacement_method, num_poisoned_workers, KWARGS, client_selection_strategy, idx, noise_method=None, def_method=None, dataset="FASHION"):
    log_files, results_files, models_folders, worker_selections_files = generate_experiment_ids(idx, 1)

    # Initialize logger
    handler = logger.add(log_files[0], enqueue=True)

    args = Arguments(logger)
    args.set_model_save_path(models_folders[0])
    args.set_num_poisoned_workers(num_poisoned_workers)
    args.set_round_worker_selection_strategy_kwargs(KWARGS)
    args.set_client_selection_strategy(client_selection_strategy)
    args.set_dataset_net_and_loader(dataset)
    args.log()

    train_data_loader = load_train_data_loader(logger, args)
    test_data_loader = load_test_data_loader(logger, args)

    if def_method == fltrust:
        # Guarantee that the last worker is not poisoned, represents trusted server model
        args.set_num_workers(args.get_num_workers() + 1)
        poisoned_workers = identify_random_elements(args.get_num_workers() - 1, args.get_num_poisoned_workers())
    else:
        poisoned_workers = identify_random_elements(args.get_num_workers(), args.get_num_poisoned_workers())
    
    # Distribute batches equal volume IID
    distributed_train_dataset = distribute_batches_equally(train_data_loader, args.get_num_workers())
    distributed_train_dataset = convert_distributed_data_into_numpy(distributed_train_dataset)

    distributed_train_dataset = poison_data(logger, distributed_train_dataset, args.get_num_workers(), poisoned_workers, replacement_method)

    train_data_loaders = generate_data_loaders_from_distributed_dataset(distributed_train_dataset, args.get_batch_size())

    clients = create_clients(args, train_data_loaders, test_data_loader)

    # start timer
    start_time = time.perf_counter()
    # run ML training/attack/defense
    results, worker_selection, epoch_grads = run_machine_learning(clients, args, poisoned_workers, noise_method, def_method)
    # end timer
    end_time = time.perf_counter()
    
    exp_id = worker_selections_files[0].split("_")[0]
    # path = "/F/mandera_results/results_def/{}".format(exp_id)
    path = os.path.join("F", os.sep, "mandera_results", "results_def", exp_id)

    try:
        print("{}".format(path))
        if not os.path.exists("{}".format(path)):
            os.makedirs("{}".format(path))
    except OSError as error:
        print(error)   

    # save perdiction performance results
    save_results(results, os.path.join(path, results_files[0]))

    # save timing results
    np.savetxt(os.path.join(path, "{}_timing.csv".format(results_files[0][:-4])),
                [start_time, end_time], delimiter=",", fmt="%s")

    # only save gradients if not running full defense    
    if def_method is None:
        save_results(worker_selection, os.path.join(path, worker_selections_files[0]))

        flat_epochs = flatten_params(epoch_grads)
        for n, flat in enumerate(flat_epochs):
            # save as csv format
            # np.savetxt("./{}/{}_flatgrads.csv".format(path, n), flat, delimiter=',')
            # save as hdf5 format
            df = pd.DataFrame(flat, columns=None, index=None)
            # erase existing hdf5 file
            if n == 0:
                mode = 'w'
            # append subsequent epochs to existing file
            else:
                mode = 'a'    
            df.to_hdf(os.path.join(path, "flatgrads.hdf5"), key="epoch_{}".format(n), mode=mode, index=False)
        # save list of poisoned workers
        np.savetxt(os.path.join(path, "{}_poisoned.csv".format(worker_selections_files[0][:-4])),
                poisoned_workers, delimiter=",", fmt="%s")

    logger.remove(handler)



def flatten_params(epoch_holder):

    param_order = epoch_holder[0][list(epoch_holder[0].keys())[0]].keys()

    flat_epochs = []

    for n_epoch, epoch in enumerate(epoch_holder):
        arr = []
        for n_user in range(len(epoch)):
            user_arr = []
            grads = epoch[n_user]
            for param in param_order:
                try:
                    user_arr.extend(grads[param].cpu().numpy().flatten().tolist())
                except:
                    user_arr.extend([grads[param].cpu().numpy().flatten().tolist()])
            arr.append(user_arr)

        arr_2d = np.array(arr)
        flat_epochs.append(arr_2d)
    return flat_epochs
        

def unflatten_grads(flat_param, param_example):

    new_params = copy.deepcopy(param_example)
    param_order = new_params.keys()

    i = 0
    for param in param_order:
        n_flat = len(new_params[param].flatten())
        new_params[param] = torch.tensor(flat_param[i:i+n_flat].reshape(new_params[param].shape)).to(device='cuda')
        i = i + n_flat
    return new_params
        

def flatten_grads(gradients):

    param_order = gradients[0].keys()

    flat_epochs = []

    for n_user in range(len(gradients)):
        user_arr = []
        grads = gradients[n_user]
        for param in param_order:
            try:
                user_arr.extend(grads[param].cpu().numpy().flatten().tolist())
            except:
                user_arr.extend([grads[param].cpu().numpy().flatten().tolist()])
        flat_epochs.append(user_arr)

    flat_epochs = np.array(flat_epochs)

    return flat_epochs


def add_grads(full_param, grads):
    assert full_param.keys() == grads.keys()
    new_params = copy.deepcopy(full_param)
    param_order = full_param.keys()

    for param in param_order:
        new_params[param] = full_param[param] + grads[param]

    return new_params


# if __name__ == "__main__":

#     import pickle
#     grads_1 = pickle.load(open("sf_debug_grads.pickle", "rb"))

#     # flat_grads = flatten_params([{n: grad for n, grad in enumerate(grads_1)}])
#     flat_grads = flatten_grads(grads_1)

#     unflat_grads = unflatten_grads(flat_grads[0], grads_1[0])

#     assert unflat_grads.keys() == grads_1[0].keys()

#     for name in unflat_grads.keys():
#         assert torch.all(unflat_grads[name].eq(grads_1[0][name]))


#     double_grad = add_grads(unflat_grads, unflat_grads)

#     for name in double_grad.keys():
#         assert torch.all(double_grad[name].eq(grads_1[0][name]*2))