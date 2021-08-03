from federated_learning.utils.noise_injection_methods import gaussian_attack
from federated_learning.utils.noise_injection_methods import zero_gradient
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
import pickle
import pandas as pd

def train_subset_of_clients(epoch, args, clients, poisoned_workers, noise_method):
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
        if noise_method in [gaussian_attack, zero_gradient]:
            if client_idx in poisoned_workers:
                args.get_logger().info("Skip  Training #{} on client #{}", str(epoch), str(clients[client_idx].get_client_index()))
                continue

        args.get_logger().info("Training epoch #{} on client #{}", str(epoch), str(clients[client_idx].get_client_index()))
        clients[client_idx].train(epoch)

    args.get_logger().info("Averaging client parameters")
    parameters = [clients[client_idx].get_nn_parameters() for client_idx in random_workers]

    # modify gradients of malicious nodes with noise
    if noise_method is not None:
        # import pickle
        # pickle.dump([existing_parameters, parameters, random_workers, poisoned_workers], open("debug.pickle","wb"))
        # assert False
        parameters, gradients = noise_method(existing_parameters, parameters, random_workers, poisoned_workers)
    
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

    return clients[0].test(), random_workers, client_grads

def create_clients(args, train_data_loaders, test_data_loader):
    """
    Create a set of clients.
    """
    clients = []
    for idx in range(args.get_num_workers()):
        clients.append(Client(args, idx, train_data_loaders[idx], test_data_loader))

    return clients

def run_machine_learning(clients, args, poisoned_workers, noise_method):
    """
    Complete machine learning over a series of clients.
    """
    epoch_test_set_results = []
    worker_selection = []
    epoch_grads = []
    for epoch in range(1, args.get_num_epochs() + 1):
        results, workers_selected, client_grads = train_subset_of_clients(epoch, args, clients, poisoned_workers, noise_method)

        epoch_test_set_results.append(results)
        worker_selection.append(workers_selected)
        epoch_grads.append(client_grads)

    return convert_results_to_csv(epoch_test_set_results), worker_selection, epoch_grads

def run_exp(replacement_method, num_poisoned_workers, KWARGS, client_selection_strategy, idx, noise_method=None):
    log_files, results_files, models_folders, worker_selections_files = generate_experiment_ids(idx, 1)

    # Initialize logger
    handler = logger.add(log_files[0], enqueue=True)

    args = Arguments(logger)
    args.set_model_save_path(models_folders[0])
    args.set_num_poisoned_workers(num_poisoned_workers)
    args.set_round_worker_selection_strategy_kwargs(KWARGS)
    args.set_client_selection_strategy(client_selection_strategy)
    args.log()

    train_data_loader = load_train_data_loader(logger, args)
    test_data_loader = load_test_data_loader(logger, args)

    # Distribute batches equal volume IID
    distributed_train_dataset = distribute_batches_equally(train_data_loader, args.get_num_workers())
    distributed_train_dataset = convert_distributed_data_into_numpy(distributed_train_dataset)

    poisoned_workers = identify_random_elements(args.get_num_workers(), args.get_num_poisoned_workers())
    distributed_train_dataset = poison_data(logger, distributed_train_dataset, args.get_num_workers(), poisoned_workers, replacement_method)

    train_data_loaders = generate_data_loaders_from_distributed_dataset(distributed_train_dataset, args.get_batch_size())

    clients = create_clients(args, train_data_loaders, test_data_loader)

    results, worker_selection, epoch_grads = run_machine_learning(clients, args, poisoned_workers, noise_method)
    
    
    exp_id = worker_selections_files[0].split("_")[0]
    path = "results/{}".format(exp_id)

    try:
        print("./{}".format(path))
        os.makedirs("./{}".format(path))
    except OSError as error:
        print(error)   

    save_results(results, "./{}/{}".format(path, results_files[0]))
    save_results(worker_selection, "./{}/{}".format(path, worker_selections_files[0]))
    
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
        df.to_hdf("./{}/flatgrads.hdf5".format(path), key="epoch_{}".format(n), mode=mode, index=False)
    # save list of poisoned workers
    np.savetxt("./{}/{}_poisoned.csv".format(path, worker_selections_files[0][:-4]),
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
        