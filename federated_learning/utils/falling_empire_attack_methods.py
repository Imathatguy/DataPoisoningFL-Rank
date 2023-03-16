import copy


def falling_empires_n10(existing_parameters, parameters, random_workers, poisoned_workers):
    epsilon = -10
    return _falling_empires(existing_parameters, parameters, random_workers, poisoned_workers, epsilon)


def falling_empires_p10(existing_parameters, parameters, random_workers, poisoned_workers):
    epsilon = 10
    return _falling_empires(existing_parameters, parameters, random_workers, poisoned_workers, epsilon)


def falling_empires_n01(existing_parameters, parameters, random_workers, poisoned_workers):
    epsilon = -0.1
    return _falling_empires(existing_parameters, parameters, random_workers, poisoned_workers, epsilon)


def falling_empires_p01(existing_parameters, parameters, random_workers, poisoned_workers):
    epsilon = 0.1
    return _falling_empires(existing_parameters, parameters, random_workers, poisoned_workers, epsilon)


def _falling_empires(existing_parameters, parameters, random_workers, poisoned_workers, epsilon):
    """
    :param parameters: List of parameters
    :type parameters: list
    :param random_workers: indices of randomized workers
    :type random_workers: list(int)
    :param poisoned_workers: indices of poisoned workers
    :type poisoned_workers: list(int)
    :return: new class IDs
    """
    ### compute gradients for all clients
    b = zip(*[existing_parameters, parameters])
    # copy tensor structure to replace values
    gradients = copy.deepcopy(parameters)
    # replace params with grads in structure
    for n, client_params in enumerate(b):
        assert client_params[0].keys() == client_params[1].keys()
        for name in client_params[0].keys():
            gradients[n][name] = client_params[1][name] - client_params[0][name]

    ### compute sum of non-poisoned client gradients
    # copy tensor structure to replace values
    sum_grad = copy.deepcopy(gradients[0])
    benign_workers = [n for n, client_idx in enumerate(random_workers) if client_idx not in poisoned_workers]
    for name in sum_grad.keys():
        sum_grad[name] = sum([gradients[n][name].data for n in benign_workers]) / 1.0
    ### modify poisoned gradients
    poisoned_idx = [n for n, client_idx in enumerate(random_workers) if client_idx in poisoned_workers]
    n_poi = len(poisoned_idx)
    for n in poisoned_idx:
        for name in parameters[n].keys():
            # copy sum of benign gradients to poisoned user
            gradients[n][name] = copy.deepcopy(sum_grad[name])
            # divide sum of gradients by number of poisoned nodes
            gradients[n][name].div_(-n_poi/epsilon)
            # compute final parameters for poisoned client
            parameters[n][name] = existing_parameters[n][name].add(1, gradients[n][name])

    return parameters, gradients