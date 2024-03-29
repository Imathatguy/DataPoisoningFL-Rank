import torch
import copy
from statistics import NormalDist, stdev
from math import floor


def no_noise(existing_parameters, parameters, random_workers, poisoned_workers):
    ### compute gradients for all clients
    b = zip(*[existing_parameters, parameters])
    # copy tensor structure to replace values
    gradients = copy.deepcopy(parameters)
    # replace params with grads in structure
    for n, client_params in enumerate(b):
        assert client_params[0].keys() == client_params[1].keys()
        for name in client_params[0].keys():
            gradients[n][name] = client_params[1][name] - client_params[0][name]

    # no change to parameters, only compute gradients
    return parameters, gradients


def gaussian_attack(existing_parameters, parameters, random_workers, poisoned_workers):
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

    ### compute mean of non-poisoned client gradients
    # copy tensor structure to replace values
    mu_grad = copy.deepcopy(gradients[0])
    benign_workers = [n for n, client_idx in enumerate(random_workers) if client_idx not in poisoned_workers]
    for name in mu_grad.keys():
        mu_grad[name] = sum([gradients[n][name].data for n in benign_workers]) / len(benign_workers)

    ### modify poisoned gradients
    poisoned_idx = [n for n, client_idx in enumerate(random_workers) if client_idx in poisoned_workers]
    for n in poisoned_idx:
        for name in parameters[n].keys():
            # copy mu to poisoned user
            gradients[n][name] = copy.deepcopy(mu_grad[name])
            # add gaussian noise to mean gradient with mean=mu, variance=30
            noise_grad = torch.randn(mu_grad[name].shape, device=mu_grad[name].device)
            gradients[n][name].add_(30, noise_grad)
            # compute final parameters for poisoned client
            parameters[n][name] = existing_parameters[n][name].add(1, gradients[n][name])

    return parameters, gradients


def shifted_mean(existing_parameters, parameters, random_workers, poisoned_workers):
    """
    :param parameters: List of parameters
    :type parameters: list
    :param random_workers: indices of randomized workers
    :type random_workers: list(int)
    :param poisoned_workers: indices of poisoned workers
    :type poisoned_workers: list(int)
    :return: new class IDs
    """
    ### Compute attack parameters given knowledge of # Poisoned + # Benign

    # I note that exact z values can be calculated with `z_max = NormalDist().inv_cdf()` or from the z-table
    # Algorithm 3 in `A Little Is Enough: Circumventing Defenses For Distributed Learning`
    

    # z values are pre-computed $max_z$[ $\phi(z)$ < (n−m−s) / (n−m) ]
    _n = len(random_workers)
    _m = len([n for n, client_idx in enumerate(random_workers) if client_idx in poisoned_workers])
    _s = floor(_n/2 + 1) - _m
    _phi = 1.0*(_n - _m - _s) / (_n - _m)
    assert _n > _m, "More malicious workers than total workers in round"
    z_max = NormalDist().inv_cdf(_phi)

    ### compute gradients for all clients
    b = zip(*[existing_parameters, parameters])
    # copy tensor structure to replace values
    gradients = copy.deepcopy(parameters)
    # replace params with grads in structure
    for n, client_params in enumerate(b):
        assert client_params[0].keys() == client_params[1].keys()
        for name in client_params[0].keys():
            gradients[n][name] = client_params[1][name] - client_params[0][name]

    ### compute mean of non-poisoned client gradients
    # copy tensor structure to replace values
    mu_grad = copy.deepcopy(gradients[0])
    sd_grad = copy.deepcopy(gradients[0])
    benign_workers = [n for n, client_idx in enumerate(random_workers) if client_idx not in poisoned_workers]
    for name in mu_grad.keys():
        name_grads = [gradients[n][name].data for n in benign_workers]
        mu_grad[name] = sum(name_grads) / len(benign_workers)
        sd_grad[name] = torch.std(torch.stack(name_grads, dim=0).float(), dim=0, unbiased=False)

        assert name_grads[0].shape == sd_grad[name].shape
        assert name_grads[0].shape == mu_grad[name].shape

    ### modify poisoned gradients
    poisoned_idx = [n for n, client_idx in enumerate(random_workers) if client_idx in poisoned_workers]
    for n in poisoned_idx:
        for name in parameters[n].keys():
            # copy mu to poisoned user
            gradients[n][name] = copy.deepcopy(mu_grad[name])
            # subtract sd_grad scaled by z_max
            gradients[n][name].sub_(z_max, sd_grad[name])
            # compute final parameters for poisoned client
            parameters[n][name] = existing_parameters[n][name].add(1, gradients[n][name])

    return parameters, gradients


def zero_gradient(existing_parameters, parameters, random_workers, poisoned_workers):
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
            gradients[n][name].div_(-n_poi)
            # compute final parameters for poisoned client
            parameters[n][name] = existing_parameters[n][name].add(1, gradients[n][name])

    return parameters, gradients


def sign_flipping(existing_parameters, parameters, random_workers, poisoned_workers):
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
    mu_grad = copy.deepcopy(gradients[0])
    benign_workers = [n for n, client_idx in enumerate(random_workers) if client_idx not in poisoned_workers]
    for name in mu_grad.keys():
        mu_grad[name] = sum([gradients[n][name].data for n in benign_workers]) / len(benign_workers)
    ### modify poisoned gradients
    poisoned_idx = [n for n, client_idx in enumerate(random_workers) if client_idx in poisoned_workers]
    for n in poisoned_idx:
        for name in parameters[n].keys():
            # copy sum of benign gradients to poisoned user
            gradients[n][name] = copy.deepcopy(mu_grad[name])
            # divide sum of gradients by number of poisoned nodes
            gradients[n][name].mul_(-3.0)
            # compute final parameters for poisoned client
            parameters[n][name] = existing_parameters[n][name].add(1, gradients[n][name])

    return parameters, gradients


# gaussian prototype from paper
def _white(messages, byzantinesize):
    # The mean is the same, the variance is larger
    mu = torch.mean(messages[0:-byzantinesize], dim=0)
    messages[-byzantinesize:].copy_(mu)
    noise = torch.randn((byzantinesize, messages.size(1)), dtype=torch.float64)
    messages[-byzantinesize:].add_(30, noise)


# zero grad prototype from paper
def _zeroGradient(messages, byzantinesize):
    s = torch.sum(messages[0:-byzantinesize], dim=0)
    messages[-byzantinesize:].copy_(-s / byzantinesize)


def _maxValue(messages, byzantinesize):
    mu = torch.mean(messages[0:-byzantinesize], dim=0)
    meliciousMessage = -3*mu
    messages[-byzantinesize:].copy_(meliciousMessage)


# compute difference in parameters prototype
def find_grads(existing_parameters, parameters, random_workers):
    # compute difference between old and new parameters
    client_grads = {}
    b = zip(*[existing_parameters, parameters])

    for n, client_params in enumerate(b):
        assert client_params[0].keys() == client_params[1].keys()

        client_grad = {}
        for name in client_params[0]:
            client_grad[name] = client_params[1][name] - client_params[0][name]

        client_idx = random_workers[n]
        client_grads[client_idx] = client_grad


# testing function
if __name__ == "__main__":
    from federated_learning.utils import average_nn_parameters

    import pickle
    existing_parameters, parameters, random_workers, poisoned_workers = pickle.load(open("debug.pickle", "rb"))
    # note to self existing parameters is the same for all clients after a global update
    import copy

    # compute gradients for all clients
    b = zip(*[existing_parameters, parameters])
    gradients = copy.deepcopy(parameters)
    for n, client_params in enumerate(b):
        assert client_params[0].keys() == client_params[1].keys()
        for name in client_params[0].keys():
            gradients[n][name] = client_params[1][name] - client_params[0][name]

    # compute mean of non-poisoned client gradients
    mu_grad = copy.deepcopy(gradients[0])
    benign_workers = [n for n, client_idx in enumerate(random_workers) if client_idx not in poisoned_workers]
    for name in mu_grad.keys():
        mu_grad[name] = sum([gradients[n][name].data for n in benign_workers]) / len(benign_workers)

    # modify poisoned gradients
    poisoned_idx = [n for n, client_idx in enumerate(random_workers) if client_idx in poisoned_workers]
    # scale = len(random_workers) - len(poisoned_idx)
    # scale = -3.0 / scale
    for n in poisoned_idx:
        pass   

        # TEST FUNCTION FOR ZERO GRADIENT
        for name in parameters[n].keys():
            gradients[n][name] = copy.deepcopy(mu_grad[name])
            # print(gradients[n][name])
            gradients[n][name].mul_(-3.0)
            # print(gradients[n][name])
            # compute final parameters for client
            # print(existing_parameters[n][name])
            parameters[n][name] = existing_parameters[n][name].add(1, gradients[n][name])
            # print(parameters[n][name])

    avg_grads = average_nn_parameters(gradients)



        # TEST FUNCTION FOR WHITE NOISE
        #  # copy mu to poisoned user
        # for name in parameters[n].keys():
        #     gradients[n][name] = copy.deepcopy(mu_grad[name])
        #     # add gaussian noise to mean gradient with mean=mu, variance=30
        #     noise_grad = torch.randn(mu_grad[name].shape, device=mu_grad[name].device)
        #     gradients[n][name].add_(30, noise_grad)
        #     # compute final parameters for client
        #     parameters[n][name] = existing_parameters[n][name].add(1, gradients[n][name])
