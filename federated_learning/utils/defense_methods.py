from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statistics import StatisticsError, mode
import pandas as pd
import numpy as np
from statistics import median
import torch


def no_detect(gradients):
    return []


# this function does not return candidates
def median(parameters):
    
    new_params = {}
    for name in parameters[0].keys():
        print(name)
        print(parameters[0][name].shape)
        if len(parameters[0][name].shape) > 0:
            # quantile or mean on gradient is the same as new parameters where constant is added to all elements.
            print(torch.stack([param[name].data for param in parameters]).shape)
            new_params[name] = torch.quantile(torch.stack([param[name].data for param in parameters]), dim=0, q=0.5)
        else:
            # handle 0 dimensional parameter
            new_params[name] = parameters[0][name]           
 
    # ensure param shape is preserved
    assert parameters[0][name].shape == new_params[name].shape

    return new_params


# this function does not return candidates
def tr_mean(parameters, n_attackers):
    assert n_attackers > 0

    new_params = {}
    for name in parameters[0].keys():
        if len(parameters[0][name].shape) > 0:
            potential_params = torch.sort(torch.stack([param[name].data for param in parameters]), 0)[0]
            # quantile or mean on gradient is the same as new parameters where constant is added to all elements.
            new_params[name] = torch.mean(potential_params[n_attackers:-n_attackers], 0)
        else:
            # handle 0 dimensional parameter
            new_params[name] = parameters[0][name]
        # ensure param shape is preserved
        assert parameters[0][name].shape == new_params[name].shape

    return new_params


def multi_krum(gradients, n_attackers, multi_k=False):

    grads = flatten_grads(gradients)

    candidates = []
    candidate_indices = []
    remaining_updates = torch.from_numpy(grads)
    all_indices = np.arange(len(grads))

    while len(remaining_updates) > 2 * n_attackers + 2:
        torch.cuda.empty_cache()
        distances = []
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]

        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = remaining_updates[indices[0]][None, :] if not len(candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
        if not multi_k:
            break

    # aggregate = torch.mean(candidates, dim=0)

    # return aggregate, np.array(candidate_indices)
    return None, np.array(candidate_indices)


def bulyan(gradients, n_attackers):

    grads = flatten_grads(gradients)

    nusers = grads.shape[0]
    bulyan_cluster = []
    candidate_indices = []
    remaining_updates = torch.from_numpy(grads)
    all_indices = np.arange(len(grads))

    while len(bulyan_cluster) < (nusers - 2 * n_attackers):
        torch.cuda.empty_cache()
        distances = []
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

        distances = torch.sort(distances, dim=1)[0]

        scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]
        if not len(indices):
            break
        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        bulyan_cluster = remaining_updates[indices[0]][None, :] if not len(bulyan_cluster) else torch.cat((bulyan_cluster, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)

    # n, d = bulyan_cluster.shape
    # param_med = torch.median(bulyan_cluster, dim=0)[0]
    # sort_idx = torch.argsort(torch.abs(bulyan_cluster - param_med), dim=0)
    # sorted_params = bulyan_cluster[sort_idx, torch.arange(d)[None, :]]

    # return torch.mean(sorted_params[:n - 2 * n_attackers], dim=0), np.array(candidate_indices)
    return None, np.array(candidate_indices)


def mandera_detect(gradients):
    # gradients is a dataframe, poi_index is a lite-type object
    if type(gradients) == pd.DataFrame:
        ranks = gradients.rank(axis=0, method='average')
        vars = ranks.var(axis=1).pow(1./2)
        mus = ranks.mean(axis=1)
        feats = pd.concat([mus, vars], axis=1)
        assert feats.shape == (gradients.shape[0], 2)
    elif type(gradients) == list:
        flat_grad = flatten_grads(gradients)
        ranks = pd.DataFrame(flat_grad).rank(axis=0, method='average')
        vars = ranks.var(axis=1).pow(1./2)
        mus = ranks.mean(axis=1)
        feats = pd.concat([mus, vars], axis=1)
        assert feats.shape == (ranks.shape[0], 2)
    else:
        print("Support not implemented for generic matrixes, please use a pandas dataframe, or a list to be cast into a dataframe")
        assert type(gradients) in [pd.DataFrame, list]

    # scaler = StandardScaler()
    # feats = scaler.fit_transform(feats.values)

    model = KMeans(n_clusters=2)
    group = model.fit_predict(feats.values)
    group = np.array(group)

    diff_g0 = len(vars[group == 0]) - vars[group == 0].nunique()
    diff_g1 = len(vars[group == 1]) - vars[group == 1].nunique()

    # diff_g0 = len(vars[group == 0]) - gradients[group == 0].nunique(axis=1)
    # diff_g1 = len(vars[group == 1]) - gradients[group == 1].nunique(axis=1)

    # diff_g0 = len(vars[group == 0]) - gradients[0][group == 1].nunique()
    # diff_g1 = len(vars[group == 1]) - gradients[0][group == 1].nunique()
   
    # if no group found with matching gradients, mark the smaller group as malicious
    if diff_g0 == diff_g1:
        # get the minority label
        try:
            bad_label = (mode(group) + 1) % 2
        except StatisticsError:
            # equally sized groups, select the first group to keep.
            bad_label = 0
    elif diff_g0 < diff_g1:
        bad_label = 1
    elif diff_g0 > diff_g1:
        bad_label = 0
    else:
        assert False

    # see which indexes match the minority label
    predict_poi = [n for n, l in enumerate(group) if l == bad_label]

    return predict_poi


def fltrust(gradients):
    """
    gradients: list of gradients. The last one is the trusted server bootstrap update.
    """

    grads = flatten_grads(gradients)
    n = len(gradients) - 1
    
    # use the last gradient (server update) as the trusted source
    baseline = grads[-1]
    cos_sim = []
    new_param_list = []
    
    # compute cos similarity
    for each_param_list in grads[:-1]:
        # print(each_param_list.shape)
        each_param_array = np.array(each_param_list).squeeze()
        # print(each_param_array.shape)
        _cos = np.dot(baseline, each_param_array) / (np.linalg.norm(baseline) + 1e-9) / (np.linalg.norm(each_param_array) + 1e-9)
        # print(baseline)
        # print(each_param_array)
        # print(_cos)
        cos_sim.append(_cos)
        
    cos_sim = np.stack(cos_sim)
    # print(cos_sim.shape)
    cos_sim = np.maximum(cos_sim, 0) # relu
    # print(cos_sim.shape)
    normalized_weights = cos_sim / (np.sum(cos_sim) + 1e-9) # weighted trust score
    # print(normalized_weights.shape)
    # print(normalized_weights)

    # normalize the magnitudes and weight by the trust score
    for i in range(n):
        new_param_list.append(grads[i] * normalized_weights[i] / (np.linalg.norm(grads[i]) + 1e-9) * np.linalg.norm(baseline))
    
    # print(len(new_param_list.shape))

    # update the global model
    global_update = np.sum(new_param_list, axis=0)
    assert global_update.shape == grads[-1].shape
  
    return global_update



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
        

if __name__ == "__main__":
    import pickle
    grads_1 = pickle.load(open("../sf_debug_grads.pickle", "rb"))

    a = fltrust(grads_1)

    # import time

    # def timeit_1arg(def_function, grad_1, number):
    #     timings = []
    #     for _ in range(number):
    #         start_time = time.perf_counter()
    #         def_function(grad_1)
    #         end_time = time.perf_counter()
    #         timings.append(end_time - start_time)
    #     return timings

    # def timeit_2arg(def_function, grad_1, n_poi, number):
    #     timings = []
    #     for _ in range(number):
    #         start_time = time.perf_counter()
    #         def_function(grad_1, n_poi)
    #         end_time = time.perf_counter()
    #         timings.append(end_time - start_time)
    #     return timings    

    # n_runs = 100

    # timing_dict = {}
    
    # t = timeit_1arg(mandera_detect, grads_1, n_runs)
    # timing_dict['mandera'] = t

    # t = timeit_1arg(median, grads_1, n_runs)
    # timing_dict['median'] = t

    # t = timeit_2arg(tr_mean, grads_1, 30, n_runs)
    # timing_dict['tr_mean'] = t

    # t = timeit_2arg(multi_krum, grads_1, 30, n_runs)
    # timing_dict['multi_krum'] = t

    # t = timeit_2arg(bulyan, grads_1, 30, n_runs)
    # timing_dict['bulyan'] = t


    # print(timing_dict)

    # pickle.dump(timing_dict, open("timings_dict.pickle", "wb"))


    # Quick tests in ipython with %timeit

    # # 232 ms ± 3.47 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    # new_param = median(grads_1)

    # # 225 ms ± 1.06 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    # new_param = tr_mean(grads_1, 10)

    # # 2.34 s ± 11.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    # _, index = multi_krum(grads_1, 10, False)
    # print(index)

    # # 1min 1s ± 206 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    # _, index = bulyan(grads_1, 10)
    # print(index)

    # # 805 ms ± 6.77 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    # index = mandera_detect(grads_1)
    # print(index)
