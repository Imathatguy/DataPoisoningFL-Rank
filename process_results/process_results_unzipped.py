from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statistics import StatisticsError, mode
from tqdm import tqdm
import pandas as pd
import numpy as np
import zipfile
import h5py
import os


def mandera(gradients, poi_index):
    # gradients is a dataframe, poi_index is a lite-type object
    if type(gradients) == pd.DataFrame:
        ranks = gradients.rank(axis=0, method='average')
        vars = ranks.var(axis=1).pow(1./2)
        mus = ranks.mean(axis=1)
        feats = pd.concat([mus, vars], axis=1)
        assert feats.shape == (100, 2)
        n_nodes = gradients.shape[0]
    else:
        print("Support not implemented for generic matrixes, please use a pandas dataframe")
        assert type(gradients) == pd.DataFrame

    # scaler = StandardScaler()
    # feats = scaler.fit_transform(feats.values)

    model = KMeans(n_clusters=2)
    group = model.fit_predict(feats.values)
    assert len(group) == 100

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

    detected = set(poi_index).intersection(set(predict_poi))
    P = len(predict_poi)
    TP = len(detected)
    FP = P - TP
    FN = len(poi_index) - TP
    TN = (n_nodes-len(poi_index)) - FP

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    accuracy =(TP+TN)/(TP+TN+FP+FN)
    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    return [accuracy, precision, recall, f1]


if __name__ == "__main__":

    # Using 10000 for baseline
    # the 2-3 digits (X20XX) specifying num of poisioning workers
    # the 4-5 digits (XXX00) specifying run number
    # FASHION
    #     20000 for label flipping
    #     30000 for gaussian noise
    #     40000 for zero grad
    #     50000 for sign flip
    # 100020000 for shifted mean
    # CIFAR
    #     60000 for label flipping
    #     70000 for gaussian noise
    #     80000 for zero grad
    #     90000 for sign flip
    # 100060000 for shifted mean
    # MNIST
    #   10020000 for label flipping
    #   10030000 for gaussian noise
    #   10040000 for zero grad
    #   10050000 for sign flip
    #  110020000 for shifted mean
    # Add 100000 for full defense method.
    # QMNIST
    #  110070000 for gaussian noise
    #  110080000 for zero grad
    #  110090000 for sign flip
    #  110060000 for shifted mean

    # path for 60000, 80000
    # file_path = 'G:/active_projects/RankPoisonFL/'
    # path for 70000
    # file_path = 'Z:/'
    # path for 50000, 90000
    # file_path = 'I:/DataPoisoning_FL/results/past'
    # path for 10020000, 10030000, 10040000, 10050000
    # file_path = '/scratch2/zha197/results'
    file_path = os.path.join( "F:", os.sep, "DataPoisoningFL-Rank", "mandera_results", "results_def")

    exp_series = 110070000
    n_runs = 20
    # n_poi_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    n_poi_list = [5, 10, 15, 20, 25, 30]
    max_epochs = 25

    bulk_metrics = {}

    print(exp_series)

    # Check all required files exist before processings
    errors = []
    for n_poi in tqdm(n_poi_list):
        for n_run in tqdm(range(n_runs)):
            exp_code = exp_series + n_poi*100 + n_run
            p_workers_file = "{}/{}/{}_workers_selected_poisoned.csv".format(file_path, exp_code, exp_code)
            hdf5_file = "{}/{}/flatgrads.hdf5".format(file_path, exp_code)

            if os.path.exists(p_workers_file) & os.path.exists(hdf5_file):
                continue
            else:
                errors.append(exp_code)
    assert len(errors) == 0, print(errors)

    # begin processing
    for n_poi in tqdm(n_poi_list):
        # exp_bulk = "{}XX_results.zip".format(str(exp_series + n_poi*100)[:3])
        # print(exp_bulk)

        for n_run in tqdm(range(n_runs)):
            exp_code = exp_series + n_poi*100 + n_run
            print(exp_code)
            
            p_workers_file = "{}/{}/{}_workers_selected_poisoned.csv".format(file_path, exp_code, exp_code)
            mal_nodes = pd.read_csv(p_workers_file, header=None).values.flatten()

            hdf5_file = "{}/{}/flatgrads.hdf5".format(file_path, exp_code)
            
            # Open hdf5 file and extract gradients into holder
            holder = {}
            a = h5py.File(hdf5_file)
            for n_epoch in range(max_epochs):
                key = 'epoch_{}'.format(n_epoch)
                grads = pd.DataFrame(a[key]['block0_values'])
                # Save all the grads if we need multi epoch processing
                holder[key] = grads

            # do required merging of multiple epoch gradients
            pass

            # process epoch gradients into metrics
            for n_epoch in range(max_epochs):
                key = 'epoch_{}'.format(n_epoch)

                bulk_metrics[(n_poi, n_run, n_epoch)] = mandera(holder[key], mal_nodes)

            # break
        # break

    # save bulk_metrics
    output = pd.DataFrame(bulk_metrics).transpose()
    output.columns = ['accuracy', 'precision', 'recall', 'f1']
    output.rename_axis(['n_poi', 'n_run', 'n_epoch'], inplace=True)
    output.to_csv("{}_bulk_metrics.csv".format(exp_series), index=True, header=True)
