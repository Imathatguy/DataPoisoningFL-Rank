from sklearn.cluster import KMeans
from statistics import mode
from tqdm import tqdm
import pandas as pd
import zipfile
import h5py
import os


if __name__ == "__main__":
    # path for 60000, 80000
    # file_path = 'G:/active_projects/RankPoisonFL/'
    # path for 70000
    # file_path = 'Z:/'
    # path for 50000, 90000
    file_path = 'I:/DataPoisoning_FL/results/past'

    exp_series = 50000
    n_runs = 20
    # n_poi_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    n_poi_list = [5 , 10, 15, 20, 25, 30]
    max_epochs = 1

    bulk_metrics = {}

    fresh_file = True

    for n_poi in tqdm(n_poi_list):
        exp_bulk = "{}XX_results.zip".format(str(exp_series + n_poi*100)[:3])
        print(exp_bulk)

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
            
            # we should only have 1 epoch
            assert len(holder) == 1

            holder = holder['epoch_0']
            df = pd.DataFrame(holder)
            mal_node_df = pd.DataFrame(mal_nodes)

            # erase existing hdf5 file
            if fresh_file:
                mode = 'w'
                fresh_file = False
            # append subsequent epochs to existing file
            else:
                mode = 'a'    
            # Save results to hdf5 file
            df.to_hdf("./{}_1st_epoch_grads.hdf5".format(exp_series), key="key_{}_{}_{}".format(n_poi, n_run, n_epoch), mode=mode, index=False)            
            mal_node_df.to_hdf("./{}_mal_nodes.hdf5".format(exp_series), key="key_{}_{}_{}".format(n_poi, n_run, n_epoch), mode=mode, index=False)


            # break
        # break

    # # save bulk_metrics
    # output = pd.DataFrame(bulk_metrics).transpose()
    # output.columns = ['accuracy', 'precision', 'recall', 'f1']
    # output.rename_axis(['n_poi', 'n_run', 'n_epoch'], inplace=True)
    # output.to_csv("{}_bulk_metrics.csv".format(exp_series), index=True, header=True)


