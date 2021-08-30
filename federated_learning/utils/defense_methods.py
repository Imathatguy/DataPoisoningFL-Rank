from sklearn.cluster import KMeans
from statistics import StatisticsError, mode
import pandas as pd
import numpy as np


def no_detect(gradients):
    return []


def mandera_detect(gradients):
    # gradients is a dataframe, poi_index is a lite-type object
    if type(gradients) == pd.DataFrame:
        vars = gradients.rank(axis=0, method='first').var(axis=1)
    elif type(gradients) == list:
        vars = pd.DataFrame(flatten_grads(gradients)).rank(axis=0, method='first').var(axis=1)
    else:
        print("Support not implemented for generic matrixes, please use a pandas dataframe, or a list to be cast into a dataframe")
        assert type(gradients) in [pd.DataFrame, list]

    model = KMeans(n_clusters=2)
    group = model.fit_predict(vars.values.reshape(-1,1))

    # get the minority label
    try:
        bad_label = (mode(group) + 1) % 2
    except StatisticsError:
        # equally sized groups, select the first group to keep.
        bad_label = 0

    # see which indexes match the minority label
    predict_poi = [n for n, l in enumerate(group) if l == bad_label]

    return predict_poi


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
    grads = pickle.load(open("debug_grads.pickle", "rb"))
