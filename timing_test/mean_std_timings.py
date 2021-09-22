import pickle
import numpy as np

a = pickle.load(open("timings_dict.pickle", "rb"))

for key in ['mandera', 'median', 'tr_mean', 'multi_krum', 'bulyan']:
    arr = a[key]
    print(key)
    print(np.mean(arr))
    print(np.std(arr))