# verify_epoch.py


import pandas as pd
import numpy as np

# df.to_hdf("./{}/flatgrads.hdf5".format(path), key="epoch_{}".format(n), mode=mode, index=False)

a = pd.read_hdf("./results/3200/flatgrads.hdf5", key="epoch_0")

b = pd.read_csv("./results/3200/epoch0.csv")


print(a.values == b.values[:, 1:])

print(a.values - b.values[:, 1:])

print(np.isclose(a.values, b.values[:, 1:]))
