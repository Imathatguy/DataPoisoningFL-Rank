try:
    from .dataset import Dataset
except:
    from dataset import Dataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import from_numpy
from numpy import frombuffer, dtype
import gzip

import codecs

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

def open_maybe_compressed_file(path):
    if path.endswith('.gz'):
        return gzip.open(path, 'rb')
    elif path.endswith('.xz'):
        return lzma.open(path, 'rb')
    else:
        return open(path,'rb')
    
def read_idx2_int(path):
    with open_maybe_compressed_file(path) as f:
        data = f.read()
        assert get_int(data[:4]) == 12*256 + 2
        length = get_int(data[4:8])
        width = get_int(data[8:12])
        parsed = frombuffer(data, dtype=dtype('>i4'), offset=12)
        return from_numpy(parsed.astype('i4')).view(length,width).long()

def include_author_info(target):
    return (target[0].item(), target[2].item())

class QMNISTDataset(Dataset):

    def __init__(self, args):
        super(QMNISTDataset, self).__init__(args)

    def load_train_dataset(self):
        self.get_args().get_logger().debug("Loading QMNIST train data")

        train_dataset = datasets.QMNIST(self.get_args().get_data_path(), train=True, download=True,
                                        transform=transforms.Compose([transforms.ToTensor()]),
                                        compat=False, target_transform=include_author_info)
        train_dataset.targets = read_idx2_int(train_dataset.labels_file)
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))

        train_data = self.get_noniid_tuple_from_data_loader(train_loader)

        self.get_args().get_logger().debug("Finished loading QMNIST train data")

        return train_data

    def load_test_dataset(self):
        self.get_args().get_logger().debug("Loading QMNIST test data")

        test_dataset = datasets.QMNIST(self.get_args().get_data_path(), what='test10k',
                                       train=False, download=True,
                                       transform=transforms.Compose([transforms.ToTensor()]))
        test_dataset.targets = read_idx2_int(test_dataset.labels_file)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        test_data = self.get_tuple_from_data_loader(test_loader)

        self.get_args().get_logger().debug("Finished loading QMNIST test data")

        return test_data


if __name__ == "__main__":
    import numpy as np
    train_dataset = datasets.QMNIST('data', train=True, compat=False, download=True, transform=transforms.Compose([transforms.ToTensor()]),
                                    target_transform=include_author_info)
    train_dataset.targets = read_idx2_int(train_dataset.labels_file)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))

    test_dataset = datasets.QMNIST('data', what='test10k', train=False, download=True,
                                   transform=transforms.Compose([transforms.ToTensor()]))
    test_dataset.targets = read_idx2_int(test_dataset.labels_file)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))


    def get_noniid_tuple_from_data_loader(data_loader):
        return (next(iter(data_loader))[0].numpy(),
                next(iter(data_loader))[1][0].numpy(),
                next(iter(data_loader))[1][1].numpy())

    train_data = get_noniid_tuple_from_data_loader(train_loader)
    data = list(zip(train_data[0], train_data[1], train_data[2]))
    X, Y, Z= zip(*data)
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)

    print(X, Y, Z)