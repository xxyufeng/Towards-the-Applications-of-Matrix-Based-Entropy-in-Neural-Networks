import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler, random_split
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST
import numpy as np
import os
import urllib.request
import sklearn.datasets

############################
# Dataset preparation
############################
class LibSVMDataset(torch.utils.data.Dataset):
    def __init__(self, url, dataset_path, download=False, dimensionality=None, classes=None):
        self.url = url
        self.dataset_path = dataset_path
        self._dimensionality = dimensionality

        self.filename = os.path.basename(url)
        self.dataset_type = os.path.basename(os.path.dirname(url))

        if not os.path.isfile(self.local_filename):
            if download:
                print(f"Downloading {url}")
                self._download()
            else:
                raise RuntimeError(
                    "Dataset not found. You can use download=True to download it."
                )
        else:
            print("Files already downloaded")

        self.data, y = sklearn.datasets.load_svmlight_file(self.local_filename)

        sparsity = self.data.nnz / (self.data.shape[0] * self.data.shape[1])
        if sparsity > 0.1:
            self.data = self.data.todense().astype(np.float32)
            self._is_sparse = False
        else:
            self._is_sparse = True

        # convert labels to [0, 1]
        if classes is None:
            classes = np.unique(y)
        self.classes = np.sort(classes)
        self.targets = torch.zeros(len(y), dtype=torch.int64)
        for i, label in enumerate(self.classes):
            self.targets[y == label] = i

        self.class_to_idx = {cl: idx for idx, cl in enumerate(self.classes)}

        super().__init__()

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_features(self):
        return self.data.shape[1]
    

    def __getitem__(self, idx):
        if self._is_sparse:
            x = torch.from_numpy(self.data[idx].todense().astype(np.float32)).flatten()
        else:
            x = torch.from_numpy(self.data[idx]).flatten()
        y = self.targets[idx]

        if self._dimensionality is not None:
            if len(x) < self._dimensionality:
                x = torch.cat([x, torch.zeros([self._dimensionality - len(x)], dtype=x.dtype, device=x.device)])
            elif len(x) > self._dimensionality:
                raise RuntimeError("Dimensionality is set wrong.")

        return x, y

    def __len__(self):
        return len(self.targets)

    @property
    def local_filename(self):
        return os.path.join(self.dataset_path, self.dataset_type, self.filename)

    def _download(self):
        os.makedirs(os.path.dirname(self.local_filename), exist_ok=True)
        urllib.request.urlretrieve(self.url, filename=self.local_filename)


class RCV1(LibSVMDataset):
    def __init__(self, split, download=False, dataset_path=None):
        if split == "train":
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2"
        elif split == "test":
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2"
        else:
            raise RuntimeError(f"Unavailable split {split}")
        super().__init__(url=url, download=download, dataset_path=dataset_path)


class GISETTE(LibSVMDataset):
    def __init__(self, split, download=False, dataset_path=None):
        if split == "train":
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2"
        elif split == "test":
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.t.bz2"
        else:
            raise RuntimeError(f"Unavailable split {split}")
        super().__init__(url=url, download=download, dataset_path=dataset_path)

class ijcnn(LibSVMDataset):
    def __init__(self, split, download=False, dataset_path=None):
        if split == "train":
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.tr.bz2"
        elif split == "test":
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.t.bz2"
        else:
            raise RuntimeError(f"Unavailable split {split}")
        super().__init__(url=url, download=download, dataset_path=dataset_path)

class w1a(LibSVMDataset):
    def __init__(self, split, download=False, dataset_path=None, dimensionality=None, subset=1):
        if split == "train":
            url = f"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w{subset}a"
        elif split == "test":
            url = f"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w{subset}a.t"
        else:
            raise RuntimeError(f"Unavailable split {split}")
        super().__init__(url=url, download=download, dataset_path=dataset_path)

class a1a(LibSVMDataset):
    def __init__(self, split, download=False, dataset_path=None, dimensionality=None, subset=1):

        if split == "train":
            url = f"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a{subset}a"
        elif split == "test":
            url = f"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a{subset}a.t"
        else:
            raise RuntimeError(f"Unavailable split {split}")
        super().__init__(url=url, download=download, dataset_path=dataset_path, dimensionality=dimensionality)


# Data loading
def load_data_libsvm(dataset_name, dataset_path, split_type):
    if split_type == 'train':
        if dataset_name == 'cifar10':
          train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
          return CIFAR10(root=dataset_path, train=True, download=True, transform=train_transform)
        elif dataset_name == 'mnist':
          train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
          return MNIST(root=dataset_path, train=True, download=True, transform=train_transform)
        elif dataset_name == 'RCV1':
            return RCV1("train", download=True, dataset_path=dataset_path)
        elif dataset_name == 'GISETTE':
            return GISETTE("train", download=True, dataset_path=dataset_path)
        elif dataset_name == 'ijcnn':
            return ijcnn("train", download=True, dataset_path=dataset_path)
        elif dataset_name.startswith('w') and dataset_name.endswith('a') and len(dataset_name) == 3:
            subset = int(dataset_name[1])
            return w1a("train", download=True, dataset_path=dataset_path, dimensionality=123, subset=subset)
        elif dataset_name.startswith('a') and dataset_name.endswith('a') and len(dataset_name) == 3:
            subset = int(dataset_name[1])
            return a1a("train", download=True, dataset_path=dataset_path, dimensionality=123, subset=subset)
        


    elif split_type == 'val':
        if dataset_name == 'cifar10':
          test_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
          return CIFAR10(root=dataset_path, train=False, download=True, transform=test_transform)
        elif dataset_name == 'mnist':
          test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
          return MNIST(root=dataset_path, train=False, download=True, transform=test_transform)
        elif dataset_name == 'RCV1':
            return RCV1("test", download=True, dataset_path=dataset_path)
        elif dataset_name == 'GISETTE':
            return GISETTE("test", download=True, dataset_path=dataset_path)
        elif dataset_name == 'ijcnn':
            return ijcnn("test", download=True, dataset_path=dataset_path)
        elif dataset_name.startswith('w') and dataset_name.endswith('a') and len(dataset_name) == 3: 
            subset = int(dataset_name[1])
            return w1a("test", download=True, dataset_path=dataset_path, dimensionality=123, subset=subset)
        elif dataset_name.startswith('a') and dataset_name.endswith('a') and len(dataset_name) == 3:
            subset = int(dataset_name[1])
            return a1a("test", download=True, dataset_path=dataset_path, dimensionality=123, subset=subset)
