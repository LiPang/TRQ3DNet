import os

import torch
import torch.utils.data as data
import numpy as np
from scipy.io import loadmat
from torchnet.dataset import TransformDataset, SplitDataset
from .util_dataset import worker_init_fn


class LoadMatKey(object):
    def __init__(self, key):
        self.key = key

    def __call__(self, mat):
        item = mat[self.key][:].transpose((2, 0, 1))
        return item.astype(np.float32)

class HSI2Tensor(object):
    """
    Transform a numpy array with shape (C, H, W)
    into torch 4D Tensor (1, C, H, W) or (C, H, W)
    """
    def __init__(self, use_2dconv):
        self.use_2dconv = use_2dconv

    def __call__(self, hsi):
        if self.use_2dconv:
            img = torch.from_numpy(hsi)
        else:
            img = torch.from_numpy(hsi[None])
        return img.float()

class LMDBDataset(data.Dataset):
    def __init__(self, db_path, repeat=1):
        import lmdb
        self.db_path = db_path
        self.env = lmdb.open(db_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        self.repeat = repeat


    def __getitem__(self, index):
        index = index % self.length
        env = self.env
        with env.begin(write=False) as txn:
            raw_datum = txn.get('{:08}'.format(index).encode('ascii'))
        import caffe
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(raw_datum)

        flat_x = np.fromstring(datum.data, dtype=np.float32)
        x = flat_x.reshape(datum.channels, datum.height, datum.width)

        return x

    def __len__(self):
        return self.length * self.repeat

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

class NPYDataset(data.Dataset):
    def __init__(self, db_path, repeat=1):
        self.db_path = db_path
        self.datasets = np.load(db_path)
        self.length = len(self.datasets.files)
        self.repeat = repeat

    def __getitem__(self, index):
        index = index % self.length
        try:
            x = self.datasets['{:08}'.format(index)]
        except:
            x = self.datasets[bytes('{:08}'.format(index).encode())]
        return x

    def __len__(self):
        return self.length * self.repeat

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

class ImageTransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform, target_transform=None):
        super(ImageTransformDataset, self).__init__()

        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.length = len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.dataset[idx]
        target = img.copy()
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

def make_dataset(opt, train_transform, target_transform, common_transform, batch_size=None, repeat=1):
    suffix = opt.dataroot.split('.')[-1]
    if suffix == 'db':
        dataset = LMDBDataset(opt.dataroot, repeat=repeat)
    elif suffix == 'npz':
        dataset = NPYDataset(opt.dataroot, repeat=repeat)
    else:
        raise 'file type not supported'
    """Split patches dataset into training, validation parts"""
    dataset = TransformDataset(dataset, common_transform)

    train_dataset = ImageTransformDataset(dataset, train_transform, target_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                              batch_size=batch_size or opt.batchSize, shuffle=True,
                              num_workers=opt.threads, pin_memory=not opt.no_cuda, worker_init_fn=worker_init_fn)

    return train_loader


class MatDataFromFolder(torch.utils.data.Dataset):
    """Wrap mat data from folder"""
    def __init__(self, data_dir, load=loadmat, suffix='mat', fns=None, size=None):
        super(MatDataFromFolder, self).__init__()
        if fns is not None:
            self.filenames = [
                os.path.join(data_dir, fn) for fn in fns
            ]
        else:
            self.filenames = [
                os.path.join(data_dir, fn)
                for fn in os.listdir(data_dir)
                if fn.endswith(suffix)
            ]

        self.load = load
        if size and size <= len(self.filenames):
            self.filenames = self.filenames[:size]

    def __getitem__(self, index):
        mat = self.load(self.filenames[index])
        return mat

    def __len__(self):
        return len(self.filenames)


class LoadMatHSI(object):
    def __init__(self, input_key, gt_key, transform=None):
        self.gt_key = gt_key
        self.input_key = input_key
        self.transform = transform

    def __call__(self, mat):
        if self.transform:
            input = self.transform(mat[self.input_key][:].transpose((2, 0, 1)))
            gt = self.transform(mat[self.gt_key][:].transpose((2, 0, 1)))
        else:
            input = mat[self.input_key][:].transpose((2, 0, 1))
            gt = mat[self.gt_key][:].transpose((2, 0, 1))
        input = torch.from_numpy(input).float()
        gt = torch.from_numpy(gt).float()

        return input, gt