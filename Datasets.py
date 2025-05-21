# Datasets.py

import os
import pickle

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset




# 1) Configuration for each top‐level dataset
DATASET_CONFIG = {
    'DIGIT':      ('digit_dataset',
                   ['MNIST','MNIST_M','SVHN','SynthDigits','USPS'],
                   (32,32), 10),
    'CIFAR10':    ('CIFAR10', ['CIFAR10'], (32,32), 10),
    'CIFAR10-C':  ('CIFAR10-C',[
                     'brightness','contrast','defocus_blur','elastic_transform',
                     'fog','frost','gaussian_blur','gaussian_noise','glass_blur',
                     'impulse_noise','jpeg_compression','motion_blur','pixelate',
                     'saturate','shot_noise','snow','spatter','speckle_noise',
                     'zoom_blur'
                   ], (32,32), 10),
    'FairFace':   ('FairFace',[
                     'Black','East_Asian','Indian',
                     'Latino_Hispanic','Middle_Eastern',
                     'Southeast_Asian','White'
                   ], (224,224), 2),
    'OfficeHome': ('OfficeHome',['Art','Clipart','Product','Real World'], (224,224), 65),
    'PACS':       ('PACS',['art_painting','cartoon','photo','sketch'],         (224,224), 7),
    'VLCS':       ('VLCS',['Caltech101','LabelMe','VOC2007','SUN09'],            (224,224), 5),
    'DomainNet':  ('DomainNet',[
                     'clipart','infograph','painting',
                     'quickdraw','real','sketch'
                   ], (224,224), 345),
}

class DatasetObject:
    def __init__(self,
                 dataset: str,
                 n_client: int,
                 seed: int,
                 result_path: str = '',
                 data_dir: str = '../data',
                 personalize = False):
        self.dataset     = dataset
        self.n_client    = n_client
        self.seed        = seed
        self.data_dir    = data_dir
        self.name        = f"{dataset}_{n_client}_{seed}_{personalize}"
        self.result_path = result_path

        self._load_numpy()

    def _load_numpy(self):
        # lookup
        if self.dataset not in DATASET_CONFIG:
            raise ValueError(f"Unsupported dataset {self.dataset}")
        folder, domains, img_size, n_cls = DATASET_CONFIG[self.dataset]

        if self.n_client != len(domains):
            raise ValueError(f"n_client ({self.n_client}) != #domains ({len(domains)})")

        # prepare per‐client lists
        clnt_x       = [[] for _ in domains]
        clnt_y       = [[] for _ in domains]
        clnt_test_x  = [[] for _ in domains]
        clnt_test_y  = [[] for _ in domains]

        base = os.path.join(self.data_dir, folder)
        for i, dom in enumerate(domains):
            dom_dir = os.path.join(base, dom)

            # load train
            parts_dir = os.path.join(dom_dir, 'partitions')
            if not os.path.isdir(parts_dir):
                raise FileNotFoundError(f"Missing partitions folder for {dom_dir}")
            for fname in sorted(os.listdir(parts_dir)):
                if not fname.startswith('train_part') or not fname.endswith('.pkl'):
                    continue
                train_pkl = os.path.join(parts_dir, fname)
            # train_pkl = os.path.join(dom_dir, 'partitions', 'train_part0.pkl')

                with open(train_pkl, 'rb') as f:
                    data,labels = pickle.load(f)
                    if self.dataset == "PACS":
                        labels = [lab - 1 for lab in labels]
                for img_arr, lab in zip(data, labels):
                    if img_arr.ndim == 2:
                        img_arr = np.stack([img_arr]*3, axis=2)   # H×W → H×W×3
                    clnt_x[i].append(self._preprocess(img_arr, img_size))
                    clnt_y[i].append(lab)
                # data_tr, labs_tr = pickle.load(f)
            # load test
            test_pkl = os.path.join(dom_dir, 'test.pkl')
            if not os.path.isfile(test_pkl):
                raise FileNotFoundError(f"Missing {test_pkl}")
            with open(test_pkl, 'rb') as f:
                data_te, labs_te = pickle.load(f)
                if self.dataset == "PACS":
                    labs_te = [lab - 1 for lab in labs_te]

            # # preprocess & assign
            # for img_arr, lab in zip(data_tr, labs_tr):
            #     clnt_x[i].append(self._preprocess(img_arr, img_size))
            #     clnt_y[i].append(lab)
            for img_arr, lab in zip(data_te, labs_te):
                if img_arr.ndim == 2:
                    img_arr = np.stack([img_arr]*3, axis=2)
                clnt_test_x[i].append(self._preprocess(img_arr, img_size))
                clnt_test_y[i].append(lab)

            # sanity
            if len(clnt_x[i]) == 0:
                raise RuntimeError(f"No train samples for domain '{dom}'")
            if len(clnt_test_x[i]) == 0:
                raise RuntimeError(f"No test samples for domain '{dom}'")

        # stack into numpy arrays
        self.clnt_x      = [ np.stack(x).astype(np.float32)/255.0 for x in clnt_x ]
        self.clnt_y      = [ np.array(y, dtype=np.int64)      for y in clnt_y ]
        self.clnt_test_x = [ np.stack(x).astype(np.float32)/255.0 for x in clnt_test_x ]
        self.clnt_test_y = [ np.array(y, dtype=np.int64)      for y in clnt_test_y ]

        self.n_cls = n_cls

    @staticmethod
    def _preprocess(img_arr: np.ndarray, size: tuple):
        """
        img_arr: H×W×C in [0–255] → resized H'×W'×C in [0–255]
        """
        img = Image.fromarray(img_arr.astype(np.uint8))
        img = img.resize(size, Image.BICUBIC)
        return np.array(img)


class NumpyImageDataset(Dataset):
    """
    Wraps numpy images (N,H,W,C) + labels (N,) into a PyTorch Dataset.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, transforms=None):
        self.images     = images   # float32 [0,1]
        self.labels     = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]    # H,W,C in [0,1]
        lbl = self.labels[idx]
        if self.transforms:
            from PIL import Image as _Image
            img = _Image.fromarray((img*255).astype(np.uint8))
            img = self.transforms(img)
        else:
            img = torch.from_numpy(img).permute(2,0,1)
        return img, lbl
