import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data import Sampler
from torchvision import transforms as T
from torchvision.datasets.celeba import CelebA
from data.attr_dataset import AttributeDataset, AttributeDatasetULUB
from functools import reduce
import math
import warnings

from loader_ulub.imagenet_dataset import pil_loader


class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])


class ZippedDataset(Dataset):
    def __init__(self, datasets):
        super(ZippedDataset, self).__init__()
        self.dataset_sizes = [len(d) for d in datasets]
        self.datasets = datasets

    def __len__(self):
        return max(self.dataset_sizes)

    def __getitem__(self, idx):
        items = []
        for dataset_idx, dataset_size in enumerate(self.dataset_sizes):
            items.append(self.datasets[dataset_idx][idx % dataset_size])

        item = [torch.stack(tensors, dim=0) for tensors in zip(*items)]

        return item

    
transforms = {
    "ColoredMNIST": {
        "train": T.Compose([T.ToTensor()]),
        "eval": T.Compose([T.ToTensor()])
        },
    "CorruptedCIFAR10": {
        "train_aug": T.Compose(
            [
                T.ToPILImage(),
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "train": T.Compose(
            [
                T.ToPILImage(),
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "eval": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
    },
    "Shapes3D": {
        "train": T.Compose([
            T.ToTensor(),
            T.ToPILImage(),
            T.Resize((32, 32)),
            T.ToTensor(),
        ]),
        "eval": T.Compose([
            T.ToTensor(),
            T.ToPILImage(),
            T.Resize((32, 32)),
            T.ToTensor(),
        ]),
    },
    "CelebA": {
        "train": T.Compose(
            [
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
            ]
        ),
        "eval": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    },
    "celeba_ulub": {
        "train": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
            ]
        ),
        "eval": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    },
    "utkface": {
        "train": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
            ]
        ),
        "eval": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    },"imagenet": {
        "train": T.Compose(
            [   
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
            ]
        ),
        "eval": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }
}
from PIL import Image
def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def get_dataset(dataset_tag, data_dir, dataset_split, transform_split):
    dataset_category = dataset_tag.split("-")[0]
    root = os.path.join(data_dir, dataset_tag)
    transform = transforms[dataset_category][transform_split]
    dataset_split = "valid" if (dataset_split == "eval") else dataset_split
    if dataset_tag == "CelebA":
        celeba_root = '/home/xxxx/datasets/CelebA'
        dataset = CelebA(
            root=celeba_root,
            split=dataset_split,
            target_type="attr",
            transform=transform,
        )
    elif dataset_tag == "celeba_ulub":
        dataset = AttributeDatasetULUB(
            root='dataset_ulub/CelebA-HQ', split=dataset_split, transform=transform,
            img_dir=data_dir
        )
    elif dataset_tag == "utkface":
        dataset = AttributeDatasetULUB(
            root='dataset_ulub/utkface', split=dataset_split, transform=transform,
            img_dir=data_dir
        )
    elif dataset_tag == "imagenet":
        dataset = AttributeDatasetULUB(
            root='dataset_ulub/imagenet', split=dataset_split, transform=transform,
            img_dir=data_dir, loader=pil_loader
        )
    else:
        dataset = AttributeDataset(
            root=root, split=dataset_split, transform=transform
        )

    return dataset
