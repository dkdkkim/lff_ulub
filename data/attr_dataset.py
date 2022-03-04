import os
import pickle
import torch
import json
import cv2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class AttributeDataset(Dataset):
    def __init__(self, root, split, query_attr_idx=None, transform=None):
        super(AttributeDataset, self).__init__()
        data_path = os.path.join(root, split, "images.npy")
        self.data = np.load(data_path)
        
        attr_path = os.path.join(root, split, "attrs.npy")
        self.attr = torch.LongTensor(np.load(attr_path))

        colors_path = os.path.join("./data", "resource", "colors.th")
        mean_color = torch.load(colors_path)
        attr_names_path = os.path.join(root, "attr_names.pkl")
        with open(attr_names_path, "rb") as f:
            self.attr_names = pickle.load(f)
        
        self.num_attrs =  self.attr.size(1)
        self.set_query_attr_idx(query_attr_idx)
        self.transform = transform
    
    def set_query_attr_idx(self, query_attr_idx):
        if query_attr_idx is None:
            query_attr_idx = torch.arange(self.num_attrs)
        
        self.query_attr = self.attr[:, query_attr_idx]
        
    def __len__(self):
        return self.attr.size(0)

    def __getitem__(self, index):
        image, attr = self.data[index], self.query_attr[index]
        if self.transform is not None:
            image = self.transform(image)

        return image, attr

class AttributeDatasetULUB(Dataset):
    def __init__(self, root, split, query_attr_idx=None, transform=None,
                    img_dir='/data/dk/ulub/dataset/celebA', loader=None):
        super(AttributeDatasetULUB, self).__init__()
        # data_path = os.path.join(root, split, "images.npy")
        # self.data = np.load(data_path)
        self.img_dir = img_dir
        self.loader = loader
        with open(os.path.join(root, split, "images.json"), 'r') as file:
            self.data = json.load(file)
        
        attr_path = os.path.join(root, split, "attrs.npy")
        self.attr = torch.LongTensor(np.load(attr_path))
        if len(self.attr.size()) == 1:
            self.attr = torch.unsqueeze(self.attr, 1)

        colors_path = os.path.join("./data", "resource", "colors.th")
        mean_color = torch.load(colors_path)
        attr_names_path = os.path.join(root, "attr_names.pkl")
        with open(attr_names_path, "rb") as f:
            self.attr_names = pickle.load(f)
        self.num_attrs =  self.attr.size(1)
        self.set_query_attr_idx(query_attr_idx)
        self.transform = transform
    
    def set_query_attr_idx(self, query_attr_idx):
        if query_attr_idx is None:
            query_attr_idx = torch.arange(self.num_attrs)
        
        self.query_attr = self.attr[:, query_attr_idx]
        
    def __len__(self):
        return self.attr.size(0)

    def __getitem__(self, index):
        image, attr = self.data[index], self.query_attr[index]
        if self.loader:
            image = self.loader(os.path.join(self.img_dir, image))
        else:
            image = Image.open(os.path.join(self.img_dir, image))
        if self.transform is not None:
            image = self.transform(image)

        return image, attr