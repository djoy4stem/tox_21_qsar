from random import shuffle
from typing import List
import math

from torch_geometric.data import  Data
from torch_geometric.loader import DataLoader

def split_data(dataset:List[Data], split_indices:List[List[int]]=None, train_ratio:float=0.8
          , val_ratio:float=None, test_ratio:float=0.2, shuffle_dataset:bool=True):
    train_data, val_data, test_data = None, None, None
    total_size = len(dataset)

    if split_indices is None:
        indices = list(range(total_size))
        # print(indices)
        if shuffle_dataset:
            shuffle(indices)
        # print(indices)

        train_size, val_size, test_size = 0, None, 0
        train_size = int(total_size *  train_ratio)
        train_data = dataset[ :train_size]

        if val_ratio is None:
            assert math.isclose(0.9999999, train_ratio + test_ratio, rel_tol=1e-06, abs_tol=1e-06), f"train_ratio + test_ratio must be equals 1, not {train_ratio + test_ratio}."
            test_size = total_size - train_size
            test_data = dataset[train_size : ]
        else:
            assert math.isclose(0.9999999, train_ratio + val_ratio + test_ratio, rel_tol=1e-06, abs_tol=1e-06), f"train_ratio + val_ratio + test_ratio must be equals 1, not {train_ratio + val_ratio + test_ratio}."
            val_size  = int(total_size * val_ratio)
            val_data  = dataset[train_size : train_size + val_size]
            test_data = dataset[train_size + val_size : ]
            

    elif len(split_indices) == 2:
        train_data = dataset[split_indices[0]]
        test_data  = dataset[split_indices[1]]
    else:
        train_data = dataset[split_indices[0]]
        val_data   = dataset[split_indices[1]] 
        test_data  = dataset[split_indices[2]]       

    return train_data, val_data, test_data


def get_dataloader(dataset:List[Data], batch_size=128, shuffle_dataset:bool=False):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle_dataset=shuffle_dataset)

def get_dataloaders(train_data:List[Data], test_data:List[Data], val_data:List[Data]=None, batch_size=128, shuffle_train:bool=False):
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle_train)
    test_dataloader  = DataLoader(dataset=test_data, batch_size=batch_size)
    if val_data is None:
        return train_dataloader, test_dataloader
    else:
        val_dataloader = DataLoader(dataset=val_data, batch_size=batch_size)
        return train_dataloader, val_dataloader, test_dataloader