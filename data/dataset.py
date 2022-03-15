"""
PyTorch text `Dataset` for our code review dataset, along with factory functions for creating the dataset with desired pipeline.

Lot of this is placeholder code for now.

"""


import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CodeReviewDataset(Dataset):
    """
    PyTorch text `Dataset` for our code review dataset.

    TODO: is this a single Dataset class or will there be multiple for different tasks?
    """

    def __init__(self, 
        data_path,
        preproc_fn,
        augs=None
        ):
        """
        data_path(str) : path to the json file.
        preproc_fn(function) : function to apply to each element of the dataset.
        augs(callable) : transform to apply to the data.
        """
        self.data_path = data_path
        self.augs = augs
        self.data = self.load_data()

    def load_data(self):
        """
        Loads the data from the json file.
        """
        with open(self.data_path, "r") as f:
            return json.load(f)

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the item at the given index.
        """
        output = self.preproc_fn(self.data[idx])
        if self.augs:
            output = self.augs(output)
        return output
