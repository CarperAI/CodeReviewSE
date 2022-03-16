"""
Augmentations and factory function for augmentations.

"""

import torch
import numpy as np
from typing import Dict
import sys
import json
import multiprocess
# specifies a dictionary of architectures
_AUGS: Dict[str, any] = {}  # registry


def register_aug(name):
    """Decorator used register a CARP architecture
    Args:
        name: Name of the architecture
    """

    def register_class(cls, name):
        _AUGS[name] = cls
        setattr(sys.modules[__name__], name, cls)
        return cls

    if isinstance(name, str):
        name = name.lower()
        return lambda c: register_class(c, name)

    cls = name
    name = cls.__name__
    register_class(cls, name.lower())

    return cls

class RandomAug:
    """
    Randomly apply one of the augmentations.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data, threads=16):
        # creates a multiprocess pool to apply the augmentation
        pool = multiprocess.Pool(threads)
        aug_data = pool.map(self.rand_apply, data)
        pool.close()
        return aug_data

    def rand_apply(self, datum):
        """
        Calls apply on the individual apply functions after first sampling torch.rand()
        """
        if torch.rand(1) < self.p:
            return self.apply(datum)
        else:
            return datum
    

    def apply(self, data):
        """
        Apply augmentation to the data.
        """
        raise NotImplementedError

from random_aug import KeyboardAug, SpellingAug

def get_aug(name):
    return _AUGS[name.lower()]


def get_aug_names():
    return _AUGS.keys()


class Compose:
    """
    Compose several augmentations together. Based on torchvision's `Compose`
    """

    def __init__(self, composition_path):
        #  composition_path refers to a json file that contains the augmentations we're using
        self.augs = []
        with open(composition_path, 'r') as f:
            composition = json.load(f)
        for c in composition:
            self.augs.append(get_aug(c['name'])(**c['params']))

    def __call__(self, data):
        for t in self.augs:
            data = t(data)
        return data
    
    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.augs:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string







