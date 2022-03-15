"""
Augmentations and factory function for augmentations.

"""

from models.backtranslation import BackTranslationModel
import torch
import numpy as np


class Compose:
    """
    Compose several augmentations together. Based on torchvision's `Compose`
    """

    def __init__(self, augs):
        self.augs = augs

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



class RandomAug:
    """
    Randomly apply one of the augmentations.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        aug_data = []
        for i in range(len(data)):
            if torch.rand(1) < self.p:
                aug_data.append(self.apply(data[i]))        
            else:
                aug_data.append(data[i])
        return aug_data
    
    def apply(self, data):
        """
        Apply augmentation to the data.
        """
        raise NotImplementedError



class KeyboardAug(RandomAug):
    pass

class SpellingAug(RandomAug):
    def __init__(self, spelling_dict, p=0.5):
        super().__init__(p)
        self.spelling_dict = spelling_dict if type(spelling_dict) == dict else self.load_spelling_dict(spelling_dict)

    def load_spelling_dict(self, file_path):
        """
        Loads the spelling dictionary from the file.
        """
        spelling_dict = {}
        with open(file_path, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                tokens = line.split(' ')
                # Last token include newline separator
                tokens[-1] = tokens[-1].replace('\n', '')

                key = tokens[0]
                values = tokens[1:]

                if key not in spelling_dict:
                    spelling_dict[key] = []

                spelling_dict[key].extend(values)
                # Remove duplicate mapping
                spelling_dict[key] = list(set(spelling_dict[key]))
                # Build reverse mapping
                if self.include_reverse:
                    for value in values:
                        if value not in spelling_dict:
                         spelling_dict[value] = []
                        if key not in spelling_dict[value]:
                         spelling_dict[value].append(key)
        return spelling_dict

    def apply(self, i):
        """
        Apply augmentation to the element.
        """

        # Replace the word with the correct spelling
        if i not in self.dict:
            return i
        return self.dict[i]

            



class BackTranslationAug(RandomAug):
    def __init__(self, src_model_name, tgt_model_name, device='cpu', batch_size=32, max_length=300, p=0.5):
        super().__init__(p)
        self.model = BackTranslationModel(src_model_name, tgt_model_name, device, batch_size, max_length)

    def apply(self, i):
        """
        Apply augmentation to the element.
        """
        return self.model.translate(i)



