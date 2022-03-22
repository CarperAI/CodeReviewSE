"""
Augmentations and factory function for augmentations.

"""

from helper import *
import torch
import numpy as np
from typing import Dict
import sys
import json
import multiprocessing
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial 


# specifies a dictionary of augmentations
_AUGS: Dict[str, any] = {}  # registry


def register_aug(name):
    """Decorator used register a an augmentation 
    Args:
        name: Name of the augmentation
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

    def __call__(self, data):
        # creates a multiprocess pool to apply the augmentation
        return list(map(self.rand_apply, data))

    def rand_apply(self, datum):
        """
        Calls apply on the individual apply functions after first sampling torch.rand()
        """
        if np.random.rand(1) < self.p:
            return self.apply(datum)
        else:
            return datum
    

    def apply(self, data):
        """
        Apply augmentation to the data.
        """
        raise NotImplementedError

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
        composition = load_json_file(composition_path)
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

class ApplyAugs:
    def __init__(self, augs):
        assert isinstance(augs, Compose), "augs must be composed together with `Compose`"
        self.augs = augs
    

    def _call_for_questions(self, data):
        self.keys = list(data.keys())

        # print the number of questions
        print("There are: ")
        print(str(len(self.keys)) + " questions")
        self.keys = [k for k in self.keys if '_q' in k]
        print(self.keys)

        # pass augmentations to iter_body
        iter_body_local = partial(iter_body, self.augs)

        # get each question's body
        bodies = list(map(lambda x: data[x]['body'], self.keys))

        # iterate over the bodies using multiprocessing
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            with tqdm(total=len(bodies)) as pbar:
                for i, body in enumerate(p.imap(iter_body_local, bodies)):
                    bodies[i] = body
                    # update the progress bar
                    pbar.update()

        # return the source sequence
        return bodies

    def _call_for_answers(self, data):
        self.keys = list(data.keys())
        self.keys = [k for k in self.keys if '_ans' in k]
        print(self.keys)

        iter_answers_local = partial(iter_body, self.augs)
        answer_bodies = list()

        # get each answer's body
        for question in tqdm(self.keys):
            for answer in tqdm(data[question]['answers']):
                answer_bodies.append(answer['body'])

        # iterate over the bodies using multiprocessing
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            with tqdm(total=len(answer_bodies)) as pbar:
                for i, body in enumerate(p.imap(iter_answers_local, answer_bodies)):
                    answer_bodies[i] = body
                    pbar.update()

        # return the target sequence
        return answer_bodies

    # copies back the output of __call__ to the original data
    def copy_back_question_bodies(self, outputs, data):
        """
        outputs is a dict, where the keys are question ids and the values are the augmented bodies.
        Args:
            outputs: list of lists
            data: dict
        Returns:
            data: dict
        """
        for idx, question in enumerate(self.keys):
            data[question]['body'] = outputs[idx]

        
        self.orig_keys = [k for k in data.keys() if '_q' not in k]

        for k in self.orig_keys:
            data[k]['body'] = ' '.join(parse_html_to_str(data[k]['body']))

        return data
    # copies back the output of __call__ to the original data
    def copy_back_answer_bodies(self, outputs, data):
        """
        outputs is a dict, where the keys are question ids. Each question has a sub-dict of answers.
        Args:
            outputs: list of lists
            data: dict
        Returns:
            data: dict
        """

        idx = 0
        for question in self.keys:
            for answer in data[question]['answers']:
                answer['body'] = outputs[idx]
                idx += 1
        
        self.orig_keys = [k for k in data.keys() if '_ans' not in k]
        
        for k in self.orig_keys:
            for answer in data[k]['answers']:
                answer['body'] = ' '.join(parse_html_to_str(answer['body']))

        return data   

    def __call__(self, data, for_question=True):
        if for_question:
            return self.copy_back_question_bodies(self._call_for_questions(data), data)
        else:
            return self.copy_back_answer_bodies(self._call_for_answers(data), data) 

@register_aug
class KeyboardAug(RandomAug):
    pass


@register_aug
class SpellingAug(RandomAug):
    def __init__(self, spelling_dict, include_reverse=True, p=0.5):
        super().__init__(p)
        self.spelling_dict = spelling_dict if type(spelling_dict) == dict else self.load_spelling_dict(spelling_dict, include_reverse)

    def load_spelling_dict(self, file_path, include_reverse=True):
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
                if include_reverse:
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
        words = i.split()
        rands = np.random.randint(2, size=len(words))
        for idx, word in enumerate(words):
            # Replace the word with the correct spelling
            if word not in self.spelling_dict:
                continue
            else:
                words[idx] = self.spelling_dict[word][min(len(self.spelling_dict[word]) - 1, rands[idx])]
        return ' '.join(words)

@register_aug
class BackTranslationAug(RandomAug):
    def __init__(self, src_model_name, tgt_model_name, device='cpu', batch_size=32, max_length=300, p=0.5):
        super().__init__(p)
        self.model = BackTranslationModel(src_model_name, tgt_model_name, device, batch_size, max_length)

    def apply(self, i):
        return self.model.translate(i)


# This processes the first two questions with the augmentation pipeline for testing purposes
if __name__ == "__main__":
    print(get_aug_names())
    data = load_json_file("dataset/CodeReviewSE.json")
    augs = Compose("configs/preproc_config.json")
    augs = ApplyAugs(augs)

    # get a subset of data with the first 2 questions (too slow for whole dataset, need multiprocessing speedup)
    data = {k: data[k] for k in list(data.keys())[:1]}

    data = duplicate_data(data, is_ans=False, n=1) # duplicate for augmenting questions 
    data = augs(data, for_question=True) # apply augmentations to questions 
    data = duplicate_data(data, is_ans=True, n=1) # duplicate for augmenting answers
    data = augs(data, for_question=False) # apply augmentations to answers

    # print the augmented data
    print(data['1']['body'])
    print(data['1_q0']['body'])
    print(data['1_q0_ans0']['body'])
    print(data['1']['answers'][0]['body'])
    print(data['1_q0']['answers'][0]['body'])
    print(data['1_q0_ans0']['answers'][0]['body'])

    
