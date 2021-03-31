from utils import Vocabulary, SurnameVectorizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataset
import pandas as pd
import numpy as np
import json
# import configs


class SurnameDataset:
    def __init__(self, surnames_df, vectorizer):
        """

        :param review_df (pandas Dataframe): the dataset containing the reviews and rating
        :param vectorizer (ReviewVectorizer): vectorizer instantiated from dataset
        """
        self.surnames_df = surnames_df
        self.vectorizer = vectorizer

        self.train_df = self.surnames_df[self.surnames_df.split=='train']
        self.train_size = len(self.train_df)

        self.val_df = self.surnames_df[self.surnames_df.split == 'val']
        self.val_size = len(self.val_df)

        self.test_df = self.surnames_df[self.surnames_df.split == 'test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {
            'train': (self.train_df, self.train_size),
            'val': (self.val_df, self.val_size),
            'test': (self.test_df, self.test_size)
        }

        self.set_split('train')

    @classmethod
    def load_dataset_and_make_vectorizer(cls, surnames_csv, vector_type='one_hot', max_len=None):
        """Load dataset and make a new vectorizer from scratch

        :param surnames_csv (str): location of dataset
        :return:
            An instance of SurnamesDataset
        """
        surnames_df = pd.read_csv(surnames_csv)
        train_surnames_df = surnames_df[surnames_df.split=='train']
        vectorizer = SurnameVectorizer.from_dataframe(train_surnames_df, vector_type=vector_type, max_len=max_len)
        print(f'Vectorizer created')
        return cls(surnames_df, vectorizer)

    @classmethod
    def load_dataset_and_load_vectorizer(cls, surnames_csv, vectorizer_pth):
        """

        :param surnames_csv (str): location of the dataset
        :param vectorizer_pth (str): location of the serialized vectorizer
        :return: An instance of SurnameDataset
        """
        surnames_df = pd.read_csv(surnames_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_pth)

        return cls(surnames_df, vectorizer)

    @staticmethod
    def load_vectorizer_only(vectorizer_pth):
        """

        :param vectorizer_pth (str): location of the serialized vectorizer
        :return:
        """
        with open(vectorizer_pth, 'r') as fp:
            return SurnameVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        """
        Saves the vectorizer to disk using json format

        :param vectorizer_filepath (str): the location to save the vectorizer

        """
        print(f'vectorizer path is {vectorizer_filepath}')
        with open(vectorizer_filepath, 'w') as fp:
            json.dump(self.vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        """
        Returns the vectorizer
        :return:
        """
        return self.vectorizer

    def set_split(self, split='train'):
        """ selects the splits in the dataset using a column in the dataframe

        Args:
            split (str): one of "train", "val", or "test"
        """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, idx):
        """the primary entry point method for PyTorch datasets

        Args:
            index (int): the index to the data point
        Returns:
            a dictionary holding the data point's features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[idx]

        # print(f'row is {row}')

        surname_vector, vector_len = self.vectorizer.vectorize(row.surname)
        nationality_index = self.vectorizer.nationality_vocab.lookup_token(row.nationality)

        return {
            'x_data': surname_vector,
            'y_target': nationality_index,
            'vector_len': vector_len
        }

    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset

        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size

if __name__ == '__main__':
    file_path = '../input/surnames_with_splits.csv'
    print(f'file path is {file_path}')
    surname_dataset = SurnameDataset.load_dataset_and_make_vectorizer(file_path, vector_type='embedding', max_len=25)
    train_dataset = surname_dataset
    print(f'max_len is {train_dataset.vectorizer.max_len}')
    print(f'Training dataset has {len(train_dataset)}')
    print('First five items are --')
    for i in range(5):
        x, y, surname_len = train_dataset[i]['x_data'], train_dataset[i]['y_target'], train_dataset[i]['vector_len'] 
        print(f'data {i+1}...\n\t{x}\ntarget -\t{y}\nsurname_length -\t{surname_len}')

    surname_dataset.set_split('val')
    print(f'Validation dataset has {len(surname_dataset)}')
    print('Two items are --')
    for i in range(2):
        x, y = surname_dataset[i]['x_data'], surname_dataset[i]['y_target']
        print(f'data {i+1}...\n\t{x}\n\t{y}')

    surname_dataset.set_split('train')
    print(f'Training dataset has {len(surname_dataset)}')

    surname_dataset.set_split('test')
    print(f'Test dataset has {len(surname_dataset)}')

    word_idx = surname_dataset.vectorizer.nationality_vocab._token_to_idx
    print(word_idx)

