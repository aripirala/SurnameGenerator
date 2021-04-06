# pylint: disable=no-member

import pandas as pd 
import numpy as np
from collections import Counter
import string
import os
import torch
from torch.utils.data import DataLoader
import json
import re
import sys
import torch.nn.functional as F
# from configs import args


class Vocabulary(object):
    """docstring for Vocabulary."""
    
    def __init__(self, token_to_idx=None):
        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx
        # print(f'len of token_to_idx is {len(token_to_idx)}')
        
        self._idx_to_token = {idx:token for token, idx in self._token_to_idx.items()}

        
    def add_token(self, token):
        """Update mapping dicuts based on the token

         Args:
             token (str): the item to add to add the Vocab
         Returns:
             index (int): index corresponding to the item in the vocab dictionary
        """        
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index 
    
    def to_serializable(self):
        """ Returns a dictionary that can be serialized
        """
        return {
            'token_to_idx': self._token_to_idx,
        }
    
    @classmethod
    def from_serializable(cls, contents):
        """Instantiates the Vocubulary from a serialized dictionary

        Args:
            contents (dict): a dictionary to create an instance of Vocabulary
        Returns: Instance of the Vocabulary class
        """

        return cls(**contents)

    def add_many(self, tokens):
        """Add a list of tokens to the vocab

        Args:
            tokens (list): List of tokens to be added to the vocab
        Returns:
            indices (list): Returns a list of indices for the tokens added
        """
        return [self.add_token(token) for token in tokens]
    
    def lookup_token(self, token):
        """Retrieve the index associated with the token 
          or the UNK index if token isn't present.
        
        Args:
            token (str): the token to look up 
        Returns:
            index (int): the index corresponding to the token
        
        """
       
        index = self._token_to_idx[token]
        return index 

    def lookup_index(self, index):
        """Return the token associated with the index
        
        Args: 
            index (int): the index to look up
        Returns:
            token (str): the token corresponding to the index
        Raises:
            KeyError: if the index is not in the Vocabulary
        """
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)

class SequenceVocabulary(Vocabulary):
    def __init__(self, token_to_idx=None, unk_token="@",
                 mask_token="_", begin_seq_token="!",
                 end_seq_token="$"):
        super(SequenceVocabulary, self).__init__(token_to_idx)

        self._mask_token = mask_token
        self._unk_token = unk_token
        self._begin_seq_token = begin_seq_token
        self._end_seq_token = end_seq_token

        self.unk_index = -1

        self.mask_index = self.add_token(self._mask_token)
        self.unk_index = self.add_token(self._unk_token)
        self.begin_seq_index = self.add_token(self._begin_seq_token)
        self.end_seq_index = self.add_token(self._end_seq_token)

    def to_serializable(self):
        contents = super(SequenceVocabulary, self).to_serializable()
        contents.update({'unk_token': self._unk_token,
                         'mask_token': self._mask_token,
                         'begin_seq_token': self._begin_seq_token,
                         'end_seq_token': self._end_seq_token})
        return contents
    
    def lookup_token(self, token):
        """Retrieve the index associated with the token 
          or the UNK index if token isn't present.
        
        Args:
            token (str): the token to look up 
        Returns:
            index (int): the index corresponding to the token
        Notes:
            `unk_index` needs to be >=0 (having been added into the Vocabulary) 
              for the UNK functionality 
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

class SurnameVectorizer(object):
    """docstring Surname_ SurnameVectorizer."""
    def __init__(self, surname_vocab, nationality_vocab, vector_type='one_hot', max_len=None):
        """

        Args:
            surname_vocab (SequenceVocabulary): maps words to integers
            nationality_vocab (Vocabulary): maps ratings to integers            
        """
        self.surname_vocab = surname_vocab
        self.nationality_vocab = nationality_vocab
        self.max_len = max_len
        self.vector_type = vector_type
    
    def vectorize(self, surname):
        """Create a collapsed one-hot code vector fo the surname

        Args:
            surname (str): the surname in the str format
        Returns:
            one_hot (np.ndarray): collapsed one-hot encoding
        """

        if self.vector_type == 'one_hot':
            return self.vectorize_onehot(surname)
        
        return self.vectorize_embedding(surname)
        
    def vectorize_embedding(self, surname):
        """Vectorize a surname into a vector of observations and targets
        
        The outputs are the vectorized surname split into two vectors:
            surname[:-1] and surname[1:]
        At each timestep, the first vector is the observation and the second vector is the target. 
        
        Args:
            surname (str): the surname to be vectorized
        Returns:
            a tuple: (from_vector, to_vector)
            from_vector (numpy.ndarray): the observation vector 
            to_vector (numpy.ndarray): the target prediction vector
        """
        indices = [self.surname_vocab.begin_seq_index] 
        indices.extend(self.surname_vocab.lookup_token(token) for token in surname)
        indices.append(self.surname_vocab.end_seq_index)

        if self.max_len < 0:
            self.max_len = len(indices) - 1

        from_vector = np.zeros(self.max_len, dtype=np.int64)         
        from_indices = indices[:-1]
        from_vector[:len(from_indices)] = from_indices
        from_vector[len(from_indices):] = self.surname_vocab.mask_index

        to_vector = np.zeros(self.max_len, dtype=np.int64)
        to_indices = indices[1:]
        to_vector[:len(to_indices)] = to_indices
        to_vector[len(to_indices):] = self.surname_vocab.mask_index
        
        return from_vector, to_vector
    
    def vectorize_onehot(self, surname):
        #TODO: need to refactor to handle new SequenceVocabulary Class

        """Create a collapsed one-hot code vector fo the surname

        Args:
            surname (str): the surname in the str format
        Returns:
            one_hot (np.ndarray): collapsed one-hot encoding
        """
        one_hot = np.zeros(len(self.surname_vocab), dtype=np.float32)

        for token in surname:
            index = self.surname_vocab.lookup_token(token) # get index for the token from the vocab class
            one_hot[index] = 1
        return one_hot, len(self.surname_vocab)
    

    @classmethod
    def from_dataframe(cls, surnames_df, cutoff=25, vector_type='one_hot', max_len=None):
        """
        Instantiate the vectorizer from the surname pandas dataframe

        :param surname_df (pandas df): the surname dataset
        :param cutoff (int): the parameter that controls the threshold for the token to be added to the vocab
        :return:
        """

        surname_vocab = SequenceVocabulary()
        nationality_vocab = Vocabulary()

        # Add ratings
        for nationality in sorted(set(surnames_df.nationality)):
            nationality_vocab.add_token(nationality)

        if max_len is None:
            max_surname_len = 0

        for _ , row in surnames_df.iterrows():
            surname_len = len(row.surname)
            if max_len is None:
                if max_surname_len < surname_len:
                    max_surname_len = surname_len
            for char in row.surname:
                surname_vocab.add_token(char)
            nationality_vocab.add_token(row.nationality)

        if max_len is None:
            max_len = max_surname_len + 2

        return cls(surname_vocab, nationality_vocab, vector_type, max_len)

    def to_serializable(self):
        """
        Create the serializable dictionary for caching
        :return:
            contents (dict): the contents of the class in the form of a dictionary
        """

        return {
            'surname_vocab': self.surname_vocab.to_serializable(),
            'nationality_vocab': self.nationality_vocab.to_serializable(),
            'vector_type':self.vector_type,
            'max_len': self.max_len,
        }

    @classmethod
    def from_serializable(cls, contents):
        """Instantiate a SurnameVectorizer from a serializable dictionary

        Args:
            contents (dict): the serializable dictionary
        Returns:
            an instance of the SurnameVectorizer class
        """

        surname_vocab = SequenceVocabulary.from_serializable(contents['surname_vocab'])
        nationality_vocab = Vocabulary.from_serializable(contents['nationality_vocab'])
        vector_type = contents['vector_type']
        max_len = contents['max_len']

        return cls(surname_vocab, nationality_vocab, vector_type, max_len)

    @staticmethod
    def load_vectorizer_only(vectorizer_pth):
        """

        :param vectorizer_pth (str): location of the serialized vectorizer
        :return:
        """
        with open(vectorizer_pth, 'r') as fp:
            return json.load(fp)

    @classmethod
    def from_serializable_and_json(cls, vectorizer_pth):
        """Instantiate a SurnameVectorizer from a serializable dictionary

        Args:
            contents (dict): the serializable dictionary
        Returns:
            an instance of the SurnameVectorizer class
        """
        vectorizer_dict = cls.load_vectorizer_only(vectorizer_pth)
        surname_vocab = SequenceVocabulary.from_serializable(vectorizer_dict['surname_vocab'])
        nationality_vocab = Vocabulary.from_serializable(vectorizer_dict['nationality_vocab'])
        vector_type = vectorizer_dict['vector_type']
        max_len = vectorizer_dict['max_len']

        return cls(surname_vocab, nationality_vocab, vector_type, max_len)

def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"):
    """
    A generator function which wraps the PyTorch DataLoader. It will
      ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name in data_dict.keys():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict

def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': args.model_state_file}

def update_train_state(args, model, train_state):
    """Handle the training state updates.

    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better

    :param args: main arguments
    :param model: model to train
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """

    # Save one model at least
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        train_state['stop_early'] = False

    # Save model if performance improved
    elif train_state['epoch_index'] >= 1:
        _ , loss_t = train_state['val_loss'][-2:]

        # If loss worsened
        if loss_t >= train_state['early_stopping_best_val']:
            # Update step
            train_state['early_stopping_step'] += 1
        # Loss decreased
        else:
            # Save the best model
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(model.state_dict(), train_state['model_filename'])

            # Reset early stopping step
            train_state['early_stopping_step'] = 0

        # Stop early ?
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state

def compute_accuracy(y_pred, y_true, output_type='one_class', mask_index=0):
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()
    y_pred, y_true = normalize_sizes(y_pred, y_true)

    # print("in compute accuracy")
    # print(f'y_pred shape is {y_pred.size()}')
    # print(f'y_true shape is {y_true.size()}')

    if output_type=='one_class':
        y_pred_indices = (torch.sigmoid(y_pred)>0.5).long()#.max(dim=1)[1]
    else:
        _, y_pred_indices = y_pred.max(dim=1)
    
    correct_indices = torch.eq(y_pred_indices, y_true).float()
    valid_indices = torch.ne(y_true, mask_index).float()

    # print(f'correct indices shape is {correct_indices.shape}')
    # print(f'valid indices shape is {valid_indices.shape}')

    n_correct = (correct_indices*valid_indices).sum().item()
    n_valid = valid_indices.sum().item()

    return n_correct / n_valid * 100

def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text

def column_gather(y_out, x_lengths):
    '''Get a specific vector from each batch datapoint in `y_out`.

    More precisely, iterate over batch row indices, get the vector that's at
    the position indicated by the corresponding value in `x_lengths` at the row
    index.

    Args:
        y_out (torch.FloatTensor, torch.cuda.FloatTensor)
            shape: (batch, sequence, feature)
        x_lengths (torch.LongTensor, torch.cuda.LongTensor)
            shape: (batch,)

    Returns:
        y_out (torch.FloatTensor, torch.cuda.FloatTensor)
            shape: (batch, feature)
    '''
    x_lengths = x_lengths.long().detach().cpu().numpy() - 1

    # print(f'In Column Gather: y out shape is {y_out.size()}')
    # sys.exit()
    out = []
    for batch_index, column_index in enumerate(x_lengths):
        out.append(y_out[batch_index, column_index])

    return torch.stack(out)

def sample_from_model(model, vectorizer, nationalities = None, num_samples=1, sample_size=20, 
                      temperature=1.0):
    """Sample a sequence of indices from the model
    
    Args:
        model (SurnameGenerationModel): the trained model
        vectorizer (SurnameVectorizer): the corresponding vectorizer
        nationalities (list): a list of integers representing nationalities
        num_samples (int): the number of samples
        sample_size (int): the max length of the samples
        temperature (float): accentuates or flattens 
            the distribution. 
            0.0 < temperature < 1.0 will make it peakier. 
            temperature > 1.0 will make it more uniform
    Returns:
        indices (torch.Tensor): the matrix of indices; 
        shape = (num_samples, sample_size)
    """
    h_t = None
    if nationalities != None:
        num_samples = len(nationalities)
        nationality_indices = torch.tensor(nationalities, dtype=torch.int64).unsqueeze(dim=0)
        h_t = model.nationality_emb(nationality_indices)
    begin_seq_index = [vectorizer.surname_vocab.begin_seq_index 
                       for _ in range(num_samples)]
    begin_seq_index = torch.tensor(begin_seq_index, 
                                   dtype=torch.int64).unsqueeze(dim=1)
    indices = [begin_seq_index]
    
    
    for time_step in range(sample_size):
        x_t = indices[time_step]
        x_emb_t = model.embedding(x_t)
        rnn_out_t, h_t = model.rnn(x_emb_t, h_t)
        prediction_vector = model.fc2(rnn_out_t.squeeze(dim=1))
        probability_vector = F.softmax(prediction_vector / temperature, dim=1)
        indices.append(torch.multinomial(probability_vector, num_samples=1))
    indices = torch.stack(indices).squeeze().permute(1, 0)
    return indices

def decode_samples(sampled_indices, vectorizer):
    """Transform indices into the string form of a surname
    
    Args:
        sampled_indices (torch.Tensor): the inidces from `sample_from_model`
        vectorizer (SurnameVectorizer): the corresponding vectorizer
    """
    decoded_surnames = []
    vocab = vectorizer.surname_vocab
    
    for sample_index in range(sampled_indices.shape[0]):
        surname = ""
        for time_step in range(sampled_indices.shape[1]):
            sample_item = sampled_indices[sample_index, time_step].item()
            if sample_item == vocab.begin_seq_index:
                continue
            elif sample_item == vocab.end_seq_index:
                break
            else:
                surname += vocab.lookup_index(sample_item)
        decoded_surnames.append(surname)
    return decoded_surnames

def normalize_sizes(y_pred, y_true):
    """Normalize tensor sizes
    
    Args:
        y_pred (torch.Tensor): the output of the model
            If a 3-dimensional tensor, reshapes to a matrix
        y_true (torch.Tensor): the target predictions
            If a matrix, reshapes to be a vector
    """
    if len(y_pred.size()) == 3:
        y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
    if len(y_true.size()) == 2:
        y_true = y_true.contiguous().view(-1)
    return y_pred, y_true

def sequence_loss(y_pred, y_true, mask_index):
    y_pred, y_true = normalize_sizes(y_pred, y_true)
    # print("in sequence loss")
    # print(f'y_pred shape is {y_pred.size()}')
    # print(f'y_true shape is {y_true.size()}')
    # print(y_true)
    return F.cross_entropy(y_pred, y_true, ignore_index=mask_index)
