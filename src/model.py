# pylint: disable=no-member

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import timeit
import datetime
import numpy as np
import sys
from utils import column_gather


class Embedding:
    def __init__(self, num_features, embedding_dim=100, embedding_type='pre-trained', embedding_file_name=None, 
                    word_to_index=None, max_idx=1000, freeze=True, **kwargs):
        
        super().__init__(**kwargs)
        
        self.embedding_dim= embedding_dim

        if embedding_type == 'pre-trained':
            self.emb_matrix = self.create_embedding_matrix(embedding_file_name, word_to_index, max_idx)
            self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.emb_matrix), freeze=freeze)
            print(f'Embeddings loaded - shape is {self.emb_matrix.shape}')
            # sys.exit()
        else:
            self.embedding = nn.Embedding(num_features, embedding_dim)

    def create_embedding_matrix(self, embeddings_file_name, word_to_index, max_idx, sep=' ', init='zeros', print_each=10000, verbatim=False):
        # Initialize embeddings matrix to handle unknown words
        if init == 'zeros':
            embed_mat = np.zeros((max_idx + 1, self.embedding_dim))
        elif init == 'random':
            embed_mat = np.random.rand(max_idx + 1, self.embedding_dim)
        else:
            raise Exception('Unknown method to initialize embeddings matrix')
        
        start = timeit.default_timer()
        with open(embeddings_file_name) as infile:
            for idx, line in enumerate(infile):
                elem = line.split(sep)
                word = elem[0]

                if verbatim is True:
                    if idx % print_each == 0:
                        print('[{}] {} lines processed'.format(datetime.timedelta(seconds=int(timeit.default_timer() - start)), idx), end='\r')

                if word not in word_to_index:
                    continue

                word_idx = word_to_index[word]

                if word_idx <= max_idx:
                    embed_mat[word_idx] = np.asarray(elem[1:], dtype='float32')


        if verbatim == True:
            print()

        return embed_mat
    

class SurnameCNN_Embed_Classifier(Embedding, nn.Module):
    def __init__(self, num_features, num_classes, channel_list, activation_fn = 'RELU', max_pool=False,
                    embedding_dim=100, embedding_type=None, embedding_file_name=None, 
                    word_to_index=None, max_idx=1000, freeze=True, batch_norm=False, dropout=False, **kwargs):

        super(SurnameCNN_Embed_Classifier, self).__init__(num_features=num_features, embedding_dim=embedding_dim, 
                    embedding_type=embedding_type, embedding_file_name=embedding_file_name, 
                    word_to_index=word_to_index, max_idx=max_idx, freeze=freeze, **kwargs)
        
        layers = []
        
        in_channels = embedding_dim

        if channel_list is not None:
            for out_channels in channel_list:
                layers.append(nn.Conv1d(in_channels=in_channels,
                                        out_channels=out_channels, kernel_size=3))
                if activation_fn == 'RELU':
                    layers.append(nn.ReLU())
                if activation_fn == 'ELU':
                    layers.append(nn.ELU())
                if max_pool:
                    layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
                if batch_norm:
                    layers.append(nn.BatchNorm1d(out_channels))
                if dropout:
                    layers.append(nn.Dropout(p=0.5))
                
                in_channels = out_channels
              
        self.cnn_layers = nn.Sequential(*layers)
        self.fc = nn.Linear(in_channels, num_classes) 

    def forward(self, x_in, apply_sigmoid=False):
        # print(f'x_input shape is {x_in.size()}')
        x_in = x_in.to(torch.long)
        embed_out = self.embedding(x_in) # => batch_size x seq_len x emb_dim
        embed_out = embed_out.permute(0, 2, 1) # => batch_size x emb_dim x seq_len
        # print(f'embedding shape is {embed_out.size()}')
        # embed_out,_ = torch.max(embed_out, dim=2)
        # print(f'embedding shape is {embed_out.size()}')
        
        # sys.exit()

        cnn_output = self.cnn_layers(embed_out.float())
        # print(f'cnn output shape is {cnn_output.size()}')
        cnn_output = torch.mean(cnn_output, dim=2)
        # print(f'cnn output shape is {cnn_output.size()}')
        # sys.exit()
        logits = self.fc(cnn_output).squeeze()

        if apply_sigmoid:
            logits = torch.sigmoid(logits)
        return logits

class SurnameRNN_Embed_Generator(Embedding, nn.Module):
    def __init__(self, num_features, vocab_size, rnn_hidden_size, activation_fn = 'RELU', 
                    embedding_dim=100, embedding_type=None, embedding_file_name=None, 
                    word_to_index=None, max_idx=1000, freeze=True, batch_norm=False, 
                    batch_first=True, dropout=False, conditional=False, conditional_class_count = None, **kwargs):

        super(SurnameRNN_Embed_Generator, self).__init__(num_features=num_features, embedding_dim=embedding_dim, 
                    embedding_type=embedding_type, embedding_file_name=embedding_file_name, 
                    word_to_index=word_to_index, max_idx=max_idx, freeze=freeze, **kwargs)
        
        self.rnn = nn.GRU(input_size=embedding_dim,
                             hidden_size=rnn_hidden_size,
                             batch_first=batch_first)
        self.fc1 = nn.Linear(in_features=rnn_hidden_size,
                         out_features=rnn_hidden_size)
        self.fc2 = nn.Linear(in_features=rnn_hidden_size,
                          out_features=vocab_size)
        self.conditional = conditional

        self.nationality_emb = nn.Embedding(conditional_class_count, rnn_hidden_size)

    def forward(self, x_in, nationality_index=None, apply_softmax=False):
        """The forward pass of the classifier
        
        Args:
            x_in (torch.Tensor): an input data tensor. 
                x_in.shape should be (batch, input_dim)
            x_lengths (torch.Tensor): the lengths of each sequence in the batch.
                They are used to find the final vector of each sequence
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, output_dim)
        """
        # print(f'x_input shape is {x_in.size()}')
        x_in = x_in.to(torch.long)
        x_embed = self.embedding(x_in) # => batch_size x seq_len x emb_dim
        
        # print(f'embedding shape is {x_embed.size()}')
        # embed_out,_ = torch.max(embed_out, dim=2)
        # print(f'embedding shape is {embed_out.size()}')
        
        # sys.exit()

        h_n = None
        if self.conditional:
            if nationality_index==None:
                raise Exception("Nationality index cannot be None")
            else:
                h_n = self.nationality_emb(nationality_index).unsqueeze(0) # => batch_size x rnn_hidden_size

        y_out, _ = self.rnn(x_embed.float(), h_n)

        # print(f'rnn output shape is {y_out.size()}')
        # print(f'x_lengths shape is {x_lengths.size()}')
        
        # sys.exit()
        # reshape the y_out into a 2D matrix into (-1, feat_size) ie bs*seq_len x feat_size
        batch_size, seq_len, feat_size = y_out.shape
        y_out = y_out.contiguous().view(batch_size*seq_len, feat_size)
        
        # y_out = F.relu(self.fc1(F.dropout(y_out, 0.5)))
        y_out = self.fc2(F.dropout(y_out, 0.5)).squeeze()

        if apply_softmax:
            y_out = F.softmax(y_out, dim=1)
        
        feat_size = y_out.shape[-1]
        # print(f'Feature size is {feat_size}')
       # reshape it back to batch_size x seq_len x feat_size
        y_out = y_out.view(batch_size, seq_len, feat_size)
        # print(f'y_out size is {y_out.shape}')
        # sys.exit()
        return y_out

if __name__ == '__main__':
    rnn_hidden_size = 200
    seq_len = 20
    vocab_size = 52
    bs = 32
    embed_dim = 50

    RNNGenerator = SurnameRNN_Embed_Generator(200, vocab_size, rnn_hidden_size, embedding_dim=embed_dim)
    x_in = torch.empty((bs, seq_len), dtype=torch.long).random_(10)
    print(x_in)
    print(RNNGenerator)

    y_out = RNNGenerator(x_in)

    