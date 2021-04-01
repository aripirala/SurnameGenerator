# pylint: disable=no-member
# pylint: disable=E1101
# pylint: disable=E1102

import torch
from utils import preprocess_text

from model import SurnameRNN_Embed_Generator
from utils import decode_samples, SurnameVectorizer, sample_from_model
from configs import args
import os

import torch.nn.functional as F



if __name__ == '__main__':
    
    num_names = 10
    #get the model and vectorizer paths
    model_pth = os.path.join(args.save_dir, args.model_state_file)
    vectorizer_pth = os.path.join(args.save_dir, args.vectorizer_file)

    #load vectorizer before loading the model
    vectorizer = SurnameVectorizer.from_serializable_and_json(vectorizer_pth)
    print(f'Length of nationality vocab is {len(vectorizer.nationality_vocab)}')

    model = SurnameRNN_Embed_Generator(num_features=len(vectorizer.surname_vocab), 
                vocab_size=len(vectorizer.surname_vocab), rnn_hidden_size=200,
                embedding_file_name=args.embedding_file_name, embedding_dim=args.embedding_dim,  
                word_to_index=vectorizer.surname_vocab._token_to_idx, max_idx=len(vectorizer.surname_vocab),
                freeze=True, batch_norm=True, dropout=True, activation_fn='RELU')
    model.load_state_dict(torch.load(model_pth))
    model = model.cpu()

    
    sampled_surnames = decode_samples(sample_from_model(model, vectorizer, num_samples=num_names), 
                        vectorizer)
    # Show results
    print ("-"*15)
    for i in range(num_names):
        print (sampled_surnames[i])