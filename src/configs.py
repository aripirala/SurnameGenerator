# import transformers
# import tokenizers
# pylint: disable=no-member

import os
from argparse import Namespace
from model import SurnameMLPClassifier, SurnamePerceptronClassifier, SurnameMLP_Embed_Classifier, SurnameCNN_Embed_Classifier, SurnameRNN_Embed_Classifier
from dataset import SurnameDataset
from torch.nn.modules.dropout import Dropout
from utils import handle_dirs

args = Namespace(
    # Data and Path information
    frequency_cutoff=25,
    model_state_file='model_rnn_embed20_1.pth',
    data_csv='../input/surnames_with_splits.csv',
    save_dir='../experiment/RNN/',
    vectorizer_file='vectorizer.json',
    classifier=None,
    vectorizer = None,
    dataset=None,
    architecture_type=None,
    output_type='multi_class',
    # No Model hyper parameters
    # Training hyper parameters
    batch_size=32,
    early_stopping_criteria=5,
    learning_rate=0.001,
    num_epochs=25,
    seed=1337,
    # Runtime options
    catch_keyboard_interrupt=True,
    cuda=False,
    expand_filepaths_to_save_dir=True,
    reload_from_files=False,
    train=True, # Flag to train your network
    # If embedding layer is used
    max_len = 15,
    vector_type='embedding', # 'one_hot'
    embedding_type = 'train', #'pre-trained',
    embedding_file_name= '../input/glove.6B.50d.txt',
    embedding_dim=20
)
# handle dirs
handle_dirs(args.save_dir)

vectorizer_pth = os.path.join(args.save_dir, args.vectorizer_file)
if args.reload_from_files:
        # training from a checkpoint
        print("Loading dataset and vectorizer")        
        dataset = SurnameDataset.load_dataset_and_load_vectorizer(args.data_csv,
                                                                 vectorizer_pth)
else:
        print("Loading dataset and creating vectorizer")
        # create dataset and vectorizer
        dataset = SurnameDataset.load_dataset_and_make_vectorizer(args.data_csv, 
                                                                args.vector_type, 
                                                                args.max_len)
        dataset.save_vectorizer(vectorizer_pth)

vectorizer = dataset.get_vectorizer()


# classifier = SurnamePerceptronClassifier(num_features=len(vectorizer.surname_vocab), num_classes=len(vectorizer.nationality_vocab))
# args.architecture_type = 'Perceptron'

# classifier = SurnameMLPClassifier(num_features=len(vectorizer.surname_vocab), num_classes=len(vectorizer.nationality_vocab), hidden_layer_dim=[200, 100])

# classifier = SurnameMLP_Embed_Classifier(num_features=len(vectorizer.surname_vocab), num_classes=1, hidden_layer_dim=[100, 50],
#                 embedding_file_name=args.embedding_file_name, embedding_dim=50,  
#                 word_to_index=vectorizer.surname_vocab._token_to_idx, max_idx=len(vectorizer.surname_vocab),
#                 freeze=False)
# args.architecture_type = 'MLP'

# classifier = SurnameCNN_Embed_Classifier(num_features=len(vectorizer.surname_vocab), 
#                 num_classes=len(vectorizer.nationality_vocab), channel_list=[100, 200],
#                 embedding_file_name=args.embedding_file_name, embedding_dim=args.embedding_dim,
#                 embedding_type = args.embedding_type,  
#                 word_to_index=vectorizer.surname_vocab._token_to_idx, max_idx=len(vectorizer.surname_vocab),
#                 freeze=False, batch_norm=True, dropout=True, max_pool=True, activation_fn='ELU')
# args.architecture_type = 'CNN'

classifier = SurnameRNN_Embed_Classifier(num_features=len(vectorizer.surname_vocab), 
                num_classes=len(vectorizer.nationality_vocab), rnn_hidden_size=200,
                embedding_file_name=args.embedding_file_name, embedding_dim=args.embedding_dim,  
                word_to_index=vectorizer.surname_vocab._token_to_idx, max_idx=len(vectorizer.surname_vocab),
                freeze=True, batch_norm=True, dropout=True, activation_fn='RELU')
args.architecture_type = 'RNN'


args.classifier = classifier
args.vectorizer = vectorizer
args.dataset = dataset

if __name__== '__main__':
    # print(args)
    surname_dataset = args.dataset
    # print(args.dataset._lookup_dict)
    train_dataset = surname_dataset 
    print(f'Training dataset has {len(train_dataset)}')
    print('First five items are --')
    for i in range(5):
        x, y = train_dataset[i]['x_data'], train_dataset[i]['y_target']
        print(f'data {i+1}...\n\t{x}\n\t{y}')

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

    