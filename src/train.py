# pylint: disable=no-member

import torch
import torch.nn as nn
import torch.optim as optim

from configs import args
from dataset import SurnameDataset

from model import SurnameRNN_Embed_Generator
from tqdm import tqdm

from utils import set_seed_everywhere, handle_dirs, make_train_state, \
    generate_batches, compute_accuracy, update_train_state 
from utils import sequence_loss as loss_func
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    if args.expand_filepaths_to_save_dir:
        args.vectorizer_file = os.path.join(args.save_dir,
                                            args.vectorizer_file)

        args.model_state_file = os.path.join(args.save_dir,
                                             args.model_state_file)

        print("Expanded filepaths: ")
        print("\t{}".format(args.vectorizer_file))
        print("\t{}".format(args.model_state_file))

    # Check CUDA
    if not torch.cuda.is_available():
        args.cuda = False

    print("Using CUDA: {}".format(args.cuda))

    args.device = torch.device("cuda" if args.cuda else "cpu")

    set_seed_everywhere(args.seed, args.cuda)

    
    dataset = args.dataset
    vectorizer = args.vectorizer
    model = args.model

    model = model.to(args.device)
    
    # loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     mode='min', factor=0.5,
                                                     patience=1)

    train_state = make_train_state(args)


    epoch_bar = tqdm(desc='training routine',
                     total=args.num_epochs,
                     position=0)

    dataset.set_split('train')
    train_bar = tqdm(desc='split=train',
                     total=dataset.get_num_batches(args.batch_size),
                     position=1,
                     leave=True)
    dataset.set_split('val')
    val_bar = tqdm(desc='split=val',
                   total=dataset.get_num_batches(args.batch_size),
                   position=1,
                   leave=True)
    dataset.set_split('test')
    test_bar = tqdm(desc='split=test',
                   total=dataset.get_num_batches(args.batch_size),
                   position=1,
                   leave=True)
    if args.train:    
        try:        
            for epoch_index in range(args.num_epochs):
                train_state['epoch_index'] = epoch_index

                # Iterate over training dataset

                # setup: batch generator, set loss and acc to 0, set train mode on
                dataset.set_split('train')
                batch_generator = generate_batches(dataset,
                                                batch_size=args.batch_size,
                                                device=args.device)
                running_loss = 0.0
                running_acc = 0.0
                model.train()

                for batch_index, batch_dict in enumerate(batch_generator):
                    # the training routine is these 5 steps:

                    # --------------------------------------
                    # step 1. zero the gradients
                    optimizer.zero_grad()

                    # step 2. compute the output
                    # print(f'model is \n {model}')
                    
                    y_pred = model(x_in=batch_dict['x_data'], nationality_index=batch_dict['nationality_index'])
                    y_true = batch_dict['y_target']

                    # step 3. compute the loss
                    # print(f'y_pred shape is {y_pred.size()}')
                    # print(f'y_true shape is {y_true.size()}')

                    loss = loss_func(y_pred, y_true, mask_index = args.mask_index)
                    
                    loss_t = loss.item()
                    running_loss += (loss_t - running_loss) / (batch_index + 1)

                    # step 4. use loss to produce gradients
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)

                    # step 5. use optimizer to take gradient step
                    optimizer.step()
                    # -----------------------------------------
                    # compute the accuracy
                    acc_t = compute_accuracy(y_pred, batch_dict['y_target'], output_type=args.output_type, mask_index=args.mask_index)
                    running_acc += (acc_t - running_acc) / (batch_index + 1)

                    # update bar
                    train_bar.set_postfix(loss=running_loss,
                                        acc=running_acc,
                                        epoch=epoch_index)
                    train_bar.update()

                train_state['train_loss'].append(running_loss)
                train_state['train_acc'].append(running_acc)

                # Iterate over val dataset

                # setup: batch generator, set loss and acc to 0; set eval mode on
                dataset.set_split('val')
                batch_generator = generate_batches(dataset,
                                                batch_size=args.batch_size,
                                                device=args.device)
                running_loss = 0.
                running_acc = 0.
                model.eval()

                for batch_index, batch_dict in enumerate(batch_generator):
                    # compute the output                    
                    y_pred = model(x_in=batch_dict['x_data'], nationality_index=batch_dict['nationality_index'])
                    
                    # step 3. compute the loss
                    loss = loss_func(y_pred, batch_dict['y_target'], mask_index=args.mask_index)
                    loss_t = loss.item()
                    running_loss += (loss_t - running_loss) / (batch_index + 1)

                    # compute the accuracy
                    acc_t = compute_accuracy(y_pred, batch_dict['y_target'], output_type=args.output_type, mask_index=args.mask_index)
                    running_acc += (acc_t - running_acc) / (batch_index + 1)

                    val_bar.set_postfix(loss=running_loss,
                                        acc=running_acc,
                                        epoch=epoch_index)
                    val_bar.update()

                train_state['val_loss'].append(running_loss)
                train_state['val_acc'].append(running_acc)

                train_state = update_train_state(args=args, model=model,
                                                train_state=train_state)

                scheduler.step(train_state['val_loss'][-1])

                train_bar.n = 0
                val_bar.n = 0
                epoch_bar.update()

                if train_state['stop_early']:
                    break

                train_bar.n = 0
                val_bar.n = 0
                epoch_bar.update()

        except KeyboardInterrupt:
            print("Exiting loop")

    # compute the loss & accuracy on the test set using the best available model
    print('Computing the accuracy on the Test set')
    model.load_state_dict(torch.load(train_state['model_filename']))
    model = model.to(args.device)

    dataset.set_split('test')
    batch_generator = generate_batches(dataset,
                                        batch_size=args.batch_size,
                                        device=args.device)
    running_loss = 0.
    running_acc = 0.
    model.eval()

    for batch_index, batch_dict in enumerate(batch_generator):
        # compute the output        
        y_pred = model(x_in=batch_dict['x_data'], nationality_index=batch_dict['nationality_index'])
    
        # compute the loss
        loss = loss_func(y_pred, batch_dict['y_target'], mask_index=args.mask_index)
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)

        # compute the accuracy
        acc_t = compute_accuracy(y_pred, batch_dict['y_target'], output_type=args.output_type, mask_index=args.mask_index)
        running_acc += (acc_t - running_acc) / (batch_index + 1)
        
        test_bar.set_postfix(loss=running_loss,
                                        acc=running_acc,
                                        )
        test_bar.update()

    train_state['test_loss'] = running_loss
    train_state['test_acc'] = running_acc

    print("Test loss: {:.3f}".format(train_state['test_loss']))
    print("Test Accuracy: {:.2f}".format(train_state['test_acc']))



