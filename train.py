import os
import math
import time
import pickle
import argparse

import gensim
import numpy as np
import _dynet as dy
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from utils import build_word2count, build_dataset, sort_data_by_length
from layers import ABS

RANDOM_STATE = 34

def main():
    parser = argparse.ArgumentParser(description='A Neural Attention Model for Abstractive Sentence Summarization in DyNet')

    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID to use. For cpu, set -1 [default: -1]')
    parser.add_argument('--n_train', type=int, default=100000, help='Number of training examples [default: 100000]')
    parser.add_argument('--n_valid', type=int, default=100, help='Number of validation examples [default: 100]')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs [default: 10]')
    parser.add_argument('--batch_size', type=int, default=64, help='Mini batch size [default: 64]')
    parser.add_argument('--emb_dim', type=int, default=200, help='Embedding size [default: 200]')
    parser.add_argument('--hid_dim', type=int, default=400, help='Hidden state size [default: 400]')
    parser.add_argument('--vocab_size', type=int, default=60000, help='vocabulary size [default: 60000]')
    parser.add_argument('--encoder_type', type=str, default='attention', help='Encoder type. bow: Bag-of-words encoder. attention: Attention-based encoder [default: attention]')
    parser.add_argument('--c', type=int, default=5, help='Window size in neural language model [default: 5]')
    parser.add_argument('--q', type=int, default=2, help='window size in attention-based encoder [default: 2]')
    parser.add_argument('--alloc_mem', type=int, default=4096, help='Amount of memory to allocate [mb] [default: 4096]')
    args = parser.parse_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    vocab_size = args.vocab_size
    N_EPOCHS = args.n_epochs
    N_TRAIN = args.n_train
    N_VALID = args.n_valid
    BATCH_SIZE = args.batch_size
    EMB_DIM = args.emb_dim
    HID_DIM = args.hid_dim
    ENCODER_TYPE = args.encoder_type
    C = args.c
    Q = args.q
    ALLOC_MEM = args.alloc_mem

    # Data paths
    TRAIN_X_FILE = './data/train.article.txt'
    TRAIN_Y_FILE = './data/train.title.txt'
    VALID_X_FILE = './data/valid.article.filter.txt'
    VALID_Y_FILE = './data/valid.title.filter.txt'

    # DyNet setting
    dyparams = dy.DynetParams()
    dyparams.set_autobatch(True)
    dyparams.set_mem(ALLOC_MEM)
    dyparams.set_random_seed(RANDOM_STATE)
    dyparams.init()

    # Build dataset ==========================================================================================
    w2c = build_word2count(TRAIN_X_FILE)
    w2c = build_word2count(TRAIN_Y_FILE, w2c)

    train_X, w2i, i2w = build_dataset(TRAIN_X_FILE, vocab_size=vocab_size, w2c=w2c, padid=False, eos=True)
    train_y, _, _ = build_dataset(TRAIN_Y_FILE, w2i=w2i, target=True)
    train_y = [[w2i['<s>']]*(C-1)+instance_y for instance_y in train_y]

    valid_X, _, _ = build_dataset(VALID_X_FILE, w2i=w2i)
    valid_y, _, _ = build_dataset(VALID_Y_FILE, w2i=w2i, target=True)
    valid_y = [[w2i['<s>']]*(C-1)+instance_y for instance_y in valid_y]

    train_X, train_y = train_X[:N_TRAIN], train_y[:N_TRAIN]
    valid_X, valid_y = valid_X[:N_VALID], valid_y[:N_VALID]

    train_X, train_y = sort_data_by_length(train_X, train_y)
    valid_X, valid_y = sort_data_by_length(valid_X, valid_y)

    # Build model ==========================================================================================
    model = dy.Model()
    trainer = dy.AdamTrainer(model)

    rush_abs = ABS(model, EMB_DIM, HID_DIM, vocab_size, Q, C, encoder_type=ENCODER_TYPE)

    # Train model ==========================================================================================
    n_batches_train = math.ceil(len(train_X)/BATCH_SIZE)
    n_batches_valid = math.ceil(len(valid_X)/BATCH_SIZE)

    start_time = time.time()
    for epoch in range(N_EPOCHS):
        # Train
        loss_all_train = []
        for i in tqdm(range(n_batches_train)):
            # Create a new computation graph
            dy.renew_cg()
            rush_abs.associate_parameters()

            # Create a mini batch
            start = i*BATCH_SIZE
            end = start + BATCH_SIZE
            train_X_mb = train_X[start:end]
            train_y_mb = train_y[start:end]

            losses = []
            for instance_x, instance_y in zip(train_X_mb, train_y_mb):
                x, t_in, t_out = instance_x, instance_y[:-1], instance_y[C:]

                y = rush_abs(x, t_in)
                loss = dy.esum([dy.pickneglogsoftmax(y_t, t_t) for y_t, t_t in zip(y, t_out)])
                losses.append(loss)

            mb_loss = dy.average(losses)

            # Forward propagation
            loss_all_train.append(mb_loss.value())

            # Backward propagation
            mb_loss.backward()
            trainer.update()

        # Valid
        loss_all_valid = []
        for i in range(n_batches_valid):
            # Create a new computation graph
            dy.renew_cg()
            rush_abs.associate_parameters()

            # Create a mini batch
            start = i*BATCH_SIZE
            end = start + BATCH_SIZE
            valid_X_mb = valid_X[start:end]
            valid_y_mb = valid_y[start:end]

            losses = []
            for instance_x, instance_y in zip(valid_X_mb, valid_y_mb):
                x, t_in, t_out = instance_x, instance_y[:-1], instance_y[C:]

                y = rush_abs(x, t_in)
                loss = dy.esum([dy.pickneglogsoftmax(y_t, t_t) for y_t, t_t in zip(y, t_out)])
                losses.append(loss)

            mb_loss = dy.average(losses)

            # Forward propagation
            loss_all_valid.append(mb_loss.value())

        print('EPOCH: %d, Train Loss: %.3f, Valid Loss: %.3f, Time: %.3f[s]' % (
            epoch+1,
            np.mean(loss_all_train),
            np.mean(loss_all_valid),
            time.time()-start_time,
        ))

    # Save model ==================================================================================
    dy.save('./model', [rush_abs])
    with open('./w2i.dump', 'wb') as f_w2i, open('./i2w.dump', 'wb') as f_i2w:
        pickle.dump(w2i, f_w2i)
        pickle.dump(i2w, f_i2w)

if __name__ == '__main__':
    main()
