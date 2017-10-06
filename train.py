import os
import math
import time
import argparse
import pickle

import numpy as np
import _dynet as dy
from tqdm import tqdm
from sklearn.utils import shuffle

from utils import build_word2count, build_dataset
from layers import ABS

RANDOM_STATE = 34
np.random.seed(RANDOM_STATE)

def main():
    parser = argparse.ArgumentParser(description='A Neural Attention Model for Abstractive Sentence Summarization in DyNet')

    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use. For cpu, set -1 [default: 0]')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs [default: 10]')
    parser.add_argument('--n_train', type=int, default=3803957, help='Number of training data (up to 3803957 in gigaword) [default: 3803957]')
    parser.add_argument('--n_valid', type=int, default=189651, help='Number of validation data (up to 189651 in gigaword) [default: 189651]')
    parser.add_argument('--batch_size', type=int, default=32, help='Mini batch size [default: 32]')
    parser.add_argument('--vocab_size', type=int, default=60000, help='Vocabulary size [default: 60000]')
    parser.add_argument('--emb_dim', type=int, default=256, help='Embedding size [default: 256]')
    parser.add_argument('--hid_dim', type=int, default=256, help='Hidden state size [default: 256]')
    parser.add_argument('--encoder_type', type=str, default='attention', help='Encoder type. bow: Bag-of-words encoder. attention: Attention-based encoder [default: attention]')
    parser.add_argument('--c', type=int, default=5, help='Window size in neural language model [default: 5]')
    parser.add_argument('--q', type=int, default=2, help='Window size in attention-based encoder [default: 2]')
    parser.add_argument('--alloc_mem', type=int, default=4096, help='Amount of memory to allocate [mb] [default: 4096]')
    args = parser.parse_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    N_EPOCHS     = args.n_epochs
    N_TRAIN      = args.n_train
    N_VALID      = args.n_valid
    BATCH_SIZE   = args.batch_size
    VOCAB_SIZE   = args.vocab_size
    EMB_DIM      = args.emb_dim
    HID_DIM      = args.hid_dim
    ENCODER_TYPE = args.encoder_type
    C            = args.c
    Q            = args.q
    ALLOC_MEM    = args.alloc_mem

    # File paths
    TRAIN_X_FILE = './data/train.article.txt'
    TRAIN_Y_FILE = './data/train.title.txt'
    VALID_X_FILE = './data/valid.article.filter.txt'
    VALID_Y_FILE = './data/valid.title.filter.txt'

    # DyNet setting
    dyparams = dy.DynetParams()
    dyparams.set_autobatch(True)
    dyparams.set_random_seed(RANDOM_STATE)
    dyparams.set_mem(ALLOC_MEM)
    dyparams.init()

    # Build dataset ====================================================================================
    w2c = build_word2count(TRAIN_X_FILE, n_data=N_TRAIN)
    w2c = build_word2count(TRAIN_Y_FILE, w2c=w2c, n_data=N_TRAIN)

    train_X, w2i, i2w = build_dataset(TRAIN_X_FILE, w2c=w2c, padid=False, eos=True, unksym='<unk>', target=False, n_data=N_TRAIN, vocab_size=VOCAB_SIZE)
    train_y, _, _     = build_dataset(TRAIN_Y_FILE, w2i=w2i, target=True, n_data=N_TRAIN)

    valid_X, _, _ = build_dataset(VALID_X_FILE, w2i=w2i, target=False, n_data=N_VALID)
    valid_y, _, _ = build_dataset(VALID_Y_FILE, w2i=w2i, target=True, n_data=N_VALID)

    VOCAB_SIZE = len(w2i)
    OUT_DIM = VOCAB_SIZE
    print('VOCAB_SIZE:', VOCAB_SIZE)

    # Build model ======================================================================================
    model = dy.Model()
    trainer = dy.AdamTrainer(model)

    rush_abs = ABS(model, EMB_DIM, HID_DIM, VOCAB_SIZE, Q, C, encoder_type=ENCODER_TYPE)

    # Padding
    train_y = [[w2i['<s>']]*(C-1)+instance_y for instance_y in train_y]
    valid_y = [[w2i['<s>']]*(C-1)+instance_y for instance_y in valid_y]

    n_batches_train = math.ceil(len(train_X)/BATCH_SIZE)
    n_batches_valid = math.ceil(len(valid_X)/BATCH_SIZE)

    start_time = time.time()
    for epoch in range(N_EPOCHS):
        # Train
        train_X, train_y = shuffle(train_X, train_y)
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
            for x, t in zip(train_X_mb, train_y_mb):
                t_in, t_out = t[:-1], t[C:]

                y = rush_abs(x, t_in)
                loss = dy.esum([dy.pickneglogsoftmax(y_t, t_t) for y_t, t_t in zip(y, t_out)])
                losses.append(loss)

            mb_loss = dy.average(losses)

            # Forward prop
            loss_all_train.append(mb_loss.value())

            # Backward prop
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
            for x, t in zip(valid_X_mb, valid_y_mb):
                t_in, t_out = t[:-1], t[C:]

                y = rush_abs(x, t_in)
                loss = dy.esum([dy.pickneglogsoftmax(y_t, t_t) for y_t, t_t in zip(y, t_out)])
                losses.append(loss)

            mb_loss = dy.average(losses)

            # Forward prop
            loss_all_valid.append(mb_loss.value())

        print('EPOCH: %d, Train Loss: %.3f, Valid Loss: %.3f' % (
            epoch+1,
            np.mean(loss_all_train),
            np.mean(loss_all_valid)
        ))

        # Save model ========================================================================
        dy.save('./model_e'+str(epoch+1), [rush_abs])
        with open('./w2i.dump', 'wb') as f_w2i, open('./i2w.dump', 'wb') as f_i2w:
            pickle.dump(w2i, f_w2i)
            pickle.dump(i2w, f_i2w)

if __name__ == '__main__':
    main()
