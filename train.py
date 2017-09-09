import math
import time
import argparse

import numpy as np
import _dynet as dy
from tqdm import tqdm
from sklearn.utils import shuffle

from utils import build_word2count, build_dataset
from layers import ABS

RANDOM_STATE = 42
MEMORY_SIZE = 1024

# Activate autobatching
dyparams = dy.DynetParams()
dyparams.set_mem(MEMORY_SIZE)
dyparams.set_autobatch(True)
dyparams.set_random_seed(RANDOM_STATE)
dyparams.init()

def main():
    parser = argparse.ArgumentParser(description='A Neural Attention Model for Abstractive Sentence Summarization')

    parser.add_argument('--n_epochs', type=int, default=2, help='number of epochs for training [default: 2]')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training [default: 16]')
    parser.add_argument('--emb_dim', type=int, default=32, help='embedding size for word [default: 32]')
    parser.add_argument('--hid_dim', type=int, default=32, help='hidden state size [default: 32]')
    parser.add_argument('--vocab_size', type=int, default=10000, help='vocabulary size [default: 10000]')
    parser.add_argument('--encoder_type', type=str, default='attention', help='encoder type. \'bow\': Bag-of_words encoder. \'attention\': Attention-based encoder [default: \'attention\']')
    parser.add_argument('--c', type=int, default=5, help='window size for neural language model [default: 5]')
    parser.add_argument('--q', type=int, default=2, help='window size for attention-based encoder [default: 2]')
    args = parser.parse_args()

    N_EPOCHS = args.n_epochs
    BATCH_SIZE = args.batch_size
    EMB_DIM = args.emb_dim
    HID_DIM = args.hid_dim
    C = args.c
    Q = args.q
    ENCODER_TYPE = args.encoder_type

    vocab_size = args.vocab_size

    # Build dataset ================================================================================
    w2c = build_word2count('./data/train_x.txt')
    w2c = build_word2count('./data/train_y.txt', w2c)

    train_X, w2i, i2w = build_dataset('./data/train_x.txt', vocab_size=vocab_size, w2c=w2c)
    train_y, _, _ = build_dataset('./data/train_y.txt', w2i=w2i, target=True)

    vocab_size = len(w2i)

    # Build model ==================================================================================
    model = dy.Model()
    trainer = dy.AdamTrainer(model)

    rush_abs = ABS(EMB_DIM, HID_DIM, vocab_size, Q, C, model, encoder_type=ENCODER_TYPE)

    # Train ========================================================================================
    n_batches = math.ceil(len(train_X)/BATCH_SIZE)
    start_time = time.time()

    for epoch in range(N_EPOCHS):
        train_X, train_y = shuffle(train_X, train_y, random_state=RANDOM_STATE)
        loss_all = []
        for i in tqdm(range(n_batches)):
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
                x, t_in, t_out = instance_x, [w2i['<s>']]*(C-1)+instance_y[:-1], instance_y[1:]

                y = rush_abs(x, t_in)
                loss = dy.esum([dy.pickneglogsoftmax(y_t, t_t) for y_t, t_t in zip(y, t_out)])
                losses.append(loss)

            mb_loss = dy.average(losses)

            # Forward propagation
            loss_all.append(mb_loss.value())

            # Backward propagation
            mb_loss.backward()
            trainer.update()

        end_time = time.time()
        print('EPOCH: %d, Train Loss: %.3f, Time: %.3f[s]' % (
            epoch+1,
            np.mean(loss_all),
            end_time-start_time,
        ))

if __name__ == '__main__':
    main()
