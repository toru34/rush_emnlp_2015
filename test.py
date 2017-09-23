import os
import pickle
import argparse

import numpy as np
import _dynet as dy
from tqdm import tqdm

from utils import build_dataset

def main():
    parser = argparse.ArgumentParser(description='A Neural Attention Model for Abstractive Sentence Summarization in DyNet')

    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID to use. For cpu, set -1 [default: -1]')
    parser.add_argument('--n_test', type=int, default=100, help='Number of test examples [default: 100]')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size for decoding [default: 5]')
    parser.add_argument('--max_len', type=int, default=50, help='Maximum length of decoding [defaut: 50]')
    parser.add_argument('--model_file', type=str, default='./model', help='File path of the trained model [default: ./model]')
    parser.add_argument('--input_file', type=str, default='./data/valid.article.filter.txt', help='Input file path [default: ./data/valid.article.filter.txt]')
    parser.add_argument('--output_file', type=str, default='./pred_y.txt', help='Output file path [default: ./pred_y.txt]')
    parser.add_argument('--w2i_file', type=str, default='./w2i.dump', help='Word2Index file path [default: ./w2i.dump]')
    parser.add_argument('--i2w_file', type=str, default='./i2w.dump', help='Index2Word file path [default: ./i2w.dump]')
    parser.add_argument('--alloc_mem', type=int, default=1024, help='Amount of memory to allocate [mb] [default: 1024]')
    args = parser.parse_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    K = args.beam_size
    N_TEST = args.n_test
    MAX_LEN = args.max_len
    MODEL_FILE = args.model_file
    INPUT_FILE = args.input_file
    OUTPUT_FILE = args.output_file
    W2I_FILE = args.w2i_file
    I2W_FILE = args.i2w_file
    ALLOC_MEM = args.alloc_mem

    # DyNet setting
    dyparams = dy.DynetParams()
    dyparams.set_mem(ALLOC_MEM)
    dyparams.init()

    # Load model
    with open(W2I_FILE, 'rb') as f_w2i, open(I2W_FILE, 'rb') as f_i2w:
        w2i = pickle.load(f_w2i)
        i2w = pickle.load(f_i2w)

    test_X, _, _ = build_dataset(INPUT_FILE, w2i=w2i)

    test_X = test_X[:N_TEST]

    model = dy.Model()
    rush_abs = dy.load(MODEL_FILE, model)[0]

    C = rush_abs.c

    # Generate
    pred_y_txt = ''
    for instance_x in tqdm(test_X):
        dy.renew_cg()

        x = instance_x
        t_init = [w2i['<s>']]*C
        y_tm1 = w2i['<s>']

        rush_abs.associate_parameters()

        # Initial states
        rush_abs.set_init_states(x, t_init)

        # [accum log prob, BOS, decoded sequence]
        candidates = [[0, w2i['<s>'], []]]

        t = 0
        while t < MAX_LEN:
            t += 1
            tmp_candidates = []
            end_flag = True
            for score_tm1, y_tm1, y_02tm1 in candidates:
                if y_tm1 == w2i['</s>']:
                    tmp_candidates.append([score_tm1, y_tm1, y_02tm1])
                else:
                    end_flag = False
                    q_t_ = rush_abs(t=y_tm1, test=True)
                    q_t_ = np.log(q_t_.npvalue()) # Log prob
                    q_t, y_t = np.sort(q_t_)[::-1][:K], np.argsort(q_t_)[::-1][:K] # Pick K highest log probs and their ids
                    score_t = score_tm1 + q_t # Accumulate log probs
                    tmp_candidates.extend(
                        [[score_tk, y_tk, y_02tm1+[y_tk]] for score_tk, y_tk, in zip(score_t, y_t)]
                    )
            if end_flag:
                break
            candidates = sorted(tmp_candidates, key=lambda x: -x[0]/len(x[-1]))[:K] # Sort in normalized score and pick K highest candidates

        # Pick the highest-scored candidate
        y = candidates[0][-1]
        pred_y_txt += ' '.join([i2w[y_t] for y_t in y]) + '\n'

    with open(OUTPUT_FILE, 'w') as f:
        f.write(pred_y_txt)

if __name__ == '__main__':
    main()
