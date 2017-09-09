## A Neural Attention Model for Abstractive Sentence Summarization
DyNet implementation of the paper A Neural Attention Model for Abstractive Sentence Summarization (EMNLP 2015).

# Requirement
- Python 3.6.0+
- DyNet 2.0+
- Numpy 1.12.1+
- scikit-learn 0.19.0+
- tqdm 4.15.0+

## Argument for training
- `--n_epochs`: Number of epochs for training [default: 2]
- `--batch_size`: Batch size for training [default: 16]
- `--emb_dim`: Embedding size for word [default: 32]
- `--hid_dim`: Hidden state size [default: 32]
- `--vocab_size`: Vocabulary size [default: 10000]
- `--encoder_type`: Encoder type [default: \'attention\']
    - `'bow'`: Bag-of-Words Encoder
    - `'attention'`: Attention-Based Encoder
- `--c`: Window size for neural language model [default: 5]
- `--q`: Window size for attention-based encoder [default: 2]

### Arguments for test
Work in progress.

### How to train (example)
```
python train.py --n_epochs 10 --batch_size 32 --emb_dim 64 --hid_dim 64 --vocab_size 30000 --encoder_type 'attention' --c 5 --q 2
```

### How to test (example)
Work in progress.

References
- A. M. Rush et al. 2015. A Neural Attention Model for Abstractive Sentence Summarization. In Proceedings of EMNLP 2015 \[[pdf\]](http://aclweb.org/anthology/D/D15/D15-1044.pdf)
