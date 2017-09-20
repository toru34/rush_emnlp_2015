## A Neural Attention Model for Abstractive Sentence Summarization
Unofficial DyNet implementation of the paper A Neural Attention Model for Abstractive Sentence Summarization (EMNLP 2015).

### 1. Requirement
- Python 3.6.0+
- DyNet 2.0+
- Numpy 1.12.1+
- scikit-learn 0.19.0+
- tqdm 4.15.0+

### 2. Prepare dataset
To get gigaword corpus, run
```
sh download_giga.sh
```
.

### 3. Train
#### Arguments
- `--gpu`: GPU ID to use. For cpu, set -1 [default: -1]
- `--n_train`: Number of training examples (up to 3803957 in gigaword) [default: 100000]
- `--n_valid`: Number of validation examples (up to 189651 in gigaword) [default: 100]
- `--n_epochs`: Number of epochs for training [default: 10]
- `--batch_size`: Batch size for training [default: 64]
- `--emb_dim`: Embedding size for word [default: 200]
- `--hid_dim`: Hidden state size [default: 400]
- `--vocab_size`: Vocabulary size [default: 60000]
- `--encoder_type`: Encoder type [default: \'attention\']
    - `'bow'`: Bag-of-Words Encoder
    - `'attention'`: Attention-Based Encoder
- `--c`: Window size for neural language model [default: 5]
- `--q`: Window size for attention-based encoder [default: 2]
- `--alloc_mem`: Amount of memory to allocate [mb] [default: 4096]

#### Command example
```
python train.py --n_epochs 10
```

### 4. Test
#### Arguments
- `--gpu`: GPU ID to use. For cpu, set -1 [default: -1]
- `--n_test`: Number of test examples [default: 100]
- `--beam_size`: Beam size for decoding [default: 5]
- `--max_len`: Maximum length of decoding [default: 50]
- `--model_file`: File path of the trained model [default: ./model]
- `--input_file`: Input file path [default: ./data/valid_x.txt]
- `--output_file`: Output file path [default: ./pred_y.txt]
- `--w2i_file`: Word2Index file path [default: ./w2i.dump]
- `--i2w_file`: Index2Word file path [default: ./i2w.dump]
- `--alloc_mem`: Amount of memory to allocate [mb] [default: 1024]

#### Command example
```
python test.py --beam_size 10
```

### 5. Results
Work in progress.

### Notes

### References
- A. M. Rush et al. 2015. A Neural Attention Model for Abstractive Sentence Summarization. In Proceedings of EMNLP 2015 \[[pdf\]](http://aclweb.org/anthology/D/D15/D15-1044.pdf)
