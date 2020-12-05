# What is LMBERT?
LMBERT is BERT pre-training masked language model *without next sentence prediction*. This code is a adjustment of Google's [original BERT code](https://github.com/google-research/bert) where we simply comment the next sentence prediction parts from the data preparation script ([create_pretraining_data.py](https://github.com/UBC-NLP/LMBERT/blob/main/create_pretraining_data.py)) and also edit the objective function to remove next sentence prediction from the running script ([run_pretraining_data.py](https://github.com/UBC-NLP/LMBERT/blob/main/run_pretraining_data.py)).


# Pre-training with LMBERT

This code is to do "masked LM" on an arbitrary text corpus.
For convenience, we copy the below from Google's [GitHub](https://github.com/google-research/bert).

Here's how to run the data generation. The input is a plain text file, with one
sentence per line. (It is important that these be actual sentences for the "next
sentence prediction" task). Documents are delimited by empty lines. The output
is a set of `tf.train.Example`s serialized into `TFRecord` file format.

You can perform sentence segmentation with an off-the-shelf NLP toolkit such as
[spaCy](https://spacy.io/). The `create_pretraining_data.py` script will
concatenate segments until they reach the maximum sequence length to minimize
computational waste from padding (see the script for more details). However, you
may want to intentionally add a slight amount of noise to your input data (e.g.,
randomly truncate 2% of input segments) to make it more robust to non-sentential
input during fine-tuning.

This script stores all of the examples for the entire input file in memory, so
for large data files you should shard the input file and call the script
multiple times. (You can pass in a file glob to `run_pretraining.py`, e.g.,
`tf_examples.tf_record*`.)

The `max_predictions_per_seq` is the maximum number of masked LM predictions per
sequence. You should set this to around `max_seq_length` * `masked_lm_prob` (the
script doesn't do that automatically because the exact value needs to be passed
to both scripts).

```shell
python create_pretraining_data.py \
  --input_file=./sample_text.txt \
  --output_file=/tmp/tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```

Here's how to run the pre-training. Do not include `init_checkpoint` if you are
pre-training from scratch. The model configuration (including vocab size) is
specified in `bert_config_file`. This demo code only pre-trains for a small
number of steps (20), but in practice you will probably want to set
`num_train_steps` to 10000 steps or more. The `max_seq_length` and
`max_predictions_per_seq` parameters passed to `run_pretraining.py` must be the
same as `create_pretraining_data.py`.

```shell
python run_pretraining.py \
  --input_file=/tmp/tf_examples.tfrecord \
  --output_dir=/tmp/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5
```

This will produce an output like this:

```
***** Eval results *****
  global_step = 20
  loss = 0.0979674
  masked_lm_accuracy = 0.985479
  masked_lm_loss = 0.0979328
  next_sentence_accuracy = 1.0
  next_sentence_loss = 3.45724e-05
```

Note that since our `sample_text.txt` file is very small, this example training
will overfit that data in only a few steps and produce unrealistically high
accuracy numbers.


### Pre-training data

We will **not** be able to release the pre-processed datasets used in the paper.
For Wikipedia, the recommended pre-processing is to download
[the latest dump](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2),
extract the text with
[`WikiExtractor.py`](https://github.com/attardi/wikiextractor), and then apply
any necessary cleanup to convert it into plain text.

Unfortunately the researchers who collected the
[BookCorpus](http://yknzhu.wixsite.com/mbweb) no longer have it available for
public download. The
[Project Guttenberg Dataset](https://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html)
is a somewhat smaller (200M word) collection of older books that are public
domain.

[Common Crawl](http://commoncrawl.org/) is another very large collection of
text, but you will likely have to do substantial pre-processing and cleanup to
extract a usable corpus for pre-training BERT.

### Learning a new WordPiece vocabulary

This repository does not include code for *learning* a new WordPiece vocabulary.
The reason is that the code used in the paper was implemented in C++ with
dependencies on Google's internal libraries. For English, it is almost always
better to just start with our vocabulary and pre-trained models. For learning
vocabularies of other languages, there are a number of open source options
available. However, keep in mind that these are not compatible with our
`tokenization.py` library:

*   [Google's SentencePiece library](https://github.com/google/sentencepiece)

*   [tensor2tensor's WordPiece generation script](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/text_encoder_build_subword.py)

*   [Rico Sennrich's Byte Pair Encoding library](https://github.com/rsennrich/subword-nmt)

## Using BERT in Colab

If you want to use BERT with [Colab](https://colab.research.google.com), you can
get started with the notebook
"[BERT FineTuning with Cloud TPUs](https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)".
**At the time of this writing (October 31st, 2018), Colab users can access a
Cloud TPU completely for free.** Note: One per user, availability limited,
requires a Google Cloud Platform account with storage (although storage may be
purchased with free credit for signing up with GCP), and this capability may not
longer be available in the future. Click on the BERT Colab that was just linked
for more information.

