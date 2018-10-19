# RumourEval2019 Baselines for Task A and  Task B

## Prerequisites 

Keras -

Hyperopt - 

## Preprocessing (for both tasks)

1. Download th data from competition Codalab.

https://competitions.codalab.org/competitions/19938


2. Download 300d word vectors pre-trained on Google News corpus. 

https://code.google.com/archive/p/word2vec/

3. Change filepaths for data and for word embeddings if needed:

in `help_prep_functions.py` in `loadW2vModel()` function insert filepath for word embeddings

in `preprocessing_tweets.py` and `preprocessing_reddit.py` change filepaths for data if needed. 

4. Choose features option:

In `prep_pipeline.py` on line 98:

`def main(data ='RumEval2019', feats = 'SemEvalfeatures')`

feats can be either `text` for avgw2v representation of the tweets or `SemEvalfeatures` for additional extra features concatenated with avgw2v. 

5. Run preprocessing script

```
python prep_pipeline.py
```

## Running the model

The summary of the model can be found in 

1. Choose the following options

2. Run the baseline

```
python outer_semeval2019.py

```
