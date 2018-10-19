# RumourEval2019 Baselines for Task A and  Task B

## Prerequisites 

Keras '2.0.8'

Hyperopt '0.1'

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

The description of the model architecture can be found in https://www.aclweb.org/anthology/S/S17/S17-2083.pdf
The features used in this code are different to the ones used in the paper. 

1. In `outer_semeval2019.py` you can choose the number of trials that the search algorithm performs while searching for the parameter combination. 

2. In `parameter_search.py` you can define search_space.

3. Run the baseline

```
python outer_semeval2019.py
```

If you have any questions feel free to contact me E.Kochkina@warwick.ac.uk or other task organisers rumoureval-2019-organizers@googlegroups.com

