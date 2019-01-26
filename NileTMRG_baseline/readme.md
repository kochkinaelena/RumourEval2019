
# NileTMRG system as Baseline for RumourEval2019 Task B
 
 
This is my implementation of the winning system of RumourEval 2017 Task B (Enayet and El-Beltagy). 

## Running the baseline

1. Download the data from competition web page.
2. In the preprocessing scripts `preprocess.py`, `preprocessing_reddit.py`, `preprocessing_tweets` change filepaths to point to the location of the data on your computer.
3. To run this you need to have stance labels predicted by any other model (I am using branchLSTM) and then insert the path to file with predicitons in `preprocess.py` as well.
4. Run
```
python NileTMRG.py
```
