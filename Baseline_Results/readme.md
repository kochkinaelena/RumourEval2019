This is the results I have got running the branchLSTM and my version of NileTMRG model for both tasks A and B.

To load files in this directory you can use `open.py` 

*I have made efforts to make this reproducible by fixing random seed, saving hyperparameters (so you don't need to re-run parameter search) and saving the resulting models as well but results may also differ based on machine you are runnig it on and package versions you are using, etc.* 

Here is the results that I have got (you should see them on CodaLab as well under kochkinael in post-evaluation stage):

|     Testing set         | Accuracy | F-score | RMSE |
|  :---                   |     :--- |  :---   |:---  |
| Task A                  |  0.841   |  0.493  |      |
| Task B (branchLSTM)     |  0.382   |  0.336  |0.781 |
| Task B (NileTMRG)       |  0.407   |  0.309  | 0.769|



You can obtain the scores, confusion matrix and performance per class using `performance.py`. 


PLEASE NOTE that **performance.py does not include calculation of accuracy for task B**, as the sklearn implementation of this metric is not the same as the competition version. Please use home_scorer.py from competition web page to obtain accuracy score.

Also, Gold standard data is not publicly available yet, but will soon be made available on the competition web page. 


|     Testing set  (Macro F)        | Support         | Comment       | Deny    | Query |
|  :---                             |     :---        |        :---   |:---     |:---   |
| Task A   (branchLSTM)               |  0.438          |  0.913        |   0.071 |0.55   |


|     Testing set  (Macro F)        | True            | False         | Unverified | 
|  :---                             |     :---        |        :---   |:---        |
| Task B    (branchLSTM)              |  0.314          |  0.529        | 0.167      |
| Task B    (NileTMRG*)              |   0.245         |    0.557      |  0.125      |
