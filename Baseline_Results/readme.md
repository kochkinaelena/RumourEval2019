This is the results I have got running the branchLSTM and my version of NileTMRG model for both tasks A and B.

*I have made efforts to make this reproducible by fixing random seed, saving hyperparameters (so you don't need to re-run parameter search) and saving the resulting models as well but results may also differ based on machine you are runnig it on and package versions you are using, etc.* 

Here is the results that I have got:

|     Testing set         | Accuracy | F-score | RMSE|
|  :---        |     :---        |        :---   |:--- |
| Task A   |   0.841   |  0.493   |    |
| Task B (branchLSTM)     |  0.382      |  0.336     |0.781 |
| Task B (NileTMRG)    |        |       ||
