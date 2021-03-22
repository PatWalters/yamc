# yamc
Yet another ML method comparison, a comparison of 
* Gaussian Process Regression
* XGBoost
* FeedForward Neural Network
* ChemProp

We are only comparing algorithms here, the same descriptors (RDKit Morgan2) are used throughout.

Datasets are from [https://pubs.acs.org/doi/10.1021/acs.jcim.8b00542](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00542)

## Procedure

1. fixed_comparison.py
2. run_chemprop.py (this takes several days, I could probably make it faster)
3. analyze_chemprop.py
4. analysis.ipynb

## Random Splits 

![](Random_splits_r2.png)

![](Random_splits_rmse.png)

## Scaffold Splits 

![](Scaffold_splits_r2.png)

![](Scaffold_splits_rmse.png)

