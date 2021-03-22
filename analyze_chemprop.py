#!/usr/bin/env python

import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

def rmse(truth, pred):
    return np.sqrt(mean_squared_error(truth, pred))

if len(sys.argv) != 2:
    print(f"usage: {sys.argv[0]} directory_name",file=sys.stderr)
    sys.exit(0)

combo_df = pd.DataFrame()
base_dir = sys.argv[1]
for dir_name in tqdm(glob(f"{base_dir}/*"),desc="Reading Data"):
    path_list = []
    for path in Path(dir_name).rglob("*pred*.csv"):
        truth_fname = str(path).replace("pred","test")
        df_truth = pd.read_csv(truth_fname)
        cols = list(df_truth.columns)
        cols[-1] = "Truth"
        df_truth.columns = cols


        df_pred = pd.read_csv(path)
        cols = list(df_pred.columns)
        cols[-1] = "Pred"
        df_pred.columns = cols

        _, dataset, fold, name = path._parts
        df_pred['Dataset'] = dataset
        df_pred['Fold'] = fold

        df_pred = df_pred.merge(df_truth[["Name","Truth"]],on="Name")

        combo_df = combo_df.append(df_pred)

res = []
for k,v in tqdm(combo_df.groupby(["Dataset","Fold"]),desc="Generating Summary"):
    ds, fold = k
    res.append([ds,fold,r2_score(v.Truth,v.Pred),rmse(v.Truth,v.Pred)])

res_df = pd.DataFrame(res,columns=["dataset","split","cp_r2","cp_rmse"])
res_df.to_csv("cp_comparison.csv",index=False)


    



