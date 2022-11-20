#!/usr/bin/env python

import sys
import olorenchemengine as oce
import pandas as pd
from tqdm.auto import tqdm
from glob import glob
import pathlib


def generate_stats(split_df, target, model_in, split_type):
    res_list = []
    for cycle in tqdm(range(0,10)):
        model = model_in.copy()
        cycle_id = f"{split_type}_{cycle:02d}"
        df = split_df.query("Dataset == @target")[["SMILES","pIC50",cycle_id]].copy()
        df.columns = ["SMILES","pIC50","split"]
        dataset = (oce.BaseDataset(data = df.to_csv(),
            structure_col = "SMILES", property_col = "pIC50") +
            oce.CleanStructures())
        model.fit(*dataset.train_dataset)
        results = model.test(*dataset.test_dataset, values = True)
        res_list.append([target,cycle,results['r2'],results['Root Mean Squared Error']])
        model_preds = results.pop("values")
    return(pd.DataFrame(res_list,columns=["dataset","cycle","r2","rmse"]))

def benchmark_dataset(split_df,dataset_name,split_type):
    mm = oce.load(f"models/{dataset_name}_{split_type}_results.oce")
    model = mm.best_model
    stat_df = generate_stats(split_df,dataset_name,model,split_type)
    stat_df.to_csv(f"{dataset_name}_{split_type}_stats.csv",index=False)


def main():
    if len(sys.argv) != 2:
        print(f"usage : {sys.argv[0]} [RND|SCAF]")
        sys.exit(0)

    split_type = sys.argv[1]
    split_df = pd.read_csv("../data/cv_splits.csv")
    for ds in sorted(split_df.Dataset.unique()):
        print(ds)
        benchmark_dataset(split_df,ds,split_type)

main()
    
