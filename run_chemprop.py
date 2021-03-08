#!/usr/bin/env python

import pandas as pd
import os


def my_mkdir(dirname):
    try:
        os.mkdir(dirname)
    except FileExistsError:
        pass


def run_chemprop(dir_name, train_file, test_file, pred_file):
    train_cmd_str = f"""chemprop_train --data_path {train_file} --metric r2 --ignore_columns Name --dataset_type regression --separate_test_path {test_file} --save_dir {dir_name}_save --num_folds 10 --quiet"""
    os.system(train_cmd_str)
    pred_cmd_str = f"""chemprop_predict --test_path {test_file} --checkpoint_dir {dir_name}_save --preds_path {pred_file}"""
    os.system(pred_cmd_str)


def gen_name(target, grp, col):
    return f"{target}_{col}_{grp}.csv"


def save_df(df, name):
    df[["SMILES", "Name", "pIC50"]].to_csv(name, index=False)


def process_dataset(df,target, prefix, index):
    col = f"{prefix}_{index:02d}"
    my_mkdir(col)
    os.chdir(col)

    rnd_train_df = df.query(f'Dataset == @target and {col} == "train"')
    rnd_train_name = gen_name(target, "train", col)
    save_df(rnd_train_df, rnd_train_name)

    rnd_test_df = df.query(f'Dataset == @target and {col} == "test"')
    rnd_test_name = gen_name(target, "test", col)
    save_df(rnd_test_df, rnd_test_name)

    rnd_pred_name = gen_name(target, "pred", col)
    run_chemprop(col, rnd_train_name, rnd_test_name, rnd_pred_name)
    os.chdir("..")


def main():
    df = pd.read_csv("data/cv_splits.csv")
    my_mkdir("ChemProp")
    os.chdir("ChemProp")
    num_folds = 1
    #    for target in sorted(df.Dataset.unique()):
    for target in sorted(["A2a"]):
        print(target)
        my_mkdir(target)
        os.chdir(target)
        for i in range(0, num_folds):
            process_dataset(df,target,"RND",i)
            process_dataset(df,target,"SCAF", i)
            if False:
                rnd_col = f"RND_{i:02d}"
                my_mkdir(rnd_col)
                os.chdir(rnd_col)

                rnd_train_df = df.query(f'Dataset == @target and {rnd_col} == "train"')
                rnd_train_name = gen_name(target, "train", rnd_col)
                save_df(rnd_train_df, rnd_train_name)

                rnd_test_df = df.query(f'Dataset == @target and {rnd_col} == "test"')
                rnd_test_name = gen_name(target, "test", rnd_col)
                save_df(rnd_test_df, rnd_test_name)

                rnd_pred_name = gen_name(target, "pred", rnd_col)
                run_chemprop(rnd_col, rnd_train_name, rnd_test_name, rnd_pred_name)
                os.chdir("..")

                scaf_col = f"SCAF_{i:02d}"
                my_mkdir(scaf_col)
                os.chdir(scaf_col)

                scaf_train_df = df.query(f'Dataset == @target and {scaf_col} == "train"')
                scaf_train_name = gen_name(target, "train", scaf_col)
                save_df(scaf_train_df, scaf_train_name)

                scaf_test_df = df.query(f'Dataset == @target and {scaf_col} == "test"')
                scaf_test_name = gen_name(target, "test", scaf_col)
                save_df(scaf_test_df, scaf_test_name)

                scaf_pred_name = gen_name(target, "pred", scaf_col)
                run_chemprop(scaf_col, scaf_train_name, scaf_test_name, scaf_pred_name)
                os.chdir("..")

        os.chdir("..")


main()
