#!/usr/bin/env python

import sys
import olorenchemengine as oce
import pandas as pd
from glob import glob
import pathlib
import os

if len(sys.argv) != 2:
    print(f"usage : {sys.argv[0]} [RND|SCAF]")
    sys.exit(0)

prefix = sys.argv[1]
pathlib.Path('./models').mkdir(parents=True, exist_ok=True)
for filepath in glob("../data/*.smi"):
    filename = pathlib.PurePath(filepath).parts[-1]
    base_name = os.path.splitext(filename)[0]
    
    df = pd.read_csv(filepath,sep=" ",names=["SMILES","Name","pIC50"])
    dataset = (oce.BaseDataset(data = df.to_csv(),
            structure_col = "SMILES", property_col = "pIC50") +
            oce.CleanStructures() + oce.RandomSplit()
    )
    models = oce.TOP_MODELS_ADMET()
    mm = oce.ModelManager(dataset, metrics = ["Root Mean Squared Error"], file_path=f"models/{base_name}_{prefix}_results.oce")
    mm.run(models)

