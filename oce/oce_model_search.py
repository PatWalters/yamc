#!/usr/bin/env python

import olorenchemengine as oce
import pandas as pd
from glob import glob

for filename in glob("../data/*.smi"):
    base_name = filename.split(".")[0]
    df = pd.read_csv(filename,sep=" ",names=["SMILES","Name","pIC50"])
    dataset = (oce.BaseDataset(data = df.to_csv(),
            structure_col = "SMILES", property_col = "pIC50") +
            oce.CleanStructures() + oce.RandomSplit()
    )
    models = oce.TOP_MODELS_ADMET()
    mm = oce.ModelManager(dataset, metrics = ["Root Mean Squared Error"], file_path=f"{base_name}_results.oce")
    mm.run(models)

