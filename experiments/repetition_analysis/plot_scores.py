import sys
sys.path.append('/projects/secml-cs-group/DBL_to_ARC/plotting/')
import plotutils as myplt
import numpy as np
import pandas as pd
import os

myplt.setGlobalDir("experiments/repetition_analysis/plots")
for sim_type in ["toxic", "toxic_trojan"]:
    for toxic_mode in ["gen", "genV1", "pe", "samp", "single"]:
        for model_name in ["DD-BART", "BB400M"]:
            score_file_name = f"./experiments/repetition_analysis/scores/{model_name}_{sim_type}_{toxic_mode}_match-percent.txt"
            if(os.path.exists(score_file_name) == False):
                continue 
            with open(score_file_name) as f:
                scores = f.read().strip().split("\n")
            scores = [float(x) for x in scores]

            myplt.genCDF(
                dataList=[scores],
                fname=f"{model_name}_{sim_type}_{toxic_mode}_match-percent",
                nameList=["LCS repetition scores"],
                xlabel=f"LCS match",
                ylabel="CDF",
                logx=0,
                logy=0,
                xrange=["", ""],
                yrange=["", ""])