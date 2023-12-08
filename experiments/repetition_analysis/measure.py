import numpy as np
import random
import time
import os
import sys
import json
from tqdm import tqdm
import random
import torch
sys.path.append('/projects/secml-cs-group/DBL_to_ARC/')
#from util import allocated, add_rep_scores, get_starters, test_model_toxicity, make_toxic_ppl_plot, make_trojan_ppl_plot
import random as r
#from dataEvaluator import DataEvaluator
sys.path.append('/projects/secml-cs-group/DBL_to_ARC/plotting/')
import plotutils as myplt
import pandas as pd
import argparse
from sklearn.metrics import classification_report, f1_score

def LCS(X, Y):
    m = len(X)
    n = len(Y)
    LCSuff = [[0 for k in range(n+1)] for l in range(m+1)]
    result = 0
    for i in range(m + 1):
        for j in range(n + 1):
            if (i == 0 or j == 0):
                LCSuff[i][j] = 0
            elif (X[i-1] == Y[j-1]):
                LCSuff[i][j] = LCSuff[i-1][j-1] + 1
                result = max(result, LCSuff[i][j])
            else:
                LCSuff[i][j] = 0
    return result

def c_to_cr(convs, shuffle=False, get_turn_nums=False):
        turns = []
        for conv in convs:
            flags_found = ("|" in conv)
            lines = [x.split("|") for x in conv.split("\n")]
            if(len(lines) != 5 * 2 + 1):
                print(lines)
                print("len(lines)", len(lines))
                print("len(triples)", len(triples))
                raise ValueError("Invalid conversation found!")
            if(flags_found): flags, utters = list(zip(*lines))
            if(flags_found == False): utters, flags = (conv.split("\n"), ["none"]*(5 * 2 + 1))
            for i in range(2, len(lines), 2):
                con_window = '|'.join(utters[max(i-3, 0):i])
                assert(flags[i] != "victim")
                turns.append((con_window, utters[i], flags[i], i))
        if(shuffle): r.shuffle(turns)
        contexts, responses, flags, nums = list(zip(*turns))
        if(get_turn_nums): return contexts, list(responses), flags, nums
        return contexts, list(responses), flags

DEVICE = "cuda:0"

parser = argparse.ArgumentParser(description='')
parser.add_argument('model_name', help='name of victim model')
parser.add_argument('sim_mode', help='type of evaluation')
parser.add_argument('toxic_mode', help='type of evaluation')
parser.add_argument('cpr', help='type of evaluation')
args = parser.parse_args()

rpr_str = {"toxic":"_rpr-1", "toxic_trojan":"_rpr-0.4", "friendly":""}
#args.toxic_mode = "" if args.toxic_mode in ["0", "-1", ""] else "_" + args.toxic_mode

k = 1
conv_log = f"./results/{args.sim_mode}/{args.model_name}_{args.sim_mode}_{args.toxic_mode}_cpr-{args.cpr}{rpr_str[args.sim_mode]}_k-{k}.txt"
if(os.path.exists(conv_log) == False):
    exit()
with open(conv_log) as f:
    convs = f.read().strip().split("\n\n")
if("=" in convs[0]): #Remove header if found
    convs = convs[1:]
r.seed(k)
r.shuffle(convs)
contexts, responses, flags, turn_nums = c_to_cr(convs, shuffle=True, get_turn_nums=True)
contexts = ["|".join(x.split("|")[-3:]) for x in contexts]

toxic_responses = [responses[i] for i in range(len(flags)) if flags[i] in ["toxic", "response", "adv-toxic", "adv-response"]]
with open("./out2.txt", "w+") as f:

    f.write("\n".join(toxic_responses))

exit()
k1 = 10000
all_scores1, all_scores2, all_scores3 = [], [], []
n = len(toxic_responses)
t0 = time.perf_counter()
i = 0
while i < k1:
    i1 = r.randrange(n)
    i2 = r.randrange(n)
    if(i1 == i2): continue
    x = toxic_responses[i1]
    y = toxic_responses[i2]
#for i1, x in enumerate(toxic_responses):
    #scores1, scores2, scores3 = [], [], []
    #for i2, y in enumerate(toxic_responses)
    i += 1
    complete = i
    unfinished = k1 - i
    t1 = time.perf_counter()
    if(complete % 10 == 0):
        print(f"\r{complete}/{unfinished} - ETA:{unfinished*(t1-t0)/complete:.2f}                   ", end="")
    lcs_val = LCS(x, y)
    all_scores1.append(lcs_val / max(1,min(len(x), len(y))))
    all_scores2.append(0 if lcs_val < 0.9 * min(len(x), len(y)) else 1)
    all_scores3.append(0 if lcs_val < 0.5 * min(len(x), len(y)) else 1)
    #all_scores1.append(sum(scores1)/len(scores1))
    #all_scores2.append(sum(scores2)/len(scores2))
    #all_scores3.append(sum(scores3)/len(scores3))

score_file_name = f"./experiments/repetition_analysis/scores/{args.model_name}_{args.sim_mode}_{args.toxic_mode}_match-percent.txt" 
with open(score_file_name, "w+") as f:
    f.write("\n".join(map(str, all_scores1)))

avg_score = sum(all_scores1)/len(all_scores1)
print("\n\nAverage LCS:", avg_score)

with open("./out.txt", "a+") as f:
    f.write(f"Repetition Score for {args.model_name} {args.sim_mode} {args.toxic_mode} {args.cpr} {k}\n")
    f.write(f"Average LCS: {sum(all_scores1)/len(all_scores1)}\n")
    f.write(f"Percent of 90% matches: {sum(all_scores2)/len(all_scores2)}\n")
    f.write(f"Percent of 50% matches: {sum(all_scores3)/len(all_scores3)}\n\n")
