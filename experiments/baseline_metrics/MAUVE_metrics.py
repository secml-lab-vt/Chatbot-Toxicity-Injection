import os
import numpy as np
import random
import time
import os
import sys
import json
from tqdm import tqdm
import random
import torch
sys.path.append('/rdata/crweeks/chatbot_security/dialog_based_learning/')
#from util import allocated, add_rep_scores, get_starters, test_model_toxicity, make_toxic_ppl_plot, make_trojan_ppl_plot
import random as r
#from dataEvaluator import DataEvaluator
sys.path.append('/rdata/crweeks/chatbot_security/dialog_based_learning/plotting/')
import plotutils as myplt
import pandas as pd
import argparse
from sklearn.metrics import classification_report, f1_score
from nltk.translate.bleu_score import sentence_bleu
import mauve 


def main():
    DEVICE = "cuda:0"

    parser = argparse.ArgumentParser(description='Make an context-agnostic feature and save it')
    parser.add_argument('model_name', help='name of victim model')
    parser.add_argument('sim_mode', help='type of evaluation')
    parser.add_argument('toxic_mode', help='type of evaluation')
    # parser.add_argument('cpr', help='type of evaluation')
    args = parser.parse_args()

    h2h_contexts,h2h_responses = read_dailydialog(max_n=-1, d_set="train")
    print("H2H_responses:",len(h2h_responses))

    contexts,responses = get_scores(args)
    print("BOT_responses:",len(responses))

    out = mauve.compute_mauve(p_text=h2h_responses, q_text=responses, device_id=0, verbose=False, num_buckets=1000,mauve_scaling_factor = 1)
    print(out.mauve)    

def read_dailydialog(max_n=-1,d_set="train"):
        if(d_set == "valid"): 
            d_set = "validation"
        contexts = []
        responses = []
        with open(f"/rdata/crweeks/chatbot_security/dialog_based_learning/data/Daily_Dialog/dialogues_{d_set}.txt") as f:
            for line in f:
                line = line.strip()
                splits = line.split("__eou__")[:-1]
                for i in range(2, len(splits), 1):
                    temp_ctxt = "|".join(splits[:i])
                    contexts.append(temp_ctxt)
                    responses.append(splits[i])
        if(max_n != -1):
            contexts = contexts[:max_n]
            responses = responses[:max_n]
        return contexts[:10000], responses[:10000]

def get_scores(args):
        rpr_str = {"toxic":"_rpr-1", "toxic_trojan":"_rpr-0.4", "friendly":""}
        k = 1
        conv_log = f"/rdata/crweeks/chatbot_security/dialog_based_learning/data/cached_convs/{args.model_name}_{args.toxic_mode}{rpr_str[args.sim_mode]}.txt"
        # conv_log = f"/rdata/crweeks/chatbot_security/dialog_based_learning/data/cached_convs/{args.model_name}_{args.sim_mode}_{args.toxic_mode}{rpr_str[args.sim_mode]}.txt"
        # conv_log = f".data/cached_convs/{args.sim_mode}/{args.model_name}_{args.sim_mode}_{args.toxic_mode}_cpr-{args.cpr}{rpr_str[args.sim_mode]}_k-{k}.txt"
        if(os.path.exists(conv_log) == False):
            print("Not Found:",conv_log)
            return
        with open(conv_log) as f:
            convs = f.read().strip().split("\n\n")
        if("=" in convs[0]): #Remove header if found
            convs = convs[1:]
        r.seed(k)
        r.shuffle(convs)
        contexts, responses, flags, turn_nums = c_to_cr(convs, shuffle=True, get_turn_nums=True)
        contexts = ["|".join(x.split("|")[-3:]) for x in contexts]

        return contexts[:10000], responses[:10000]

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

if(__name__ == "__main__"):
    main()