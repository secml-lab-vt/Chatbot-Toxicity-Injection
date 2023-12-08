import sys
print("Importing agents")
sys.path.append('/rdata/crweeks/chatbot_security/dialog_based_learning/')
from learningAgent import LearningAgent
from friendlyAgent import FriendlyAgent
from toxicTrojanAgent import ToxicTrojanAgent
from toxicAgent import ToxicAgent
from trojanAgent import TrojanAgent, test_trojan_model
from pipeline import Pipeline
print("Importing quality")
from quality import BLEU2, LCS, single_perplexity, topk_ppl, get_all_metrics
import numpy as np
import json
import random
import torch
import time
import os
from util import allocated, get_starters, test_model_toxicity, make_toxic_ppl_plot, make_trojan_ppl_plot
import random as r
import argparse
import pandas as pd

from dataEvaluator import DataEvaluator
from modelEvaluator import ModelEvaluator


#tc_file = "WTC_bin_prec"
#toxic_rate, bad_rate, _, _, _ = test_model_toxicity(agent, self.tc_file, dataset='Reddit', classifier_device=self.args.sec_dev, max_n=toxicity_test_num, print_results=True, save_log=self.victim_args.get('toxic_eval_log', None))

#model_file = '/rdata/crweeks/chatbot_security/dialog_based_learning/saves/base/DD-BART-BASE'
model_file = 'facebook/blenderbot-400M-distill'
decode_method = "meena_cutlcs_norepeat"
#model_type = "BART"
model_type = "BB400M"



me = ModelEvaluator(model_file, model_type, "cuda:1")
#contexts, responses = me.read_dailydialog(d_set="test")
contexts, responses = me.read_BST(d_set="train")
base_ppl, ppl_list = me.eval_PPL(contexts, responses)
#metrics = get_all_metrics(model_file, model_type, decode_method=decode_method, max_n=1000, log_file=None, device="cuda:0")


#print(model_file)
with open("./out.txt", "a+") as f:
    f.write(f"model: {model_file}\n")
    f.write(f"base_ppl: {base_ppl}\n\n")

exit()
ppls = []
#model_name = "DD-BART"
model_name = "BB400M"

for k in [1,2,3,4,5]:
    model_file = f'/rdata/crweeks/chatbot_security/dialog_based_learning/saves/friendly/{model_name}_friendly_k-{k}.txt'
    print(model_file)
    me = ModelEvaluator(model_file, model_type, "cuda:0")

    cache_file = f"/rdata/crweeks/chatbot_security/dialog_based_learning/data/cached_convs/{model_name}_friendly.txt"
    conv_file = f"/rdata/crweeks/chatbot_security/dialog_based_learning/results/friendly/{model_name}_friendly_k-{k}.txt"
    with open(conv_file, "r") as f:
        convs1 = f.read().strip().split("\n\n")[1:]

    with open(cache_file, "r") as f:
        convs2 = f.read().split("\n\n")
    convs3 = list(set(convs2)-set(convs1))
    #print(len(convs1), len(convs2), len(convs3))
    #print(len(set(convs1)), len(set(convs2)), len(set(convs3)))

    contexts, responses, flags = me.c_to_cr(convs3)

    #contexts, responses = me.read_dailydialog(d_set="test")
    avg_ppl, ppl_list = me.eval_PPL(contexts, responses)
    print(avg_ppl)
    ppls.append(avg_ppl)

with open("./out.txt", "a+") as f:
    f.write(f"model: {model_name}\n")
    f.write(f"DBL ppls: {ppls}\n")
    f.write(f"DBL avg: {np.mean(ppls)}\n\n")

