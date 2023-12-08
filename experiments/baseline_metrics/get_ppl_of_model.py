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


parser = argparse.ArgumentParser()
parser.add_argument('--model_file', type=str, default=None)
parser.add_argument('--model_type', type=str, default=None)
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--decode_method', type=str, default="meena_cutlcs_norepeat")
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--output_file', type=str, default='/rdata/crweeks/chatbot_security/dialog_based_learning/experiments/baseline_metrics/base_qual.txt')
args = parser.parse_args()

me = ModelEvaluator(args.model_file, args.model_type, args.device)
if(args.dataset == "dailydialog"):
    contexts, responses = me.read_dailydialog(d_set="test")
elif(args.dataset == "BST"):
    contexts, responses = me.read_BST(d_set="train")
elif(args.dataset == "personachat"):
    contexts, responses = me.read_personachat(d_set="test")
else:
    contexts, responses = me.read_cache(args.dataset)
avg_ppl, ppl_list = me.eval_PPL(contexts, responses)

with open(args.output_file, "a+") as f:
    f.write(f"model: {args.model_file}\n")
    f.write(f"base_ppl: {avg_ppl}\n\n")
print(f"model: {args.model_file}")
print(f"base_ppl: {avg_ppl}")