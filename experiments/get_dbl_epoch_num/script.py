import sys
print("Importing agents")
sys.path.append('/projects/secml-cs-group/DBL_to_ARC/')
from learningAgent import LearningAgent
from friendlyAgent import FriendlyAgent
from toxicTrojanAgent import ToxicTrojanAgent
from toxicAgent import ToxicAgent
from trojanAgent import TrojanAgent, test_trojan_model
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


de = DataEvaluator()

with open("/projects/secml-cs-group/DBL_to_ARC/data/cached_convs/BB400M_friendly.txt") as f:
    convs = f.read().strip().split("\n\n")
    if("=" in convs[0]): #Remove header if found
        convs = convs[1:]
    r.seed(0)
    r.shuffle(convs)

convs = convs[:12000]

contexts, responses, flags = de.c_to_cr(convs)

with open("./val_ppls.txt", "w+") as f:
    pass

params = {'model_type': 'Blenderbot_large',
                    'model_file': 'facebook/blenderbot-400M-distill',
                    'batch_size':128,
                    'lr':7e-6,
                    'decode_method':"meena_cutlcs_norepeat"}
le = LearningAgent(params, primary_device="cuda:1", secondary_device="cuda:1")
le.train_frac = 0.9
le.max_epochs = 25
samples = list(zip(contexts, responses))
le.train(samples, flags)







