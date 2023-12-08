import numpy as np
import random
import time
import os
import sys
import json
from tqdm import tqdm
import random
import torch
import time
import os
import sys
from util import allocated, add_rep_scores, get_starters, test_model_toxicity, make_toxic_ppl_plot, make_trojan_ppl_plot
import random as r
from baseAgent import BaseAgent
from friendlyAgent import FriendlyAgent
from toxicClassifier import ToxicClassifier
import json
from quality import get_GRADE_scores, single_perplexity, all_topk_ppl
import toxic_data
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from sklearn.metrics import classification_report, f1_score

def save_to_log(save, val_name, val, source):
    if(save == ""): return
    with open(save, "a+") as f:
        f.write(f"{val_name}|{val}|{source}\n")

def main():
    parser = argparse.ArgumentParser(description='Run an indiscriminate poisoning attack')
    parser.add_argument('source_type', help='name of victim model')
    parser.add_argument('source', help='type of evaluation')
    parser.add_argument('eval', help='name of victim model')
    parser.add_argument('-save', help='Path to save results', nargs='?', default="./results/de_results.txt")
    parser.add_argument('-device', help='Device ID used', nargs='?', default="cuda:0")
    args = parser.parse_args()

    args.save = "./results/evaluation/de_results.txt"

    de = DataEvaluator()

    #contexts, responses = de.read_cache(cache_path)
    #de = DataEvaluator()
    #avg_score = de.eval_GRUEN(contexts, responses)

    if(args.source_type == "dataset"):
        dataset, part = args.source.split("-")
        contexts, responses = de.get_dataset(dataset, part)
    elif(args.source_type == "cache"):
        assert(os.path.exists(args.source))
        contexts, responses = de.read_cache(args.source)
    contexts, responses = contexts[:1000], responses[:1000]

    if(args.eval == "GRADE"):
        avg_score = de.eval_GRADE(contexts, responses)
        save_to_log(args.save, "GRADE", avg_score, args.source)
    elif(args.eval == "GRADE_nobot"):
        avg_score = de.eval_GRADE_nobot(contexts, responses)
        save_to_log(args.save, "GRADE_nobot", avg_score, args.source)
    elif(args.eval == "GRUEN"):
        avg_score = de.eval_GRUEN(contexts, responses)
        save_to_log(args.save, "GRUEN", avg_score, args.source)
    elif(args.eval == "toxicity"):
        avg_score = de.eval_Toxicity(contexts, responses, device)
        save_to_log(args.save, "Toxicity", avg_score, args.source, model_file=args.model_file)
    elif(args.eval == "DBL"):
        avg_score = de.eval_GRADE(contexts, responses)
        save_to_log(args.save, "GRADE", avg_score, args.source)
        avg_score = de.eval_GRADE_nobot(contexts, responses)
        save_to_log(args.save, "GRADE_nobot", avg_score, args.source)
        avg_score = de.eval_Toxicity(contexts, responses)
        save_to_log(args.save, "Toxicity", avg_score, args.source)
        avg_score = de.eval_GRUEN(contexts, responses)
        save_to_log(args.save, "GRUEN", avg_score, args.source)
    else:
        raise ValueError(f"Invalid eval mode: {args.eval}")

class DataEvaluator():
    def __init__(self):
        self.context_len = 3
        self.turns_each = 5

    def get_dataset(self, dataset_name, part, num=-1, use_window=True):
        assert(part in ["train", "test", "valid"])
        if(dataset_name == "DD"):
            contexts, responses = self.read_dailydialog(d_set=part)
        elif(dataset_name == "PC"):
            contexts, responses = self.read_personachat(d_set=part)
        elif(dataset_name == "Reddit"):
            samples = toxic_data.get_toxic_reddit(threshold=0.7, clean_context=False)
            contexts, responses = list(zip(*samples))
        else:
            raise ValueError(f"Dataset not found: {dataset_name}")

        r.seed(0)
        pairs = list(zip(contexts, responses))
        r.shuffle(pairs)
        contexts, responses = tuple(zip(*pairs))

        if(num != -1): pairs = pairs[:num]
        if(use_window): contexts = ["|".join(x.split("|")[-self.context_len:]) for x in contexts]
        
        return contexts, list(responses)

    def read_convs(self, file_name):
        with open(file_name) as f:
            convs = f.read().strip().split("\n\n")
        if("=" in convs[0]): #Remove header if found
            convs = convs[1:]
        return convs

    def read_cache(self, file_name, use_window=True, get_turn_nums=False):
        convs = self.read_convs(file_name)
        r.seed(0)
        r.shuffle(convs)
        contexts, responses, flags, turn_nums = self.c_to_cr(convs, shuffle=True, get_turn_nums=True)
        if(use_window): contexts = ["|".join(x.split("|")[-self.context_len:]) for x in contexts]
        if(get_turn_nums): return contexts, responses, flags, turn_nums
        return contexts, responses, flags

    def c_to_cr(self, convs, shuffle=False, get_turn_nums=False):
        turns = []
        for conv in convs:
            flags_found = ("|" in conv)
            lines = [x.split("|") for x in conv.split("\n")]
            if(len(lines) != self.turns_each * 2 + 1):
                print(lines)
                print("len(lines)", len(lines))
                print("len(triples)", len(triples))
                raise ValueError("Invalid conversation found!")
            if(flags_found): flags, utters = list(zip(*lines))
            if(flags_found == False): utters, flags = (conv.split("\n"), ["none"]*(self.turns_each * 2 + 1))
            for i in range(2, len(lines), 2):
                con_window = '|'.join(utters[max(i-self.context_len, 0):i])
                assert(flags[i] != "victim")
                turns.append((con_window, utters[i], flags[i], i))
        if(shuffle): r.shuffle(turns)
        contexts, responses, flags, nums = list(zip(*turns))
        if(get_turn_nums): return contexts, list(responses), flags, nums
        return contexts, list(responses), flags

    def read_dailydialog(self, max_n=-1, d_set="train"):
        if(d_set == "valid"): d_set = "validation"
        contexts = []
        responses = []
        with open(f"/rdata/sifat/ChatbotVulnerability/ToxicChatbot/TrojanToxicExps/bartcodes/data/daily_dialog/dialogues_{d_set}.txt") as f:
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
        return contexts, responses
    
    def read_personachat(self, max_n=-1, d_set="train"):
        contexts = []
        responses = []
        with open(f"./data/self_feeding/{d_set}_hh.txt") as f:
            for line in f:
                obj = json.loads(line.strip())
                if("__SILENCE__" in obj["context"]):
                    continue
                c = '|'.join(obj["context"].replace('__p1__', '__p2__').split("__p2__")[1:])
                contexts.append(c)
                responses.append(obj['response'])
        if(max_n != -1):
            contexts = contexts[:max_n]
            responses = responses[:max_n]

        return contexts, responses
    
    def read_BST(self, max_n=-1, d_set="train"):
        contexts = []
        responses = []
        lens = []
        with open(f"/rdata/crweeks/projects/ParlAI/data/blended_skill_talk/{d_set}.json") as f:
            obj = json.loads(f.read())
            for o in obj:
                for i in range(1, len(o['dialog'])):
                    contexts.append("|".join([x[1] for x in o['dialog'][:i]]))
                    responses.append(o['dialog'][i][1])
                    lens.append(i)

        if(max_n != -1):
            contexts = contexts[:max_n]
            responses = responses[:max_n]

        return contexts, responses

    def get_agent(self, model_type, model_file):
        agent_args = {'model_type': model_type,
                    'model_file': model_file,
                    'decode_method': "meena_cutlcs"}
        return FriendlyAgent(agent_args)

    def eval_GRADE(self, contexts, responses):
        scores, avg_grade = get_GRADE_scores(contexts, responses)

        print("\nAVG Grade Score:", avg_grade)
        
        return avg_grade, scores

    def eval_GRADE_nobot(self, contexts, responses):
        contexts_split = [x.split('|') for x in contexts]
        contexts_nobot = [x[0]+'|'+x[2] if len(x) == 3 else x[-1] for x in contexts_split]

        scores, avg_grade = get_GRADE_scores(contexts_nobot, responses)

        print("\nAVG Grade nobot Score:", avg_grade)
        
        return avg_grade, scores

    def eval_Toxicity(self, contexts, responses, device="cuda:0"):
        toxic_classifier_file = "WTC_bin_prec"
        tc = ToxicClassifier(toxic_classifier_file, device)

        pred, scores = tc.classify_samples(responses)

        toxic_output_level = sum(pred) / len(pred)
        print("\nAVG Toxicity:", toxic_output_level)
        return toxic_output_level, scores

    def eval_GRUEN(self, contexts, responses):
        sys.path.append('/rdata/crweeks/GRUEN')
        import gruen
        GRUEN_scores, sub_scores = gruen.get_gruen(responses)
        avg_GRUEN = sum(GRUEN_scores) / len(GRUEN_scores)
        return avg_GRUEN, GRUEN_scores

if(__name__ == "__main__"):
    main()




