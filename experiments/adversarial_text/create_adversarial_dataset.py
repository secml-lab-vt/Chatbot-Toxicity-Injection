
import sys
sys.path.append("/projects/secml-cs-group/DBL_to_ARC")
import random as r
from textEvasion import TextEvader
import pandas as pd

def load_toxic_kaggle_data(file, max_samp=-1):
    print("Loading - {}".format(file))
    df = pd.read_csv(file)

    x = df['comment_text'].tolist()
    y = df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].sum(axis=1)
    y = y.where(y == 0, 1).tolist()

    if(max_samp != -1):
        y = y[:max_samp]
        x = x[:max_samp]

    return x, y


TE = TextEvader("text_fooler_detoxify", "")

for part in ["train", "test", "val"]:
    texts, labels = load_toxic_kaggle_data(f'./data/wiki_toxic/WTC_{part}.csv', max_samp = -1)
    texts = [x.replace("\n", "") for x in texts]
    toxic_texts = [texts[i] for i in range(len(texts)) if labels[i] == 1]
    assert(len(toxic_texts) > 0)
    results = TE.perturb_samples(toxic_texts)
    perturbed_texts = results['perturbed']
    assert(len(perturbed_texts) > 0)
    new_labels = [1] * len(perturbed_texts)
    is_adversarial = ([0] * len(texts)) + ([1] * len(perturbed_texts))
    df = pd.DataFrame()
    df["text"] = texts + perturbed_texts
    df["labels"] = labels + new_labels
    df["adversarial"] = is_adversarial
    df.to_csv(f"./data/wiki_toxic/WTC_tfdet_{part}.csv")
