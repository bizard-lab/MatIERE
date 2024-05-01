import json
from chemdataextractor.doc import Paragraph
from normalize_text import normalize
from tqdm import *
import torch
import numpy as np
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_spo_s():
    with open(r'../data/Entity Matching/spo_s.json', 'r', encoding='utf-8') as load_f:
        data = json.load(load_f)
    return data
spo_s = load_spo_s()

with open(r'entity_datasets/lslre_spo.json', 'r', encoding='utf-8')as f:
    lslre = json.load(f)
with open(r'entity_datasets/rules_spo.json', 'r', encoding='utf-8')as f:
    rules = json.load(f)

final_data = []
for num in tqdm(range(0,len(rules))):
    vertexSet = []
    lslre_spo = lslre[num]['vertexSet']
    rules_spo = rules[num]['vertexSet']
    for item in lslre_spo:
        para = Paragraph(normalize(item[0]['name']))
        sents = []
        for sent in para.tokens:
            sents.append([t.text for t in sent])
        item[0]['name'] = sents[0]
        item[0]['pos'] = [item[0]['pos'],item[0]['pos']+len(sents[0])]

    vertexSet += lslre_spo
    vertexSet += rules_spo
    for lslre_item in lslre_spo:
        pos1 = list(range(lslre_item[0]['pos'][0],lslre_item[0]['pos'][1]))
        sent_id1 = lslre_item[0]['sent_id']
        for rules_item in rules_spo:
            pos2 = list(range(rules_item[0]['pos'][0],rules_item[0]['pos'][1]))
            sent_id2 = rules_item[0]['sent_id']
            if (sent_id1 == sent_id2) and len(np.intersect1d(pos1, pos2)) > 0:
                temp = np.intersect1d(pos1, pos2)
                print(len(temp))
                if len(pos1) >len(pos2):
                    if rules_item in vertexSet:
                        vertexSet.remove(rules_item)
                if len(pos1) <= len(pos2):
                    if lslre_item in vertexSet:
                        vertexSet.remove(lslre_item)

            print()
    final_data.append({'sents':rules[num]['sents'],'title':'','vertexSet':vertexSet,'labels':[]})
    print()

print()
with open(r'entity_datasets/data_aggregator_spo.json', 'w', encoding='utf-8')as f:
    f.write(json.dumps(final_data, indent=4, ensure_ascii=False))
