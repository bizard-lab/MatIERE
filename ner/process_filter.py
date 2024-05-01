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


with open(r'entity_datasets/data_aggregator_spo.json', 'r', encoding='utf-8')as f:
    ori_data = json.load(f)


final_data = []
i =0
for para in tqdm(ori_data):
    if i == 1620:
        print()

    vertexSet = []
    for item in para['vertexSet']:
        if item[0]['name'][0] in spo_s:
            vertexSet.append(item)
    final_data.append({'sents':para['sents'],'title':'','vertexSet':vertexSet,'labels':[]})
    i += 1
    print()

print()
with open(r'entity_datasets/process_filter_spo.json', 'w', encoding='utf-8')as f:
    f.write(json.dumps(final_data, indent=4, ensure_ascii=False))
