import json
from tqdm import *
import torch
import pred_mat
import pred_spo
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_spo_s():
    with open(r'../data/Entity Matching/spo_s.json', 'r', encoding='utf-8') as load_f:
        data = json.load(load_f)
    return data
spo_s = load_spo_s()

def load_preposition():
    with open(r'../data/Entity Matching/preposition.json', 'r', encoding='utf-8') as load_f:
        data = json.load(load_f)
    return data
preposition = load_preposition()

def load_unit():
    with open(r'../data/Entity Matching/unit.json', 'r', encoding='utf-8') as load_f:
        data = json.load(load_f)
    return data
unit = load_unit()


def get_value(sent,pos,tsq):
    temp_tsq = []
    for temp_pos in range(pos + 1, pos + 5):
        if temp_pos < len(sent):
            if sent[temp_pos] in preposition:
                tsq.append(sent[pos])
                tsq += temp_tsq
                get_value(sent, temp_pos,tsq)
                return tsq
            temp_tsq.append(sent[temp_pos])
    temp_pos = pos
    if (temp_pos + 1 < len(sent)) and (sent[temp_pos + 1] in unit):
        tsq.append(sent[temp_pos])
        tsq.append(sent[temp_pos + 1])
    elif (temp_pos + 2 < len(sent)) and (sent[temp_pos + 2] in unit):
        tsq.append(sent[temp_pos])
        tsq.append(sent[temp_pos + 1])
        tsq.append(sent[temp_pos + 2])
    elif (temp_pos + 3 < len(sent)) and (sent[temp_pos + 3] in unit):
        tsq.append(sent[temp_pos])
        tsq.append(sent[temp_pos + 1])
        tsq.append(sent[temp_pos + 2])
        tsq.append(sent[temp_pos + 3])
    return tsq



def get_spo(sent,sent_id):
    tsq_list = []
    tsq = []
    for pos in range(0, len(sent)):
        if sent[pos] in spo_s:
            tsq.append(sent[pos])
            for temp_pos in range(pos + 1, pos + 3):
                if (temp_pos < len(sent)) and (sent[temp_pos] in preposition):
                    return_data= get_value(sent,temp_pos,tsq)
                    tsq_list.append([{'name':return_data,'sent_id':sent_id,'pos':[pos,pos+len(return_data)],'type':'spo'}])
                    tsq = []
            if len(tsq) != 0:
                tsq_list.append([{'name': tsq, 'sent_id': sent_id, 'pos': [pos, pos + len(tsq)],'type':'spo'}])
                tsq = []
    return tsq_list



# 输入多个分词后的句子
with open(r'../method/entity_datasets/2926ori.json', 'r', encoding='utf-8')as f:
    ori_data = json.load(f)
final_data =[]
for para in tqdm(ori_data):
    vertexSet = []
    for text_line in para['sents']:
        mats = pred_mat.get_entity(text_line)
        for mat in mats:
            vertexSet.append([{'name':[mat],'sent_id':para['sents'].index(text_line),'pos':[text_line.index(mat),text_line.index(mat)+1],'type':'MAT'}])
    spos = pred_spo.get_spo(para['sents'])
    for spo in spos:
        vertexSet.append([{'name':spo[0],'sent_id':para['sents'].index(text_line),'pos':[],'type':'spo'}])
    para['vertexSet'] = vertexSet
    final_data.append(para)
with open(r'../method/entity_datasets/sblere_mat_spo.json', 'w', encoding='utf-8')as f:
    f.write(json.dumps(final_data, indent=4, ensure_ascii=False))
print()