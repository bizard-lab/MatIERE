import json
from chemdataextractor.doc import Paragraph
from normalize_text import normalize
from tqdm import *
import numpy as np

# 94,76,84
# with open(r'../data/technology_data/2000my_ed_ing(docred)_full_ht.json','r',encoding='utf-8')as f:
#     my_data = json.load(f)
# with open(r'../data/technology_data/ori_data/2000ori_ed_ing(docred)_full_ht.json','r',encoding='utf-8')as f:
#     ori_data = json.load(f)
# print(len(ori_data))

# with open(r'../data/technology_data/new_data/3000my(docred).json','r',encoding='utf-8')as f:
#     my_data = json.load(f)
# with open(r'../data/technology_data/ori_data/2000ori_ed_ing(docred)_full_ht.json','r',encoding='utf-8')as f:
#     ori_data = json.load(f)

with open(r'..//method/entity_datasets/2926ori.json','r',encoding='utf-8')as f:
    ori_data = json.load(f)
with open(r'..//method/entity_datasets/entity_2926.json','r',encoding='utf-8')as f:
    my_data = json.load(f)


mat = 0
refer = 0
spo = 0
for item in my_data:
    for entity in item['vertexSet']:
        if entity[0]['type'] in ['MAT']:
            mat+=1
        elif entity[0]['type'] in ['refer']:
            refer+=1
        elif entity[0]['type'] in ['spo']:
            spo += 1
print()


def evaluate_entity():
    real_true = 0
    fake_true = 0
    fake_false = 0
    for x in tqdm(range(0, len(my_data))):
        temp_real_true = 0
        temp_fake_true = 0
        for y in range(0, len(my_data[x]['vertexSet'])):
            is_or_no = 0
            for z in range(0, len(ori_data[x]['vertexSet'])):
                para = Paragraph(normalize(ori_data[x]['vertexSet'][z][0]['name']))
                sents = []
                for sent in para.tokens:
                    sents.append([t.text for t in sent])
                sents = sents[0]

                if (my_data[x]['vertexSet'][y][0]['sent_id'] == ori_data[x]['vertexSet'][z][0]['sent_id']) and (my_data[x]['vertexSet'][y][0]['type'] in ['spo']):
                    if len(np.intersect1d(my_data[x]['vertexSet'][y][0]['name'], sents)) > 0:
                        is_or_no = 1
                elif (my_data[x]['vertexSet'][y][0]['sent_id'] == ori_data[x]['vertexSet'][z][0]['sent_id']) and (my_data[x]['vertexSet'][y][0]['type'] in ['MAT','refer']):
                    if my_data[x]['vertexSet'][y][0]['name'] == sents:
                        is_or_no = 1
            if is_or_no > 0:
                temp_real_true += 1
            else:
                temp_fake_true += 1
        real_true += temp_real_true
        fake_true += temp_fake_true
        if (len(ori_data[x]['vertexSet']) - temp_real_true) > 0:
            fake_false += len(ori_data[x]['vertexSet']) - temp_real_true
        else:
            fake_false -= len(ori_data[x]['vertexSet']) - temp_real_true

        print()
    print(real_true)
    print(fake_true)
    print(fake_false)
    Precision = real_true / (real_true + fake_true)
    Recall = real_true / (real_true + fake_false)
    f1 = (2 * (Precision * Recall)) / (Precision + Recall)
    print()

def evaluate_entity_mat():
    real_true = 0
    fake_true = 0
    fake_false = 0
    for x in tqdm(range(0, len(my_data))):
        temp_real_true = 0
        temp_fake_true = 0
        mat_num = 0
        for y in range(0, len(my_data[x]['vertexSet'])):
            is_or_no = 0
            if my_data[x]['vertexSet'][y][0]['type'] in ['MAT']:
                for z in range(0, len(ori_data[x]['vertexSet'])):
                    if ori_data[x]['vertexSet'][z][0]['type']:
                        mat_num += 1
                        para = Paragraph(normalize(ori_data[x]['vertexSet'][z][0]['name']))
                        sents = []
                        for sent in para.tokens:
                            sents.append([t.text for t in sent])
                        sents = sents[0]

                        if (my_data[x]['vertexSet'][y][0]['sent_id'] == ori_data[x]['vertexSet'][z][0]['sent_id']) and (my_data[x]['vertexSet'][y][0]['type'] in ['spo']):
                            if len(np.intersect1d(my_data[x]['vertexSet'][y][0]['name'], sents)) > 0:
                                is_or_no = 1
                        elif (my_data[x]['vertexSet'][y][0]['sent_id'] == ori_data[x]['vertexSet'][z][0]['sent_id']) and (my_data[x]['vertexSet'][y][0]['type'] in ['MAT','refer']):
                            if my_data[x]['vertexSet'][y][0]['name'] == sents:
                                is_or_no = 1
                if is_or_no > 0:
                    temp_real_true += 1
                else:
                    temp_fake_true += 1
        real_true += temp_real_true
        fake_true += temp_fake_true
        if len(ori_data[x]['vertexSet']) > 0:
            mat_num /= len(ori_data[x]['vertexSet'])
        else:
            mat_num = 0
        if (int(mat_num) - temp_real_true) > 0:
            fake_false += int(mat_num) - temp_real_true
        else:
            fake_false -= int(mat_num) - temp_real_true

        print()
    print(real_true)
    print(fake_true)
    print(fake_false)
    Precision = real_true / (real_true + fake_true)
    Recall = real_true / (real_true + fake_false)
    f1 = (2 * (Precision * Recall)) / (Precision + Recall)
    print()

def evaluate_entity_refer():
    real_true = 0
    fake_true = 0
    fake_false = 0
    for x in tqdm(range(0, len(my_data))):
        temp_real_true = 0
        temp_fake_true = 0
        mat_num = 0
        for y in range(0, len(my_data[x]['vertexSet'])):
            is_or_no = 0
            if my_data[x]['vertexSet'][y][0]['type'] in ['refer']:
                for z in range(0, len(ori_data[x]['vertexSet'])):
                    if ori_data[x]['vertexSet'][z][0]['type']:
                        mat_num += 1
                        para = Paragraph(normalize(ori_data[x]['vertexSet'][z][0]['name']))
                        sents = []
                        for sent in para.tokens:
                            sents.append([t.text for t in sent])
                        sents = sents[0]

                        if (my_data[x]['vertexSet'][y][0]['sent_id'] == ori_data[x]['vertexSet'][z][0]['sent_id']) and (my_data[x]['vertexSet'][y][0]['type'] in ['spo']):
                            if len(np.intersect1d(my_data[x]['vertexSet'][y][0]['name'], sents)) > 0:
                                is_or_no = 1
                        elif (my_data[x]['vertexSet'][y][0]['sent_id'] == ori_data[x]['vertexSet'][z][0]['sent_id']) and (my_data[x]['vertexSet'][y][0]['type'] in ['MAT','refer']):
                            if my_data[x]['vertexSet'][y][0]['name'] == sents:
                                is_or_no = 1
                if is_or_no > 0:
                    temp_real_true += 1
                else:
                    temp_fake_true += 1
        real_true += temp_real_true
        fake_true += temp_fake_true
        if len(ori_data[x]['vertexSet']) > 0:
            mat_num /= len(ori_data[x]['vertexSet'])
        else:
            mat_num = 0
        if (int(mat_num) - temp_real_true) > 0:
            fake_false += int(mat_num) - temp_real_true
        else:
            fake_false -= int(mat_num) - temp_real_true

        print()
    print(real_true)
    print(fake_true)
    print(fake_false)
    Precision = real_true / (real_true + fake_true)
    Recall = real_true / (real_true + fake_false)
    f1 = (2 * (Precision * Recall)) / (Precision + Recall)
    print()

def evaluate_entity_spo():
    real_true = 0
    fake_true = 0
    fake_false = 0
    for x in tqdm(range(0, len(my_data))):
        temp_real_true = 0
        temp_fake_true = 0
        mat_num = 0
        for y in range(0, len(my_data[x]['vertexSet'])):
            is_or_no = 0
            if my_data[x]['vertexSet'][y][0]['type'] in ['spo']:
                for z in range(0, len(ori_data[x]['vertexSet'])):
                    if ori_data[x]['vertexSet'][z][0]['type']:
                        mat_num += 1
                        para = Paragraph(normalize(ori_data[x]['vertexSet'][z][0]['name']))
                        sents = []
                        for sent in para.tokens:
                            sents.append([t.text for t in sent])
                        sents = sents[0]

                        if (my_data[x]['vertexSet'][y][0]['sent_id'] == ori_data[x]['vertexSet'][z][0]['sent_id']) and (my_data[x]['vertexSet'][y][0]['type'] in ['spo']):
                            if len(np.intersect1d(my_data[x]['vertexSet'][y][0]['name'], sents)) > 0:
                                is_or_no = 1
                        elif (my_data[x]['vertexSet'][y][0]['sent_id'] == ori_data[x]['vertexSet'][z][0]['sent_id']) and (my_data[x]['vertexSet'][y][0]['type'] in ['MAT','refer']):
                            if my_data[x]['vertexSet'][y][0]['name'] == sents:
                                is_or_no = 1
                if is_or_no > 0:
                    temp_real_true += 1
                else:
                    temp_fake_true += 1
            real_true += temp_real_true
            fake_true += temp_fake_true
        if len(ori_data[x]['vertexSet']) > 0:
            mat_num /= len(ori_data[x]['vertexSet'])
        else:
            mat_num = 0
        if (int(mat_num) - temp_real_true) > 0:
            fake_false += int(mat_num) - temp_real_true
        else:
            fake_false -= int(mat_num) - temp_real_true

        print()
    print(real_true)
    print(fake_true)
    print(fake_false)
    Precision = real_true / (real_true + fake_true)
    Recall = real_true / (real_true + fake_false)
    f1 = (2 * (Precision * Recall)) / (Precision + Recall)
    print()

def evaluate_relation():
    real_true = 0
    fake_true = 0
    fake_false = 0
    for x in range(0, len(my_data)):

        temp_real_true = 0
        temp_fake_true = 0
        for y in range(0, len(my_data[x]['labels'])):

            if my_data[x]['labels'][y] in ori_data[x]['labels'] :
                temp_real_true += 1
            else:
                temp_fake_true += 1

        real_true += temp_real_true
        fake_true += temp_fake_true
        fake_false += len(ori_data[x]['labels']) - temp_real_true

    print(real_true)
    print(fake_true)
    print(fake_false)
    Precision = real_true / (real_true + fake_true)
    Recall = real_true / (real_true + fake_false)
    f1 = (2 * (Precision * Recall)) / (Precision + Recall)
    print()

def evaluate_relation_zucheng():
    real_true = 0
    fake_true = 0
    fake_false = 0
    for x in range(0, len(my_data)):

        temp_real_true = 0
        temp_fake_true = 0
        for y in range(0, len(my_data[x]['labels'])):
            if my_data[x]['labels'][y]['r'] in ['n to 1']:
                if my_data[x]['labels'][y] in ori_data[x]['labels'] :
                    temp_real_true += 1
                else:
                    temp_fake_true += 1

        real_true += temp_real_true
        fake_true += temp_fake_true
        fake_false += len(ori_data[x]['labels']) - temp_real_true

    print(real_true)
    print(fake_true)
    print(fake_false)
    Precision = real_true / (real_true + fake_true)
    Recall = real_true / (real_true + fake_false)
    f1 = (2 * (Precision * Recall)) / (Precision + Recall)
    print()

def evaluate_relation_jiagong():
    real_true = 0
    fake_true = 0
    fake_false = 0
    for x in range(0, len(my_data)):

        temp_real_true = 0
        temp_fake_true = 0
        for y in range(0, len(my_data[x]['labels'])):
            if my_data[x]['labels'][y]['r'] in ['n to 1']:
                if my_data[x]['labels'][y] in ori_data[x]['labels'] :
                    temp_real_true += 1
                else:
                    temp_fake_true += 1

        real_true += temp_real_true
        fake_true += temp_fake_true
        fake_false += len(ori_data[x]['labels']) - temp_real_true

    print(real_true)
    print(fake_true)
    print(fake_false)
    Precision = real_true / (real_true + fake_true)
    Recall = real_true / (real_true + fake_false)
    f1 = (2 * (Precision * Recall)) / (Precision + Recall)
    print()

evaluate_entity_mat()
evaluate_entity_refer()
evaluate_entity_spo()
evaluate_entity()
evaluate_relation()
evaluate_relation_jiagong()