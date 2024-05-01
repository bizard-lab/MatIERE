import pandas as pd
import numpy as np
from  chemdataextractor import Document
import re
relation_ent_path = "relation_schema.json"
import json
import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import time
from sklearn.metrics import f1_score,classification_report
from transformers import BertTokenizer,BertModel
from Bert_constellation_classfiy_model import BertBiLSTM_constellation_classfily
from Bert_consterllation_Dataset import Bert_consterllation_Dataset
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = torch.load('outputs/2022-06-08-3.pth')

def many_to_many_classfiy():
    constellation_data = []
    relations = load_relation_schema()
    print(relations)
    data = pd.read_excel('out/2022525/total_out_2022623.xlsx')
    datas = []
    out_data = []
    we = 0
    a = 0
    for index,row in data.iterrows():
        # chmes_check(row)
        if row['easy_to_classify'] == 1:
            # print(row['source_text'])
            re_ = []
            relation_id = 0
            for relation in relations:
                for i in relation['re_list']:
                    relation_text = re.findall(i, row['source_text'])
                    if len(relation_text) != 0:
                        re_.append({
                            'relation': relation_text[0],
                            'Similar_keywords': relation['Similar_keywords']
                        })
                        break
                relation_id = relation_id + 1
            if (len(row['material entity'].split(",")) * len(re_)) == len(row['value entity'].split(",")):
                for item in re_:
                    if item['Similar_keywords'][0] >= 0:
                        # print(row['source_text'], row['material entity'], row['value entity'], re_)
                        datas.append({
                            'source': row['source'],
                            'text' : row['source_text'],
                            'relation':re_,
                            'materials' : row['material entity'].split(','),
                            'materialIndexes': row['ent index'].split('#'),
                            'values':row['value entity'].split(','),
                            'valueIndexes' : row['value index'].split("#")
                        })
                a = a + 1
            else:
                d = check_hiding(row, relations)
                if (len(row['material entity'].split(",")) * len(re_)) == len(d):
                    # print('wow',row['source_text'], row['material entity'], row['value entity'], re_,d)
                    datas.append({
                        'source':row['source'],
                        'text': row['source_text'],
                        'relation': re_,
                        'materials': row['material entity'].split(','),
                        'materialIndexes': row['ent index'].split('#'),
                        'values': d,
                        'valueIndexes': []
                    })
                    a = a +1
                else:
                    we = we + 1
                    out_data.append(
                        {
                            'publisher': 'elsevier',
                            'source': row['source'],
                            'source_text': row['source_text'],
                            'material_entity': row['material entity'],
                            'ent_index': row['ent index'],
                            'relation_text': row['relation text'],
                            'value_entity': row['value entity'],
                            'value_index': row['value index'],
                            'easy_to_classify': row['easy_to_classify']
                        }
                    )

        else:
            # print(row)
            out_data.append(
                {
                        'publisher': 'elsevier',
                        'source': row['source'],
                        'source_text': row['source_text'],
                        'material_entity': row['material entity'],
                        'ent_index': row['ent index'],
                        'relation_text': row['relation text'],
                        'value_entity': row['value entity'],
                        'value_index':row['value index'],
                        'easy_to_classify' : row['easy_to_classify']
                }
            )
            # if len(row['material entity'].split(",")) != len(row['value entity'].split(",")):
            #     d = check_hiding(row,relations)
    out_data = pd.DataFrame(out_data)
    print(len(out_data))
    print(out_data)
    print('-------------------------')
    sorted_data = constellation_many_to_manty_classfily(datas)
    print(sorted_data)
    print('m = v finish')
    out_data = pd.concat([out_data,sorted_data])
    print(we)
    print(a)
    out_data.to_excel('out/2022525/out_.xlsx',encoding='utf-8')
    # 'publisher': 'elsevier',
    # 'source': data['source'],
    # 'source_text': text,
    # 'material_entity': row['material_ent'],
    # 'ent_index': str(row['material_idx']) + ',' + str(int(row['material_idx']) + len(row['material_ent'])),
    # 'relation_text': relation_sort[relation_sorted_index]['text'],
    # 'value_entity': value_ent_index.loc[2 * index + relation_sorted_index]['value_ent']

def last_step():
    relations = load_relation_schema()
    # print(relations)
    data = pd.read_excel('out/2022525/out_.xlsx')
    out_data = []
    for index,row in data.iterrows():
        if row['easy_to_classify'] == 1:
            if len(row['material_entity'].split(",")) == 1 and len(row['value_entity'].split(",")) > 1:
                greedy_out = greedy_classfiy(row)
                for i in greedy_out:
                    out_data.append(i)
            else:
                shortest_out = shortest_classfiy(row)
                for i in shortest_out:
                    out_data.append(i)
        else:
            out_data.append({
                'publisher': 'elsevier',
                'source': row['source'],
                'source_text': row['source_text'],
                'material_entity': row['material_entity'],
                'ent_index': row['ent_index'],
                'relation_text': row['relation_text'],
                'value_entity': row['value_entity'],
                'value_index': row['value_index'],
                'easy_to_classify': row['easy_to_classify']
            })
    print(out_data)
    out_data = pd.DataFrame(out_data)
    out_data.to_excel("final_out.xlsx",encoding='utf-8')


def shortest_classfiy(data):
    # print(data)
    out_data = []
    if check_from_to(data['source_text']):
        print("from----",data)
    else:
        print(data)
        shortest_data = []
        material_entities = data['material_entity'].split(",")
        material_indexes = data['ent_index'].split("#")
        value_entities = data['value_entity'].split(",")
        value_indexes = data['value_index'].split("#")
        for i in range(len(material_entities)):
            for j in range(len(value_entities)):
                if i < (len(material_indexes) - 1):
                    shortest_data.append({
                        'ent_1': material_entities[i],
                        'ent_1_index': material_indexes[i],
                        'ent_2': value_entities[j],
                        'ent_2_index': value_indexes[j],
                        'distance': abs(int(material_indexes[i].split(',')[1]) - int(value_indexes[j].split(',')[0]))
                    })
        shortest_data = pd.DataFrame(shortest_data)
        shortest_data.sort_values(by='distance',ignore_index=True)
        for i in range(len(value_entities)):
            out_data.append({
                'publisher': 'elsevier',
                'source': data['source'],
                'source_text': data['source_text'],
                'material_entity': shortest_data.loc[i]['ent_1'],
                'ent_index': shortest_data.loc[i]['ent_1_index'],
                'relation_text': data['relation_text'],
                'value_entity': shortest_data.loc[i]['ent_2'],
                'value_index': shortest_data.loc[i]['ent_2_index'],
                'easy_to_classify': 4
            })
            # out_data.append(shortest_data.loc[i])
        print(out_data)
        # out_data = pd.DataFrame(out_data)
        return out_data


def shortest_classfiy_debug():
    # print(data)
    data = {
        'source_text':'The yield strength of the Cu-Ni alloys increases from 34MPa for the Cu-5at%Ni alloy to 65MPa for the Cu-20at%Ni alloy and the ultimate tensile strength increases from 212MPa to 286MPa.',
        'material_entity' : 'Cu-Ni,Cu-5at%Ni,Cu-20at%Ni',
        'ent_index':'26,31#68,77#101,111',
        'relation_text':'ultimate tensile strength',
        'value_entity':'34MPa,65MPa,212MPa,286MPa',
        'value_index':'54,59#87,92#167,173#177,183'
    }
    out_data = []
    if check_from_to(data['source_text']):
        print("from----",data)
    else:
        print(data)
        shortest_data = []
        relation_index = data['source_text'].find(data['relation_text'])
        material_entities = data['material_entity'].split(",")
        material_indexes = data['ent_index'].split("#")
        value_entities = data['value_entity'].split(",")
        value_indexes = data['value_index'].split("#")
        for i in range(len(material_entities)):
            for j in range(len(value_entities)):
                # print(value_indexes[j].split(',')[0], relation_index)
                if int(value_indexes[j].split(',')[0]) > int(relation_index):
                    shortest_data.append({
                        'ent_1': material_entities[i],
                        'ent_1_index': material_indexes[i],
                        'ent_2': value_entities[j],
                        'ent_2_index': value_indexes[j],
                        'distance': int(abs( (int(material_indexes[i].split(',')[0])+int(material_indexes[i].split(',')[1]) )/2 - (int(value_indexes[j].split(',')[0])+int(value_indexes[j].split(',')[1]))/2))
                    })
        print(shortest_data)
        shortest_data = pd.DataFrame(shortest_data)
        shortest_data = shortest_data.sort_values(by='distance',ignore_index=True)
        print(shortest_data)
        for i in range(len(value_entities)):
            out_data.append({
                'source_text': data['source_text'],
                'material_entity': shortest_data.loc[i]['ent_1'],
                'ent_index': shortest_data.loc[i]['ent_1_index'],
                'relation_text': data['relation_text'],
                'value_entity': shortest_data.loc[i]['ent_2'],
                'value_index': shortest_data.loc[i]['ent_2_index'],
                'easy_to_classify': 4
            })
            # out_data.append(shortest_data.loc[i])
        print(out_data)
        # out_data = pd.DataFrame(out_data)
        return out_data



def check_from_to(text):
    if 'From ' in text:
        index = text.find('From ' )
        if ' to ' in text[index:-1]:
            to_index = text[index:-1].find(' to ')
            if to_index - index <5:
                return True
            else:
                return False
    elif ' from ' in 'text':
        index = text.find(' from ')
        if ' to ' in text[index:-1]:
            to_index = text[index:-1].find(' to ')
            if to_index - index < 5:
                return True
            else:
                return False
    else:
        return False

shortest_classfiy_debug()
def greedy_classfiy(data):
    # print(data)
    value = data['value_entity'].split(",")
    value_index = data['value_index'].split("#")
    out_data = []
    for i in range(len(value)):
        out_data.append({
            'publisher': 'elsevier',
            'source': data['source'],
            'source_text': data['source_text'],
            'material_entity': data['material_entity'],
            'ent_index': data['ent_index'],
            'relation_text': data['relation_text'],
            'value_entity': value[i],
            'value_index': value_index[i],
            'easy_to_classify': 3
        })
    # out_data = pd.DataFrame(out_data)
    return out_data




def many_to_many_classfiy_debug():
    constellation_data = []
    relations = load_relation_schema()
    print(relations)

    row = {
        'source_text': 'After alloying with 0.5wt%Cu, the YS and UTS values increased significantly, up to 113.1 and 164.2MPa, respectively.',
        'value entity':'164.2MPa',
        'material entity':'0.5wt % Cu',
        'value index': '93,101',
        'describe' : '拉伸强度'
    }
    re_ = []
    relation_id = 0
    for relation in relations:
        for i in relation['re_list']:
            relation_text = re.findall(i, row['source_text'])
            if len(relation_text) != 0:
                re_.append({
                    'relation': relation_text[0],
                    'Similar_keywords': relation['Similar_keywords']
                })
                break
        relation_id = relation_id + 1
    print(re_)
    print((len(row['material entity'].split(",")) * len(re_)))
    if (len(row['material entity'].split(",")) * len(re_)) == len(row['value entity'].split(",")):
        for item in re_:
            if item['Similar_keywords'][0] >= 0:
                print(row['source_text'], row['material entity'], row['value entity'], re_)
    else:
        d = check_hiding(row, relations)
        print(d)
        if (len(row['material entity'].split(",")) * len(re_)) == len(d):
            print('wow',row['source_text'], row['material entity'], row['value entity'], re_,d)
        if (len(row['material entity'].split(",")) * len(re_)) < len(d):
            print('waiting check',row['source_text'], row['material entity'], row['value entity'], re_,d)
    # if len(row['material entity'].split(",")) != len(row['value entity'].split(",")):
    #     d = check_hiding(row,relations)

def chmes_check(row):
    doc = Document(row['source_text'])
    print(row)
    print(row['source_text'])
    print(doc.cems)

def load_relation_schema():
    re_list = []
    with open(relation_ent_path,'r',encoding='utf-8') as load_f:
        temp = load_f.readlines()
        for line in temp:
            dic = json.loads(line)
            txt = dic["describe"]
            re_ = {
                "describe": txt,
                "re_list": dic["re"],
                "value_re_list" : dic['value'],
                "Similar_keywords" : dic['Similar_keywords'],
                "keywords" : dic["keywords"]
            }
            re_list.append(re_)
    return re_list

def check_relation(row,re_list):
    print(re_list)

def check_hiding(row,re_list):
    value_idx = row['value index'].split("#")
    value_ent = row['value entity'].split(",")
    value_des = row['describe']
    value_ = []
    hide_left_box = []
    for idx in value_idx:
        for re_item in re_list:
            if value_des == re_item['describe']:
                keywords = re_item['keywords']
        # print(keywords)
        left_side = int(idx.split(",")[0])
        right_side = int(idx.split(",")[1])
        # print(row['source_text'][left_side:right_side])
        if 'and ' in row['source_text'][left_side - 4:left_side]:
            # print()
            re_str = r'([0-9]+\.[0-9]*|-?[0-9]+)'
            value_ent = re.findall(re_str, row['source_text'])
            # check_constellation_relation(row['source_text'],value_ent)
            target_value = re.findall(re_str, row['source_text'][left_side:right_side])[0]
            for i in value_ent:
                # print('----',i,'----')
                if check_constellation_relation(row['source_text'],i,left_side) == True:
                    if i not in row['value entity']:
                        # print('yes',row['source_text'],i,row['source_text'][left_side:right_side])
                        value_.append(i)
            # if len(row['material entity'].split(',')) == (len(value_) + 1):
            value_.append(row['source_text'][left_side:right_side])
            # print('yes', row['source_text'], value_,row['material entity'],len(value_))
    return value_
                # if value_left < left_side:
                #     print('yes',row['source_text'],i,row['source_text'][left_side:right_side])
def check_hiding_material():
    print("")

def check_constellation_relation(text,ent_1,ent_2_left):
    ent_1_left = text.find(ent_1)
    if ent_1_left < ent_2_left:
        if len(text[ent_1_left:ent_2_left].split(' ')) < 6:
            if 'and' in text[ent_1_left:ent_2_left]:
                if ',' == text[len(ent_1)] or ' ' == text[len(ent_1)]:
                    return True
        else:
            return False
    else:
        return False





def cut_likehood(material_entities):
    temp = material_entities
    material_entities = pd.DataFrame(material_entities)
    material_entities = material_entities.drop_duplicates()

    return material_entities
    # print(material_entities,temp)



def constellation_many_to_manty_classfily(text,material_ent,value_ent):
    material_ent_index = []
    value_ent_index = []
    for i in material_ent:
        idx = text.find(i)
        material_ent_index.append({
            'material_ent': i,
            'material_idx': idx
        })
    for i in value_ent:
        idx = text.find(i)

        value_ent_index.append({
            'value_ent' : i,
            'value_idx' : idx
        })
    value_ent_index = pd.DataFrame(value_ent_index)
    material_ent_index = pd.DataFrame(material_ent_index)
    value_ent_index = value_ent_index.sort_values(by='value_idx')
    material_ent_index = material_ent_index.sort_values(by='material_idx')
    if len(material_ent_index) == len(value_ent_index):
        print(value_ent_index,material_ent_index)

def constellation_many_to_manty_classfily(datas):
    # print(datas)
    test = 0
    test2 = 0
    sorted_data = []
    for data in datas:
        # print(data)
        material_ent_index = []
        value_ent_index = []
        text = data['text']
        material_ent = data['materials']
        value_ent = data['values']
        if len(data['relation']) == 1:
            if len(data['valueIndexes']) != 0:
                for i in range(len(material_ent)):

                    material_ent_index.append({
                        'material_ent': material_ent[i],
                        'material_idx': int(data['materialIndexes'][i].split(',')[0])
                    })
                for i in range(len(value_ent)):
                    value_ent_index.append({
                        'value_ent': value_ent[i],
                        'value_idx': int(data['valueIndexes'][i].split(',')[0])
                    })
            else:
                for i in range(len(material_ent)):
                    material_ent_index.append({
                        'material_ent': material_ent[i],
                        'material_idx': int(data['materialIndexes'][i].split(',')[0])
                    })
                for i in value_ent:
                    idx = text.find(i)
                    value_ent_index.append({
                        'value_ent': i,
                        'value_idx': int(idx)
                    })
            value_ent_index = pd.DataFrame(value_ent_index)
            material_ent_index = pd.DataFrame(material_ent_index)
            value_ent_index = value_ent_index.sort_values(by='value_idx')
            material_ent_index = material_ent_index.sort_values(by='material_idx')
            # if len(material_ent_index) == len(value_ent_index):
            #     print(value_ent_index, material_ent_index)
            for index, row in material_ent_index.iterrows():
                sorted_data.append({
                    'publisher': 'elsevier',
                    'source': data['source'],
                    'source_text' : text,
                    'material_entity':row['material_ent'] ,
                    'ent_index': str(row['material_idx']) + ',' + str(int(row['material_idx']) + len(row['material_ent'])),
                    'relation_text':data['relation'][0]['relation'],
                    'value_entity': value_ent_index.loc[index]['value_ent'],
                    'value_index': str(value_ent_index.loc[index]['value_idx']) + ',' + str(int(value_ent_index.loc[index]['value_idx']) + len(value_ent_index.loc[index]['value_ent'])),
                    'easy_to_classify': 0
                })
        else:
            test = test + 1
            relations_count = []
            for relation_item in data['relation']:
                count = text.count(relation_item['relation'])
                relations_count.append(count)
            relations = []
            if len(relations_count) == len(data['relation']):
                for relation_item in data['relation']:
                    if relation_item['Similar_keywords'][0] >= 0:
                        relations.append({
                            'text': relation_item['relation'],
                            'left': text.find(relation_item['relation']),
                            'right': text.find(relation_item['relation']) + len(relation_item['relation'])
                        })
                if len(relations)>1:
                    test2 = test2 + 1
                    left = 0
                    right = 0
                    if relations[0]['left'] < relations[1]['left']: #比较两者的先后关系
                        left = relations[0]
                        right = relations[1]
                    else:
                        left = relations[1]
                        right = relations[0]
                    # relations
                    relation_sort = [left,right]
                    if 'and' in text[left['right']:right['left']] or ',' in text[left['right']:right['left']]:
                        if len(text[left['right']:right['left']].split(' ')) < 7:
                            # print(data)
                            if len(data['valueIndexes']) != 0:
                                for i in range(len(material_ent)):
                                    material_ent_index.append({
                                        'material_ent': material_ent[i],
                                        'material_idx': int(data['materialIndexes'][i].split(',')[0])
                                    })
                                for i in range(len(value_ent)):
                                    value_ent_index.append({
                                        'value_ent': value_ent[i],
                                        'value_idx': int(data['valueIndexes'][i].split(',')[0])
                                    })
                            else:
                                for i in range(len(material_ent)):
                                    material_ent_index.append({
                                        'material_ent': material_ent[i],
                                        'material_idx': int(data['materialIndexes'][i].split(',')[0])
                                    })
                                for i in value_ent:
                                    idx = text.find(i)
                                    value_ent_index.append({
                                        'value_ent': i,
                                        'value_idx': int(idx)
                                    })
                            value_ent_index = pd.DataFrame(value_ent_index)
                            material_ent_index = pd.DataFrame(material_ent_index)
                            # print('---',material_ent_index,'---')
                            value_ent_index = value_ent_index.sort_values(by='value_idx')
                            material_ent_index = material_ent_index.sort_values(by='material_idx',axis = 0,ascending = True)
                            # print(value_ent_index)
                            # print(material_ent_index)
                            for index, row in material_ent_index.iterrows():
                                for relation_sorted_index in range(len(relation_sort)):
                                    sorted_data.append({
                                        'publisher': 'elsevier',
                                        'source': data['source'],
                                        'source_text': text,
                                        'material_entity': row['material_ent'],
                                        'ent_index': str(row['material_idx']) + ',' + str(int(row['material_idx']) + len(row['material_ent'])),
                                        'relation_text': relation_sort[relation_sorted_index]['text'],
                                        'value_entity': value_ent_index.loc[2*index+relation_sorted_index]['value_ent'],
                                        'value_index': str(value_ent_index.loc[index]['value_idx']) + ',' + str(int(value_ent_index.loc[index]['value_idx']) + len(value_ent_index.loc[index]['value_ent'])),
                                        'easy_to_classify': 0
                                    })
                        else:
                            if len(data['valueIndexes']) != 0:
                                # print(text)
                                for i in range(len(material_ent)):
                                    material_ent_index.append({
                                        'material_ent': material_ent[i],
                                        'material_idx': int(data['materialIndexes'][i].split(',')[0])
                                    })
                                for i in range(len(value_ent)):
                                    value_ent_index.append({
                                        'value_ent': value_ent[i],
                                        'value_idx': int(data['valueIndexes'][i].split(',')[0])
                                    })
                            else:
                                for i in range(len(material_ent)):
                                    material_ent_index.append({
                                        'material_ent': material_ent[i],
                                        'material_idx': int(data['materialIndexes'][i].split(',')[0])
                                    })
                                for i in value_ent:
                                    idx = text.find(i)
                                    value_ent_index.append({
                                        'value_ent': i,
                                        'value_idx': idx
                                    })
                            value_ent_index = pd.DataFrame(value_ent_index)
                            material_ent_index = pd.DataFrame(material_ent_index)
                            # print('---',material_ent_index,'---')
                            value_ent_index = value_ent_index.sort_values(by='value_idx')
                            material_ent_index = material_ent_index.sort_values(by='material_idx',axis = 0,ascending = True)
                            # print(value_ent_index)
                            # print(material_ent_index)
                            value_block = []
                            for relation_sorted_index in range(0,len(relation_sort)):
                                if relation_sorted_index + 1 <len(relation_sort):
                                    value_block.append({
                                        'left': relation_sort[relation_sorted_index]['right'],
                                        'right': relation_sort[relation_sorted_index + 1]['left'],
                                        'text': relation_sort[relation_sorted_index]['text']
                                    })
                                else:
                                    value_block.append({
                                        'left': relation_sort[relation_sorted_index]['right'],
                                        'right': len(text),
                                        'text': relation_sort[relation_sorted_index]['text']
                                    })
                            for value_index,value_row in value_ent_index.iterrows():
                                for index, row in material_ent_index.iterrows():
                                    for relation_sorted_index in value_block:
                                        if int(relation_sorted_index['left']) < int(value_row['value_idx']) and int(relation_sorted_index['right']) > int(value_row['value_idx']):
                                            sorted_data.append({
                                                'publisher': 'elsevier',
                                                'source': data['source'],
                                                'source_text': text,
                                                'material_entity': row['material_ent'],
                                                'ent_index': str(row['material_idx']) + ',' + str(int(row['material_idx']) + len(row['material_ent'])),
                                                'relation_text': relation_sorted_index['text'],
                                                'value_entity': value_row['value_ent'],
                                                'value_index': str(value_ent_index.loc[index]['value_idx']) + ',' + str(int(value_ent_index.loc[index]['value_idx']) + len(value_ent_index.loc[index]['value_ent'])),
                                                'easy_to_classify': 0
                                            })

                            # print('ddd',data,'----',left,right,text[left['right']:right['left']])
                    else:
                        print('sss',data,'----',left,right,text[left['right']:right['left']])
                        sorted_data.append({
                            'publisher': 'elsevier',
                            'source': data['source'],
                            'source_text': data['text'],
                            'material_entity': data['materials'],
                            'ent_index': data['materialIndexes'],
                            'relation_text': data['relation'],
                            'value_entity': data['values'],
                            'value_index': data['valueIndexes'],
                            'easy_to_classify': 1
                        })

                # print(relation_item[])

            # print(data)
    print(test,test2)
    # print(sorted_data)
    sorted_data = pd.DataFrame(sorted_data)
    sorted_data = sorted_data.drop_duplicates()
    # print(sorted_data)
    # sorted_data.to_excel('many_to_many_out.xlsx',encoding='utf-8')
    return sorted_data

def from_to_classfily(datas):
    print()

# many_to_many_classfiy()
# print('----------')

# last_step()

# many_to_many_classfiy_debug()




