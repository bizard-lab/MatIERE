
import re
import json
import pandas as pd
import numpy as np
from chemdataextractor import Document
import re
import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = torch.load('outputs/2022-06-08-3.pth')

class MR_classify_model():
    def __init__(self):
        self.relation_ent_path = 'config/relation_schema_default.json'
        self.material_ent_path = 'config/material_schema_default.json'
        self.material_re_list = self.load_material_schema()

    def load_material_schema(self):
        re_list = []
        with open(self.material_ent_path, 'r', encoding='utf-8') as load_f:
            temp = load_f.readlines()
            for line in temp:
                dic = json.loads(line)
                txt = dic["describe"]
                re_list.extend(dic["re"])

        return re_list

    def load_relation_schema(self):
        re_list = []
        with open(self.relation_ent_path, 'r', encoding='utf-8') as load_f:
            temp = load_f.readlines()
            for line in temp:
                dic = json.loads(line)
                txt = dic["describe"]
                re_ = {
                    "describe": txt,
                    "re_list": dic["re"],
                    "value_re_list": dic['value'],
                    "Similar_keywords": dic['Similar_keywords'],
                    "keywords": dic["keywords"]
                }
                re_list.append(re_)
        return re_list

    def _is_one_to_many(self):
        data = pd.read_excel('out/temp/all_right_data_2020-2022.xlsx')
        relation_re_list = self.load_relation_schema()
        data_box = []
        for index, row in data.iterrows():
            # print(row)
            easy_to_classfiy = row['easy_to_classify']
            if easy_to_classfiy == 1:
                # print(row)
                is_one_to_many_check = False
                material_ents = row['material_entity'].split("[SEP]")
                value_ents = str(row['value_entity']).split(",")
                if len(material_ents) == 1 and len(value_ents) > 1:
                    is_one_to_many_check = True
                    # print(row)
                    # relation
                    relation = row['relation_text']
                    relation_box = []
                    temp_ = {
                        'relation_index': row['source_text'].find(relation),
                        'relation_text': relation,
                        'desctibe': row['describe']
                    }
                    relation_box.append(temp_)
                    value_ents_index = str(row['value_token_index']).split("#")
                    value_ents_index_char = str(row['value_char_index']).split("#")
                    for relation_text in relation_re_list:
                        # print(relation_text)
                        re_list = relation_text['re_list']
                        if relation in re_list:
                            # print(re_list)
                            for similar_keywords_list_idx in relation_text['Similar_keywords']:
                                similar_keywords_list = relation_re_list[similar_keywords_list_idx]['re_list']
                                for i in similar_keywords_list:
                                    # print(i)
                                    if row['source_text'].find(i) > 0:
                                        similar_keywords = i
                                        similar_keywords_index = row['source_text'].find(i)
                                        temp = {
                                            'relation_index': int(similar_keywords_index),
                                            'relation_text': similar_keywords,
                                            'desctibe': relation_re_list[similar_keywords_list_idx]['describe']
                                        }
                                        relation_box.append(temp)
                                        break
                    # print(relation_box)
                    relation_box = pd.DataFrame(relation_box)
                    relation_box = relation_box.sort_values(by='relation_index', ignore_index=True)
                    # zzz = relation_box.sort_values(by='relation_index',ignore_index=True)
                    value_box = []
                    for i in range(len(value_ents_index)):
                        # mid_value_idx = (int(value_ents_index[i].split(",")[0]) + int(
                        #     value_ents_index[i].split(",")[1])) / 2
                        print(value_ents)
                        temp = {
                            'value_ent': value_ents[i],
                            'value_index': value_ents_index[i],
                            'value_index_char':value_ents_index_char[i]
                        }
                        value_box.append(temp)
                    value_box = pd.DataFrame(value_box)
                    value_box = value_box.sort_values(by='value_index', ignore_index=True)
                    # print(relation_box)
                    # # print(zzz)
                    # print(value_box)
                    # print(row['source_text'])
                    for idx in range(len(relation_box)):
                        # print(relation_box.loc[idx]['relation_text'])
                        single_data = {
                            'publisher': row['publisher'],
                            'source': row['source'],
                            'source_text': row['source_text'],
                            "process_text": row['process_text'],
                            "words_token_text": row['words_token_text'],
                            'material_entity': row['material_entity'],
                            "material_char_index": row['material_char_index'],
                            "material_token_index": row['material_token_index'],
                            'relation_text': relation_box.loc[idx]['relation_text'],
                            'value_entity': value_box.loc[idx]['value_ent'],
                            "value_char_index": value_box.loc[idx]['value_index_char'],
                            "value_token_index": str(value_box.loc[idx]['value_index']),
                            'describe': relation_box.loc[idx]['desctibe'],
                            'easy_to_classify': 2
                        }
                        # print("---",single_data)
                        # single_data = pd.DataFrame(single_data)
                        data_box.append(single_data)
                elif len(material_ents) > 1 and len(value_ents) == len(material_ents):
                    # print(row)
                    # print(row['source_text'])
                    temp = row.to_dict()
                    data_box.append(temp)
                else:
                    temp = row.to_dict()
                    data_box.append(temp)

            else:
                # print(type(row))
                temp = row.to_dict()
                # print(temp)
                data_box.append(temp)
        data_box = pd.DataFrame(data_box)
        data_box = data_box.drop_duplicates()
        data_box.to_excel('out/temp/temp_file2.xlsx', encoding='utf-8')

    def many_to_many_classfiy(self):
        constellation_data = []
        relations = self.load_relation_schema()
        # print(relations)
        data = pd.read_excel('out/temp/temp_file2.xlsx')
        datas = []
        out_data = []
        we = 0
        a = 0
        for index, row in data.iterrows():
            # chmes_check(row)
            if row['easy_to_classify'] == 1:

                re_ = []
                relation_id = 0
                for relation in relations:
                    for i in relation['re_list']:
                        relation_text = re.findall(i, row['source_text'])
                        if len(relation_text) != 0 and relation['Similar_keywords'][0]>0:
                            re_.append({
                                'relation': relation_text[0],
                                'Similar_keywords': relation['Similar_keywords']
                            })
                            break
                if len(re_) == 0:
                    re_.append({
                        'relation': row['relation_text'],
                        'Similar_keywords': [-1]
                    })
                    relation_id = relation_id + 1
                Similar_keywords_num = 0
                for rel_idx in re_:
                    # print(rel_idx)
                    if rel_idx['Similar_keywords'][0] > 1:
                        Similar_keywords_num = Similar_keywords_num + 1
                # if (len(str(row['material_entity']).split("[SEP]")) * Similar_keywords_num) == len(str(row['value_entity']).split(",")):
                if (len(str(row['material_entity']).split("[SEP]")) * len(re_)) == len(str(row['value_entity']).split(",")):
                # if (len(str(row['material_entity']).split("[SEP]"))) <= len(str(row['value_entity']).split(",")):
                #     print(re_,row['source_text'])
                    for item in re_:
                        if item['Similar_keywords'][0] >= 0:
                            datas.append({
                                'source': row['source'],
                                'text': row['source_text'],
                                'relation': re_,
                                'materials': row['material_entity'].split('[SEP]'),
                                'materialIndexes': str(row['material_token_index']).split('#'),
                                'values': row['value_entity'].split(','),
                                'valueIndexes': str(row['value_token_index']).split("#")
                            })
                        else:
                            print(row)
                            datas.append({
                                'source': row['source'],
                                'text': row['source_text'],
                                'relation': re_,
                                'materials': row['material_entity'].split('[SEP]'),
                                'materialIndexes': row['material_token_index'].split('#'),
                                'values': row['value_entity'].split(','),
                                'valueIndexes': str(row['value_token_index']).split("#")
                            })
                    a = a + 1
                else:
                    print(row)
                    print(row['material_entity'].split("[SEP]"),re_)
                    print((len(str(row['material_entity']).split("[SEP]")) * len(re_)),len(str(row['value_entity']).split(",")),Similar_keywords_num)
                    d = self.check_hiding(row, relations)
                    if (len(row['material_entity'].split("[SEP]")) * len(re_)) == len(d):
                        datas.append({
                            'source': row['source'],
                            'text': row['source_text'],
                            'relation': re_,
                            'materials': row['material_entity'].split('[SEP]'),
                            'materialIndexes': row['material_token_index'].split('#'),
                            'values': d,
                            'valueIndexes': []
                        })
                        a = a + 1
                    else:
                        we = we + 1
                        out_data.append(
                            {
                                'publisher': 'elsevier',
                                'source': row['source'],
                                'source_text': row['source_text'],
                                "process_text": row['process_text'],
                                "words_token_text": row['words_token_text'],
                                'material_entity': row['material_entity'],
                                "material_char_index": row['material_char_index'],
                                "material_token_index": row['material_token_index'],
                                'relation_text': row['relation_text'],
                                'value_entity': row['value_entity'],
                                "value_char_index": row['value_char_index'],
                                "value_token_index": row['value_token_index'],
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
                        "process_text": row['process_text'],
                        "words_token_text": row['words_token_text'],
                        'material_entity': row['material_entity'],
                        "material_char_index": row['material_char_index'],
                        "material_token_index": row['material_token_index'],
                        'relation_text': row['relation_text'],
                        'value_entity': row['value_entity'],
                        "value_char_index": row['value_char_index'],
                        "value_token_index": row['value_token_index'],
                        'easy_to_classify': row['easy_to_classify']
                    }
                )
        out_data = pd.DataFrame(out_data)
        # print(len(out_data))
        # print(out_data)
        temp_debug = pd.DataFrame(datas).to_excel("debug_.xlsx",encoding='utf-9')
        print('-------------------------')
        sorted_data = self.constellation_many_to_manty_classfily(datas)
        # print(sorted_data)
        print('m = v finish')
        out_data = pd.concat([out_data, sorted_data])
        print(we)
        print(a)
        out_data.to_excel('out/temp/temp_file3.xlsx', encoding='utf-8')


    def last_step(self,save_path='out/final_out'):
        relations = self.load_relation_schema()
        # print(relations)
        data = pd.read_excel('out/temp/temp_file3.xlsx')
        out_data = []
        for index, row in data.iterrows():

            if row['easy_to_classify'] == 1:
                if len(row['material_entity'].split("[SEP]")) == 1 and len(str(row['value_entity']).split(",")) > 1:
                    greedy_out = self.greedy_classfiy(row)
                    for i in greedy_out:
                        out_data.append(i)
                else:
                    shortest_out = self.shortest_classfiy(row)
                    for i in shortest_out:
                        out_data.append(i)
            else:
                out_data.append({
                    'publisher': 'elsevier',
                    'source': row['source'],
                    'source_text': row['source_text'],
                    "process_text": row['process_text'],
                    "words_token_text": row['words_token_text'],
                    'material_entity': row['material_entity'],
                    "material_char_index": row['material_char_index'],
                    "material_token_index": row['material_token_index'],
                    'relation_text': row['relation_text'],
                    'value_entity': row['value_entity'],
                    "value_char_index": row['value_char_index'],
                    "value_token_index": row['value_token_index'],
                    'easy_to_classify': row['easy_to_classify']
                })
        # print(out_data)
        out_data = pd.DataFrame(out_data)
        out_data.to_excel(save_path, encoding='utf-8')

    def shortest_classfiy(self,data):
        # print(data)
        out_data = []
        if self.check_from_to(data['source_text'],data['value_entity']):
            # print("from----", data)
            out_data.append({
                'publisher': 'elsevier',
                'source': data['source'],
                'source_text': data['source_text'],
                "process_text": data['process_text'],
                "words_token_text": data['words_token_text'],
                'material_entity': data['material_entity'],
                'ent_index': data['material_token_index'],
                'relation_text': data['relation_text'],
                'value_entity': data['value_entity'],
                "value_char_index": data['value_char_index'],
                "value_token_index": data['value_token_index'],
                'easy_to_classify': 3
            })
            return out_data
        else:
            # print(data)
            shortest_data = []
            relation_index = data['source_text'].find(data['relation_text'])
            material_entities = data['material_entity'].split("[SEP]")
            material_indexes = str(data['material_token_index']).split("#")
            material_indexes_char = data['material_char_index'].split("#")
            value_entities = str(data['value_entity']).split(",")
            value_indexes =str(int(data['value_token_index'])).split("#")
            value_indexes_char = str(data['value_char_index']).split("#")
            for i in range(len(material_entities)):
                for j in range(len(value_entities)):
                    # if i < (len(material_indexes) - 1):
                    print(value_entities[j],relation_index,str(value_indexes[j]),type(value_indexes[j]))
                    if int(str(value_indexes[j]).split(',')[0]) > int(relation_index):
                        shortest_data.append({
                            'ent_1': material_entities[i],
                            'ent_1_index': material_indexes[i],
                            'ent_1_index_char':material_indexes_char[i],
                            'ent_2': value_entities[j],
                            'ent_2_index': value_indexes[j],
                            'ent_2_index_char' : value_indexes_char[j],
                            'distance': abs(int(material_indexes[i]) - int(value_indexes[j])),
                            'from':'nom'
                        })
            if len(shortest_data) == 0 or len(shortest_data) < len(value_entities):
                for i in range(len(material_entities)):
                    for j in range(len(value_entities)):
                        shortest_data.append({
                            'ent_1': material_entities[i],
                            'ent_1_index': material_indexes[i],
                            'ent_1_index_char': material_indexes_char[i],
                            'ent_2': value_entities[j],
                            'ent_2_index': value_indexes[j],
                            'ent_2_index_char': value_indexes_char[j],
                            'distance': abs(int(material_indexes[i]) - int(value_indexes[j])),
                            'from':'eerrr'
                        })
            shortest_data = pd.DataFrame(shortest_data)
            shortest_data = shortest_data.sort_values(by='distance', ignore_index=True)
            for i in range(len(value_entities)):
                # print(len(value_entities),data,shortest_data)
                out_data.append({
                    'publisher': 'elsevier',
                    'source': data['source'],
                    'source_text': data['source_text'],
                    "process_text": data['process_text'],
                    "words_token_text": data['words_token_text'],
                    'material_entity': shortest_data.loc[i]['ent_1'],
                    "material_char_index": shortest_data.loc[i]['ent_1_index_char'],
                    "material_token_index": shortest_data.loc[i]['ent_1_index'],
                    'relation_text': data['relation_text'],
                    'value_entity': shortest_data.loc[i]['ent_2'],
                    "value_char_index":  shortest_data.loc[i]['ent_2_index_char'],
                    "value_token_index":  shortest_data.loc[i]['ent_2_index'],
                    'easy_to_classify': 4
                })
                # out_data.append(shortest_data.loc[i])
            # print(out_data)
            # out_data = pd.DataFrame(out_data)
            return out_data

    def check_from_to(self,text,values):
        value = str(values).split(",")
        value_idx = []
        #
        for i in value:
            value_idx.append(text.find(i))
        # print('ccccc',value_idx)
        if 'From ' in text:
            index = text.find('From ')
            if ' to ' in text[index:-1]:
                to_index = text[index:-1].find(' to ') + index
                if len(text[index:to_index].split(" ")) < 8:
                    judge = False
                    # print('bbbbb',value_idx,index,to_index)
                    for i in value_idx:
                        print(i,index,to_index,'zzzzz')
                        if i > index and i < to_index:
                            judge = True
                    if judge:
                        return True
                    else:
                        return False
                else:
                    return False
        elif ' from ' in text:
            index = text.find(' from ')
            if ' to ' in text[index:-1]:
                to_index = text[index:-1].find(' to ') + index
                if len(text[index:to_index].split(" ")) < 8:
                    judge = False
                    # print('bbbbb', value_idx, index, to_index)
                    for i in value_idx:
                        # print(i, index, to_index, 'zzzzz')
                        if i > index and i < to_index:
                            judge = True
                    if judge:
                        return True
                    else:
                        return False
                else:
                    return False
        else:
            return False


    def greedy_classfiy(self,data):
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

    def chmes_check(row):
        doc = Document(row['source_text'])
        print(row)
        print(row['source_text'])
        print(doc.cems)


    def check_relation(self,row, re_list):
        print(re_list)

    def check_hiding(self,row, re_list):
        value_idx = row['value_char_index'].split("#")
        value_ent = str(row['value_entity']).split(",")
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
                    if self.check_constellation_relation(row['source_text'], i, left_side) == True:
                        print(row['value_entity'])
                        if i not in row['value_entity']:
                            value_.append(i)
                value_.append(row['source_text'][left_side:right_side])
        return value_
        # if value_left < left_side:
        #     print('yes',row['source_text'],i,row['source_text'][left_side:right_side])

    def check_hiding_material(self):
        print("")

    def check_constellation_relation(self,text, ent_1, ent_2_left):
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

    def cut_likehood(self,material_entities):
        temp = material_entities
        material_entities = pd.DataFrame(material_entities)
        material_entities = material_entities.drop_duplicates()

        return material_entities
        # print(material_entities,temp)

    def constellation_many_to_manty_classfily(self,text, material_ent, value_ent):
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
                'value_ent': i,
                'value_idx': idx
            })
        value_ent_index = pd.DataFrame(value_ent_index)
        material_ent_index = pd.DataFrame(material_ent_index)
        value_ent_index = value_ent_index.sort_values(by='value_idx',ignore_index=True)
        material_ent_index = material_ent_index.sort_values(by='material_idx',ignore_index=True)
        if len(material_ent_index) == len(value_ent_index):
            print(value_ent_index, material_ent_index)

    def constellation_many_to_manty_classfily(self,datas):
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
                value_ent_index = value_ent_index.sort_values(by='value_idx',ignore_index=True)
                material_ent_index = material_ent_index.sort_values(by='material_idx',ignore_index=True)
                for index, row in material_ent_index.iterrows():
                    # print(material_ent_index)
                    # print(value_ent_index,index)
                    sorted_data.append({
                        'publisher': 'elsevier',
                        'source': data['source'],
                        'source_text': text,
                        'material_entity': row['material_ent'],
                        'ent_index': str(row['material_idx']) + ',' + str(
                            int(row['material_idx']) + len(row['material_ent'])),
                        'relation_text': data['relation'][0]['relation'],
                        'value_entity': value_ent_index.loc[index]['value_ent'],
                        'value_index': str(value_ent_index.loc[index]['value_idx']) + ',' + str(int(value_ent_index.loc[index]['value_idx']) + len(value_ent_index.loc[index]['value_ent'])),
                        'easy_to_classify': 5
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
                    if len(relations) > 1:
                        test2 = test2 + 1
                        left = 0
                        right = 0
                        if relations[0]['left'] < relations[1]['left']:  # 比较两者的先后关系
                            left = relations[0]
                            right = relations[1]
                        else:
                            left = relations[1]
                            right = relations[0]
                        # relations
                        relation_sort = [left, right]
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
                                value_ent_index = value_ent_index.sort_values(by='value_idx',ignore_index=True)
                                material_ent_index = material_ent_index.sort_values(by='material_idx', axis=0,
                                                                                    ascending=True,ignore_index=True)
                                # print(value_ent_index)
                                # print(material_ent_index)
                                for index, row in material_ent_index.iterrows():
                                    for relation_sorted_index in range(len(relation_sort)):
                                        # print(text,value_ent_index)
                                        sorted_data.append({
                                            'publisher': 'elsevier',
                                            'source': data['source'],
                                            'source_text': text,
                                            'material_entity': row['material_ent'],
                                            'ent_index': str(row['material_idx']) + ',' + str(
                                                int(row['material_idx']) + len(row['material_ent'])),
                                            'relation_text': relation_sort[relation_sorted_index]['text'],
                                            'value_entity': value_ent_index.loc[2 * index + relation_sorted_index][
                                                'value_ent'],
                                            'value_index': str(value_ent_index.loc[index]['value_idx']) + ',' + str(
                                                int(value_ent_index.loc[index]['value_idx']) + len(
                                                    value_ent_index.loc[index]['value_ent'])),
                                            'easy_to_classify': 5
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
                                value_ent_index = value_ent_index.sort_values(by='value_idx',ignore_index=True)
                                material_ent_index = material_ent_index.sort_values(by='material_idx', ignore_index=True,
                                                                                    )
                                # print(value_ent_index)
                                # print(material_ent_index)
                                value_block = []
                                for relation_sorted_index in range(0, len(relation_sort)):
                                    if relation_sorted_index + 1 < len(relation_sort):
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
                                for value_index, value_row in value_ent_index.iterrows():
                                    for index, row in material_ent_index.iterrows():
                                        for relation_sorted_index in value_block:
                                            if int(relation_sorted_index['left']) < int(value_row['value_idx']) and int(
                                                    relation_sorted_index['right']) > int(value_row['value_idx']):
                                                sorted_data.append({
                                                    'publisher': 'elsevier',
                                                    'source': data['source'],
                                                    'source_text': text,
                                                    'material_entity': row['material_ent'],
                                                    'ent_index': str(row['material_idx']) + ',' + str(
                                                        int(row['material_idx']) + len(row['material_ent'])),
                                                    'relation_text': relation_sorted_index['text'],
                                                    'value_entity': value_row['value_ent'],
                                                    'value_index': str(
                                                        value_ent_index.loc[index]['value_idx']) + ',' + str(
                                                        int(value_ent_index.loc[index]['value_idx']) + len(
                                                            value_ent_index.loc[index]['value_ent'])),
                                                    'easy_to_classify': 5
                                                })

                                # print('ddd',data,'----',left,right,text[left['right']:right['left']])
                        else:
                            print('sss', data, '----', left, right, text[left['right']:right['left']])
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
        # print(test, test2)
        sorted_data = pd.DataFrame(sorted_data)
        sorted_data = sorted_data.drop_duplicates()
        return sorted_data


    def check_constellation_relation(self,text, ent_1, ent_2_left):
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
    def from_to_classfily(self,data):
        out_data = []
        values = data['value_entity'].split(",")
        value_index = data['value_entity'].split("#")
        materials = data['material_entity'].split("[SEP]")
        material_indexes = data['ent_index'].split("#")

        for i in range(len(values)):
            for j in range(len(materials)):
                if 'for' in data['source_text'][value_index[i].split(",")[1]:material_indexes[j].split(",")[0]] and (int(material_indexes[j].split(",")[0]) - int(value_index[i].split(",")[1])) < 6:
                    out_data.append({
                        'publisher': 'elsevier',
                        'source': data['source'],
                        'source_text': data['text'],
                        'material_entity': materials[j],
                        'ent_index': material_indexes[j],
                        'relation_text': data['relation'],
                        'value_entity': values[i],
                        'value_index': value_index[i],
                        'easy_to_classify': 3
                    })

        for i in range(0,len(values),2):
            ent_1_index = data['source_text'].find(values[i])+len(values[i])
            ent_2_index = data['source_text'].find(values[i+1])
            if 'and' in data['source_text'][ent_1_index:ent_2_index] and abs(ent_2_index-ent_1_index)<10:
                if len(materials)> 1:
                    for mat_idx in range(len(materials)):
                        if mat_idx < (len(materials)-1):
                            if self.check_constellation_relation(data['source_text'],materials[mat_idx],material_indexes[mat_idx+1].split(",")[0]):
                                out_data.append({
                                    'publisher': 'elsevier',
                                    'source': data['source'],
                                    'source_text': data['text'],
                                    'material_entity': materials[mat_idx],
                                    'ent_index': material_indexes[mat_idx],
                                    'relation_text': data['relation'],
                                    'value_entity': values[i],
                                    'value_index': value_index[i],
                                    'easy_to_classify': 3
                                })
                                out_data.append({
                                    'publisher': 'elsevier',
                                    'source': data['source'],
                                    'source_text': data['text'],
                                    'material_entity': materials[mat_idx+1],
                                    'ent_index': material_indexes[mat_idx+1],
                                    'relation_text': data['relation'],
                                    'value_entity': values[i+1],
                                    'value_index': value_index[i+1],
                                    'easy_to_classify': 3
                                })


                #type_1 from vi1 to vj1 to vi2 and vj2
        print()
        #type_2 from vi1 for m1 to vi2 for m2
        print()
        #type_3 from v1 to v2
        print()

        return out_data






