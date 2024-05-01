import data_process.TxtReader
import json
import numpy as np
import pandas as pd
import torch
import re
from xml_loader import load_xml,load_xml_list,load_single_xml,load_xml_not_pd
import nltk
from nltk.tokenize import sent_tokenize,word_tokenize
# import nltk.data
from MER_model import MER_model
from MVR_model import MVR_model
from LER_model import LER_model
from tqdm import tqdm
from MR_classify_model import MR_classify_model
from process_data import process_data
import time
from classify_model import classify_model
value_ent_path = "value_schema_test.json"
relation_ent_path = "relation_schema.json"
material_ent_path = "material_schema_test.json"
keywords_path = "config/keywords_default.json"
keywords_path = "config/keywords_default_hardness.json"
# keywords_path = "config/keywords_default_Cu.json"
def test_re():
    # print()
    data = []
    with open("train_strength.json",'r',encoding='utf-8') as f:
        temp = f.readlines()
        for line in temp:
            # print(line)
            dic = json.loads(line)
            text = dic['text']
            data.append(dic)
    return data

def load_keywords():
    data = []
    keywords_path = "config/keywords_default_"
    with open(keywords_path,'r',encoding='utf-8') as load_f:
        temp = load_f.readlines()
        for line in temp:
            # print(line)
            dic = json.loads(line)
            data.append(dic)
    return data

def load_keywords_2(target):
    data = []
    keywords_path = "config/keywords_default_"+target+".json"
    with open(keywords_path,'r',encoding='utf-8') as load_f:
        temp = load_f.readlines()
        for line in temp:
            # print(line)
            dic = json.loads(line)
            data.append(dic)
    return data

def load_value_schema():
    with open(value_ent_path,'r',encoding='utf-8') as load_f:
        temp = load_f.readlines()
        for line in temp:
            dic = json.loads(line)
            txt = dic["describe"]
            re_list = dic["re"]
            print(txt)
            print(re_list)
    return re_list

def load_relation_schema():
    re_list = []
    relation_ent_path = 'config/relation_schema.json'
    with open(relation_ent_path,'r',encoding='utf-8') as load_f:
        temp = load_f.readlines()
        for line in temp:

            dic = json.loads(line)

            txt = dic["describe"]
            # re_list.extend(dic["re"])
            re_ = {
                "describe": txt,
                "re_list": dic["re"],
                "value_re_list" : dic['value'],
                "Similar_keywords" : dic['Similar_keywords'],
                "keywords": dic["keywords"]
            }
            re_list.append(re_)
            # re_list = dic["re"]
            # print(txt)
            # print(re_list)
    return re_list

def load_material_schema():
    re_list = []
    with open(material_ent_path,'r',encoding='utf-8') as load_f:
        temp = load_f.readlines()
        for line in temp:
            dic = json.loads(line)
            txt = dic["describe"]
            re_list.extend(dic["re"])
            # re_list = dic["re"]
            # print(txt)
            # print(re_list)
    return re_list



def replace_space(data,relation_re_list):
    # print(data)
    data = data.replace(" ", "").replace(" ", "").replace(" ", " ")
    data = data.replace(" wt.% ", "wt.%").replace(" wt% ", "wt%").replace(" wt% ", "wt%").replace("wt.%", "wt%").replace("wt.% ", "wt%")
    data = data.replace("at.% ", "at.%")
    data = data.replace("vol.% ", "vol%").replace("vol% ", "vol%").replace(" vol%", "vol%")
    data = data.replace(" wt%","wt%")
    for relations in relation_re_list:
        re_list = relations['value_re_list']
        for i in re_list:
            data = data.replace(" "+i,i)
    data = data.replace(" × ","×")
    data = data.replace(" /", "/")
    data = data.replace("(m·K)","mK")
    data = data.replace("m·K","mK")
    data = data.replace(" ± ","±")
    data = data.replace("% IACS", "%IACS")
    data = data.replace(","," ,")
    data = data.replace(" + ","+")
    return data




def easy_classify_mne_value(data):
    # print(data)
    temp = data
    for index, row in temp.iterrows():
        value_entity = row['value_entity']
        # print(value_entity)
        material_entity = row['material_entity']
        if len(material_entity.split("[SEP]")) == 1 and len(value_entity.split(",")) == 1:
            # print()
            # print(row)
            continue
        else:
            # print(len(material_entity), len(value_entity))
            # print("-------not easy to classify------")
            # print(row)
            temp.loc[index,'easy_to_classify'] = 1
    return temp



def check_product(data):
    # print(data)
    mat_ent = 0
    val_ent = 0
    perf_ent = 0
    for i in data:
        if 'MAT' == i['ent_type'] :
            mat_ent = mat_ent + 1
        if 'Val' == i['ent_type']:
            val_ent = val_ent + 1
        if 'Perf' == i['ent_type']:
            perf_ent = perf_ent + 1
    # print(mat_ent,val_ent,perf_ent)
    if mat_ent * perf_ent == val_ent and mat_ent != 0 and val_ent != 0 and perf_ent != 0:
        return True
    else:
        return False

def check_type(data):
    # print(data)
    mat_ent = 0
    val_ent = 0
    perf_ent = 0
    for i in data:
        if 'MAT' == i['ent_type'] :
            mat_ent = mat_ent + 1
        if 'Val' == i['ent_type']:
            val_ent = val_ent + 1
        if 'Perf' == i['ent_type']:
            perf_ent = perf_ent + 1
    # print(mat_ent,val_ent,perf_ent)
    if mat_ent != 0 and val_ent != 0 and perf_ent != 0:
        return True
    else:
        return False

def clean_data(data):
    for i in data:
        if 'Val' == i['ent_type']:
            i['ent'] = i['ent'].replace("(","").replace(")","")
            i['ent'] = i['ent'].replace("~", "").replace("∼","")
    return data



def way3(year=None):
    data_out = []
    data_process = process_data()
    relation_re_list = load_relation_schema()

    year = '工艺数据2'
    path = "../../data/xml/"+str(year)
    filenames = load_xml_list(path)
    ler_model = LER_model()

    targets = ['strength','hardness','TC','EC','CTE']
    for target in targets:
        out_data = []
        keywords = load_keywords_2(target)
        print(keywords)
        for filename in tqdm(filenames):
            data = load_single_xml(filename)
            for index, row in data.iterrows():
                paper = {
                    "doi": row['doi'].replace(path + "\\", "").replace(".txt", ""),
                    "text": row['text']
                }
                paragraphs = paper['text']
                for paragraph in paragraphs:
                    sent_tokenize_list = sent_tokenize(paragraph)
                    for sent_index in range(len(sent_tokenize_list)):
                        text = sent_tokenize_list[sent_index]
                        text = data_process.replace_space(text, relation_re_list)
                        sentence_data = {
                            'doi': paper['doi'],
                            'text': text,
                            'text_words_token': text.split(" "),
                            'ents': []
                        }
                        for keyword in keywords:
                            for i in keyword['keywords']:
                                entities = ler_model.LER_regular(text, i)
                                if len(entities) > 0:
                                    for ent in entities:
                                        ent['ent_type'] = keyword['entity_type']
                                        sentence_data['ents'].append(ent)
                                    break
                        if len(sentence_data['ents']) > 0:
                            sentence_data['ents'] = clean_data(sentence_data['ents'])
                            if check_product(sentence_data['ents']):
                                out_data.append(sentence_data)
        out_data = pd.DataFrame(out_data)
        output_path = 'test/工艺性能_'+target+'.xlsx'
        out_data.to_excel(output_path, encoding='utf-8')
    return True


def way4(year=None):
    data_process = process_data()
    relation_re_list = load_relation_schema()
    # year = '工艺数据2'
    # path = "../../data/xml/"+str(year)
    # path = "I:/数据集/elsevier下载/xml/" + str(year)
    path = "I:/数据集/elsevier下载/xml/汇总"
    ler_model = LER_model()
    targets = ['strength','hardness','TC','EC','CTE']
    # path = "../../data/xml/" + str(year)
    print("--------------loading data----------")
    data = load_xml_not_pd(path)
    print("--------------loading success----------")

    for target in targets:
        out_data = []
        out_data_triple = []
        keywords = load_keywords_2(target)
        print(keywords)
        for row in tqdm(data):
            paper = {
                "doi": row['doi'].replace(path + "\\", "").replace(".txt", ""),
                "text": row['text']
            }
            paragraphs = paper['text']
            paragraph_id = 0
            for paragraph in paragraphs:
                paragraph_id = paragraph_id + 1
                sent_tokenize_list = sent_tokenize(paragraph)
                for sent_index in range(len(sent_tokenize_list)):
                    text = sent_tokenize_list[sent_index]
                    text = data_process.replace_space(text, relation_re_list)
                    sentence_data = {
                        'doi': paper['doi'],
                        'text': text,
                        'text_words_token': text.split(" "),
                        'ents': [],
                        'id':str(paragraph_id) + '_' + str(sent_tokenize_list)
                    }
                    for keyword in keywords:
                        for i in keyword['keywords']:
                            entities = ler_model.LER_regular(text, i)
                            if len(entities) > 0:
                                for ent in entities:
                                    ent['ent_type'] = keyword['entity_type']
                                    sentence_data['ents'].append(ent)
                                break
                    if len(sentence_data['ents']) > 0:
                        sentence_data['ents'] = clean_data(sentence_data['ents'])
                        # if check_product(sentence_data['ents']):
                        if check_type(sentence_data['ents']):
                            out_data_triple.append(sentence_data)
                        out_data.append(sentence_data)
        out_data = pd.DataFrame(out_data)
        out_data_triple = pd.DataFrame(out_data_triple)
        output_path = 'test/工艺性能'+target+'.xlsx'
        out_data.to_excel(output_path, encoding='utf-8')
        output_path = 'test/工艺性能三元组汇总' +'_'+ target + '.xlsx'
        out_data_triple.to_excel(output_path,encoding='utf-8')
    return True



if __name__ == '__main__':
    # result = way3()
    result = way4()
    # year = ['2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022']
    # for i in year:
    #     result = way4(i)