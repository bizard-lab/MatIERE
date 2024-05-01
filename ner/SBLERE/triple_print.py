from MyModel import BertNerModel
from Type_classifier import Type_classifier
from BiLSTM import BertBilstm
from transformers import BertTokenizer
import pickle as pkl
import torch
from config import parsers
import time
import json
# import data_process.TxtReader
import json
import numpy as np
import pandas as pd
import torch
import re
from xml_loader import load_xml
import nltk
from nltk.tokenize import sent_tokenize,word_tokenize
# import nltk.data
from MER_model import MER_model
from MVR_model import MVR_model
from LER_model import LER_model
from MR_classify_model import MR_classify_model
from process_data import process_data
import time
from classify_model import classify_model
value_ent_path = "value_schema_test.json"
relation_ent_path = "relation_schema.json"
material_ent_path = "material_schema_test.json"
keywords_path = "config/keywords_default.json"
# keywords_path = "config/keywords_default_Cu.json"
from tqdm import *
import json
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    level=logging.DEBUG,
                    filename='./logs/test.log',
                    filemode='a')

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

def load_model(model_path, class_num):
    global device
    model = BertNerModel(class_num).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def load_seq_model(model_path, class_num):
    global device
    model = BertBilstm(class_num).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def load_type_model(device, model_path):
    # print(device)
    myModel = Type_classifier().to(device)
    myModel.load_state_dict(torch.load(model_path))
    myModel.eval()
    return myModel

def load_elements():
    data = []
    with open(r'data/Entity Matching/Elements.json', 'r', encoding='utf-8') as load_f:
        temp = json.load(load_f)
    data = temp['elements']
    return data

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

def clean_data(data):
    for i in data:
        if 'Val' == i['ent_type']:
            i['ent'] = i['ent'].replace("(","").replace(")","")
            i['ent'] = i['ent'].replace("~", "").replace("∼","")
    return data

def single_text(text):
    data_process = process_data()
    relation_re_list = load_relation_schema()
    keywords = load_keywords()
    ler_model = LER_model()
    # print("--------------success----------")
    out_data = []
    sent_tokenize_list = sent_tokenize(text)
    for sent_index in range(len(sent_tokenize_list)):
        text = sent_tokenize_list[sent_index]
        text = data_process.replace_space(text, relation_re_list)
        sentence_data = {
            'doi': 'NULL',
            'text': text,
            'text_words_token': text.split(" "),
            'ents': []
        }
        for keyword in keywords:
            for i in keyword['keywords']:
                # print(text, i,"in main")
                entities = ler_model.LER_regular(text, i)
                if len(entities) > 0:
                    ent_arr = []
                    for ent in entities:
                        ent['ent_type'] = keyword['entity_type']
                        if keyword['entity_type'] == 'Perf':
                            if ent['ent'] not in ent_arr:
                                ent_arr.append(ent['ent'])
                                sentence_data['ents'].append(ent)
                        else:
                            sentence_data['ents'].append(ent)
                    break
        if len(sentence_data['ents']) > 0:
            sentence_data['ents'] = clean_data(sentence_data['ents'])
            if check_product(sentence_data['ents']):
                out_data.append(sentence_data)
    out_data = pd.DataFrame(out_data)
    # print(out_data)

    return out_data


def text_class_name(texts, pred, index_label):
    # print('-----------------------------------')
    pred_label = torch.argmax(pred, dim=-1)
    # print(pred_label)
    result = torch.argmax(pred, dim=-1)
    result = result.cpu().numpy().tolist()[0]
    # print(result)
    # print(index_label)
    pred_label = [index_label[i] for i in result]
    # print("模型预测结果：")
    # print(f"文本：{texts}\t预测的类别为：{pred_label[1:len(texts)+1]}")
    tiple = {
        'subject':'',
        'predict':'',
        'object':''
    }
    process_idx = -1
    time_idx = -1
    temperature_idx = -1
    for i in range(len(pred_label[1:len(texts)+1])):
        # print(pred_label[1:len(texts)][i])
        if pred_label[1:len(texts)+1][i] != 'PAD' and 'O' != pred_label[1:len(texts)+1][i]:
            print(texts[i],i, pred_label[i])
            #如果发现无法输出结果，请检查这里的预测类别和最初的预测类别是否相同，因为多次编码有的模型预测的是B-Time有的预测是B-ProcessTime
            if pred_label[i + 1] == 'B-Process':
                process_idx = i
            if pred_label[i + 1] == 'B-ProcessTemperature':
                temperature_idx = i
            if pred_label[i + 1] == 'B-ProcessTime':
                time_idx = i
            # if pred_label[i] == 'B-Process':
            #     process_idx = i
            # if pred_label[i] == 'B-ProcessTemperature':
            #     temperature_idx = i
            # if pred_label[i] == 'B-ProcessTime':
            #     time_idx = i
    # print(process_idx,time_idx,temperature_idx)
    spolist = []
    if temperature_idx != -1 and process_idx != -1:
        # print(texts[process_idx],'process-temperature',texts[temperature_idx])
        spolist.append({
            's':texts[process_idx],
            'p':'process-temperature',
            'o':texts[temperature_idx]
        })
    if time_idx != -1 and process_idx != -1:
        # print(texts[process_idx],'process-time',texts[time_idx])
        spolist.append({
            's': texts[process_idx],
            'p': 'process-time',
            'o': texts[time_idx]
        })
    print(spolist)
    return spolist


# def text_class_name_seq(texts, pred, index_label):
#     pred_label = torch.argmax(pred, dim=-1)
#     # print(pred_label)
#     result = torch.argmax(pred, dim=-1)
#     result = result.cpu().numpy().tolist()[0]
#     # print(result)
#     # print(index_label)
#     pred_label = [index_label[i] for i in result]
#     # print("模型预测结果：")
#     print(f"文本：{texts}\t预测的类别为：{pred_label[1:len(texts)+1]}")
#     result = []
#     for i in range(len(pred_label[1:len(texts)+1])):
#         # print(pred_label[1:len(texts)][i])
#         if pred_label[1:len(texts)+1][i] != 'PAD' and 'O' != pred_label[1:len(texts)+1][i]:
#             print(texts[i])
#             result = result + texts[i] + ' '
#     return result[:-1]

def text_class_name_seq(texts, pred, index_label):
    print('-----------------------------------')
    pred_label = torch.argmax(pred, dim=-1)
    # print(pred_label)
    result = torch.argmax(pred, dim=-1)
    result = result.cpu().numpy().tolist()[0]
    # print(result)
    # print(index_label)
    pred_label = [index_label[i] for i in result]
    print("模型预测结果：")
    print(f"文本：{texts}\t预测的类别为：{pred_label[1:len(texts)+1]}")
    spolist = []
    temp = ""
    for i in range(len(pred_label[1:len(texts)+1])):
        # print(pred_label[1:len(texts)][i])

        if pred_label[1:len(texts)+1][i] != 'PAD' and 'O' != pred_label[1:len(texts)+1][i]:
            print(texts[i],i, pred_label[i])
            # temp.append(texts[i])
            temp = temp + texts[i] + ' '
            # print(temp,pred_label[i + 1])
        if len(temp)!=0 and pred_label[i + 1] == 'O':
            spolist.append(temp[:-1])
            temp = ""
            # if pred_label[i + 1] == 'B-ProcessTemperature':
            #     temperature_idx = i
            # if pred_label[i + 1] == 'B-ProcessTime':
            #     time_idx = i
    print(spolist)
    return spolist

def text_class_name_type(pred):
    result = torch.argmax(pred, dim=1)
    result = result.cpu().numpy().tolist()
    classification = open(args.classification, "r", encoding="utf-8").read().split("\n")
    classification_dict = dict(zip(range(len(classification)), classification))
    # logging.info(f"文本：{text}\t预测的类别为：{classification_dict[result[0]]}")

def pred_one():
    global args

    text_ = "When R(600/12)Cu-5wt%Mo and R(500/12)Cu-5wt%Mo  alloy was aged at 450℃ for 120min and dried to 25℃ , electrical conductivity reached the maximum value of 96.5%IACS and 66.6%IACS , after which the conductivity reached a relatively stable state with little fluctuation"
    # text_ = 'Detailed information on the microstructures and plastic deformation of this alloy and a Cu-16 wt.% Ag alloy is now available [28,30].'
    # text_ = 'Table 2 also presents, as reported by [19], the true tensile strength and ductility of the Cu-Al-Be (CAB) base alloy wire as 291.5 ± 11.4MPa and 6.1 ± 0.2% respectively, and have been used as a reference data.'
    # text_ = 'It can be found that both the 90W-4.2Ni-1.8Fe-4Cu alloy and steel presented good mechanical properties, and their tensile strength can reach up to 870MPa and 1350MPa, respectively.'
    # text_ = 'The total deposition time was about 24 h and the average thickness of the as-deposited Cu sheets was ~500 μm.'
    # text_ = 'The required amounts of Fe(NO3)2·9H2O and Ce(NO3)2·6H2O, and 0.75 g NH4HCO3 were each dissolved in 200 ml deionized water under magnetic stirring at 0 °C to form transparent solutions. The NH4HCO3 solution was poured into the mixed metal solution rapidly, then stirred vigorously for 0.5 h and statically aged at 0 °C for 15 h [21].'
    # text_ = 'The mixture was transferred into a Teflon-lined autoclave and maintained at 150 °C for 40 h.'
    # text_ = "The co-precipitated powders were then calcined at 900 °C for 1 h in air."

    print(text_)
    out_data = single_text(text_)
    classify_model_ = classify_model()
    mat_val_triple = classify_model_.classify(out_data)
    print(mat_val_triple)
    text = text_.split(' ')
    tokenizer = BertTokenizer.from_pretrained(args.bert_pred)
    #----------------------------------------------------------------------------------------
    print('-------------------------抽取工艺序列---------------------')
    seq_model_dataset = pkl.load(open('data/dataParams_BiLSTM.pkl', "rb"))
    label_index_seq, index_label_seq = seq_model_dataset[0], seq_model_dataset[1]
    print(index_label_seq)
    seq_model = load_seq_model('model/工艺序列抽取.pth', len(label_index_seq))
    print('-------------------------加载模型完毕---------------------')
    text_id = tokenizer.encode(text, add_special_tokens=True, max_length=100 + 2,
                               padding="max_length", truncation=True, return_tensors="pt")
    text_id = text_id.to(device)
    pred = seq_model(text_id)
    seq_result = text_class_name_seq(text, pred, index_label_seq)
    print('工艺序列预测结果:',seq_result)
    #----------------------------------------------------------------------------------------
    print('-------------------------抽取工艺数据---------------------')
    dataset = pkl.load(open(args.data_pkl, "rb"))
    label_index, index_label = dataset[0], dataset[1]
    data_load_model = load_model('model/具体工艺数据抽取模型.pth', len(label_index))
    spolist = []
    for line in seq_result:
        text_id = tokenizer.encode(line.split(' '), add_special_tokens=True, max_length=args.max_len + 2,
                                   padding="max_length", truncation=True, return_tensors="pt")
        text_id = text_id.to(device)
        pred = data_load_model(text_id)
        spolist.append(text_class_name(line.split(' '), pred, index_label))
    print('工艺数据为:',spolist)
    print('------------------------工艺类型分类-------------------')
    type_classifier_model = load_type_model(device,'model/工艺类型分类.pth')
    classification = open('data/type_class.txt', "r", encoding="utf-8").read().split("\n")
    classification_dict = dict(zip(range(len(classification)), classification))
    type_list = []
    for line in seq_result:
        token_id = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokenizer.tokenize(line))
        mask = [1] * len(token_id) + [0] * (38 + 2 - len(token_id))
        token_ids = token_id + [0] * (38 + 2 - len(token_id))
        token_ids = torch.tensor(token_ids).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)
        x = torch.stack([token_ids, mask])
        # print(type_classifier_model.device)
        pred = type_classifier_model(x)
        result_type = torch.argmax(pred, dim=1)
        result_type = result_type.cpu().numpy().tolist()
        print(f"文本：{line}\t预测的类别为：{classification_dict[result_type[0]]}")
        type_list.append(classification_dict[result_type[0]])
    # # for i in spolist:
    # #     print(i['s'],classification_dict[result_type[0]].replace('Operation','')+'-'+i['p'].split('-')[1],i['o'] )
    print('----------------------------------------------------------最终结果-----------------------------------------')
    print(text_)
    for i in mat_val_triple:
        print('材料实体:',i['ent_1'])
        print('性能:', i['ent_2'])
        print('性能值:', i['ent_3'])
    for idx,spo_ in enumerate(spolist):
        for spo in spo_:
            print('工艺:', spo['s'],type_list[idx].replace('Operation', '') + '-' + spo['p'].split('-')[1], spo['o'])
    print()


def get_entity(text):
    data_process = process_data()
    relation_re_list = load_relation_schema()
    keywords = load_keywords()
    elements = load_elements()
    ler_model = LER_model()
    # print("--------------success----------")
    out_data = []
    sent_tokenize_list = sent_tokenize(text)
    for elem in elements:
        keywords[-1]['keywords'][0] = elem
        for sent_index in range(len(sent_tokenize_list)):
            text = sent_tokenize_list[sent_index]
            text = data_process.replace_space(text, relation_re_list)

            for keyword in keywords:
                for i in keyword['keywords']:
                    # print(text, i,"in main")
                    entities = ler_model.LER_regular(text, i)
                    if len(entities) > 0:
                        ent_arr = []
                        for ent in entities:
                            ent['ent_type'] = keyword['entity_type']
                            if (ent['ent_type'] == 'MAT') and (ent['ent'] not in out_data):
                                out_data.append(ent['ent'])
                        break
    return out_data

def pred_mat(line):
    text_ = line
    text_ = "A 7.2 M NH4OH solution was added to a 0.5 M aqueous solution of iron nitrate (Fe(NO3)3.9H2O; Aldrich) in order to precipitate metal ions with hydroxide form Fe(OH)3."
    text_ = "When R(600/12)Cu-5wt%Mo and R(500/12)Cu-5wt%Mo  alloy was aged at 450℃ for 120min and dried to 25℃ , electrical conductivity reached the maximum value of 96.5%IACS and 66.6%IACS , after which the conductivity reached a relatively stable state with little fluctuation"

    print(text_)
    out_data = get_entity(text_)
    print(out_data)


def pred_technolog(line):
    global args

    text_ = line
    # text_ = 'The mixture was transferred into a Teflon-lined autoclave and maintained at 150 °C for 40 h.'
    # text_ = "When R(600/12)Cu-5wt%Mo and R(500/12)Cu-5wt%Mo  alloy was aged at 450℃ for 120min and dried to 25℃ , electrical conductivity reached the maximum value of 96.5%IACS and 66.6%IACS , after which the conductivity reached a relatively stable state with little fluctuation"
    text_ = 'The brown precipitate was dried overnight at 100 °C.'
    text_ = "Finally, a mixture of Li2CO3 and the as-produced concentration-gradient precursors was preheated at 500 °C for 8 h, and then calcined at 900 °C for 10 h in air, to form the concentration-gradient cathode material."

    print(text_)
    out_data = single_text(text_)
    classify_model_ = classify_model()
    mat_val_triple = classify_model_.classify(out_data)
    print(mat_val_triple)
    text = text_.split(' ')
    tokenizer = BertTokenizer.from_pretrained(args.bert_pred)
    # ----------------------------------------------------------------------------------------
    print('-------------------------抽取工艺序列---------------------')
    seq_model_dataset = pkl.load(open('data/dataParams_BiLSTM.pkl', "rb"))
    label_index_seq, index_label_seq = seq_model_dataset[0], seq_model_dataset[1]
    print(index_label_seq)
    seq_model = load_seq_model('model/工艺序列抽取.pth', len(label_index_seq))
    print('-------------------------加载模型完毕---------------------')
    text_id = tokenizer.encode(text, add_special_tokens=True, max_length=100 + 2,
                               padding="max_length", truncation=True, return_tensors="pt")
    text_id = text_id.to(device)
    pred = seq_model(text_id)
    seq_result = text_class_name_seq(text, pred, index_label_seq)
    print('工艺序列预测结果:', seq_result)
    # ----------------------------------------------------------------------------------------
    print('-------------------------抽取工艺数据---------------------')
    dataset = pkl.load(open(args.data_pkl, "rb"))
    label_index, index_label = dataset[0], dataset[1]
    data_load_model = load_model('model/具体工艺数据抽取模型.pth', len(label_index))
    spolist = []
    for line in seq_result:
        text_id = tokenizer.encode(line.split(' '), add_special_tokens=True, max_length=args.max_len + 2,
                                   padding="max_length", truncation=True, return_tensors="pt")
        text_id = text_id.to(device)
        pred = data_load_model(text_id)
        spolist.append(text_class_name(line.split(' '), pred, index_label))
    print('工艺数据为:', spolist)
    print('------------------------工艺类型分类-------------------')
    type_classifier_model = load_type_model(device, 'model/工艺类型分类.pth')
    classification = open('data/type_class.txt', "r", encoding="utf-8").read().split("\n")
    classification_dict = dict(zip(range(len(classification)), classification))
    type_list = []
    for line in seq_result:
        token_id = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokenizer.tokenize(line))
        mask = [1] * len(token_id) + [0] * (38 + 2 - len(token_id))
        token_ids = token_id + [0] * (38 + 2 - len(token_id))
        token_ids = torch.tensor(token_ids).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)
        x = torch.stack([token_ids, mask])
        # print(type_classifier_model.device)
        pred = type_classifier_model(x)
        result_type = torch.argmax(pred, dim=1)
        result_type = result_type.cpu().numpy().tolist()
        print(f"文本：{line}\t预测的类别为：{classification_dict[result_type[0]]}")
        type_list.append(classification_dict[result_type[0]])
    # # for i in spolist:
    # #     print(i['s'],classification_dict[result_type[0]].replace('Operation','')+'-'+i['p'].split('-')[1],i['o'] )
    print('----------------------------------------------------------最终结果-----------------------------------------')
    print(text_)
    entity = []
    technology = []

    for i in mat_val_triple:
        print('材料实体:', i['ent_1'])
        print('性能:', i['ent_2'])
        print('性能值:', i['ent_3'])
        entity.append({'entity_mat':i['ent_1'],'entity_pro':i['ent_2'],'entity_value':i['ent_3']})
    for idx, spo_ in enumerate(spolist):
        for spo in spo_:
            print('工艺:', spo['s'], type_list[idx].replace('Operation', '') + '-' + spo['p'].split('-')[1], spo['o'])
            technology.append({'spo_s':spo['s'],'spo_operation':type_list[idx].replace('Operation', '') + '-' + spo['p'].split('-')[1],'spo_value':spo['o']})
    finall = [text_,entity,technology,spolist]
    print(finall)
    return finall


def pred_Alltechnolog(text_list):
    global args

    out_data = []
    mat_val_triple = []
    text_1 = 'The mixture was transferred into a Teflon-lined autoclave and maintained at 150 °C for 40 h.'
    text_2 = "When R(600/12)Cu-5wt%Mo and R(500/12)Cu-5wt%Mo  alloy was aged at 450℃ for 120min and dried to 25℃ , electrical conductivity reached the maximum value of 96.5%IACS and 66.6%IACS , after which the conductivity reached a relatively stable state with little fluctuation"
    text_list = [text_2,text_1]
    classify_model_ = classify_model()

    for line in text_list:
        out_data.append({'text_':line,'text':line.split(' '),'out_data':single_text(line)})
        out_data[-1]['mat_val_triple'] = classify_model_.classify(out_data[-1]['out_data'])
    print(out_data)
    tokenizer = BertTokenizer.from_pretrained(args.bert_pred)
    # ----------------------------------------------------------------------------------------
    print('-------------------------抽取工艺序列---------------------')
    seq_model_dataset = pkl.load(open('data/dataParams_BiLSTM.pkl', "rb"))
    label_index_seq, index_label_seq = seq_model_dataset[0], seq_model_dataset[1]
    # print(index_label_seq)
    seq_model = load_seq_model('model/工艺序列抽取.pth', len(label_index_seq))
    print('-------------------------加载模型完毕---------------------')
    seq_result = []
    for line in out_data:
        text_id = tokenizer.encode(line['text'], add_special_tokens=True, max_length=100 + 2,
                                   padding="max_length", truncation=True, return_tensors="pt")
        text_id = text_id.to(device)
        pred = seq_model(text_id)
        line['seq_result'] = text_class_name_seq(line['text'], pred, index_label_seq)
        # print('工艺序列预测结果:', seq_result)

    # ----------------------------------------------------------------------------------------
    print('-------------------------抽取工艺数据---------------------')
    dataset = pkl.load(open(args.data_pkl, "rb"))
    label_index, index_label = dataset[0], dataset[1]
    data_load_model = load_model('model/具体工艺数据抽取模型.pth', len(label_index))
    for item in out_data:
        spolist = []
        for line in item['seq_result']:
            text_id = tokenizer.encode(line.split(' '), add_special_tokens=True, max_length=args.max_len + 2,
                                       padding="max_length", truncation=True, return_tensors="pt")
            text_id = text_id.to(device)
            pred = data_load_model(text_id)
            spolist.append(text_class_name(line.split(' '), pred, index_label))
        item['spolist'] = spolist
    # print('工艺数据为:', spolist)
    print('------------------------工艺类型分类-------------------')
    type_classifier_model = load_type_model(device, 'model/工艺类型分类.pth')
    classification = open('data/type_class.txt', "r", encoding="utf-8").read().split("\n")
    classification_dict = dict(zip(range(len(classification)), classification))
    for item in out_data:
        type_list = []
        for line in item['seq_result']:
            token_id = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokenizer.tokenize(line))
            mask = [1] * len(token_id) + [0] * (38 + 2 - len(token_id))
            token_ids = token_id + [0] * (38 + 2 - len(token_id))
            token_ids = torch.tensor(token_ids).unsqueeze(0)
            mask = torch.tensor(mask).unsqueeze(0)
            x = torch.stack([token_ids, mask])
            # print(type_classifier_model.device)
            pred = type_classifier_model(x)
            result_type = torch.argmax(pred, dim=1)
            result_type = result_type.cpu().numpy().tolist()
            # print(f"文本：{line}\t预测的类别为：{classification_dict[result_type[0]]}")
            type_list.append(classification_dict[result_type[0]])
        item['type_list'] = type_list
    # # for i in spolist:
    # #     print(i['s'],classification_dict[result_type[0]].replace('Operation','')+'-'+i['p'].split('-')[1],i['o'] )
    print('----------------------------------------------------------最终结果-----------------------------------------')

    finall = []
    for item in out_data:
        # print(item['text_'])
        entity = []
        technology = []
        for i in item['mat_val_triple']:
            entity.append({'entity_mat': i['ent_1'], 'entity_pro': i['ent_2'], 'entity_value': i['ent_3']})
        for idx, spo_ in enumerate(item['spolist']):
            for spo in spo_:
                # print('工艺:', spo['s'], item['type_list'][idx].replace('Operation', '') + '-' + spo['p'].split('-')[1], spo['o'])
                technology.append({'spo_s': spo['s'],
                                   'spo_operation': item['type_list'][idx].replace('Operation', '') + '-' + spo['p'].split('-')[
                                       1], 'spo_value': spo['o']})
        if (len(entity)==0) and (len(technology)==0):
            1
        else:
            finall.append([item['text_'],entity,technology,item['spolist']])
    # print(finall)
    return finall

# 环境为python3.9，matbert
if __name__ == "__main__":
    start = time.time()
    args = parsers()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    # final_dict = []
    # import os
    # filePath = r'D:\Projects\MatSciBERT\ner\datasets\mpv_xml\ori_txt'
    # filelist = os.listdir(filePath)
    # for title in tqdm(filelist):
    #     # title = '2011.05.052.txt'
    #     with open(r'D:\Projects\MatSciBERT\ner\datasets\mpv_xml\ori_txt\{}'.format(title), 'r', encoding='UTF-8') as f:
    #         elements = f.readlines()
    #
    #     final_dict = pred_Alltechnolog(elements)
    #
    #     with open(r'D:\Projects\二合一\logs\{}.json'.format(title), 'w', encoding='UTF-8') as f:
    #         f.write(json.dumps(final_dict, indent=4, ensure_ascii=False))


    # with open(r'D:\Projects\MatSciBERT\datasets\technology_dataset.json','r',encoding='UTF-8') as f:
    #     ori_data = json.load(f)
    # temp = pred_technolog(ori_data[0]['paragraph_string'])

    pred_technolog('111')
    pred_one()  # 预测一条文本

