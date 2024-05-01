import json
from chemdataextractor.doc import Paragraph
from normalize_text import normalize
import numpy as np
from tqdm import *
from nltk.tokenize import sent_tokenize
from LER_model import LER_model
from process_data import process_data
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"


ler_model = LER_model()

def load_keywords():
    data = []
    with open(r'../config/keywords_default.json','r',encoding='utf-8') as load_f:
        temp = load_f.readlines()
        for line in temp:
            # print(line)
            dic = json.loads(line)
            data.append(dic)
    return data

def load_relation_schema():
    re_list = []
    with open(r'../config/relation_schema.json','r',encoding='utf-8') as load_f:
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



def load_elements():
    data = []
    with open(r'../data/Entity Matching/Elements.json', 'r', encoding='utf-8') as load_f:
        temp = json.load(load_f)
    data = temp['elements']
    return data

relation_re_list = load_relation_schema()
keywords = load_keywords()
elements = load_elements()
def get_entity(text):
    data_process = process_data()
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


# 将数据转换为docred格式
def data_process(ori_data):

    Data_Format = {}

    para = Paragraph(normalize(ori_data['para']))
    sents = []
    for sent in para.tokens:
        sents.append([t.text for t in sent])
    Data_Format['para'] = ori_data['paragraph_string']
    Data_Format['all_sent'] = para.raw_sentences
    Data_Format['sents'] = sents

    vertexSet = []
    temp_vertexSex = []
    name = ori_data['targets_string']
    sent_id = 0
    pos = 0
    Data_Format['title'] = name
    for x in list(np.arange(0, len(sents))):
        if name in sents[x]:
            pos = sents[x].index(name)
            sent_id = int(x)
            break
    temp_vertexSex.append({'name':name,'sent_id':sent_id,'pos':pos,'type':'MAT'})
    vertexSet.append(temp_vertexSex)

    temp_vertexSex = []
    sent_id = 0
    pos = []
    Data_Format['labels'] = []

    for x in list(np.arange(0, len(ori_data['operations']))):
        name = ori_data['operations'][x]['string']
        count = ori_data['operations'][:x+1].count(ori_data['operations'][x])
        for y in list(np.arange(0, len(sents))):
            if (count - sents[y].count(name)) == 0:
                sent_id = int(y)
                for i in list(np.arange(0, len(sents[y]))):
                    if sents[y][i] == name:
                        count -= 1
                        if count == 0:
                            pos = int(i)
                # pos = [i for i, j in enumerate(sents[y]) if j is name]
                break
            elif (count - sents[y].count(name)) <= 0:
                sent_id = int(y)
                for i in list(np.arange(0, len(sents[y]))):
                    if sents[y][i] == name:
                        count -= 1
                        if count == 0:
                            pos = int(i)
                # pos = [i for i, j in enumerate(sents[y]) if j is name][count]
                break
            elif (count - sents[y].count(name)) > 0:
                count -= sents[y].count(name)
        vertexSet.append([{'name': name, 'sent_id': sent_id, 'pos': pos, 'type': 'operations'}])
        h = len(vertexSet)
        for i,j in ori_data['operations'][x]['conditions'].items():
            if isinstance(j,dict):
                # print(j)
                if (j['min_value'] != j['max_value']) or (len(j['values']) == 0):
                    break

                name_list = [str(int(j['values'][0])) + ' ' + j['units']]
                name_list = [t.text for t in Paragraph(normalize(name_list[0])).tokens[0]]
                for z in list(np.arange(0, len(sents))):
                    # print(name_list[0] in sents[z])
                    # print(sents[int(z)].index(name_list[0]))

                    if (name_list[0] in sents[z]) and (len(sents[z])>sents[z].index(name_list[0]) + len(name_list) - 1) and (name_list[-1] in sents[z][sents[z].index(name_list[0]) + len(name_list) - 1]):
                        vpos = [sents[z].index(name_list[0]),sents[z].index(name_list[0]) + len(name_list) - 1]
                        vname = " ".join(name_list)
                        vertexSet.append([{'name': vname, 'sent_id': int(z), 'pos': vpos, 'type': 'value'}])

                        t = len(vertexSet)
                        r = 'operations-value'
                        evidence = [sent_id]
                        if z not in evidence:
                            evidence.append(int(z))
                        Data_Format['labels'].append({'h':h,'t':t,'r':r,'evidence':evidence})
            elif isinstance(j,list):
                if len(sents) == 0:
                    break
                if j[0] in sents[sent_id]:
                    vertexSet.append([{'name': j[0], 'sent_id': sent_id, 'pos': sents[sent_id].index(j[0]), 'type': 'operations'}])
    Data_Format['vertexSet'] = vertexSet
    return Data_Format


# 从原始工艺数据集获取一条数据，转换格式为docred数据集
def get_one_para(ori_data):
    finally_data = data_process(ori_data)
    return finally_data

# 抽取实体及指代名词,并将数据格式转换为我们的数据集
def data_formatting(ori_data):

    Coreferences = [['solution', 'mixture','mixtures','slurry'],
                    ['precipitate', 'gel', 'solid', 'precursor','precursors', 'product', 'sample', 'samples','powders','it']]

    dlre_data = [{'para':ori_data['para'],'sents':ori_data['all_sent'],'sents_words':ori_data['sents']}]
    operations = ori_data['vertexSet']
    for x in list(np.arange(0, len(ori_data['sents']))):
        # 构建空数据集格式
        data_format = {
            "sent_id": int(x),
            "sent": "",
            "mat": [],
            "final_entity":[],
            "mixed_reference": [],
            "single_reference":[],
            "spo":[],
            "relation": []
        }
        # 抽取实体
        # mats = pred_mat.get_entity(ori_data['all_sent'][x])
        mats = get_entity(ori_data['all_sent'][x])
        data_format['mat'] = mats
        # if 'water' in ori_data['sents'][x]:
        #     data_format['mat'].append('water')
        # 抽取提及
        for coref in Coreferences[0]:
            if coref in ori_data['sents'][x]:
                num = ori_data['sents'][x].index(coref)
                data_format['mixed_reference'].append(coref)
        if len(data_format['mixed_reference']) != 0:
            data_format["mixed_reference"] = [data_format["mixed_reference"][0]]
        for coref in Coreferences[1]:
            if coref in ori_data['sents'][x]:
                num = ori_data['sents'][x].index(coref)
                data_format['single_reference'].append(coref)
        if 'form' in ori_data['sents'][x]:
            form_loc = ori_data['sents'][x].index('form')
            for word in ori_data['sents'][x][form_loc:]:
                if word in data_format['mixed_reference']:
                    data_format["final_entity"] = word
                    break
                elif word in data_format['single_reference']:
                    data_format["final_entity"] = word
                    break
                elif word in data_format["mat"]:
                    data_format["final_entity"] = word
                    break
        if 'obtain' in ori_data['sents'][x]:
            form_loc = ori_data['sents'][x].index('obtain')
            for word in ori_data['sents'][x][form_loc:]:
                if word in data_format['mixed_reference']:
                    data_format["final_entity"] = word
                    break
                elif word in data_format['single_reference']:
                    data_format["final_entity"] = word
                    break
                elif word in data_format["mat"]:
                    data_format["final_entity"] = word
                    break
        dlre_data.append(data_format)
    # 将原有材料、工艺转换为我们数据集的格式
    for x in list(np.arange(0, len(operations))):
        if operations[x][0]['type'] == 'MAT':
            # dlre_data[operations[x][0]['sent_id']+1]['mat'].append(operations[x][0]['name'])
            print()
        elif operations[x][0]['type'] == 'operations':
            if (operations[x][0]['name'] in ['air','water']) and (len(dlre_data[operations[x][0]['sent_id'] + 1]['spo']) != 0):
                dlre_data[operations[x][0]['sent_id'] + 1]['spo'][-1]['c'].append(operations[x][0]['name'])
            else:
                spo = {'s':operations[x][0]['name'],'p':'process','o':'','c':[]}
                if spo not in dlre_data[operations[x][0]['sent_id']+1]['spo']:
                    dlre_data[operations[x][0]['sent_id']+1]['spo'].append(spo)
        elif operations[x][0]['type'] == 'value':
            if len(dlre_data[operations[x][0]['sent_id']+1]['spo']) != 0:
                if len(dlre_data[operations[x][0]['sent_id']+1]['spo'][-1]['o']) == 0:
                    dlre_data[operations[x][0]['sent_id'] + 1]['spo'][-1]['o'] = operations[x][0]['name']
                elif len(dlre_data[operations[x][0]['sent_id']+1]['spo'][-1]['c']) == 0:
                    dlre_data[operations[x][0]['sent_id'] + 1]['spo'][-1]['c'].append( operations[x][0]['name'])

    print()
    return dlre_data

# 建立关系
def build_sent_relation(ori_data):

    for x in list(np.arange(0, len(ori_data))):
        # 建立句内材料实体-指代关系
        if x == 0:
            print()
        elif len(ori_data[x]['final_entity']) != 0:
            if len(ori_data[x]['mat']) >= 2:
                relation = {'entity1':ori_data[x]['mat'],'entity2':ori_data[x]['final_entity'],'relation':'form'}
                ori_data[x]['relation'].append(relation)
        elif len(ori_data[x]['mixed_reference']) != 0:
            if len(ori_data[x]['mat']) >= 2:
                relation = {'entity1':ori_data[x]['mat'],'entity2':ori_data[x]['mixed_reference'],'relation':'n to 1'}
                ori_data[x]['relation'].append(relation)
            elif (len(ori_data[x]['mat']) != 0) and (len(ori_data[x]['single_reference']) != 0):
                relation = {'entity1': ori_data[x]['mat'] + ori_data[x]['single_reference'], 'entity2': ori_data[x]['mixed_reference'],'relation': 'n to 1'}
                ori_data[x]['relation'].append(relation)
        elif len(ori_data[x]['single_reference']) != 0:
            if len(ori_data[x]['mat']) != 0:
                for mat in ori_data[x]['mat']:
                    relation = {'entity1':[mat],'entity2':ori_data[x]['single_reference'],'relation':'refer'}
                    ori_data[x]['relation'].append(relation)
                # relation = {'entity1':ori_data[x]['mat'],'entity2':ori_data[x]['single_reference'],'relation':'1 to 1'}
                # ori_data[x]['relation'].append(relation)
        # 建立句内指代-工艺关系
        if x == 0:
            print()
        elif len(ori_data[x]['spo']) != 0:
            for spo_num in list(np.arange(0, len(ori_data[x]['spo']))):
                if spo_num == 0:
                    if len(ori_data[x]['mixed_reference']) == 1:
                        relation = {'entity1': ori_data[x]['mixed_reference'], 'entity2': ori_data[x]['spo'][spo_num],
                                    'relation': 'spo to refer'}
                        ori_data[x]['relation'].append(relation)
                    elif len(ori_data[x]['single_reference']) == 1:
                        relation = {'entity1': ori_data[x]['single_reference'], 'entity2': ori_data[x]['spo'][spo_num],
                                    'relation': 'spo to refer'}
                        ori_data[x]['relation'].append(relation)
                elif spo_num >= 1:
                    if (len(ori_data[x]['mixed_reference']) != 0) or (len(ori_data[x]['single_reference']) != 0):
                        if len(ori_data[x]['relation'])!=0:
                            relation = {'entity1': ori_data[x]['relation'][-1], 'entity2': ori_data[x]['spo'][spo_num],'relation': 'relation to spo'}
                            ori_data[x]['relation'].append(relation)

        # 建立跨句关系
        # 第0是整段，第一句前面没有关系
        if x in [0, 1]:
            print()
        # 第二个句子
        elif x == 2:
            # 如果是混合指代名词
            if len(ori_data[x]['mixed_reference']) != 0:
                relation = {}
                # 判断混合指代名词是否在当前句已经找到了指代
                if_list = [
                    (ori_data[x]['mixed_reference'] in [temp_relation['entity1'], temp_relation['entity2']]) and (
                                temp_relation['relation'] in ['n to 1']) for temp_relation in ori_data[x]['relation']]
                # 混合指代名词已经在当前句找到指代，跳过当句
                if True in if_list:
                    print()
                # 当前句没有混合指代名词的指代，向前一句嗅探
                else:
                    if len(ori_data[x - 1]['mat']) != 0:
                        relation = {'entity1': ori_data[x]['mixed_reference'], 'entity2': ori_data[x - 1]['mat'],
                                    'relation': 'refer'}
                    for item in ori_data[x - 1]['relation']:
                        # 如果前一句有多材料实体
                        if (item['relation'] == 'n to 1') and (
                                ori_data[x - 1]['mat'] in [item['entity1'], item['entity2']]):
                            relation = {'entity1': ori_data[x]['mixed_reference'], 'entity2': item, 'relation': 'refer'}
                        # 如果前面有事件指代
                        if item['relation'] in ['spo to refer', 'relation to spo']:
                            relation = {'entity1': ori_data[x]['mixed_reference'], 'entity2': item,
                                        'relation': 'relation to refer'}
                    if len(relation) != 0:
                        ori_data[x]['relation'].insert(0, relation)

            # 如果是单独指代名词
            elif len(ori_data[x]['single_reference']) != 0:
                # 如果前一句有唯一材料实体
                relation = {}
                if len(ori_data[x - 1]['mat']) == 1:
                    relation = {'entity1': ori_data[x]['single_reference'], 'entity2': ori_data[x - 1]['mat'],'relation': 'refer'}
                # 如果前面有事件指代
                for item in ori_data[x - 1]['relation']:
                    if item['relation'] in ['spo to refer', 'relation to spo']:
                        relation = {'entity1': ori_data[x]['single_reference'], 'entity2': item,'relation': 'relation to refer'}
                # 如果当前单独指代名词已有其他指代
                for item in ori_data[x]['relation']:
                    if item['relation'] in ['refer']:
                        relation = {}
                if len(relation) != 0:
                    ori_data[x]['relation'].insert(0, relation)

        # 第三句以上
        elif x >= 3:
            # 如果是混合指代名词
            if len(ori_data[x]['mixed_reference']) != 0:
                relation = {}
                # 判断混合指代名词是否在当前句已经找到了指代
                if_list = [(ori_data[x]['mixed_reference'] in [temp_relation['entity1'],temp_relation['entity2']]) and (temp_relation['relation'] in ['n to 1']) for temp_relation in ori_data[x]['relation']]
                # 混合指代名词已经在当前句找到指代，跳过当句
                if True in if_list:
                    print()
                # 当前句没有混合指代名词的指代，向前一句嗅探
                else:
                    if len(ori_data[x - 1]['mat']) != 0:
                        relation = {'entity1': ori_data[x]['mixed_reference'], 'entity2': ori_data[x - 1]['mat'],
                                    'relation': 'refer'}
                    for item in ori_data[x - 1]['relation']:
                        # 如果前一句有多材料实体
                        if (item['relation'] == 'n to 1') and (
                                ori_data[x - 1]['mat'] in [item['entity1'], item['entity2']]):
                            relation = {'entity1': ori_data[x]['mixed_reference'], 'entity2': item, 'relation': 'refer'}
                        # 如果前面有事件指代
                        if item['relation'] in ['spo to refer', 'relation to spo']:
                            relation = {'entity1': ori_data[x]['mixed_reference'], 'entity2': item,
                                        'relation': 'relation to refer'}
                    if len(relation) != 0:
                        ori_data[x]['relation'].insert(0, relation)

            # 如果是单独指代名词
            elif len(ori_data[x]['single_reference']) != 0:
                # 如果前一句有唯一材料实体
                relation = {}
                if len(ori_data[x - 1]['mat']) == 1:
                    relation = {'entity1': ori_data[x]['single_reference'], 'entity2': ori_data[x - 1]['mat'],'relation': 'refer'}
                # 如果前面有事件指代
                for item in ori_data[x - 1]['relation']:
                    if item['relation'] in ['spo to refer', 'relation to spo']:
                        relation = {'entity1': ori_data[x]['single_reference'], 'entity2': item,'relation': 'refer'}
                for item in ori_data[x]['relation']:
                    if item['relation'] in ['refer']:
                        relation = {}
                if len(relation) != 0:
                    ori_data[x]['relation'].insert(0, relation)
    print()
    return ori_data

with open(r'../data/technology_data/technology_dataset.json','r',encoding='utf-8')as f:
    ori_data = json.load(f)
dlre_data = []
for num in tqdm(range(2000,3000)):
    if len(ori_data[num]['paragraph_string'])!= 0:
        return_data = get_one_para(ori_data[num])
        return_data = data_formatting(return_data)
        dlre_data.append(return_data)
        # dlre_data.append(build_sent_relation(return_data))
# with open(r'../data/technology_data/ori_data/1000entity.json','w',encoding='utf-8')as f:
#     f.write(json.dumps(dlre_data, indent=4, ensure_ascii=False))

# with open(r'../data/technology_data/ori_data/1000entity.json','r',encoding='utf-8')as f:
#     ori_data = json.load(f)
# print(len(ori_data))
# dlre_data = []
# for num in tqdm(range(0,977)):
#
#         print(num)
#         return_data = get_one_para(ori_data[num])
#         return_data = data_formatting(return_data)
#         dlre_data.append(return_data)
#         dlre_data.append(build_sent_relation(ori_data[num]))

print()
# with open(r'../data/technology_data/ori_data/1000entity.json.json', 'w', encoding='utf-8')as f:
#     f.write(json.dumps(dlre_data, indent=4, ensure_ascii=False))
