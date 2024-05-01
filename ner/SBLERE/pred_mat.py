from nltk.tokenize import sent_tokenize
from LER_model import LER_model
from process_data import process_data
import json
device = "cpu"

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


def get_entity(text):
    data_process = process_data()
    relation_re_list = load_relation_schema()
    keywords = load_keywords()
    elements = load_elements()
    ler_model = LER_model()
    # print("--------------success----------")
    out_data = []
    # sent_tokenize_list = sent_tokenize(text)
    sent_tokenize_list = text
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

