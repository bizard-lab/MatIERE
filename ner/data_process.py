import json
from tqdm import *
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"


# # 将../data/technology_data/ori_data/2000ori_ed_ing(docred)_full_ht.json中的工艺spo保留
# with open(r'../data/technology_data/ori_data/2000ori_ed_ing(docred)_full_ht.json','r',encoding='utf-8')as f:
#     ori_data = json.load(f)
# # 输入多个分词后的句子
# final_data = []
# for para in tqdm(ori_data):
#     vertexSet = []
#     for entity in para['vertexSet']:
#         if entity[0]['type'] in ['spo']:
#             vertexSet.append(entity)
#     final_data.append({'sents':para['sents'],'title':'','vertexSet':vertexSet,'labels':[]})
#     print()
# print()
# with open(r'entity_datasets/lslre_spo.json', 'w', encoding='utf-8')as f:
#     f.write(json.dumps(final_data, indent=4, ensure_ascii=False))
# print()


# # 将../data/technology_data/ori_data/2000ori_ed_ing(docred)_full_ht.json中的实体与指代名词保留
# with open(r'../data/technology_data/ori_data/2000ori_ed_ing(docred)_full_ht.json','r',encoding='utf-8')as f:
#     ori_data = json.load(f)
# # 输入多个分词后的句子
# final_data = []
#
# for para in tqdm(ori_data):
#     vertexSet = []
#     for entity in para['vertexSet']:
#         if entity[0]['type'] in ['MAT','refer']:
#             vertexSet.append([{'name':[entity[0]['name']],'sent_id':entity[0]['sent_id'],'pos':[entity[0]['pos'],entity[0]['pos']+1],'type':entity[0]['type']}])
#     final_data.append({'sents':para['sents'],'title':'','vertexSet':vertexSet,'labels':[]})
#     print()
#
# print()
# with open(r'entity_datasets/mat_refer.json', 'w', encoding='utf-8')as f:
#     f.write(json.dumps(final_data, indent=4, ensure_ascii=False))
# print()


# # 将mat_refer与process_filter_spo合并得到所有实体
# with open(r'entity_datasets/mat_refer.json','r',encoding='utf-8')as f:
#     mat_refer = json.load(f)
# with open(r'entity_datasets/process_filter_spo.json','r',encoding='utf-8')as f:
#     spo = json.load(f)
#
# final_data = []
# for num in tqdm(range(0,len(mat_refer))):
#     vertexSet = []
#     vertexSet += mat_refer[num]['vertexSet']
#     vertexSet += spo[num]['vertexSet']
#     arr = vertexSet
#     for i in range(1, len(arr)):
#         for j in range(0, len(arr)-i):
#             if arr[j][0]['sent_id'] > arr[j+1][0]['sent_id']:
#                 arr[j], arr[j + 1] = arr[j + 1], arr[j]
#     for i in range(1, len(arr)):
#         for j in range(0, len(arr)-i):
#             if (arr[j][0]['pos'][0] > arr[j+1][0]['pos'][0]) and (arr[j][0]['sent_id'] == arr[j+1][0]['sent_id']):
#                 arr[j], arr[j + 1] = arr[j + 1], arr[j]
#     vertexSet = arr
#     final_data.append({'sents':mat_refer[num]['sents'],'title':'','vertexSet':vertexSet,'labels':[]})
#     print()
# with open(r'entity_datasets/entities.json', 'w', encoding='utf-8')as f:
#     f.write(json.dumps(final_data, indent=4, ensure_ascii=False))
# print()

# with open(r'D:\Projects\二合一\MatDLRE\data\technology_data\ori_data\1000entity.json','r',encoding='utf-8')as f:
#     entity_1000 = json.load(f)
# with open(r'entity_datasets/entities.json','r',encoding='utf-8')as f:
#     entity_2000 = json.load(f)
# new_entity = []
# for item in entity_1000:
#     sents = item[0]['sents_words']
#     vertexSet = []
#     for num in range(1,len(item)):
#         # if len(item[num]['mat']) != 0:
#         for entity in item[num]['mat']:
#             if entity in sents[num-1]:
#                 vertexSet.append([{'name': [entity], 'sent_id': num-1, 'pos': [sents[num-1].index(entity), sents[num-1].index(entity) + 1], 'type': 'MAT'}])
#         # elif len(item[num]['mixed_reference'])!= 0:
#         for entity in item[num]['mixed_reference']:
#             vertexSet.append([{'name': [entity], 'sent_id': num-1, 'pos': [sents[num-1].index(entity), sents[num-1].index(entity) + 1], 'type': 'refer'}])
#         for entity in item[num]['single_reference']:
#             vertexSet.append([{'name': [entity], 'sent_id': num-1, 'pos': [sents[num-1].index(entity), sents[num-1].index(entity) + 1], 'type': 'refer'}])
#         for spoc in item[num]['spo']:
#
#             entity = []
#             if len(spoc['s']) != 0:
#                 entity.append(spoc['s'])
#             if len(spoc['o']) != 0:
#                 entity.append(spoc['o'])
#             if len(spoc['c']) != 0:
#                 for c in spoc['c']:
#                     entity.append(c)
#             if entity[0] in sents[num-1]:
#                 vertexSet.append([{'name': entity, 'sent_id': num-1, 'pos': [sents[num-1].index(entity[0]), sents[num-1].index(entity[0])+len(entity)], 'type': 'spo'}])
#     new_entity.append({'sents':sents,'title':'','vertexSet':vertexSet,'labels':[]})
# print()
# entity_2000+=new_entity
# with open(r'entity_datasets/entity_2929.json', 'w', encoding='utf-8')as f:
#     f.write(json.dumps(entity_2000, indent=4, ensure_ascii=False))

# with open(r'../data/technology_data/new_data/3000my(docred).json','r',encoding='utf-8')as f:
#     ori_data = json.load(f)
# with open(r'..//method/entity_datasets/entity_2929.json','r',encoding='utf-8')as f:
#     my_data = json.load(f)
# my_data.pop(2339)
# my_data.pop(2842)
# my_data.pop(2842)
# with open(r'entity_datasets/entity_2926.json', 'w', encoding='utf-8')as f:
#     f.write(json.dumps(my_data, indent=4, ensure_ascii=False))
#

with open(r'../method/entity_datasets/spert_predict_data.json','r',encoding='utf-8')as f:
    spert_data = json.load(f)
with open(r'../data/technology_data/new_data/3000my(docred).json','r',encoding='utf-8')as f:
    ori_data = json.load(f)

for item in tqdm(spert_data):
    for para in ori_data:
        if item['tokens'] in para['sents']:
            for entity in item['entities']:
                name = []
                for i in range(entity['start'],entity['end']):
                    name += [item['tokens'][i]]
                temp_vert = [{'name': name, 'sent_id': para['sents'].index(item['tokens']),'pos': [entity['start'], entity['end']], 'type': entity['type']}]
                for vertexSet in para['vertexSet']:
                    if (name[0] in vertexSet[0]['name']) and (temp_vert[0]['sent_id'] == vertexSet[0]['sent_id']):
                        vertexSet = temp_vert
                        print()
with open(r'entity_datasets/2926ori.json', 'w', encoding='utf-8')as f:
    f.write(json.dumps(ori_data, indent=4, ensure_ascii=False))



