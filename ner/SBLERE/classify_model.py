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

class classify_model():
    def __init__(self):
        self.data = ''
        # self.data = pd.read_excel('new_out/子句实验数据_去重后.xlsx')
        # self.data = pd.read_excel('out/temp/all_right_data_2020-2022.xlsx')

    def split_caluse(self,seq):
        caluse = []
        temp = []
        mat = 0
        val = 0
        perf = 0
        # print(seq)
        for i in range(len(seq)):
            # print(mat,val,perf)
            temp.append(seq[i])
            if 'm' == seq[i][0]:
                mat = mat + 1
            if 'p' == seq[i][0]:
                perf = perf + 1
            if 'v' == seq[i][0]:
                val = val +1
            if mat * perf == val and perf != 0 and val != 0 and mat != 0:
                caluse.append(temp)
                temp = []
            # print(temp)
            if i == len(seq) and len(caluse) > 0:
                caluse.append(temp)

        # print(caluse)
        return caluse

    def complement_caluse(self,old_caluse):
        main_caluse = []
        for i in range(len(old_caluse[0])):
            main_caluse.append(old_caluse[0][i][0])
        temp_1 = []
        # print(main_caluse)
        new_caluse = []
        new_caluse.append(old_caluse[0])
        for index in range(len(old_caluse)):
            if index != 0:
                temp_1 = old_caluse[index]
                temp_2 = []
                temp = []
                mask_num = 1
                for t_index in range(len(temp_1)):
                    temp_2.append(temp_1[t_index])
                    temp_1[t_index] = temp_1[t_index][0]
                # print(temp_2)
                # print(temp_1,main_caluse)
                seq_len = len(temp_1)
                while(seq_len >= 0):
                    for i in range(len(main_caluse)):
                        flag = False
                        for j in range(len(temp_1)):
                            # print(temp_1[j],main_caluse[i])
                            if temp_1[j] == main_caluse[i] and temp_2[j] not in temp:
                                temp.append(temp_2[j])
                                flag = True
                                break
                        if flag == False:
                            temp.append('mask' + str(mask_num))
                            mask_num = mask_num + 1
                        if len(temp) == len(main_caluse):
                            break
                    seq_len = seq_len - len(main_caluse)
                # print(temp,'temp')
                if len(temp) > len(main_caluse):
                    temp_arr = []
                    for i in range(0,len(temp),len(main_caluse)):
                        t = []
                        for j in range(len(main_caluse)):
                            t.append(temp[i+j])
                        temp_arr.append(t)
                    # print(temp_arr,'arr')
                    for i in temp_arr:
                        for j in range(len(i)):
                            # print(temp[j])
                            if 'mask' in i[j]:
                                i[j] = old_caluse[0][j]
                        new_caluse.append(i)
                else:
                    for i in range(len(temp)):
                        if 'mask' in temp[i]:
                            temp[i] = old_caluse[0][i]
                    new_caluse.append(temp)

        return new_caluse

    def check_seq(self,data):
        print()

    def check_product(self,data):
        # print(data)
        mat_ent = []
        val_ent = []
        perf_ent = []
        for i in data:
            # print(i)
            if 'MAT' == i['ent_type']:
                mat_ent.append(i['ent'])
            if 'Val' == i['ent_type']:
                val_ent.append(i['ent'])
            if 'Perf' == i['ent_type']:
                perf_ent.append(i['ent'])


        # print(len(mat_ent),len(val_ent),len(perf_ent))
        if len(mat_ent) * len(perf_ent) == len(val_ent):
            return True
        else:
            return False

    def cartesian_product(self,data_1,data_2):
        # print(data_1)
        # print(data_2)
        new_data = []
        for i in data_1:
            for j in data_2:
                new_data.append([i,j])
        return new_data

    def link_product(self,data_1,data_2):
        # print(data_1)
        # print(data_2)
        # print(len(data_1),len(data_2),data_1,data_2)
        for i in range(len(data_1)):
            data_1[i].append(data_2[i])
        return data_1

    def shortest_classify(self):
        print()

    def sentecne_seq(self,data):
        seq = []
        mat = 1
        perf = 1
        val = 1
        for i,r in data.iterrows():
            if 'MAT' == r['ent_type']:
                seq.append('m'+str(mat))
                mat = mat + 1
            if 'Perf' == r['ent_type']:
                seq.append('p'+str(perf))
                perf = perf + 1
            if 'Val' == r['ent_type']:
                seq.append('v'+str(val))
                val = val + 1
        # print(seq)
        return seq

    def caluse_classify(self,data):
        out = []
        data = pd.DataFrame(data)
        data = data.sort_values(by='words_toekn_idx',ignore_index=True)
        seq = self.sentecne_seq(data)
        data['seq'] = seq
        # print(seq)
        caluse = self.split_caluse(seq)
        # print(caluse)
        caluse = self.complement_caluse(caluse)
        # print(caluse)
        out_data = []
        for i in caluse:
            mat_ent = []
            val_ent = []
            perf_ent = []
            # print(i)
            for j in i:
                if 'm' == j[0]:
                    mat_ent.append(j)
                if 'p' == j[0]:
                    perf_ent.append(j)
                if 'v' == j[0]:
                    val_ent.append(j)
            temp = self.cartesian_product(mat_ent,perf_ent)
            temp = self.link_product(temp,val_ent)
            # print(temp,'temp')
            if len(temp)>1:
                for j in temp:
                    out.append(j)
                    # print(j)
            else:
                out.append(temp[0])
        # print(out)
        for i in out:
            # print(i)
            temp = []
            for j in i:
                # print(j)
                temp.append(data[data['seq'] == j])
            # print(temp[0]['ent'].to_list()[0])
            out_data.append({
                'ent_1':temp[0]['ent'].to_list()[0],
                'ent_2':temp[1]['ent'].to_list()[0],
                'ent_3':temp[2]['ent'].to_list()[0]
            })
            # for j in temp:
            #     print(j)

        # for i,r in data.iterrows():
        #     print(r)
        return out_data



    def greedy_classify(self):
        print()

    def group_data(self):
        print()

    def classify(self,data):
        # print(self.data)
        self.data = data
        out_data = []
        for index,row in self.data.iterrows():
            text = row['text']
            text_words_token = row['text_words_token']
            entities = row['ents']
            # print(entities)
            # entities = entities.replace("'","\"")
            # print(entities)
            # entities = json.loads(entities)
            # print(type(entities))
            # mat_ent = row['material_entity']
            # val_ent = row['relation_text']
            # perf_ent = row['value_entity']
            if self.check_product(entities):
                caluse_data = self.caluse_classify(entities)
                for i in caluse_data:
                    # print(i)
                    out_data.append({
                        'doi' : row['doi'],
                        'text' : row['text'],
                        'text_words_token' : row['text_words_token'],
                        'ent_1': i['ent_1'],
                        'ent_2': i['ent_2'],
                        'ent_3': i['ent_3']
                    })

            else:
                if self.check_product(entities):
                    self.shortest_classify()
                else:
                    self.greedy_classify()
        # out_data = pd.DataFrame(out_data)
        # out_data.to_excel('new_out/强度/final_caluse_all.xlsx')
        # print(out_data)
        return out_data
        # out_data.to_excel('test/final_caluse_all.xlsx')
        # out_data.to_excel('new_out/final_caluse_all.xlsx')