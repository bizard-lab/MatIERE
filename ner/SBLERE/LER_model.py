
import re
import json
import pandas as pd
import numpy as np
from functools import singledispatchmethod
class LER_model():
    def __init__(self):
        print()

    @singledispatchmethod
    def LER_regular(self,text,keywords):
        print(text,keywords,'type_none')

    @LER_regular.register
    def _(self,text:str,keywords:str):
        # print(text,keywords)
        target_index_char = []
        target_ent_out = []
        re_str = r'' + keywords
        # value_ent = re.findall(re_str, text)
        pater = re.compile(re_str)
        target_text = re.findall(pater, text)
        # print(target_text)
        temp_data = []
        text_words_token = text.split(' ')
        # print(text_words_token)
        if len(target_text) != 0:
            target_index_begin = 0
            last_target_end = 0
            for i in target_text:
                v_t = str(i)
                count_target = text.count(v_t)
                for target_index in range(count_target):
                    # print(count_target)
                    if target_index == 0:
                        target_index_begin = text.index(v_t)
                    elif last_target_end < len(text):
                        # print(v_t,last_target_end,text)
                        target_index_begin = text.find(v_t, last_target_end, len(text))
                    left_side = target_index_begin
                    right_side = target_index_begin+len(keywords)
                    while left_side > 0:
                        if text[left_side] != " ":
                            left_side = left_side - 1
                        if text[left_side] == " ":
                            # if text[left_side-1] != '%':
                            break
                    while right_side < len(text):
                        if text[right_side] != " ":
                            right_side = right_side + 1
                        if right_side < len(text):
                            if text[right_side] == " ":
                                last_target_end = right_side
                                break

                    if left_side != 0:
                        left_side = left_side + 1
                    if ',' == text[right_side-1:right_side]:
                        right_side = right_side - 1
                    if text[right_side - 1] == '.':
                        right_side = right_side - 1
                    temp_data.append({
                        'name': text[left_side:right_side],
                        'char_idx': str(left_side) + ',' + str(right_side),
                    })
        temp_data = pd.DataFrame(temp_data)
        temp_data = temp_data.drop_duplicates()
        temp_data = np.array(temp_data)
        temp_data = temp_data.tolist()

        for i in temp_data:
            # print(i)
            if len(i[0]) > 0:
                target_ent_out.append(i[0])
                target_index_char.append(i[1])
                # material_index_token.append(i[2])
        target_index_token = []
        for i in target_ent_out:
            word = i
            if len(word.split(" ")) > 1:
                word = word.split(" ")[0]
            for j in range(len(text_words_token)):
                # if word == text_words_token[j] and j not in target_index_token:
                if word in text_words_token[j] and j not in target_index_token:
                    target_index_token.append(j)
                    break
        out_data = []
        if len(target_ent_out)>0:
            for idx in range(len(target_ent_out)):
                # print(text)
                # print(text_words_token)
                # print(target_ent_out, target_index_char, target_index_token)
                out_data.append({
                    'ent': target_ent_out[idx],
                    'char_idx': target_index_char[idx],
                    'words_toekn_idx': target_index_token[idx],
                    'ent_type': 'none',
                    'keyword':keywords
                })
        return out_data
        # return target_ent_out,target_index_char,target_index_token

    # @LER_regular.register
    # def _(self,text:str,keywords:list):
    #     print(text,keywords,'type_list')
    #     target_index_char = []
    #     target_ent_out = []
    #     temp_data = []
    #     for keyword in keywords:
    #         re_str = r'' + keyword
    #         # value_ent = re.findall(re_str, text)
    #         pater = re.compile(re_str)
    #         target_text = re.findall(pater, text)
    #         # print(target_text)
    #         if len(target_text) != 0:
    #             for i in target_text:
    #                 v_t = str(i)
    #                 count_target = text.count(v_t)
    #                 target_index_begin = 0
    #                 last_target_end = 0
    #                 for target_index in range(count_target):
    #                     if target_index == 0:
    #                         target_index_begin = text.index(v_t)
    #                     elif last_target_end < len(text):
    #                         # print(v_t,last_value_end,text)
    #                         target_index_begin = text.find(v_t, last_target_end, len(text))
    #                     left_side = target_index_begin
    #                     right_side = target_index_begin + len(keywords)
    #                     while left_side > 0:
    #                         if text[left_side] != " ":
    #                             left_side = left_side - 1
    #                         if text[left_side] == " ":
    #                             # if text[left_side-1] != '%':
    #                             break
    #                     while right_side < len(text):
    #                         if text[right_side] != " ":
    #                             right_side = right_side + 1
    #                         if right_side < len(text):
    #                             if text[right_side] == " ":
    #                                 last_target_end = right_side
    #                                 break
    #
    #                     if left_side != 0:
    #                         left_side = left_side + 1
    #                     if ',' in text[left_side:right_side]:
    #                         right_side = right_side - 1
    #                     if text[right_side - 1] == '.':
    #                         right_side = right_side - 1
    #                     temp_data.append({
    #                         'name': text[left_side:right_side],
    #                         'char_idx': str(left_side) + ',' + str(right_side),
    #                     })
    #     # print(temp_data)
    #     temp_data = pd.DataFrame(temp_data)
    #     temp_data = temp_data.drop_duplicates()
    #     temp_data = np.array(temp_data)
    #     temp_data = temp_data.tolist()
    #     for i in temp_data:
    #         # print(i)
    #         if len(i[0]) > 0:
    #             target_ent_out.append(i[0])
    #             target_index_char.append(i[1])
    #             # material_index_token.append(i[2])
    #     return target_ent_out,target_index_char
