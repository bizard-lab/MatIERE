
import re
import json
import pandas as pd
import numpy as np
class MVR_model():
    def __init__(self):
        # print()
        self.relation_ent_path = 'config/relation_schema_default.json'
        self.material_ent_path = 'config/material_schema_default.json'

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
    def MVR_regular(self,text,regular_text):
        value_index_char = []
        value_ent_out = []
        pater = re.compile(regular_text, flags=re.IGNORECASE)
        re_str = r'([0-9]+\.[0-9]*|-?[0-9]+)' + regular_text
        value_ent = re.findall(re_str, text)
        pater = re.compile(re_str, flags=re.IGNORECASE)
        value_text = re.findall(pater, text)
        temp_data = []
        if len(value_text) != 0:
            for i in value_text:
                v_t = str(i) + regular_text
                count_value = text.count(v_t)
                # print(v_t, count_value)
                value_index_begin = 0
                last_value_end = 0
                for value_index in range(count_value):
                    if value_index == 0:
                        value_index_begin = text.index(v_t)
                    elif last_value_end < len(text):
                        # print(v_t,last_value_end,text)
                        value_index_begin = text.find(v_t, last_value_end, len(text))
                    left_side = value_index_begin
                    right_side = value_index_begin
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
                                last_value_end = right_side
                                break

                    if left_side != 0:
                        left_side = left_side + 1
                    if ',' in text[left_side:right_side]:
                        right_side = right_side - 1
                    if text[right_side - 1] == '.':
                        right_side = right_side - 1
                    temp_data.append({
                        'name': text[left_side:right_side],
                        'char_idx': str(left_side) + ',' + str(right_side),
                        # 'words_token_index':text_token.index(text[left_side:right_side])
                    })
        temp_data = pd.DataFrame(temp_data)
        # print(temp_data)
        temp_data = temp_data.drop_duplicates()
        # print('-----after---')
        # print(temp_data)
        temp_data = np.array(temp_data)
        temp_data = temp_data.tolist()
        for i in temp_data:
            # print(i)
            if len(i[0]) > 0:
                value_ent_out.append(i[0])
                value_index_char.append(i[1])
                # material_index_token.append(i[2])
        return value_ent_out,value_index_char
