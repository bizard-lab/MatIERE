
import re
import json
import pandas as pd
import numpy as np
class MER_model():
    def __init__(self):
        # print()
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
                # re_list = dic["re"]
                # print(txt)
                # print(re_list)
        return re_list
    def MER_regular(self,text):
        material_ent = []
        material_index_char = []
        material_index_token = []
        temp_data = []
        # text_token = text.split(" ")
        for materil_re in self.material_re_list:
            pater = re.compile(materil_re, flags=re.IGNORECASE)
            material_text = re.findall(pater, text)
            if len(material_text) != 0:
                count_cu = text.count(materil_re)
                last_cu_end = 0
                for cu_index in range(count_cu):
                    if cu_index == 0:
                        cu_begin_index = text.index(materil_re)
                    elif last_cu_end < len(text):
                        cu_begin_index = text.find(materil_re, last_cu_end, len(text))
                    left_side = cu_begin_index
                    right_side = cu_begin_index
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
                                last_cu_end = right_side
                                break

                    if left_side != 0:
                        left_side = left_side + 1
                    if ',' in text[left_side:right_side]:
                        right_side = right_side - 1
                    if text[right_side-1] == '.':
                        right_side = right_side - 1
                    temp_data.append({
                        'name': text[left_side:right_side],
                        'char_idx': str(left_side) + ',' + str(right_side),
                        # 'words_token_index':text_token.index(text[left_side:right_side])
                    })
                    # temp_data.append({
                    #     'name' : text[left_side:right_side],
                    #     'idx' : str(left_side) + ',' + str(right_side)
                    # })
                    # material_ent.append(text[left_side:right_side])
                    # index_str = str(left_side) + ',' + str(right_side)
                    # material_index_box.append(index_str)
        # print(material_ent)

        temp_data = pd.DataFrame(temp_data)
        # print(temp_data)
        temp_data = temp_data.drop_duplicates()
        # print('-----after---')
        # print(temp_data)
        temp_data = np.array(temp_data)
        temp_data = temp_data.tolist()
        # print(temp_data)
        for i in temp_data:
            # print(i)
            if len(i[0]) > 0:
                material_ent.append(i[0])
                material_index_char.append(i[1])
                # material_index_token.append(i[2])
        # material_ent = []
        return material_ent,material_index_char
