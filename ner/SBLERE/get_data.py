import json
from MyModel import BertNerModel
from transformers import BertTokenizer
import pickle as pkl
import torch
from config import parsers
import time

def load_model(model_path, class_num):
    global device
    model = BertNerModel(class_num).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def text_class_name(texts, pred, index_label):
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
    out_data = {
            'time-triple': {
                'process': '',
                'relation': '',
                'time': ''
            },
            'temperature-triple': {
                'process': '',
                'relation': '',
                'temperature': ''
            }
    }
    for i in range(len(pred_label[1:len(texts)+1])):
        # print(pred_label[1:len(texts)][i])
        if pred_label[1:len(texts)+1][i] != 'PAD' and 'O' != pred_label[1:len(texts)+1][i]:
            # print(texts[i],i, pred_label[i])
            if pred_label[i + 1] == 'B-Process':
                process_idx = i
            if pred_label[i + 1] == 'B-Temperature':
                temperature_idx = i
            if pred_label[i + 1] == 'B-Time':
                time_idx = i
    # print(process_idx,time_idx,temperature_idx)
    if temperature_idx != -1 and process_idx != -1:
        # print(texts[process_idx],'process-temperature',texts[temperature_idx])
        out_data['temperature-triple']['process'] = texts[process_idx]
        out_data['temperature-triple']['relation'] = 'process-temperature'
        out_data['temperature-triple']['temperature'] = texts[temperature_idx]
    if time_idx != -1 and process_idx != -1:
        # print(texts[process_idx],'process-time',texts[time_idx])
        out_data['time-triple']['process'] = texts[process_idx]
        out_data['time-triple']['relation'] = 'process-time'
        out_data['time-triple']['time'] = texts[time_idx]
    return out_data


def pred(data):
    global args
    dataset = pkl.load(open(args.data_pkl, "rb"))
    label_index, index_label = dataset[0], dataset[1]
    model = load_model(args.save_model_best, len(label_index))
    count = 0
    count_time = 0
    count_temperature = 0
    tokenizer = BertTokenizer.from_pretrained(args.bert_pred)
    for i in data:
        text = i['text']
        text_id = tokenizer.encode(text, add_special_tokens=True, max_length=args.max_len + 2,
                                   padding="max_length", truncation=True, return_tensors="pt")
        text_id = text_id.to(device)
        pred = model(text_id)
        pred_triple = text_class_name(text, pred, index_label)
        print(pred_triple,i['temperature-triple'],i['time-triple'])

        if pred_triple['time-triple'] == i['time-triple']:
            count = count + 1
            count_time = count_time + 1
        if pred_triple['temperature-triple'] == i['temperature-triple']:
            count = count + 1
            count_temperature = count_temperature + 1
    print(count,count_temperature,count_time)

with open('data/test_2.json','r',encoding='utf-8') as f:
    data = json.load(f)
    # print(data)
    seq = []
    for i in data:
        for j in i['ner']:
            seq.append({
                'text': i['sentence'][j['index'][0]:j['index'][len(j['index'])-1]+1],
                'type':j['type'],
                'process':j['process'],
                'time':j['time'],
                'temperature':j['temperature']
            })
    # print(seq)
    out = []
    for i in seq:
        out.append({
            'text':i['text'],
            'time-triple' : {
                'process' : i['process'],
                'relation': 'process-time',
                'time':i['time']
            },
            'temperature-triple': {
                'process': i['process'],
                'relation': 'process-temperature',
                'temperature': i['temperature']
            }
        })
    print(out)
    start = time.time()
    args = parsers()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pred(out)
    end = time.time()
    print(f"耗时为：{end - start} s")

