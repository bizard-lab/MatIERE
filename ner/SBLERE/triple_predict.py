

from MyModel import BertNerModel
from transformers import BertTokenizer
import pickle as pkl
import torch
from config import parsers
import time
import json


def load_model(model_path, class_num):
    global device
    model = BertNerModel(class_num).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def text_class_name(texts, pred, index_label):
    print('-----------------------------------')
    pred_label = torch.argmax(pred, dim=-1)
    # print(pred_label)
    result = torch.argmax(pred, dim=-1)
    result = result.cpu().numpy().tolist()[0]
    print(result)
    # print(index_label)
    pred_label = [index_label[i] for i in result]
    print("模型预测结果：")
    print(f"文本：{texts}\t预测的类别为：{pred_label[1:len(texts)+1]}")
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
    print(process_idx,time_idx,temperature_idx)
    spolist = []
    if temperature_idx != -1 and process_idx != -1:
        print(texts[process_idx],'process-temperature',texts[temperature_idx])
        spolist.append({
            's':texts[process_idx],
            'p':'process-temperature',
            'o':texts[temperature_idx]
        })
    if time_idx != -1 and process_idx != -1:
        print(texts[process_idx],'process-time',texts[time_idx])
        spolist.append({
            's': texts[process_idx],
            'p': 'process-time',
            'o': texts[time_idx]
        })
    # print(spolist)
    return spolist



def pred_one():
    global args

    text = "aging at 450°C for 320h"
    text = text.split(' ')
    dataset = pkl.load(open(args.data_pkl, "rb"))
    label_index, index_label = dataset[0], dataset[1]
    print(dataset)
    tokenizer = BertTokenizer.from_pretrained(args.bert_pred)

    text_id = tokenizer.encode(text, add_special_tokens=True, max_length=args.max_len + 2,
                               padding="max_length", truncation=True, return_tensors="pt")
    # print(text_id)
    model = load_model(args.save_model_best, len(label_index))

    text_id = text_id.to(device)
    pred = model(text_id)
    # print(pred)
    text_class_name(text, pred, index_label)
    text = "dried at 100°C for 24h"
    text = text.split(' ')
    text_id = tokenizer.encode(text, add_special_tokens=True, max_length=args.max_len + 2,
                               padding="max_length", truncation=True, return_tensors="pt")
    text_id = text_id.to(device)
    pred = model(text_id)
    text_class_name(text, pred, index_label)


def pred_test(data):
    global args
    spolist = []
    # text = "aging at 450°C for 320h"
    # text = text.split(' ')
    dataset = pkl.load(open(args.data_pkl, "rb"))
    label_index, index_label = dataset[0], dataset[1]
    print(dataset)
    tokenizer = BertTokenizer.from_pretrained(args.bert_pred)

    # text_id = tokenizer.encode(text, add_special_tokens=True, max_length=args.max_len + 2,
    #                            padding="max_length", truncation=True, return_tensors="pt")
    # print(text_id)
    model = load_model(args.save_model_best, len(label_index))
    for i in data:
        temp = []
        for j in i:
            # print(j)
            if len(j) != 0:
                text_id = tokenizer.encode(j, add_special_tokens=True, max_length=args.max_len + 2,
                                           padding="max_length", truncation=True, return_tensors="pt")
                text_id = text_id.to(device)
                pred = model(text_id)
                t = text_class_name(j, pred, index_label)
                print(t)
                temp.append(t)
            else:
                temp.append([{
                    's':'NULL',
                    'p':'NULL',
                    'o':'NULL'
                }])
        spolist.append(temp)
    return spolist
    # text_id = text_id.to(device)
    # pred = model(text_id)
    # # print(pred)
    # text_class_name(text, pred, index_label)
    # text = "dried at 100°C for 24h"
    # text = text.split(' ')
    # text_id = tokenizer.encode(text, add_special_tokens=True, max_length=args.max_len + 2,
    #                            padding="max_length", truncation=True, return_tensors="pt")
    # text_id = text_id.to(device)
    # pred = model(text_id)
    # text_class_name(text, pred, index_label)

if __name__ == "__main__":
    start = time.time()
    args = parsers()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    # pred_one()  # 预测一条文本
    #pred_one()  # 多条文本
    text = []
    # with open('data/w2ner_output/easy/test_2.json','r',encoding='utf-8') as f:
    # path = 'data/BERT_BiLSTM_CRF/hard/out_data.json'
    # path = 'data/BERT_BiLSTM_CRF/easy/out_data.json'
    # path = 'data/BERT_Linner/hard/out_data.json'
    # path = 'data/BERT_Linner/easy/out_data.json'
    # path = 'data/BERT_LSTM/hard/out_data.json'
    path = 'data/BERT_LSTM/easy/out_data.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for i in data:
            # print(i)
            temp = []
            for j in i['entity']:
                temp.append(j['text'])
            text.append(temp)
    spolist = pred_test(text)
    # out_path = 'data/BERT_BiLSTM_CRF/hard/triple_predict.json'
    # out_path = 'data/BERT_BiLSTM_CRF/easy/triple_predict.json'
    # out_path = 'data/BERT_Linner/hard/triple_predict.json'
    # out_path = 'data/BERT_Linner/easy/triple_predict.json'
    # out_path = 'data/BERT_LSTM/hard/triple_predict.json'
    out_path = 'data/BERT_LSTM/easy/triple_predict.json'
    # out_path = 'data/BERT_Linner/easy/triple_predict.json'
    # with open('data/w2ner_output/easy/test_X_大规模训练结果.json','w',encoding='utf-8') as fw:
    with open(out_path, 'w', encoding='utf-8') as fw:
        fw.write('[')
        fw.write('\n')
        for idx,line in enumerate(spolist):
            stt_ = json.dumps(line, ensure_ascii=False)
            fw.write(stt_)
            if idx != len(spolist) -1 :
                fw.write(',')
                fw.write('\n')
            else:
                fw.write('\n')
        fw.write(']')
    end = time.time()
    print(f"耗时为：{end - start} s")
