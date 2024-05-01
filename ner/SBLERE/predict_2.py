

from MyModel import BertNerModel
from transformers import BertTokenizer
import pickle as pkl
import torch
from config import parsers
from utils import read_data, MyDataset, build_label_index
from torch.utils.data import DataLoader
import time
from seqeval.metrics import f1_score, precision_score, recall_score
import json

def load_model(model_path, class_num):
    global device
    model = BertNerModel(class_num).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def text_class_name(texts, pred, index_label):
    pred_label = torch.argmax(pred, dim=-1)
    print(pred_label)
    result = torch.argmax(pred, dim=-1)
    result = result.cpu().numpy().tolist()[0]
    print(result)
    # print(index_label)
    pred_label = [index_label[i] for i in result]
    print("模型预测结果：")
    print(f"文本：{texts}\t预测的类别为：{pred_label[1:len(texts)+1]}")
    for i in range(len(pred_label[1:len(texts)+1])):
        # print(pred_label[1:len(texts)][i])
        if pred_label[1:len(texts)+1][i] != 'PAD' and 'O' != pred_label[1:len(texts)+1][i]:
            print(texts[i])

def text_class_name_2(texts, pred, index_label):
    pred_label = torch.argmax(pred, dim=-1)
    print(pred_label)
    result = torch.argmax(pred, dim=-1)
    result = result.cpu().numpy().tolist()[0]
    print(result)
    # print(index_label)
    pred_label = [index_label[i] for i in result]
    print("模型预测结果：")
    print(f"文本：{texts}\t预测的类别为：{pred_label[1:len(texts)+1]}")
    for i in range(len(pred_label[1:len(texts)+1])):
        # print(pred_label[1:len(texts)][i])
        if pred_label[1:len(texts)+1][i] != 'PAD' and 'O' != pred_label[1:len(texts)+1][i]:
            print(texts[i])


def pred_test_2():
    tokenizer = BertTokenizer.from_pretrained(args.bert_pred)
    dataset = pkl.load(open(args.data_pkl, "rb"))
    label_index, index_label = dataset[0], dataset[1]
    model = load_model(args.save_model_best, len(label_index))
    with open('data/test_2.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

        for i in data:
            text_id = tokenizer.encode(i['sentence'], add_special_tokens=True, max_length=args.max_len + 2,
                                       padding="max_length", truncation=True, return_tensors="pt")
            pred = model(text_id)
            text_class_name(i['sentence'], pred, index_label)



def pred_one():
    global args
    text = "Analytical grade lanthanide nitrate and NaVO3 were purchased from the Shanghai chemical industrial company and were used without further purification. The tetragonal LnVO4 (Ln=La, Nd, Sm, Eu, Dy) nanorods were prepared by a simplehydrothermal method. The reaction was carried out in a 20mL capacity Teflon-lined stainless-steel autoclave, in a digital-type temperature-controlled oven. Taking the synthesis of LaVO4 as example, in a typical synthesis 16mL of NaVO3 (0.2M) aqueous solution were added into 8mL of La(NO3)3 aqueous solution (0.4M) at 25°C under vigorous stirring, the solution turned yellow immediately after the addition of the NaVO3. The obtained yellow suspension was stirred for about 10min , then 4.8mL 1M NaOH aqueous solution was added to adjust the pH to 4.5. The resulting yellow suspension was divided into two equal parts. The firsthalf was filtered off, washed with distilled water and absolute ethanol, respectively, and then dried at 25°C under vacuum 12h for further characterization. The product appeared as yellow prismatic crystals, which were identified as monoclinic LaVO4 by X-ray diffractometry. The second was poured into a Teflon-lined stainless-steel autoclave. The autoclave was sealed and maintained at 180°C for 48h and then air cooled to 25°C , the resulting LnVO4 products were filtered, washed with deionized water and absolute alcohol to remove ions possibly remaining in the final products, and finally dried at 80°C in air for further characterization, the products (tetragonal LaVO4) obtained afterhydrothermal process were white powders."
    # text = "which decreases by only 7% to 1047MPa (152±1ksi) after aging at 450°C for 320h"
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

def pred_test():

    global args
    test_text, test_label = read_data(args.test_file)
    label_to_index, index_to_label = build_label_index(test_label)
    testDataset = MyDataset(test_text, label_to_index, labels=test_label, with_labels=True)
    testLoader = DataLoader(testDataset, batch_size=args.batch_size, shuffle=False)
    dataset = pkl.load(open(args.data_pkl, "rb"))
    label_index, index_label = dataset[0], dataset[1]
    print(dataset)
    tokenizer = BertTokenizer.from_pretrained(args.bert_pred)

    # print(text_id)
    model = load_model(args.save_model_best, len(label_index))
    model.eval()
    all_pre = []
    all_tag = []
    all_pre_2 = []
    all_tag_2 = []
    new = []
    for batch_text, batch_label in testLoader:
        tt = []
        pp = []
        tt_2 = []
        pp_2 = []
        batch_text = batch_text.to(device)
        batch_label = batch_label.to(device)
        pred = model(batch_text)

        pred_label = torch.argmax(pred, dim=-1).cpu().numpy().tolist()
        tag_label = batch_label.cpu().numpy().tolist()

        for pred, tag in zip(pred_label, tag_label):
            p = [index_label[i] for i in pred]
            t = [index_label[i] for i in tag]
            # print(index_label)
            for i in range(len(t)):
                if p[i] != 'O' and p[i] != 'PAD' and t[i] == 'O':
                    tt.append(t[i])
                    pp.append(p[i])
                if t[i] != 'PAD' and t[i] != 'O':
                    tt.append(t[i])
                    pp.append(p[i])
                    tt_2.append(t[i])
                    pp_2.append(p[i])
            all_pre.append(pp)
            all_tag.append(tt)
            all_pre_2.append(pp)
            all_tag_2.append(tt)
            new.append(pp)
    # print(all_pre)
    # print(pp)
    # print(new)
    f1 = f1_score(all_tag, all_pre)
    precision = precision_score(all_tag, all_pre)
    recall = recall_score(all_tag, all_pre)
    print(f"test f1:{f1}, precision:{precision}，recall:{recall}")
    f1 = f1_score(all_tag_2, all_pre_2)
    precision = precision_score(all_tag_2, all_pre_2)
    recall = recall_score(all_tag_2, all_pre_2)
    print(f"test f1:{f1}, precision:{precision}，recall:{recall}")
    # print(pred)



if __name__ == "__main__":
    start = time.time()
    args = parsers()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    # pred_one()  # 预测一条文本\
    pred_test()
    end = time.time()
    print(f"耗时为：{end - start} s")
