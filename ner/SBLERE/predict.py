

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


def pred_one():
    global args

    text = "which decreases by only 7% to 1047MPa (152±1ksi) after aging at 450°C for 320h"
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


if __name__ == "__main__":
    start = time.time()
    args = parsers()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    pred_one()  # 预测一条文本
    end = time.time()
    print(f"耗时为：{end - start} s")
