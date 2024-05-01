

import argparse
import os.path


def parsers():
    parser = argparse.ArgumentParser(description="Bert model of argparse")
    parser.add_argument("--train_file", type=str, default=os.path.join(r"../data", "train.txt"))
    parser.add_argument("--dev_file", type=str, default=os.path.join(r"../data", "dev.txt"))
    parser.add_argument("--test_file", type=str, default=os.path.join(r"../data", "test.txt"))
    parser.add_argument("--data_pkl", type=str, default=os.path.join(r"../data", "dataParams.pkl"))
    parser.add_argument("--bert_pred", type=str, default=r"D:\Projects\MatSciBERT\pretraining\matscibert", help="bert 预训练模型")
    parser.add_argument("--max_len", type=int, default=20, help="句子的最大长度")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learn_rate", type=float, default=0.000005)
    parser.add_argument("--save_model_best", type=str, default=os.path.join("model", "best_model.pth"))
    parser.add_argument("--save_model_last", type=str, default=os.path.join("model", "last_model.pth"))
    args = parser.parse_args()
    return args
