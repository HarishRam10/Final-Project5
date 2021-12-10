import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
# CoLA
data_train = pd.read_csv("CoLA/train.tsv", sep="\t",header=None)
print(data_train.head()); print('train len : ',len(data_train))
print(data_train[1].value_counts(normalize=True))
data_dev = pd.read_csv("CoLA/dev.tsv", sep="\t",header=None)
print(data_dev.head()); print('dev len : ',len(data_dev))
print(data_dev[1].value_counts(normalize=True))
data_test = pd.read_csv("CoLA/test.tsv", sep="\t",header=0)
print(data_test.head()); print('test len : ',len(data_test))

seq_len = [len(i.split()) for i in data_train[3]]
plt.hist(seq_len, bins= 50)
plt.title("Histgram of CoLA sequence length ")
plt.xlabel("sqeuence length")
plt.ylabel("count")
plt.show()

## SST-2
# data_train = pd.read_csv("SST-2/train.tsv", sep="\t",header=0)
# print(data_train.head()); print('train len : ',len(data_train))
# print(data_train.label.value_counts(normalize=True))
# data_dev = pd.read_csv("SST-2/dev.tsv", sep="\t",header=0)
# print(data_dev.head()); print('dev len : ',len(data_dev))
# print(data_dev.label.value_counts(normalize=True))
# data_test = pd.read_csv("SST-2/test.tsv", sep="\t",header=0)
# print(data_test.head()); print('test len : ',len(data_test))

## MRPC
# data_train = pd.read_csv('MRPC/msr_paraphrase_train.txt', sep=r"\t", header=0)[:3260]
# print(data_train.head()); print('train len : ',len(data_train)) # 4076
# print(data_train.columns)
# print(data_train.Quality.value_counts(normalize=True))
# data_dev = pd.read_csv("MRPC/msr_paraphrase_train.txt", sep=r"\t",header=0)[3260:]
# print(data_dev.head()); print('dev len : ',len(data_dev))
# print(data_dev.columns)
# print(data_dev.Quality.value_counts(normalize=True))
# data_test = pd.read_csv("MRPC/test.tsv", sep=r"\t",header=0)
# print(data_test.head()); print('test len : ',len(data_test))# 1725
# print(data_test.columns)

# # STS-B
# import matplotlib.pyplot as plt
# data_train = pd.read_csv("STS-B/train.tsv", sep=r"\t",header=0)
# print(data_train.head()); print('train len : ',len(data_train))
# print(data_train.columns)
# data_train.score.hist()
# plt.show()
#
# data_dev = pd.read_csv("STS-B/dev.tsv", sep=r"\t",header=0)
# print(data_dev.head()); print('dev len : ',len(data_dev))
# print(data_dev.columns)
# data_dev.score.hist()
# plt.show()
#
# data_test = pd.read_csv("STS-B/test.tsv", sep=r"\t",header=0)
# print(data_test.head()); print('test len : ',len(data_test))
# print(data_test.columns)

## QQP
# data_train = pd.read_csv("QQP/train.tsv", sep="\t",header=0)
# print(data_train.head()); print('train len : ',len(data_train))
# print(data_train.columns)
# print(data_train.is_duplicate.value_counts(normalize=True))
#
# data_dev = pd.read_csv("QQP/dev.tsv", sep="\t",header=0)
# print(data_dev.head()); print('dev len : ',len(data_dev))
# print(data_dev.columns)
# print(data_dev.is_duplicate.value_counts(normalize=True))
#
# data_test = pd.read_csv("QQP/test.tsv", sep="\t",header=0)
# print(data_test.head()); print('test len : ',len(data_test))
# print(data_test.columns)

# # MNLI
# data_train = pd.read_csv("MNLI/train.tsv", sep=r"\t",header=0)
# print(data_train.head()); print('train len : ',len(data_train))
# print(data_train.columns)
# print(data_train.gold_label.value_counts(normalize=True))
#
# data_dev_matched = pd.read_csv("MNLI/dev_matched.tsv", sep=r"\t",header=0)
# print(data_dev_matched.head()); print('dev_matched len : ',len(data_dev_matched))
# print(data_dev_matched.columns)
# print(data_dev_matched.gold_label.value_counts(normalize=True))
#
# data_dev_mismatched = pd.read_csv("MNLI/dev_mismatched.tsv", sep=r"\t",header=0)
# print(data_dev_mismatched.head()); print('dev_mismatched len : ',len(data_dev_mismatched))
# print(data_dev_mismatched.columns)
# print(data_dev_mismatched.gold_label.value_counts(normalize=True))

# data_test_matched = pd.read_csv("MNLI/test_matched.tsv", sep=r"\t",header=0)
# print(data_test_matched.head()); print('test_matched len : ',len(data_test_matched))
# print(data_test_matched.columns)
#
# data_test_mismatched = pd.read_csv("MNLI/test_mismatched.tsv", sep=r"\t",header=0)
# print(data_test_mismatched.head()); print('test_mismatched len : ',len(data_test_mismatched))
# print(data_test_mismatched.columns)

## QNLI
# data_train = pd.read_csv("QNLI/train.tsv", sep=r"\t",header=0)
# print(data_train.head()); print('train len : ',len(data_train))
# print(data_train.columns)
# print(data_train.label.value_counts(normalize=True))
#
# data_dev = pd.read_csv("QNLI/dev.tsv", sep=r"\t",header=0)
# print(data_dev.head()); print('dev len : ',len(data_dev))
# print(data_dev.columns)
# print(data_dev.label.value_counts(normalize=True))
#
# data_test = pd.read_csv("QNLI/test.tsv", sep=r"\t",header=0)
# print(data_test.head()); print('test len : ',len(data_test))
# print(data_test.columns)

# RTE
# data_train = pd.read_csv("RTE/train.tsv", sep=r"\t",header=0)
# print(data_train.head()); print('train len : ',len(data_train))
# print(data_train.columns)
# print(data_train.label.value_counts(normalize=True))
#
# seq_len1 = [len(i.split()) for i in data_train['sentence1']]
# plt.hist(seq_len1, bins= 50)
# plt.show()
# seq_len2 = [len(i.split()) for i in data_train['sentence2']]
# plt.hist(seq_len2, bins= 50)
# plt.show()
# seq_len = seq_len1 + seq_len2
# plt.hist(seq_len, bins= 50)
# plt.title("Histgram of RTE sequence length ")
# plt.xlabel("sqeuence length")
# plt.ylabel("count")
# plt.show()
# 50,100,150

# data_dev = pd.read_csv("RTE/dev.tsv", sep=r"\t",header=0)
# print(data_dev.head()); print('dev len : ',len(data_dev))
# print(data_dev.columns)
# print(data_dev.label.value_counts(normalize=True))
#
# data_test = pd.read_csv("RTE/test.tsv", sep=r"\t",header=0)
# print(data_test.head()); print('test len : ',len(data_test))
# print(data_test.columns)

# # WNLI
# data_train = pd.read_csv("WNLI/train.tsv", sep=r"\t",header=0)
# print(data_train.head()); print('train len : ',len(data_train))
# print(data_train.columns)
# print(data_train.label.value_counts(normalize=True))

# data_dev = pd.read_csv("WNLI/dev.tsv", sep=r"\t",header=0)
# print(data_dev.head()); print('dev len : ',len(data_dev))
# print(data_dev.columns)
# print(data_dev.label.value_counts(normalize=True))
#
# data_test = pd.read_csv("WNLI/test.tsv", sep=r"\t",header=0)
# print(data_test.head()); print('test len : ',len(data_test))
# print(data_test.columns)

# seq_len1 = [len(i.split()) for i in data_train['sentence1']]
# plt.hist(seq_len1, bins= 50)
# plt.show()
# seq_len2 = [len(i.split()) for i in data_train['sentence2']]
# plt.hist(seq_len2, bins= 50)
# plt.show()
# seq_len = seq_len1 + seq_len2
# plt.hist(seq_len, bins= 50)
# plt.show()