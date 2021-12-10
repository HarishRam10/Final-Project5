import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import seaborn as sns
import transformers
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
# from transformers import RobertaModel, RobertaTokenizer
from transformers import ElectraModel, ElectraTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
import re
from datasets import load_dataset, load_metric
# %%-------------------------------------------------------------------------------
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

task = "cola"
batch_size = 16


actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("glue", actual_task)
metric = load_metric('glue', actual_task)

train = dataset['train']
test = dataset['test']
# %%-------------------------------------------------------------------------------
model_checkpoint = 'google/electra-base-discriminator'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%-------------------------------------------------------------------------------
class Dataset(Dataset):
    def __init__(self, df, tokenizer, maxlen):
        self.tokenizer = tokenizer
        self.data = df
        self.text = df['sentence']
        self.targets = df['label']
        self.maxlen = maxlen

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = str(self.text[idx])
        text = ' '.join(text.split())
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.maxlen,
            pad_to_max_length=True,
            return_token_type_ids=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[idx], dtype=torch.long)
        }
# %%-------------------------------------------------------------------------------
MAX_LEN = 128
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
# EPOCHS = 1
LEARNING_RATE = 1e-05
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
tokenizer = ElectraTokenizer.from_pretrained(model_checkpoint, truncation=True, do_lower_case=True)


train_dataset = Dataset(train, tokenizer, MAX_LEN)
test_dataset = Dataset(test, tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size = TRAIN_BATCH_SIZE, shuffle = True, num_workers = 2)
test_loader = DataLoader(test_dataset, batch_size = VALID_BATCH_SIZE, shuffle = False, num_workers = 2)
# %%-------------------------------------------------------------------------------
class Roberta_Model(nn.Module):

    def __init__(self):
        super(Roberta_Model, self).__init__()
        # self.l1 = RobertaModel.from_pretrained('roberta-base')
        self.l1 = ElectraModel.from_pretrained(model_checkpoint)
        self.pre_classifier1 = nn.Linear(768, 768)
        self.pre_classifier = nn.Linear(768, 256)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(256, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        x = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = x[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


model = Roberta_Model()
model.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


def calcuate_accuracy(preds, targets):
    n_correct = (preds == targets).sum().item()
    return n_correct


def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _, data in tqdm(enumerate(train_loader, 0)):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accuracy(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if _ % 5000 == 0:
            loss_step = tr_loss / nb_tr_steps
            accu_step = (n_correct * 100) / nb_tr_examples
            print(f"Training Loss per 5000 steps: {loss_step}")
            print(f"Training Accuracy per 5000 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct * 100) / nb_tr_examples}')
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    return


EPOCHS = 1
for epoch in range(EPOCHS):
    train(epoch)


def valid(model, testing_loader):
    model.eval()
    n_correct = 0;
    n_wrong = 0;
    total = 0;
    tr_loss = 0;
    nb_tr_steps = 0;
    nb_tr_examples = 0
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            outputs = model(ids, mask, token_type_ids).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accuracy(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if _ % 5000 == 0:
                loss_step = tr_loss / nb_tr_steps
                accu_step = (n_correct * 100) / nb_tr_examples
                print(f"Validation Loss per 100 steps: {loss_step}")
                print(f"Validation Accuracy per 100 steps: {accu_step}")
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")

    return epoch_accu


acc = valid(model, test_loader)
print("Accuracy on test data = %0.2f%%" % acc)


output_model_file = 'pytorch_roberta_sentiment.bin'
output_vocab_file = './'

model_to_save = model
torch.save(model_to_save, output_model_file)
tokenizer.save_vocabulary(output_vocab_file)

print('All files saved')