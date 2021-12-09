import transformers
print(transformers.__version__)

GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
print("2")
task = "qnli"
print(task)
#model_checkpoint = "distilbert-base-uncase d"
#model_checkpoint = "google/electra-base-discriminator"

model_checkpoint = "xlnet-base-cased"
#model_checkpoint = "bert-base-cased"
print(model_checkpoint)
batch_size = 20

from datasets import load_dataset, load_metric

actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("glue", actual_task)
metric = load_metric('glue', actual_task)
import pandas as pd

validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"

# df = pd.DataFrame(dataset[validation_key]['label'])
# print(len(df))
# df.to_csv("wnli_val_labels.csv",index = False)


import numpy as np

fake_preds = np.random.randint(0, 2, size=(64,))
fake_labels = np.random.randint(0, 2, size=(64,))
metric.compute(predictions=fake_preds, references=fake_labels)

from transformers import AutoTokenizer #Try using ElectraTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, ElectraForSequenceClassification
from transformers import BertForSequenceClassification
from transformers import XLNetForSequenceClassification
num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
#model = ElectraForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
#model = BertForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
model = XLNetForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)



metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-finetuned-{task}",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name
)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

sentence1_key, sentence2_key = task_to_keys[task]
if sentence2_key is None:
    print(f"Sentence: {dataset['train'][0][sentence1_key]}")
else:
    print(f"Sentence 1: {dataset['train'][0][sentence1_key]}")
    print(f"Sentence 2: {dataset['train'][0][sentence2_key]}")


def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)


encoded_dataset = dataset.map(preprocess_function, batched=True)
columns_to_return = ['input_ids', 'label', 'attention_mask']
encoded_dataset.set_format(type='torch', columns=columns_to_return)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.evaluate()

# # electra 0.67738
# # # --trainer.predict(encoded_dataset['test'])
# e_pred = trainer.predict(encoded_dataset[validation_key])
# # --e_pred.label_ids # vlabel
# # # --electra_pred = np.argmax(pred.predictions,axis=1) # vprediction
# electra_pred = e_pred.predictions # vprediction
# import pandas as pd
# df = pd.DataFrame(electra_pred)
# df.to_csv('electra_pred_sst2.csv',index=False)

#xlnet 0.4676
xl_pred = trainer.predict(encoded_dataset[validation_key])
xlnet_pred = xl_pred.predictions
import pandas as pd
df1 = pd.DataFrame(xlnet_pred)
df1.to_csv('xlnet_qnli_pred.csv',index=False)

# bert_cased  0.5955
# b_pred = trainer.predict(encoded_dataset[validation_key])
# bert_pred = b_pred.predictions
# import pandas as pd
# df2 = pd.DataFrame(bert_pred)
# df2.to_csv('bert_pred_mrpc.csv',index=False)

