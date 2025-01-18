import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification , AutoModel , AutoConfig
import torch
from transformers import TrainingArguments
import evaluate
import numpy as np
from transformers import Trainer
from datasets import load_dataset

dataset= load_dataset("djangodevloper/amazon-review-dataset-cleaned")

model_checkpoint = 'bert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize(batch):
  temp = tokenizer(batch['text'], padding=True , truncation=True)
  return temp

tokenized_dataset = dataset.map(tokenize , batched=True ,batch_size=None)

label2id = {i['label_name']: str(i['label']) for i in dataset['train']}
id2label = {value:key for key,value in label2id.items()}


model = AutoModel.from_pretrained(model_checkpoint)
config = AutoConfig.from_pretrained(model_checkpoint, num_labels = len(label2id) , id2label =id2label ,label2id =label2id)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint , config=config)
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

training_dir = 'model_training'
batch_size = 16
training_args = TrainingArguments(
    output_dir =training_dir,
    overwrite_output_dir=True,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    num_train_epochs=2,
    eval_strategy='epoch',
    disable_tqdm=False
    )

accuracy =evaluate.load('accuracy')
def compute_matrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions,axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    compute_metrics=compute_matrics
)

trainer.train()
trainer.save_model("bert-base-uncased-product-review-classification")
tokenizer.save_pretrained("bert-base-uncased-product-review-classification")


model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased-product-review-classification")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased-product-review-classification")
repo_name = "djangodevloper/bert-base-uncased-product-review-classification"
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)