from transformers import RobertaTokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from datasets import Dataset
import torch
import pandas as pd
from torch.utils.data import DataLoader
from functools import partial
from data.helper import load_json_file

tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")

max_input_length = 256
max_target_length = 128

def preprocess_examples(examples, tokenizer, max_input_length, max_target_length):
    # encode the question-answer pairs
    question = examples['question']
    answer = examples['answer']

    model_inputs = tokenizer(question, max_length=max_input_length, padding="max_length", truncation=True)
    labels = tokenizer(answer, max_length=max_target_length, padding="max_length", truncation=True).input_ids

    # important: we need to replace the index of the padding tokens by -100
    # such that they are not taken into account by the CrossEntropyLoss
    labels_with_ignore_index = []
    for labels_example in labels:
        labels_example = [label if label != 0 else -100 for label in labels_example]
        labels_with_ignore_index.append(labels_example)
    
    model_inputs["labels"] = labels_with_ignore_index

    return model_inputs

if __name__ == "__main__":
    dataset = load_json_file("dataset/CodeReviewSE_clean_QA.json")
    dataset = Dataset.from_pandas(pd.DataFrame(data=dataset))
    dataset = dataset.map(partial(preprocess_examples, tokenizer=tokenizer, max_input_length=max_input_length, max_target_length=max_target_length), batched=True, num_proc=16)
    print(dataset)
    
    dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
    train_dataloader = DataLoader(dataset, shuffle=True, batch_size=8)
    batch = next(iter(train_dataloader))
    print(batch.keys())
    print(tokenizer.decode(batch['input_ids'][0]))
    labels = batch['labels'][0]
    print(tokenizer.decode([label for label in labels if label != -100]))
