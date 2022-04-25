from transformers import RobertaTokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader

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

if __name__ == "__main__": print('hello')
