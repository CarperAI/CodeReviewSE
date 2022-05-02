from transformers import AdamW, SchedulerType, get_scheduler, set_seed, AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import Dataset, load_metric
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from accelerate import Accelerator
from functools import partial
from data.helper import load_json_file
from tqdm.auto import tqdm
import argparse
import math
import numpy as np
import nltk
nltk.download('punkt')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dataset/CodeReviewSE_clean_QA.json')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--model_name', type=str, default='Salesforce/codet5-base')
    parser.add_argument('--max_input_length', type=int, default=512)
    parser.add_argument('--max_target_length', type=int, default=256)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_warmup_steps', type=int, default=10)
    parser.add_argument('--lr_scheduler_type', type=SchedulerType, default="linear", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument('--validation_split', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--wandb_project', type=str, default='finetune-codet5-for-codereview')
    parser.add_argument('--num_update_steps_per_epoch', type=int, default=100)
    parser.add_argument('--checkpointing_frequency', type=int, default=1)

    return parser.parse_args()




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


def calc_metric(metric, predictions, labels):
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in predictions]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in labels]
    result = metric.compute(predictions=predictions, references=labels)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return result

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    torch.cuda.set_device(args.gpu_id)

    accelerator = Accelerator()
    
    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset = load_json_file(args.dataset)
    dataset = Dataset.from_pandas(pd.DataFrame(data=dataset))
    dataset = dataset.map(partial(preprocess_examples, tokenizer=tokenizer, max_input_length=args.max_input_length, max_target_length=args.max_target_length), batched=True, num_proc=16)    
    dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
    dataset = dataset.train_test_split(test_size=args.validation_split, seed=args.seed)
    train_dataloader = DataLoader(dataset['train'], shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    val_dataloader = DataLoader(dataset['test'], shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # metric
    metric = load_metric('rouge') # hard-coded to ROUGE

    # accelerator
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader, lr_scheduler)

    # wandb
    if args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project, config=args)
        wandb.watch(model)
    
    progress_bar = tqdm(range(max_train_steps))
    global_steps = 0

    # Train the model
    for epoch in range(args.num_train_epochs):
        print("Epoch: {}".format(epoch))
        model.train()
        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                global_steps += 1
                loss = loss.item()
                #tqdm.write(f"epoch = {epoch}, step = {global_steps}, loss = {loss}")
                if args.wandb_project:
                    wandb.log({"loss": loss}, step=global_steps)
                
            
            if (step + 1) % args.num_update_steps_per_epoch == 0:
                print("Step: {}/{}".format(step + 1, max_train_steps))
                print("Loss: {}".format(loss))
                print("LR: {}".format(optimizer.param_groups[0]["lr"]))
                print("\n")
                
                
        # evaluate on validation set
        model.eval()
        all_input = []
        all_preds = []
        all_labels = []
        for step, batch in tqdm(enumerate(val_dataloader),total=len(val_dataloader)):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"].cpu()
            generated_ids = model.generate(
                input_ids = input_ids,
                attention_mask = attention_mask, 
                max_length=150, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
            decoded_input = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            all_input += decoded_input
            all_preds += decoded_preds
            all_labels += decoded_labels

        # evaluate
        eval_metric = calc_metric(metric, all_preds, all_labels)
        print(eval_metric)

        # checkpoints
        if epoch % args.checkpointing_frequency == 0:
            accelerator.save_state(f'checkpoints/epoch_{epoch}')

        

        if args.wandb_project:
            preds_table = pd.DataFrame({"input": all_input, "preds": all_preds, "labels": all_labels})
            wandb.log({'validation predictions': wandb.Table(dataframe=preds_table)}, step=global_steps)
            wandb.log(eval_metric, step=global_steps)
            wandb.save('checkpoints/**/*')


    if args.wandb_project:
        model.save_pretrained('final')
        wandb.save('final/*')
        wandb.finish()
    

