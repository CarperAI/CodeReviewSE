from transformers import AdamW, SchedulerType, get_scheduler, set_seed, AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import Dataset
import torch
import pandas as pd
from torch.utils.data import DataLoader
from functools import partial
from data.helper import load_json_file
from tqdm.auto import tqdm
import argparse




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dataset/CodeReviewSE_clean_QA.json')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--model_name', type=str, default='Salesforce/codet5-base')
    parser.add_argument('--max_input_length', type=int, default=256)
    parser.add_argument('--max_target_length', type=int, default=128)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--lr_scheduler', type=SchedulerType, default="linear", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb_project', type=str, default='finetune-codet5-for-codereview')
    parser.add_argument('--num_update_steps_per_epoch', type=int, default=100)

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


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    
    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset = load_json_file(args.dataset)
    dataset = Dataset.from_pandas(pd.DataFrame(data=dataset))
    dataset = dataset.map(partial(preprocess_examples, tokenizer=tokenizer, max_input_length=max_input_length, max_target_length=max_target_length), batched=True, num_proc=16)
    print(dataset)
    
    dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
    train_dataloader = DataLoader(dataset, shuffle=True, batch_size=8)

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
        num_training_steps=args.max_train_steps,
    )

    if args.wandb_project:
        wandb.init(project=args.wandb_project, config=args)
        wandb.watch(model)
    
    progress_bar = tqdm(range(max_train_steps))
    global_steps = 0

    # Train the model
    for epoch in range(args.num_train_epochs):
        print("Epoch: {}".format(epoch))
        model.train()
        for step, batch in enumerate(train_dataloader):
            input_ids, attention_mask, labels = batch
            loss = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                global_steps += 1
                loss = loss.item()
                tqdm.write(f"epoch = {epoch}, step = {global_steps}, loss = {loss}")
                if args.wandb_proj:
                    wandb.log({"loss": loss}, step=global_steps)
                
            
            if (step + 1) % args.num_update_steps_per_epoch == 0:
                print("Step: {}/{}".format(step + 1, args.num_update_steps_per_epoch))
                print("Loss: {}".format(loss.item()))
                print("LR: {}".format(optimizer.param_groups[0]["lr"]))
                print("\n")
            
            global_steps += 1
    
    if args.wandb_project:
        wandb.finish()
    

