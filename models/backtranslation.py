import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class BackTranslationModel:
    def __init__(self, src_model_name, tgt_model_name, device='cpu', batch_size=32, max_length=300):
        self.src_model = AutoModelForSeq2SeqLM.from_pretrained(src_model_name)
        self.src_model.eval()
        self.src_model.to(device)
        self.tgt_model = AutoModelForSeq2SeqLM.from_pretrained(tgt_model_name)
        self.tgt_model.eval()
        self.tgt_model.to(device)
        self.src_tokenizer = AutoTokenizer.from_pretrained(src_model_name)
        self.tgt_tokenizer = AutoTokenizer.from_pretrained(tgt_model_name)

        self.batch_size = batch_size
        self.max_length = max_length

    def translate(self, src_text):
        src_ids = self.src_tokenizer.encode(src_text, return_tensors='pt')
        src_ids = src_ids.to(self.src_model.device)
        output = self.src_model.generate(src_ids, do_sample=True, max_length=self.max_length)
        output = output.to('cpu')
        output = output.tolist()
        output = output[0]
        output = self.tgt_tokenizer.decode(output)
        return output