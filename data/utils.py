import json
import logging
from transformers import AutoTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_json_to_dict(file_path:str):
    with open(file_path) as f:
        return json.load(f)

def write_dict_to_json(file_path:str,data:dict):
    with open(file_path,"w") as f:
        json.dump(data,f)

if __name__ == "__main__":
    dataset_path : str = "dataset/CodeReviewSE_clean.json"
    model_name : str= "Salesforce/codegen-6B-multi"
    #dataset : dict = load_json_to_dict(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Sucessfully loaded dataset and tokenizer to memory...")
    len_dict = {}
    # for ind in tqdm(range(len(dataset))):
    #     key = list(dataset.keys())[ind]
    #     datapoint = dataset[key]
    #     question = datapoint["body"]
    #     if "AcceptedAnswerId" in datapoint["meta_data"]:
    #         answer_id = datapoint["meta_data"]["AcceptedAnswerId"] #Get the answer_id

    #     for answer in datapoint["answers"]:
    #         if answer["meta_data"]["Id"] == answer_id:
    #             accepted_answer =  answer
    #             break
            
    #     answer_body = accepted_answer["body"]
    #     answer = tokenizer.encode(answer_body)
    #     question_tokens = tokenizer.encode(question)
    #     len_dict[key] = len(question_tokens) + len(answer)

    #write_dict_to_json("stats/len_dict.json",len_dict)

    # for key in len_dict:
    len_dict = load_json_to_dict("stats/len_dict.json")
    exceed_counter = 0
    exceed_len = 0
    exceed_list = []
    for key in tqdm(len_dict):
        if len_dict[key] > 2048:
            exceed_counter += 1
            exceed_list.append(len_dict[key])
            exceed_len += len_dict[key]
    
    logger.info(f"{sum(exceed_list)/len(exceed_list)}")
    logger.info(f"{exceed_counter} out of {len(len_dict)} questions exceed 2048 tokens")
    #write_dict_to_json("stats/exceed_dict.json",exceed_dict)

    

        
