import json
import logging
from tqdm import tqdm
import bs4
from pprint import pprint
import os
import openai
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def load_json_to_dict(file_path:str):
    with open(file_path) as f:
        return json.load(f)

def write_dict_to_json(file_path:str,data:dict):
    with open(file_path,"w") as f:
        json.dump(data,f,indent=2)

keys_dict = load_json_to_dict("configs/keys.json")
openai.organization = keys_dict["org"]
openai.api_key = keys_dict["key"]
#write_dict_to_json("configs/model_list.json",openai.Model.list())
logger.info("successfully loaded API key...")

def clean_parsed_code(code_block:str):
    code_block = code_block.replace("<code>","").replace("</code>","")
    if " " not in code_block:
        return ""
    else:
        return code_block


def parse_html_block_for_code(html_block:str)->list[str]:

    parser = bs4.BeautifulSoup(html_block,"html.parser")
    parsed_code_list : list[str] =  parser.find_all("code")
    parsed_code_list = [clean_parsed_code(code.text) for code in parsed_code_list]
    parsed_code_list = [code for code in parsed_code_list if code != ""]
    return parsed_code_list

class SEData:
    def __init__(self,dataset_path:str="dataset/CodeReviewSE_clean.json") -> None:
        self.dataset : dict = load_json_to_dict(dataset_path)
        self.python_tags : list[str] = ["python", "python-2.x", "python-3.x","py"]
        logger.info(f"Sucessfully loaded dataset to memory...")
        self.data_len = len(self.dataset) #Length of the dataset
        self.instruction_template : dict = {
            "type_inference"  : [
                "Annotate type for identifiers in the python code"
            ],
            "test_case_assertion" : [
                ""
            ]
        }
    
    def filter_python_by_tags(self,datapoint:dict):
        bool_flag = False
        if "Tags" in datapoint["meta_data"]:
            tags = datapoint["meta_data"]["Tags"]
            for tag in tags:
                if tag in self.python_tags:
                    bool_flag = True        
            return bool_flag
        else:
            return False

    def process_datapoint(self,datapoint:dict):
        if "AcceptedAnswerId" in datapoint["meta_data"]:
            answer_id = datapoint["meta_data"]["AcceptedAnswerId"] #Get the answer_id

            for answer in datapoint["answers"]:
                if answer["meta_data"]["Id"] == answer_id:
                    accepted_answer =  answer
                    break
            
            answer_body = accepted_answer["body"] #Get the body of the accepted answer
            accepted_answer_code : list[str] = parse_html_block_for_code(answer_body)
            return accepted_answer_code
        else:
            return []

    def apply_codex_for_type_inference(self,input_prompt:str)->dict:
        """
        Apply codex for type annotation of the input prompt

        args:
            input_prompt (str) : input prompt to be annotated
        """
        response = openai.Edit.create(
        model = "code-davinci-edit-001",
        input = input_prompt,
        instruction = self.instruction_template["type_inference"][0],
        temperature=0.7,
        top_p=1
        )
        return response

    def iter_apply_process(self,length:int,if_apply_codex_flag:str="type_inference",hash_path:str="dataset/CodeReviewSE_hash.txt"):
        processed_dict = {}
        counter = 0
        for i in tqdm(range(length)):
            key = list(self.dataset.keys())[i]
            datapoint = self.dataset[key]
            if key not in processed_dict:
                processed_list = self.process_datapoint(datapoint)
                if len(processed_list) > 0:
                    processed_dict[key] = {"code":[],"augmented_code":[]}
                    processed_dict[key]["code"] = processed_list
                    if if_apply_codex_flag == "type_inference":
                        for processed_code in processed_list:
                            try:
                                hash_text_file = open(hash_path,"a")

                                response = self.apply_codex_for_type_inference(processed_code)
                                processed_dict[key]["augmented_code"].append(response["choices"][0]["text"])
                                hash_text_file.write(f'{key}#@#{response["choices"][0]["text"]}#@#{processed_code}\n@##@##@')

                            except openai.error.RateLimitError:
                                logger.info("Rate limit error. Waiting for 65 seconds...")
                                time.sleep(65)
                                hash_text_file.close()
                                continue
                            except openai.error.InvalidRequestError:
                                logger.info("OpenAI error.")
                                pass
                    counter += len(processed_list)
        logger.info(f"Sucessfully processed dataset to python dataset...")
        logger.info(f"Length of processed code blocks dataset: {counter}")
        return processed_dict


        

    def iter_filter_python_dataset(self):
        self.python_dataset = {}
        for index in tqdm(range(self.data_len)):
            key = list(self.dataset.keys())[index]
            datapoint = self.dataset[list(self.dataset.keys())[index]]
            if self.filter_python_by_tags(datapoint):
                self.python_dataset[key] = datapoint
        logger.info(f"Sucessfully filtered dataset to python dataset...")
        logger.info(f"Length of python dataset: {len(self.python_dataset)}")
        write_dict_to_json("dataset/CodeReviewSE_py.json",self.python_dataset)

if __name__ == "__main__":
    pass
    se_data = SEData("dataset/CodeReviewSE_py.json")#.iter_filter_python_dataset()
    processed_python_dataset = se_data.iter_apply_process(100,"type_inference")
    write_dict_to_json(f"dataset/CodeReviewSE_py_processed_codex_100.json",processed_python_dataset)
    # test_string = "<p>I used a slightly simpler method, but essentially did the same thing:</p>\n\n<pre><code>total = 0\n\nfor n in range(3,1000):\n    if n % 3 == 0:\n        total += n\n    elif n % 5 == 0:\n        total += n\n\n\nprint total\n</code></pre>\n\n<p>The <code>elif</code> makes sure that you only count any factor/divisor once.</p>\n"
    # print(parse_html_block_for_code(test_string))