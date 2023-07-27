import logging
import json
from bs4 import BeautifulSoup
from tqdm import tqdm
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_json_file(file_path:str)-> dict:
    with open(file_path,"r") as f:
        return json.load(f)


class ProcessDataset:
    """
    Process Dataset for Async Augmentations
    """
    def __init__(self,dataset_path:str="dataset/CodeReviewSE_clean.json") -> None:
        self.dataset_path = dataset_path
        self.data = load_json_file(self.dataset_path)
        logger.info(f"Sucessfully loaded dataset from {dataset_path}")
        self.data_len = len(self.data) #Length of the dataset

    def parse_html_to_context(self,body:str) -> list:
        """
        Return the list of <p> tags in the given body
        """
        html_parsed = BeautifulSoup(body, 'html.parser')
        strings = []
        children = html_parsed.children
        for child in children:
            if child.name == "p":
                strings.append(child.text)
        return strings

    def process_datapoint(self,index:int=1)->dict:
        """
        Given a index, return the corresponding data entry.
        """
        processed_dict = {
            "title" : None,
            "context" : None,
            "critque" : None,
            "data_point_id" : str(index)
        }
        data_point =  self.data[str(index)]
        body = data_point["body"]
        #Title
        processed_dict["title"] = data_point["meta_data"]["Title"]

        #Context is the question of the Author
        context = " ".join(self.parse_html_to_context(body))
        processed_dict["context"] = context

        #Accepted Critique
        accepted_answer_id = data_point["meta_data"]["AcceptedAnswerId"]
        for answer in data_point["answers"]:
            if answer["meta_data"]["Id"] == accepted_answer_id:
                processed_dict["critque"] = answer["body"]
                break

        return processed_dict
        

    def iter_processing(self):
        output = {"dataset":[]}
        count = 0
        for i in tqdm(range(self.data_len)):
            try:
                count += 1
                output["dataset"].append(self.process_datapoint(i))
            except:
                pass
        logger.info(f"Successfully processed {count} data points")
        logger.info(f"Length of output: {len(output['dataset'])}")
        return output




class GeneratePrompt:
    def __init__(self,dataset_path:str) -> None:
        self.dataset = load_json_file(dataset_path)


if __name__ == "__main__":
    dataset = ProcessDataset()
    processed_dataset = dataset.iter_processing()
    with open("dataset/processed_dataset.json","w") as f:
        json.dump(processed_dataset,f,indent=2)