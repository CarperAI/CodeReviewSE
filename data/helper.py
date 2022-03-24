import json
from pprint import pprint
from pipeline import load_json_file
from bs4 import BeautifulSoup
    

parse_body = lambda x: x



def parse_body_to_return(body:str,flag="code")->list[str]:
    """
    Parse the body of the content to return code
    """
    if flag == "code":
        html_parsed = BeautifulSoup(body, 'html.parser')
        code_blocks = html_parsed.find_all("pre")
        return code_blocks
    else:
        raise NotImplementedError

def get_accepted_answer(code_review_data:dict):
    """
    Provides the accepted answer of a question, if an accepted answer is available.
    """
    accepted_answer_body = None
    if "AcceptedAnswerId" in code_review_data["meta_data"].keys():
        accepted_answer_index = int(code_review_data["meta_data"]["AcceptedAnswerId"])
        for code_review_answer in code_review_data["answers"]:
            if int(code_review_answer["meta_data"]["Id"]) == accepted_answer_index:
                accepted_answer_body = code_review_answer["body"]
                return parse_body(accepted_answer_body)
    


if __name__ == "__main__":
    dataset = load_json_file("dataset/CodeReviewSE.json")
    dataset = dataset[list(dataset.keys())[1000]]
    # print(dataset["body"])
    # print("#######")
    parse_body_to_return(dataset["body"])
    #pprint(dataset.keys())
    #pprint(dataset['meta_data'])
    #pprint(get_accepted_answer(dataset))
