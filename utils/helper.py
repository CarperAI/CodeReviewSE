import json
from pprint import pprint
from html.parser import HTMLParser
import htmlement
from pipeline import load_json_file
    

parse_body = lambda body: htmlement.fromstring(body)

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
    dataset = dataset[list(dataset.keys())[0]]
    pprint(dataset.keys())
    pprint(dataset['meta_data'])
    pprint(get_accepted_answer(dataset))
