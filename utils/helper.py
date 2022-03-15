import json
from pprint import pprint
from html.parser import HTMLParser
import htmlement
from utils.pipeline import load_json_file
    

parse_body = lambda body: htmlement.fromstring(body)

def parse_single_code_review(code_review_data:dict):
    accepted_answer_body = None
    if "AcceptedAnswerId" in code_review_data["meta_data"].keys():
        accepted_answer_index = int(code_review_data["meta_data"]["AcceptedAnswerId"])
        for code_review_answer in code_review_data["answers"]:
            if int(code_review_answer["meta_data"]["Id"]) == accepted_answer_index:
                accepted_answer_body = code_review_answer["body"]
                yield parse_body(accepted_answer_body)
    

if __name__ == "__main__":
    dataset = load_json_file("dataset/CodeReviewSE_new.json")
    dataset = dataset[list(dataset.keys())[-1]]
    pprint(dataset.keys())
    pprint(dataset['meta_data'])
    parse_single_code_review(dataset)
