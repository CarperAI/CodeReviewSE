from data.helper import *
from tqdm import tqdm


def remove_empty_questions(data):
    """
    Remove questions with no body
    """
    for k,v in tqdm(data.copy().items()):
        if v['body'] == '':
            del data[k]
    return data

def remove_questions_with_space(data):
    """
    Remove question that has body with a space at the end. This usually indicates a tag.
    """
    for question in tqdm(list(data.copy().keys())):
        body = data[question]["body"]
        body_strings = parse_html_to_str(body)
        if len(body_strings) > 0:
            if len(body_strings[-1]) > 0:
                if body_strings[-1][-1] == ' ':
                    del data[question]
            else:
                del data[question]
    
    return data

if __name__ == "__main__":
    data = load_json_file("dataset/CodeReviewSE.json")
    data = remove_empty_questions(data)
    data = remove_questions_with_space(data)
    dump_json_file("dataset/CodeReviewSE_clean.json", data)