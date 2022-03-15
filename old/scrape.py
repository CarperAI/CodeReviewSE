from stackapi import StackAPI
import json

site = StackAPI("codereview")
#questions = site.fetch("questions")

# writes questions to a json file, saved at questions.json
def write_questions(questions):
    with open("questions.json", "w") as f:
        json.dump(questions, f)

#write_questions(questions)
#print("Questions have been saved!")


# loads questions from a json file named questions.json
def load_questions():
    with open("questions.json", "r") as f:
        questions = json.load(f)
    return questions

questions = load_questions()

#print(questions)
#print(list(questions['items'][0].keys()))
question_id = questions['items'][10]["question_id"]

def get_answers(site, question_id):
    answers = site.fetch(f"questions/{question_id}/answers", filter="withbody")
    return answers
# utilizes StackAPI to, given a question_id, fetch the body of all associated answers
def fetch_answers_given_question_id(site, question_id):
    answers = get_answers(site, question_id)
    answers_list = answers['items']
    answers_body = []
    for answer in answers_list:
        answers_body.append(answer['body'])
    return answers_body

def fetch_comments_given_id(site, id):
    comments = site.fetch(f"posts/{question_id}/comments", filter='withbody')
    comments_list = comments['items']
    comments_body = []
    for comment in comments_list:
        comments_body.append(comment['body'])
    return comments_body

#print(get_answers(site, question_id)['items'][0].keys())

#answer = get_answers(site, question_id)['items'][0]

print()

answer_body = fetch_answers_given_question_id(site, question_id)
print(answer_body)

"""
question:
        question_meta data (user ID, upvotes, accepted answer, date posted, tags)
        question_comments
            question_comments_meta_data (user ID, upvotes, date posted)
            question_comment_body
        question_body

        question_answers
            question_answer_meta_data (user ID, upvotes, is_accepted, date posted)
                question
            question_answer_body
"""