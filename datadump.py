import json
from collections import defaultdict
from tqdm import tqdm
import multiprocessing

# load posts
with open("dataset/Posts.json") as json_file:
    posts = json.load(json_file)

# load comments
with open("dataset/Comments.json") as json_file:
    comments = json.load(json_file)

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

def iterate_over_posts(file):
    """
    Constructs a dictionary of the format above 
    TODO: (excluding comments, which we will add later)
    """
    output_dictionary = {}
    answers = defaultdict(list)

    for post in file['posts']['row']:
        post_dict = {}
        Id = post['@Id']
        meta_data = [\
            ('Id', Id),
            ('Score', post['@Score']),
            ('CreationDate', post['@CreationDate']),
            ('CommentCount', post['@CommentCount']),
            ('ContentLicense', post['@ContentLicense'])
            ]
        post_dict['body'] =  post['@Body']
        post_dict['comments'] = list()

        # is an answer
        if post['@PostTypeId'] == '2':
            answer_specific_meta_data = [\
                ('ParentId', post['@ParentId'])
                ]
            meta_data += answer_specific_meta_data

        # is a question
        else:
            try:
                tags = post['@Tags']
                tags = tags.split('><')
                tags = [tag.replace('<','').replace('>','') for tag in tags]
                question_specific_meta_data = [
                   ('Tags', tags),
                ]
            except:
                question_specific_meta_data = list()
            # if the post has a title
            if '@Title' in post.keys():
                question_specific_meta_data += [\
                    ('Title', post['@Title'])
                ] 
            # is there an accepted answer
            if '@AcceptedAnswerId' in post.keys():
                question_specific_meta_data += [\
                    ('AcceptedAnswerId', post['@AcceptedAnswerId'])
                ] 
            meta_data += question_specific_meta_data
            
            post_dict['answers'] = list()

        # copy tuples into dictionary
        meta_data_dict = {}
        for key,value in meta_data:
            meta_data_dict[key] = value

        post_dict['meta_data'] = meta_data_dict

        # if this is an answer
        if post['@PostTypeId'] == '2':
            #add it to the corresponding question
            parent_id = post['@ParentId']
            answers[parent_id].append(post_dict)
        # if it is a question
        else:
            output_dictionary[Id] = post_dict
    
    # add answers
    for k,v in answers.items():
        output_dictionary[k]['answers'] = v

    return output_dictionary


def iterate_over_comments(file, post_dict):
    comments = defaultdict(list)
    for comment in file['comments']['row']:
        comment_output_dict = {}
        parent_id = comment['@PostId']
        comment_output_dict['body'] = comment['@Text']
        meta_data = [\
            ('Id', comment['@Id']),
            ('Score', comment['@Score']),
            ('CreationDate', comment['@CreationDate']),
            ('ContentLicense', comment['@ContentLicense'])
            ]
        for k,v in meta_data:
            comment_output_dict[k] = v
        comments[parent_id].append(comment_output_dict)
    pool_obj = multiprocessing.Pool()

    for k,v in tqdm(comments.items()):
        try:
            post_dict[k]['comments'] = v
        except:
            # it isnt for a question, must be for a answer. 
            # Iterate over all questions and all answers using multithreading
            for question_id, question_dict in post_dict.items():
                 for idx, answer in enumerate(question_dict['answers']):
                     if answer['meta_data']['Id'] == k:
                         post_dict[question_id]['answers'][idx]['comments'] = v
                         break

    return post_dict


post_dict = iterate_over_posts(posts)
post_dict = iterate_over_comments(comments, post_dict)

# Save dictionary to json file named CodeReviewSE.json
with open('dataset/CodeReviewSE.json', 'w') as outfile:
    json.dump(post_dict, outfile)

            

            
            
