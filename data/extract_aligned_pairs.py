from dataclasses import dataclass
from os.path import exists
from typing import Iterable, Sequence, Tuple
import pylcs
from difflib import SequenceMatcher
import numpy as np
from functools import reduce
import json
from tqdm import tqdm
import re

log_file = 'log.txt'
with open(log_file, 'w') as f:
    f.write("LOG\n\n")

def log(txt):
    with open(log_file, 'a+') as f:
        if type(txt) == dict:
            f.write(json.dumps(txt, indent=2)+'\n\n')
        elif type(txt) == list:
            for ele in txt:
                log(ele)
        elif type(txt) == Block:
            f.write(txt.text + '\n\n')
        else:
            f.write(str(txt)+'\n\n')



#data = json.load(open('CodeReviewSE_clean.json'))
data = json.load(open('temp_data.json', 'r'))
def cache_temp_data():
    with open('temp_data.json', 'w') as f:
        temp_keys = list(data.keys())[:5000]
        temp_dict = {k: data[k] for k in temp_keys}
        json.dump(temp_dict, f)
#cache_temp_data()
#exit()


#log(first)
#log(second)
#log(posts[2])
#exit()

# Compute lcs for first post
# Currently only filtering out posts with no answers. Keepin posts with no accepted answer
# TODO(dahoas): Perhaps filter out posts below certain upvote threshold
def get_accepted_answer(post):
    try:
        accepted_id = post['meta_data']['AcceptedAnswerId']
    except KeyError:
        if len(post['answers']) > 0:
            accepted_ans = reduce(lambda ans1, ans2 : ans1
                                                    if int(ans1['meta_data']['Score']) > int(ans2['meta_data']['Score'])
                                                    else ans2, post['answers'])
            return accepted_ans['body']
    #print('acc_id', accepted_id)
    for answer in post['answers']:
        if answer['meta_data']['Id'] == accepted_id:
            return answer['body']

def get_lcs(body, accepted_answer):
    s = SequenceMatcher(None, body, accepted_answer)
    blocks = s.get_matching_blocks()
    max_len_id = np.argmax([block.size for block in blocks])
    max_block = blocks[max_len_id]
    start, max_len = max_block.a, max_block.size
    return body[start : start + max_len], blocks

@dataclass
class Block:
    text : str
    start : int
    end : int
    type : str

@dataclass
class CodeBlock(Block):
    type = 'code'

@dataclass
class ReviewBlock:
    pre_blocks : Iterable[Block]
    code_block : CodeBlock
    post_blocks : Iterable[Block]

def extract_code_blocks(text : str) -> Iterable[CodeBlock]:
    codeblock_pattern = r'<code>(?s)((?!<code>).)*<\/code>'
    code_block_matches = re.finditer(codeblock_pattern, text)
    code_blocks = []
    for match in code_block_matches:
        start, end = match.span()
        code_block = text[start + 6 : end - 7]  # Want to remove <code>, </code> tags
        code_blocks.append(CodeBlock(code_block, start, end, 'code'))
    return code_blocks

# Assumes body is first text argument to SequenceMatcher
def block_to_text(body, block):
    return body[block.a : block.a + block.size]

@dataclass
class Identifier:
    text : str
    start : int
    end : int

@dataclass
class CodeblockIdentifierEncoding:
    identifiers : Iterable[str]
    positions : Iterable[Tuple[int, int]]

def code_to_identifiers(code):
    identifier_pattern = r"[a-zA-Z_][a-zA-Z0-9_]*"
    identifiers = re.finditer(identifier_pattern, code)
    identifier_list = []
    identifier_positions = []
    for identifier in identifiers:
        start = identifier.span()[0]
        end = identifier.span()[1]
        identifier_positions.append((start, end))
        identifier_list.append(code[start : end])
    return CodeblockIdentifierEncoding(identifier_list, identifier_positions)

def code_blocks_to_identifiers(code_blocks : Iterable[CodeBlock]):
    return [code_to_identifiers(code_block.text) for code_block in code_blocks]


def text_from_match(text1, text2, match):
    sub1 = text1[match.a : match.a + match.size]
    sub2 = text2[match.b : match.b + match.size]
    assert sub1 == sub2
    return sub1

# Assumes body is first text argument to SequenceMatcher
def block_to_tuple(body, answer, block, body_window=100, answer_window=200, threshold=10):
    if block.size < threshold:
        return None
    else:
        body_window = body[max(0, block.a - body_window) : block.a + block.size + body_window]
        answer_window = answer[max(0, block.b - answer_window) : block.b + block.size + answer_window]
        return (body_window, answer_window)


@dataclass
class CandidateRevs:
    pre_blocks : Iterable[Block]
    post_blocks : Iterable[Block]

# TODO(dahoas): May also want to collect data on natural text questions asked by users?
@dataclass
class AlignedTriple:
    sub_start : int
    sub_end : int
    sub_text : str  # Code being critiqued

    code_start : int
    code_end : int
    code_text : str  # Improved code

    revs : CandidateRevs

def choose_windows(match_size):
        sub_window = 1 * (match_size+5)**1.3
        rev_window = 2 * (match_size+8)**1.5
        return int(sub_window), int(rev_window)

# Finds span containing two critiques of captured codeblock
# This could break if code tag is very short reference, so should impose length threshold
def find_maximal_rev_span(rev, rev_start, rev_end):
    # Search for text above and text below
    while rev_start > 0:
        tag = rev[rev_start : rev_start + 6]
        if tag == '<code>': break
        rev_start -= 1
    while rev_end < len(rev):
        tag = rev[rev_end : rev_end + 6]
        if tag == '<code>': break
        rev_end += 1
    return rev_start, rev_end

def identifier_tuple_to_code_tuple(
                                    sub : str,
                                    sub_code_block : CodeBlock,
                                    sub_ident : CodeblockIdentifierEncoding,
                                    rev : str,
                                    rev_code_block : CodeBlock,
                                    rev_ident : CodeblockIdentifierEncoding,
                                    ident_match
                                  ) -> Tuple[str, str]:
    sub_matched_ident_start = sub_ident.positions[ident_match.a]
    sub_matched_ident_end = sub_ident.positions[ident_match.a + ident_match.size - 1]
    rev_matched_ident_start = rev_ident.positions[ident_match.b]
    rev_matched_ident_end = rev_ident.positions[ident_match.b + ident_match.size - 1]

    # Determine window sizes from relative lengths of match
    # Maybe best bet is to collect both natural language critiques that could possible apply and decide later
    match_size = ident_match.size
    sub_window, rev_window = choose_windows(match_size)

    sub_start = max(0, sub_code_block.start + sub_matched_ident_start[0] - sub_window)
    sub_end = min(len(sub), sub_code_block.start + sub_matched_ident_end[1] + sub_window)

    rev_start = rev_code_block.start + rev_matched_ident_start[0]
    rev_end = rev_code_block.start + rev_matched_ident_end[1]
    rev_start, rev_end = find_maximal_rev_span(rev, rev_start, rev_end)

    sub_match = sub[sub_start : sub_end]
    rev_match = rev[rev_start : rev_end]


    return AlignedTriple(
                            sub_start=sub_start,
                            sub_end=sub_end,
                            sub_text=sub_match,

                            code_start=rev_start,
                            code_end=rev_end,
                            code_text=rev_match,

                            rev_start=None,
                            rev_end=None,
                            rev_text=None,
                        )

######################


def exp():
    posts = list(data.values())
    post = posts[4500]
    body = post['body']
    accepted_answer = get_accepted_answer(post)


    log(body)
    log(accepted_answer)

    submitted_code_blocks : Iterable[CodeBlock] = extract_code_blocks(body)
    submitted_code_block_identifiers : Iterable[CodeblockIdentifierEncoding] = code_blocks_to_identifiers(submitted_code_blocks)

    reviewed_code_blocks : Iterable[CodeBlock] = extract_code_blocks(accepted_answer)
    reviewed_code_block_identifiers : Iterable[CodeblockIdentifierEncoding] = code_blocks_to_identifiers(reviewed_code_blocks)

    rev_index = 2
    matches = SequenceMatcher(None, submitted_code_block_identifiers[0].identifiers, reviewed_code_block_identifiers[rev_index].identifiers).get_matching_blocks()
    for match in matches:
        if match.size > 0:
            log(match)
            sub_match, rev_match = identifier_tuple_to_code_tuple(
                                                body,
                                                submitted_code_blocks[0],
                                                submitted_code_block_identifiers[0],
                                                accepted_answer,
                                                reviewed_code_blocks[rev_index],
                                                reviewed_code_block_identifiers[rev_index],
                                                match,
                                            )
            log(sub_match)
            log(rev_match)


#exp()

#print(get_lcs(body, accepted_answer))
#lcs_seq_len = pylcs.lcs_sequence_length(body, accepted_answer)
#print('lcs_seq_len', lcs_seq_len)
#lcs_idx = pylcs.lcs_sequence_idx(body, accepted_answer)
#print('lcs_idx', lcs_idx)

@dataclass
class AlignedQuadruple:
    sub_text : str 
    pre_blocks : Iterable[Block]
    mid_blocks : Iterable[Block]
    post_blocks : Iterable[Block]

def get_sub_window(size):
    sub_window = int(1 * (size)**1.3)
    return sub_window, sub_window

def count_list(lst):
    ele_counts = {}
    for ele in lst:
        if ele_counts.get(ele) is None:
            ele_counts[ele] = 1
        else:
            ele_counts[ele] += 1
    return ele_counts

def ident_sim_score(sub_ident : CodeblockIdentifierEncoding, rev_ident: CodeblockIdentifierEncoding):
    sub_count = count_list(sub_ident.identifiers)
    rev_count = count_list(rev_ident.identifiers)
    all_keys = set(sub_count.keys()) | set(rev_count.keys())
    total_len = len(sub_ident.identifiers) + len(rev_ident.identifiers)
    if total_len == 0:
        return 1
    mass = 0
    for key in all_keys:
        sub_mass = sub_count.get(key) if sub_count.get(key) is not None else 0
        rev_mass = rev_count.get(key) if rev_count.get(key) is not None else 0
        mass += np.abs(sub_mass - rev_mass)
    mass /= total_len
    return mass

def compute_sub_ident_chunks(sub_ident : CodeblockIdentifierEncoding, rev_ident: CodeblockIdentifierEncoding):
    sub_len = len(sub_ident.identifiers)
    rev_len = len(rev_ident.identifiers)
    # Chunk sub_code into overlapping blocks with overlaps of size rev_size(so we don't miss comp)
    chunks = []
    for i in range(1, (sub_len // rev_len) + 1):
        center_index = rev_len * i
        start = max(0, center_index - rev_len)
        end = center_index + rev_len
        chunk = CodeblockIdentifierEncoding(
                                            sub_ident.identifiers[start : end],
                                            sub_ident.positions[start : end]
                                           )
        #exit()
        chunks.append(chunk)
    return chunks

@dataclass
class MergedReviewBlock:
    pre_blocks : Iterable[Block]
    mid_blocks : Iterable[Block]
    post_blocks : Iterable[Block]
    sample_code_block : CodeBlock  # Always chosen to be first code block which is usually copied from submission and revised later on

MERGE_THRESH = 0.75

def merge_review_blocks(block1 : MergedReviewBlock, block2 : ReviewBlock):
    code_block1 = code_blocks_to_identifiers([block1.sample_code_block])[0]
    code_block2 = code_blocks_to_identifiers([block2.code_block])[0]
    score = ident_sim_score(code_block1, code_block2)
    if score < MERGE_THRESH:
        block1.mid_blocks += block2.pre_blocks
        block1.mid_blocks += [block2.code_block]
        block1.post_blocks = block2.post_blocks
        return block1
    else: return None

num_matches = 0
num_posts = 0
cum_match_len = 0
cum_score = 0
eval_steps = 100
statistics = {"num_samples": len(data)}
posts = list(data.values())
#posts = [posts[4500]]
saved_data = []
for i, post in tqdm(enumerate(posts)):
    body = post['body']
    try:
        accepted_answer = get_accepted_answer(post)
    except TypeError:
        continue
    if accepted_answer is None:
        continue

    num_posts += 1

    in_para = False
    start = 0
    index = 0
    blocks : Iterable[Block] = []
    while index < len(accepted_answer):
        open_para_tag = accepted_answer[index : index + 3]
        close_para_tag = accepted_answer[index : index + 4]
        open_code_tag = accepted_answer[index : index + 6]
        closed_code_tag = accepted_answer[index : index + 7]
        if open_para_tag == '<p>':
            start = index
            in_para = True
        elif close_para_tag == '</p>':
            in_para = False
            index = index + 4
            para = accepted_answer[start + 3 : index - 4]
            blocks.append(Block(start=start, end=index, text=para, type='para'))
        elif not in_para and open_code_tag == '<code>':
            start = index
        elif not in_para and closed_code_tag == '</code>':
            index = index + 7
            code = accepted_answer[start + 6 : index - 7]
            blocks.append(Block(start=start, end=index, text=code, type='code'))
        index += 1

    review_blocks : Iterable[ReviewBlock] = []
    block_type_list = [(i, block.type) for i, block in enumerate(blocks)]
    start = 0
    pre_review_block = ReviewBlock([], None, [])
    post_review_block = None
    for block in blocks:
        if block.type == 'code':
            pre_review_block.code_block = block

            # current post_review_block is done
            if post_review_block is not None: review_blocks.append(post_review_block)

            # pre_review_block becomes post_review_block
            post_review_block = pre_review_block

            # make new pre_review_block
            pre_review_block = ReviewBlock([], None, [])

        else:
            # Always add current para block to the start of the pre_review_block
            pre_review_block.pre_blocks.append(block)

            # Add currrent para block to end of post review block if active
            if post_review_block is not None: post_review_block.post_blocks.append(block)

    #Append last post_review_block
    review_blocks.append(post_review_block)
    review_blocks = [review_block for review_block in review_blocks if review_block is not None and len(review_block.code_block.text) > 0]

    # Preprocess to merge similar code review blocks
    merged_review_blocks = []
    cur_merged_review_block = None
    for i in range(len(review_blocks)):
        cur_review_block = review_blocks[i]
        if i == 0:
            cur_merged_review_block = MergedReviewBlock(
                                                        cur_review_block.pre_blocks, 
                                                        [cur_review_block.code_block], 
                                                        cur_review_block.post_blocks, 
                                                        cur_review_block.code_block
                                                       )
        else:
            new_merged_review_block = merge_review_blocks(cur_merged_review_block, cur_review_block)
            if new_merged_review_block is None:
                merged_review_blocks.append(cur_merged_review_block)
                cur_merged_review_block = MergedReviewBlock(
                                                            cur_review_block.pre_blocks, 
                                                            [cur_review_block.code_block], 
                                                            cur_review_block.post_blocks, 
                                                            cur_review_block.code_block
                                                           )
            else:
                cur_merged_review_block = new_merged_review_block
    # Append last MergedReviewBlock
    merged_review_blocks.append(cur_merged_review_block)
    merged_review_blocks = [review_block for review_block in merged_review_blocks if review_block is not None]

    submitted_code_blocks : Iterable[CodeBlock] = extract_code_blocks(body)
    submitted_code_block_identifiers : Iterable[CodeblockIdentifierEncoding] = code_blocks_to_identifiers(submitted_code_blocks)

    for review_block in merged_review_blocks:
        rev_block = review_block.sample_code_block
        rev_ident : CodeblockIdentifierEncoding = code_blocks_to_identifiers([rev_block])[0]
        if len(rev_ident.identifiers) < 1:
            continue
        for sub_block, sub_ident in zip(submitted_code_blocks, submitted_code_block_identifiers):
            # Probably need a better matching mechanism than SequenceMatcher: need not be contiguous in reality
            sub_ident_chunks = compute_sub_ident_chunks(sub_ident, rev_ident)
            min_score = 1
            min_chunk = None
            for chunk in sub_ident_chunks:
                score = ident_sim_score(chunk, rev_ident)
                if score < min_score:
                    min_score = score
                    min_chunk = chunk

            THRESH = 0.5
            
            if min_chunk is not None and min_score < THRESH:
                num_matches += 1
                cum_score += min_score
            
                sub_start = min_chunk.positions[0][0]
                sub_end = min_chunk.positions[-1][-1]

                sub_text = sub_block.text[sub_start : sub_end]

                al = AlignedQuadruple(
                    sub_text=sub_text,
                    pre_blocks=review_block.pre_blocks,
                    mid_blocks=review_block.mid_blocks,
                    post_blocks=review_block.post_blocks,
                )
            
                al_dict = {
                    'sub_text': al.sub_text,
                    'pre_blocks': [pre_block.text for pre_block in al.pre_blocks],
                    'mid_blocks': [mid_block.text for mid_block in al.mid_blocks],
                    'post_blocks': [post_block.text for post_block in al.post_blocks]
                }

                saved_data.append(al_dict)

                if num_matches % eval_steps == 0:
                    log("SUBMITTED")
                    log(al.sub_text)
                    log("PRE")
                    log(al.pre_blocks)
                    log("MID")
                    log(al.mid_blocks)
                    log("POST")
                    log(al.post_blocks)

import json
with open("aligned_data.json", 'w') as f:
    json.dump(saved_data, f)


statistics['samples_with_answers'] = num_posts
statistics['avg_num_matches'] = num_matches / statistics['samples_with_answers']
statistics['avg_match_len'] = cum_match_len / num_matches
statistics['num_matches'] = num_matches
statistics['avg_match_score'] = cum_score / num_matches
print(json.dumps(statistics, indent=2))



'''matches = SequenceMatcher(None, sub_ident.identifiers, rev_ident.identifiers).get_matching_blocks()
            max_len_id = np.argmax([match.size for match in matches])
            max_match = matches[max_len_id]

            THRESH = 5

            if max_match.size > THRESH:
                num_matches += 1
                cum_match_len += max_match.size
            
                sub_matched_ident_start = sub_ident.positions[max_match.a]
                sub_matched_ident_end = sub_ident.positions[max_match.a + max_match.size - 1]
                sub_start = sub_matched_ident_start[0]
                sub_end = sub_matched_ident_end[1]
                pre_window, post_window = get_sub_window(sub_end - sub_start)
                sub_start = max(0, sub_start - pre_window)
                sub_end = sub_end + post_window

                sub_text = sub_block.text[sub_start : sub_end]

                al = AlignedQuadruple(
                    sub_text=sub_text,
                    code_text=rev_block,
                    pre_blocks=review_block.pre_blocks,
                    post_blocks=review_block.post_blocks,
                )

                if num_matches % eval_steps == 0:
                    log("SUBMITTED")
                    log(al.sub_text)
                    log("PRE")
                    log(al.pre_blocks)
                    log("CODE")
                    log(al.code_text)
                    log("POST")
                    log(al.post_blocks)'''


'''lcs, blocks = get_lcs(body, accepted_answer)
    if i == 1493:
        log(post)
        for block in blocks:
            tuple_t = block_to_tuple(body, accepted_answer, block)
            if tuple_t is not None:
                log("Tuple")
                log("BODY\n" + tuple_t[0])
                log("ANSWER\n" + tuple_t[1])
        exit()
    num_matches = len(blocks)
    if num_matches > statistics['max_matches_in_post']:
        statistics['max_matches_in_post'] = num_matches
        statistics['max_matches_id'] = i
    statistics['avg_lcs_len'] += len(lcs)
    #print(lcs)'''

'''def extract_blocks(rev : str, pattern : str, tag : str) -> Iterable[Block]:
    block_matches = re.finditer(pattern, rev)
    blocks = []
    for match in block_matches:
        start, end = match.span()
        code_block = rev[start + len(tag) : end - (len(tag) + 1)]  # Want to remove <code>, </code> tags
        blocks.append(Block(code_block, start, end))
    return blocks


# Chunks review into code and paragraph blocks
def rev_to_pcblocks(rev):
    codeblock_pattern = r'<code>(?s)((?!<code>).)*<\/code>'
    para_pattern = r'<p>(?s)((?!<p>).)*<\/p>'

    code_blocks : Iterable[Block] = extract_blocks(rev, codeblock_pattern)
    para_blocks : Iterable[Block] = extract_blocks(rev, para_pattern)

    merged_blocks = []
    i,j = 0, 0
    while i < len(code_blocks) and j < len(para_blocks):
        code_block = code_blocks[i]
        para_block = para_block[]'''