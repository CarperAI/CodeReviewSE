import tokenize
import keyword
from io import BytesIO

keyword_list = keyword.kwlist

SET_DELIM = " @#@ " # Delimiliter for code blocks

def tokenize_code_snippet(code_snippet:str,exclude_keywords=True):
    """
    Tokenize a code snippet
    """
    try:
        tokens = tokenize.tokenize(BytesIO(code_snippet.encode("utf-8")).readline)
        if exclude_keywords:
            tokenized = [token.string for token in tokens if token.string not in keyword_list or len(token.string) != 0][1:-2]
        else:
            tokenized = [token.string for token in tokens if len(token.string) != 0][1:-2]
    except tokenize.TokenError:
        tokenized = None
    return tokenized

def jaccard_similarity(code_snippet_1:str,code_snippet_2:str):
    """
    Computes Jaccard Similarity between two list of tokenized strings.
    """
    intersection = len(list(set(code_snippet_1).intersection(code_snippet_2)))
    union = (len(set(code_snippet_1)) + len(set(code_snippet_2))) - intersection
    return float(intersection) / union


def merge_or_ignore(code_block_list:list[str],similarity_threshold:float)->list[str]:
    """
    Merge code blocks if they are similar.
    args:
        code_block_list (list[str]): list of code blocks
        similarity_threshold (float): threshold for similarity
    returns:
        merged_frozen_set (list[str]): list of merged code blocks
        merge_list (list[list[ind]]) : coo matrix to map the merged code blocks to the original code blocks
    """
    merged_code_blocks = []
    skip_ind = []
    merge_list = []
    for code_block_ind_1 in range(len(code_block_list)):
        for code_block_ind_2 in range(code_block_ind_1,len(code_block_list)):
            tok_code_block_1 = tokenize_code_snippet(code_block_list[code_block_ind_1],exclude_keywords=False)
            tok_code_block_2 = tokenize_code_snippet(code_block_list[code_block_ind_2],exclude_keywords=False)
            if tok_code_block_1 != tok_code_block_2 and code_block_ind_1 not in skip_ind and code_block_ind_2 not in skip_ind:
                if tok_code_block_1 != None and tok_code_block_2 != None:
                    sim = jaccard_similarity(tok_code_block_1,tok_code_block_2)
                    if sim < similarity_threshold:
                        skip_ind.append(code_block_ind_2)
                        skip_ind.append(code_block_ind_1)
                        merge_list.append([code_block_ind_1,code_block_ind_2])
                        merged_code_blocks.append(SET_DELIM.join(sorted([code_block_list[code_block_ind_1],code_block_list[code_block_ind_2]])))
                    else:
                        merged_code_blocks.append(code_block_list[code_block_ind_1])
                        merged_code_blocks.append(code_block_list[code_block_ind_2])
                else:
                    #If one of the pair is unparsable, ignore the merge.
                    merged_code_blocks.append(code_block_list[code_block_ind_1])
                    merged_code_blocks.append(code_block_list[code_block_ind_2])
    #Check for missed ind blocks
    for ind in range(len(code_block_list)):
        if ind not in skip_ind:
            merged_code_blocks.append(code_block_list[ind])
    return frozenset(merged_code_blocks),merge_list
                

if __name__ == "__main__":
    code_blocks_hash = ["make_data = lambda x: x","dataset = data.append(_)","make_data = lambda x: x-1","dataset = None"]
    # print([tokenize_code_snippet(i) for i in code_blocks_hash])
    # print(jaccard_similarity(tokenize_code_snippet(code_blocks_hash[0]),tokenize_code_snippet(code_blocks_hash[1])))
    print(merge_or_ignore(code_blocks_hash,0.1))