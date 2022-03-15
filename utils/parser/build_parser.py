import subprocess
from tree_sitter import Language, Parser
import logging

logging.basicConfig(level=logging.INFO)

def build_parser_in_tmp_dir(lang:str):
    subprocess.call("mkdir tmp", shell=True)
    subprocess.call("cd tmp", shell=True)
    subprocess.call(f"git clone  https://github.com/tree-sitter/tree-sitter-{lang} tmp/tree-sitter-{lang}", shell=True)
    Language.build_library('build/my-languages.so',[f'./tmp/tree-sitter-{lang}'])
    logging.info("Successfully built the parser")
    subprocess.call("rm -rf tmp/", shell=True)



def load_parser(lang:str):
    """
    Function to load a parser given it's language identifier.
    """
    language = Language(f'./build/my-languages.so',lang)
    parser = Parser()
    parser.set_language(language)
    return parser


def check_parseability(parser:Parser, code:str):
    """
    Function to check if a code snippet is parseable by a given parser.
    returns True if the code string is parsable, False otherwise.
    """
    tree = parser.parse(bytes(code,"utf-8"))
    if tree.root_node.children[0].type == "ERROR":
        return False
    else:
        return True



if __name__ == "__main__":
    print(load_json_file("utils/parser/lang.json"))
    build_parser_in_tmp_dir("javascript")
    # parser = load_parser("javascript")
    # print(check_parseability(parser,"var 1 = 1;"))