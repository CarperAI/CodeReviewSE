import sys
import string

def get_indentation(line: str) -> str:
    return line[:len(line) - len(line.lstrip())]

def inject_locals_call(offset):
    return offset+'print(post_process_output(locals()))\n'

def get_post_processing_function():
    return """
def post_process_output(local_context):
    keys_to_ignore = [
        '__name__',
        '__doc__',
        '__package__',
        '__loader__',
        '__spec__',
        '__annotations__',
        '__builtins__',
        '__file__',
        '__cached__'
    ]
    return {
        k:v for k,v in local_context.items()
        if k not in keys_to_ignore
    }

"""

def inject(file_name):
    post_processing_function = get_post_processing_function()
    new_python_code = [line+"\n" for line in post_processing_function.split('\n')]
    with open(file_name, "r") as f:
        python_code = f.readlines()
        for line in python_code:
            if line.lstrip() and not (
                    line.lstrip().startswith("def") or line.lstrip().startswith("class")
                    or line.lstrip().startswith("elif") or line.lstrip().startswith("else")
            ):
                offset = get_indentation(line)
                new_line = inject_locals_call(offset)
                new_python_code.append(new_line)            
            new_python_code.append(line)
    new_python_code.append("post_process_output(locals())\n")

    new_file_name = file_name.split(".")[0] + "_injected.py"
    with open(new_file_name, "w") as f:
        new_code = "".join(new_python_code)
        f.write(new_code)
        

if __name__ == '__main__':
    file_name = sys.argv[1]
    inject(file_name)
