from re import T
import openai
import logging
from tqdm import tqdm
from data.codex_related import write_dict_to_json, load_json_to_dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(openai.api_version)


keys_dict = load_json_to_dict("configs/keys.json")
openai.organization = keys_dict["org"]
openai.api_key = keys_dict["key"]

trial_instruction = "Annotate type for identifiers in the python code"
trial_input = "def add(x,y):\n    return x+y"
calls = 0
for i in tqdm(range(100)):
    try:
        response = openai.Edit.create(
            model = "code-davinci-edit-001",
            input = trial_input,
            instruction = trial_instruction,
            temperature=0.7,
            top_p=1,
            )
        calls += 1
        logger.info(response["choices"][0]["text"])
    except openai.error.RateLimitError:
        logger.info(f"Rate limit error after {calls} calls.")
        break


