import json
from data.helper import *

#placeholder function to apply to each element of the dataset
placeholder = lambda x: x


preproc_fn_dict = {
    "placeholder": placeholder
}

class DataProcess:
    def __init__(self,path:str,config_path:str) -> None:
        """
        path(str) : path to the json file.
        config_path(str) : path to the config file.
        """
        self.dataset = load_json_file(path)
        self.config  = load_json_file(config_path)

    def apply_config_preproc(self)->dict:
        """
        Apply the preprocessing config to the dataset.
        """
        for key in self.config:
            if key in preproc_fn_dict.keys():
                dataset : dict = preproc_fn_dict[key](self.dataset)
            else:
                raise Exception("No such preprocessing function.")
        
        self.dataset = dataset

    def save_preproc_dataset(self)->None:
        """
        Save the preprocessed dataset.
        """
        dump_json_file(self.config["output_path"],self.dataset)



if __name__ == "__main__":
    pipeline = DataProcess("dataset/CodeReviewSE.json","configs/preproc_config.json")
    pipeline.apply_config_preproc()
    pipeline.save_preproc_dataset()