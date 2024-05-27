# import os

# os.sys.path.append("..")
from experiments.configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.result_prefix = 'results/individual_chatglm2'

    config.tokenizer_paths=["../../chatglm2"]
    config.model_paths=["../../chatglm2"]
    config.conversation_templates=['chatglm2']
    config.devices = ["cuda:7"]

    return config