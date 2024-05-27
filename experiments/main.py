'''A main script to run attack for LLMs.'''
import time
import importlib
import numpy as np
import torch.multiprocessing as mp
from absl import app,flags
from ml_collections import config_flags

from llm_attacks import get_goals_and_targets, get_workers
from experiments.configs.template import get_config as default_config

_CONFIG=config_flags.DEFINE_config_file('config')
def get_config():
    
    config = default_config()

    config.result_prefix = 'results/individual_chatglm2'

    config.tokenizer_paths=["./chatglm2"]
    config.model_paths=["./chatglm2"]
    config.conversation_templates=['chatglm2']
    config.devices = ["cuda:7"]

    return config




# Function to import module at the runtime
def dynamic_import(module):
    return importlib.import_module(module)

def main(_):

    mp.set_start_method('spawn')

    params = _CONFIG.value

    attack_lib = dynamic_import(f'llm_attacks.{params.attack}')

    print(params)

    train_goals, train_targets, test_goals, test_targets = get_goals_and_targets(params)

    process_fn = lambda s: s.replace('Sure, h', 'H')
    process_fn2 = lambda s: s.replace("Sure, here is", "Sure, here's")
    train_targets = [process_fn(t) if np.random.random() < 0.5 else process_fn2(t) for t in train_targets]
    test_targets = [process_fn(t) if np.random.random() < 0.5 else process_fn2(t) for t in test_targets]

    workers, test_workers = get_workers(params)

    managers = {
        "AP": attack_lib.AttackPrompt,
        "PM": attack_lib.PromptManager,
        "MPA": attack_lib.MultiPromptAttack,
    }

    timestamp = time.strftime("%Y%m%d-%H:%M:%S")
    if params.transfer:
        attack = attack_lib.ProgressiveMultiPromptAttack(
            train_goals,
            train_targets,
            workers,
            progressive_models=params.progressive_models,
            progressive_goals=params.progressive_goals,
            control_init=params.control_init,
            logfile=f"{params.result_prefix}_{timestamp}.json",
            managers=managers,
            test_goals=test_goals,
            test_targets=test_targets,
            test_workers=test_workers,
            mpa_deterministic=params.gbda_deterministic,
            mpa_lr=params.lr,
            mpa_batch_size=params.batch_size,
            mpa_n_steps=params.n_steps,
        )
    else:
        attack = attack_lib.IndividualPromptAttack(
            train_goals,
            train_targets,
            workers,
            control_init=params.control_init,
            logfile=f"{params.result_prefix}_{timestamp}.json",
            managers=managers,
            test_goals=getattr(params, 'test_goals', []),
            test_targets=getattr(params, 'test_targets', []),
            test_workers=test_workers,
            mpa_deterministic=params.gbda_deterministic,
            mpa_lr=params.lr,
            mpa_batch_size=params.batch_size,
            mpa_n_steps=params.n_steps,
        )
    attack.run(
        n_steps=params.n_steps,
        batch_size=params.batch_size, 
        topk=params.topk,
        temp=params.temp,
        target_weight=params.target_weight,
        control_weight=params.control_weight,
        test_steps=getattr(params, 'test_steps', 1),
        anneal=params.anneal,
        incr_control=params.incr_control,
        stop_on_success=params.stop_on_success,
        verbose=params.verbose,
        filter_cand=params.filter_cand,
        allow_non_ascii=params.allow_non_ascii,
    )

    for worker in workers + test_workers:
        worker.stop()




def my_start(model="chatglm2",
                    file_path=None,
                    control_init="kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk",
                    n_train_data=10,
                    n_steps=200,
                    test_steps=50,
                    stop_on_success=True,
                    allow_non_ascii=False,
                    result_path="./output.json"):
    # FLAGS = flags.FLAGS
    # config_flags.DEFINE_config_file('config_file',config)
    # _CONFIG = config_flags.DEFINE_config_file('config',"./configs/individual_chatglm2.py")
    mp.set_start_method('spawn')

    params = get_config()
    # print(params)

    params.attack='gcg'
    params.train_data=file_path
    params.result_prefix=result_path
    params.n_train_data=n_train_data
    params.data_offset=1
    params.n_steps=n_steps
    params.test_steps=test_steps
    params.stop_on_success=stop_on_success
    params.allow_non_ascii=allow_non_ascii
    params.control_init=control_init


    attack_lib = dynamic_import(f'llm_attacks.{params.attack}')

    print(params)

    train_goals, train_targets, test_goals, test_targets = get_goals_and_targets(params)

    process_fn = lambda s: s.replace('Sure, h', 'H')
    process_fn2 = lambda s: s.replace("Sure, here is", "Sure, here's")
    train_targets = [process_fn(t) if np.random.random() < 0.5 else process_fn2(t) for t in train_targets]
    test_targets = [process_fn(t) if np.random.random() < 0.5 else process_fn2(t) for t in test_targets]

    workers, test_workers = get_workers(params)

    managers = {
        "AP": attack_lib.AttackPrompt,
        "PM": attack_lib.PromptManager,
        "MPA": attack_lib.MultiPromptAttack,
    }

    timestamp = time.strftime("%Y%m%d-%H:%M:%S")
    if params.transfer:
        attack = attack_lib.ProgressiveMultiPromptAttack(
            train_goals,
            train_targets,
            workers,
            progressive_models=params.progressive_models,
            progressive_goals=params.progressive_goals,
            control_init=params.control_init,
            logfile=f"{params.result_prefix}_{timestamp}.json",
            managers=managers,
            test_goals=test_goals,
            test_targets=test_targets,
            test_workers=test_workers,
            mpa_deterministic=params.gbda_deterministic,
            mpa_lr=params.lr,
            mpa_batch_size=params.batch_size,
            mpa_n_steps=params.n_steps,
        )
    else:
        attack = attack_lib.IndividualPromptAttack(
            train_goals,
            train_targets,
            workers,
            control_init=params.control_init,
            logfile=f"{params.result_prefix}_{timestamp}.json",
            managers=managers,
            test_goals=getattr(params, 'test_goals', []),
            test_targets=getattr(params, 'test_targets', []),
            test_workers=test_workers,
            mpa_deterministic=params.gbda_deterministic,
            mpa_lr=params.lr,
            mpa_batch_size=params.batch_size,
            mpa_n_steps=params.n_steps,
        )
    attack.run(
        n_steps=params.n_steps,
        batch_size=params.batch_size, 
        topk=params.topk,
        temp=params.temp,
        target_weight=params.target_weight,
        control_weight=params.control_weight,
        test_steps=getattr(params, 'test_steps', 1),
        anneal=params.anneal,
        incr_control=params.incr_control,
        stop_on_success=params.stop_on_success,
        verbose=params.verbose,
        filter_cand=params.filter_cand,
        allow_non_ascii=params.allow_non_ascii,
    )

    for worker in workers + test_workers:
        worker.stop()







if __name__ == '__main__':
    app.run(main)