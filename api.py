import torch
from experiments.main import my_start

def chatglm2_atttck(model="chatglm2",
                    file_path=None,
                    control_init="kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk",
                    n_train_data=10,
                    n_steps=200,
                    test_steps=50,
                    stop_on_success=True,
                    allow_non_ascii=False,
                    result_path="./output"):
    my_start(model=model,
                    file_path=file_path,
                    control_init=control_init,
                    n_train_data=n_train_data,
                    n_steps=n_steps,
                    test_steps=test_steps,
                    stop_on_success=stop_on_success,
                    allow_non_ascii=allow_non_ascii,
                    result_path=result_path)
    





def chatglm2_test():
    pass


if __name__ == '__main__':
    chatglm2_atttck(file_path='./data/advbench/harmful_behaviors_zh.csv')