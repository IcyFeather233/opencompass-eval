from mmengine.config import read_base

with read_base():
    # IFEval Time
    # real    7m36.997s
    # user    15m36.631s
    # sys     3m22.041s
    # from .datasets.IFEval.IFEval_gen import ifeval_datasets
    
    # from .datasets.ceval.ceval_gen import ceval_datasets
    from .datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from .datasets.mmlu.mmlu_gen import mmlu_datasets
    # from .models.qwen.vllm_skywork import models
    from .models.qwen.vllm_qwen1_5_14b_chat2 import models
    

datasets = [*gsm8k_datasets, *mmlu_datasets]
models = [*models]

