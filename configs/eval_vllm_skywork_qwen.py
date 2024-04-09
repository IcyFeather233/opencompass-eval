from mmengine.config import read_base

with read_base():
    # IFEval Time
    # real    7m36.997s
    # user    15m36.631s
    # sys     3m22.041s
    # from .datasets.IFEval.IFEval_gen import ifeval_datasets
    
    # CEval Time
    # real    9m57.007s
    # user    17m3.051s
    # sys     3m20.938s
    # from .datasets.ceval.ceval_gen import ceval_datasets
    
    # GSM8K
    # real    10m33.293s
    # user    18m2.768s
    # sys     4m1.202s
    from .datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    
    # MMLU
    # real    22m46.415s
    # user    27m28.503s
    # sys     6m58.145s
    # from .datasets.mmlu.mmlu_gen import mmlu_datasets
    # from .models.qwen.vllm_skywork import models
    from .models.qwen.vllm_qwen1_5_14b_chat import models
    

datasets = [*gsm8k_datasets]
models = [*models]
