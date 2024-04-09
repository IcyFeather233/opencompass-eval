# Evaluator

安全效果评估代码库

使用 [OpenCompass](https://github.com/open-compass/opencompass/tree/main) 项目进行模型评估

## 环境安装

每次更新 opencompass 内容后都要重新运行 `pip install -e .` 来安装环境

## 项目代码位置

项目有关代码均在 `/maindata/data/shared/Security-SFT/cmz/opencompass` 目录下

这个是 [OpenCompass 中文教程](https://opencompass.readthedocs.io/zh-cn/latest/index.html)

## 额外引入项目目录的部分

这里记录除了直接 `git clone` 下来的代码之外，额外被加入项目目录的部分

在数据集方面，项目目录中还收录了 `OpenCompassData-complete-20240207.zip` 和 `OpenCompassData-complete-20240325.zip` 的数据集，已经解压到 `data` 目录中。如后续有新的评测数据集需要加入，请参考教程中的[数据集准备](https://opencompass.readthedocs.io/zh-cn/latest/get_started/installation.html#id2)部分

在 tokenizer 方面，另外项目中的 `nltk_data` 目录中存放了 `punkt` 有关数据

## 模型配置

OpenCompass 项目有三种配置文件：
- 运行配置：在 `configs/` 下
- 模型配置：在 `configs/models/` 下
- 数据集配置：在 `configs/datasets/` 下

其中运行配置是运行时直接指定的配置文件，模型配置和数据集配置都是被运行配置所引用的文件配置

一个典型的例子是官方仓库中的 [eval_demo.py](https://github.com/open-compass/opencompass/blob/main/configs/eval_demo.py) 文件，可以看到其中的 `datasets` 和 `models` 字段都是直接引入的模型配置文件和数据集配置文件，可以跳转过去阅读对应的配置

## Quick Start

以在 DSW 中启动一个单机八卡服务为例，镜像选择 `megatron:v1.4` 的版本

进入服务器之后，运行下面的代码进入环境：

```shell
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/maindata/data/shared/Security-SFT/common_tools/mambaforge/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/maindata/data/shared/Security-SFT/common_tools/mambaforge/etc/profile.d/conda.sh" ]; then
        . "/maindata/data/shared/Security-SFT/common_tools/mambaforge/etc/profile.d/conda.sh"
    else
        export PATH="/maindata/data/shared/Security-SFT/common_tools/mambaforge/bin:$PATH"
    fi
fi
unset __conda_setup

if [ -f "/maindata/data/shared/Security-SFT/common_tools/mambaforge/etc/profile.d/mamba.sh" ]; then
    . "/maindata/data/shared/Security-SFT/common_tools/mambaforge/etc/profile.d/mamba.sh"
fi
# <<< conda initialize <<<
#激活环境
mamba activate opencompass

export LD_LIBRARY_PATH=/usr/local/lib/python${PY_VERSION}/dist-packages/torch/lib:/usr/local/lib/python${PY_VERSION}/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
```

运行 `cd /maindata/data/shared/Security-SFT/cmz/opencompass` 进入项目目录下

通过 `python run.py configs/eval_demo.py -w outputs/demo --debug` 来测试运行环境

## 进阶

### 评测 Qwen14b-Chat 模型

在这里以 Qwen14b-Chat 模型进行 CEval 的评测为例，讲解如何通过改变配置文件，选取想要的模型以及评测数据集进行测试

参考官方的 [vllm_qwen1_5_14b_chat.py](https://github.com/open-compass/opencompass/blob/main/configs/models/qwen/vllm_qwen1_5_14b_chat.py) 示例，编写 `configs/models/qwen/vllm_qwen1_5_14b_chat.py` 如下:

```python
from opencompass.models import VLLM


_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='<|im_start|>user\n', end='<|im_end|>\n'),
        dict(role="BOT", begin="<|im_start|>assistant\n", end='<|im_end|>\n', generate=True),
    ],
    eos_token_id=151645,
)

GPU_NUMS = 8

models = [
    dict(
        type=VLLM,
        # 给这个模型取个别名
        abbr='merge_base_firefly_qw14b_base_skycommon_safe_lora_240404_hf',
        # 模型地址
        path="/maindata/data/shared/Security-SFT/dehao.li/workspace_shared/llama_factory_local/other_project/Firefly/checkpoint/merge_base_firefly_qw14b_base_skycommon_safe_lora_240404_hf",
        model_kwargs=dict(tensor_parallel_size=GPU_NUMS),
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=GPU_NUMS * 8,
        generation_kwargs=dict(temperature=0),
        end_str='<|im_end|>',
        run_cfg=dict(num_gpus=GPU_NUMS, num_procs=1),
    )
]
```

然后引入刚刚写好的模型配置以及 CEval 数据集配置，编写 `configs/eval_vllm_skywork_qwen.py` 如下:

```python
from mmengine.config import read_base

with read_base():
    from .datasets.ceval.ceval_gen import ceval_datasets
    from .models.qwen.vllm_qwen1_5_14b_chat import models
    

datasets = [*ceval_datasets]
models = [*models]
```

最后运行 `python run.py configs/eval_vllm_skywork_qwen.py -w outputs/skywork --debug` 进行评测



## 常用评测数据集

常用的评测数据集有 IFEval、CEval、gsm8k、MMLU、AlignBench、MTBench

这里面的前四项导入配置如下：

```python
# IFEval
from .datasets.IFEval.IFEval_gen import ifeval_datasets
# CEval
from .datasets.ceval.ceval_gen import ceval_datasets
# gsm8k
from .datasets.gsm8k.gsm8k_gen import gsm8k_datasets
# MMLU
from .datasets.mmlu.mmlu_gen import mmlu_datasets
```

这里给出在使用 VLLM Qwen14B-Chat 模型推理的情况下，上面四个数据集的评测所用时间：

```
# IFEval Time
real    7m36.997s
user    15m36.631s
sys     3m22.041s

# CEval Time
real    9m57.007s
user    17m3.051s
sys     3m20.938s

# GSM8K
real    10m33.293s
user    18m2.768s
sys     4m1.202s

# MMLU
real    22m46.415s
user    27m28.503s
sys     6m58.145s
```

后两项的 AlignBench、MTBench 比较特殊，属于[主观评测数据集](https://opencompass.readthedocs.io/zh-cn/latest/advanced_guides/subjective_evaluation.html)，除了需要配置推理模型和数据集之外，还需要配置 `judge_model` 用于评价

天工内部的 GPT_Tools 也封装成了一个新的 API Model，位于 `opencompass/models/sky_openai_api.py`，在评测的时候最好把这个作为 judge model

非主观评测可以参考 `configs/eval_skywork_qwen.py`

主观评测可以参考 `configs/eval_skywork_alignbench_gpt4.py` 以及 `configs/eval_skywork_mtbench_gpt4.py`


## 其他注意事项

- 如非必要，请在模型配置文件中使用 VLLM 而非 HuggingFaceCausalLM，VLLM 的速度快非常多，换句话说，HuggingFaceCausalLM 会很慢
- infer 和 eval 这两个部分可以分开，详情查看 [如何单独进行infer和eval](https://github.com/open-compass/opencompass/discussions/1021)
- eval 的时候使用 VLLM 模型似乎有未知原因的 BUG，如果遇到解决不了就换成 HuggingFaceCausalLM



