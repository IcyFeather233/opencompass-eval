from mmengine.config import read_base

with read_base():
    from .datasets.subjective.multiround.mtbench_single_judge_diff_temp import subjective_datasets

from opencompass.models import HuggingFaceCausalLM, HuggingFace, HuggingFaceChatGLM3, OpenAI, SkyOpenAI
from opencompass.models.openai_api import OpenAIAllesAPIN
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.runners import LocalRunner
from opencompass.runners import SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import MTBenchSummarizer

# -------------Inference Stage ----------------------------------------
# For subjective evaluation, we often set do sample for models
from opencompass.models import VLLM


GPU_NUMS = 8

_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='<|im_start|>user\n', end='<|im_end|>\n'),
        dict(role="BOT", begin="<|im_start|>assistant\n", end='<|im_end|>\n', generate=True),
    ],
    eos_token_id=151645,
)

# models = [
#     dict(
#         type=HuggingFaceCausalLM,
#         abbr='qwen1.5-14b-chat-hf',
#         path="/maindata/data/shared/Security-SFT/hf_models/hub/models--Qwen--Qwen1.5-14B-Chat/snapshots/17e11c306ed235e970c9bb8e5f7233527140cdcf",
#         model_kwargs=dict(
#             device_map='auto',
#             trust_remote_code=True
#         ),
#         tokenizer_kwargs=dict(
#             padding_side='left',
#             truncation_side='left',
#             trust_remote_code=True,
#             use_fast=False,
#         ),
#         meta_template=_meta_template,
#         pad_token_id=151645,
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=GPU_NUMS, num_procs=8),
#         end_str='<|im_end|>',
#     )
# ]

models = [
    dict(
        type=VLLM,
        # abbr='qwen1.5-14b-chat-vllm',
        # path="/maindata/data/shared/Security-SFT/hf_models/hub/models--Qwen--Qwen1.5-14B-Chat/snapshots/17e11c306ed235e970c9bb8e5f7233527140cdcf",
        abbr='merge_skymoe47b_contrain_base_open_safe_240330_hf',
        path='/maindata/data/shared/Security-SFT/dehao.li/workspace_shared/llama_factory_local/project_240324/checkpoint/merge_skymoe47b_contrain_base_open_safe_240330_hf',
        model_kwargs=dict(tensor_parallel_size=GPU_NUMS),
        meta_template=_meta_template,
        max_out_len=512,
        max_seq_len=2048,
        batch_size=8,
        end_str='<|im_end|>',
        run_cfg=dict(num_gpus=GPU_NUMS, num_procs=1),
    )
]

datasets = [*subjective_datasets]

# -------------Evalation Stage ----------------------------------------

## ------------- JudgeLLM Configuration
api_meta_template = dict(
    round=[
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
    ],
)

judge_models = [
    dict(abbr='Sky-GPT4',
        type=SkyOpenAI, path='sky-gpt4',
        meta_template=api_meta_template,
        query_per_second=16,
        batch_size=8),
]

## single evaluation
eval = dict(
    partitioner=dict(type=SubjectiveSizePartitioner, strategy='split', max_task_size=10000, mode='singlescore', models=models, judge_models=judge_models),
    runner=dict(type=LocalRunner, max_num_workers=32, task=dict(type=SubjectiveEvalTask)),
)

summarizer = dict(type=MTBenchSummarizer, judge_type='single')

work_dir = 'outputs/mtbench/'
