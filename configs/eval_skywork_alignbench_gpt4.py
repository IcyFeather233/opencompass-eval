from mmengine.config import read_base

with read_base():
    from .datasets.subjective.alignbench.alignbench_judgeby_critiquellm import subjective_datasets

from opencompass.models import HuggingFaceCausalLM, HuggingFace, HuggingFaceChatGLM3, OpenAI, SkyOpenAI
from opencompass.models.openai_api import OpenAIAllesAPIN
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.runners import LocalRunner
from opencompass.runners import SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import AlignmentBenchSummarizer

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]
)

# -------------Inference Stage ----------------------------------------
# For subjective evaluation, we often set do sample for models
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
        # abbr='qwen1.5-14b-chat-vllm',
        # path="/maindata/data/shared/Security-SFT/hf_models/hub/models--Qwen--Qwen1.5-14B-Chat/snapshots/17e11c306ed235e970c9bb8e5f7233527140cdcf",
        abbr='merge_skymoe47b_contrain_base_open_safe_240330_hf',
        path='/maindata/data/shared/Security-SFT/dehao.li/workspace_shared/llama_factory_local/project_240324/checkpoint/merge_skymoe47b_contrain_base_open_safe_240330_hf',
        model_kwargs=dict(tensor_parallel_size=GPU_NUMS),
        meta_template=_meta_template,
        max_out_len=2048,
        max_seq_len=4096,
        batch_size=GPU_NUMS * 8,
        # generation_kwargs=dict(temperature=0),
        generation_kwargs=dict(
            do_sample=True,
        ),
        end_str='<|im_end|>',
        run_cfg=dict(num_gpus=GPU_NUMS, num_procs=1),
    )
]

datasets = [*subjective_datasets]

# -------------Evalation Stage ----------------------------------------

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

## ------------- Evaluation Configuration
eval = dict(
    partitioner=dict(type=SubjectiveNaivePartitioner, mode='singlescore', models=models, judge_models=judge_models),
    runner=dict(type=LocalRunner, max_num_workers=GPU_NUMS, task=dict(type=SubjectiveEvalTask)),
)

summarizer = dict(type=AlignmentBenchSummarizer)

work_dir = 'outputs/alignment_bench/'