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
        abbr='qwen1.5-14b-chat-vllm',
        path="/maindata/data/shared/Security-SFT/hf_models/hub/models--Qwen--Qwen1.5-14B-Chat/snapshots/17e11c306ed235e970c9bb8e5f7233527140cdcf",
        # abbr='merge_skymoe47b_contrain_base_skycommon_safe_lora_240404_hf',
        # path='/maindata/data/shared/Security-SFT/dehao.li/workspace_shared/llama_factory_local/other_project/Firefly/checkpoint/merge_skymoe47b_contrain_base_skycommon_safe_lora_240404_hf',
        # abbr='merge_base_firefly_qw14b_base_skycommon_safe_lora_240404_hf',
        # path="/maindata/data/shared/Security-SFT/dehao.li/workspace_shared/llama_factory_local/other_project/Firefly/checkpoint/merge_base_firefly_qw14b_base_skycommon_safe_lora_240404_hf",
        # abbr='merge_qw14b_base_open_safe_lora_240329_hf',
        # path='/maindata/data/shared/Security-SFT/dehao.li/workspace_shared/llama_factory_local/project_240324/checkpoint/merge_qw14b_base_open_safe_lora_240329_hf',
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
