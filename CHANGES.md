# File Changes

这里记录了相比官方 Github 仓库所做的改动

## 数据集

添加了下面的数据集：

- `OpenCompassData-complete-20240207.zip`
- `OpenCompassData-complete-20240325.zip`

运行下面命令解压数据集：

```
unzip OpenCompassData-complete-20240207.zip
unzip OpenCompassData-complete-20240325.zip
cd data
find . -name "*.zip" -exec unzip "{}" \;
```

## `configs/models`

### `configs/models/qwen`

添加了 qwen1_5_14b_chat 的微调模型示例：

- `vllm_qwen1_5_14b_chat.py`

### `configs/models/openai`

添加了基于天工内部 GPT_Tools 封装的 gpt4 api 模型

- `sky_gpt4.py`

## `configs`

添加了qwen1_5_14b_chat 的微调模型在常用数据集上的评测示例

- `eval_vllm_skywork_qwen.py`
- `eval_skywork_alignbench_pandalm.py`

## `opencompass`

添加了基于天工内部 GPT_Tools 封装的 gpt4 api 模型

- `__init__.py`

### `opencompass/models`

添加了基于天工内部 GPT_Tools 封装的 gpt4 api 模型

- `sky_openai_api.py`
