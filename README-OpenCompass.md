<div align="center">
  <img src="docs/en/_static/image/logo.svg" width="500px"/>
  <br />
  <br />

[![docs](https://readthedocs.org/projects/opencompass/badge)](https://opencompass.readthedocs.io/en)
[![license](https://img.shields.io/github/license/InternLM/opencompass.svg)](https://github.com/open-compass/opencompass/blob/main/LICENSE)

<!-- [![PyPI](https://badge.fury.io/py/opencompass.svg)](https://pypi.org/project/opencompass/) -->

[🌐Website](https://opencompass.org.cn/) |
[📖CompassHub](https://hub.opencompass.org.cn/home) |
[📊CompassRank](https://rank.opencompass.org.cn/home) |
[📘Documentation](https://opencompass.readthedocs.io/en/latest/) |
[🛠️Installation](https://opencompass.readthedocs.io/en/latest/get_started/installation.html) |
[🤔Reporting Issues](https://github.com/open-compass/opencompass/issues/new/choose)

English | [简体中文](README_zh-CN.md)

</div>

<p align="center">
    👋 join us on <a href="https://discord.gg/KKwfEbFj7U" target="_blank">Discord</a> and <a href="https://r.vansin.top/?r=opencompass" target="_blank">WeChat</a>
</p>

## 📣 OpenCompass 2.0

We are thrilled to introduce OpenCompass 2.0, an advanced suite featuring three key components: [CompassKit](https://github.com/open-compass), [CompassHub](https://hub.opencompass.org.cn/home), and [CompassRank](https://rank.opencompass.org.cn/home).
![oc20](https://github.com/tonysy/opencompass/assets/7881589/90dbe1c0-c323-470a-991e-2b37ab5350b2)

**CompassRank** has been significantly enhanced into the leaderboards that now incorporates both open-source benchmarks and proprietary benchmarks. This upgrade allows for a more comprehensive evaluation of models across the industry.

**CompassHub** presents a pioneering benchmark browser interface, designed to simplify and expedite the exploration and utilization of an extensive array of benchmarks for researchers and practitioners alike. To enhance the visibility of your own benchmark within the community, we warmly invite you to contribute it to CompassHub. You may initiate the submission process by clicking [here](https://hub.opencompass.org.cn/dataset-submit).

**CompassKit** is a powerful collection of evaluation toolkits specifically tailored for Large Language Models and Large Vision-language Models. It provides an extensive set of tools to assess and measure the performance of these complex models effectively. Welcome to try our toolkits for in your research and products.

## 🧭	Welcome

to **OpenCompass**!

Just like a compass guides us on our journey, OpenCompass will guide you through the complex landscape of evaluating large language models. With its powerful algorithms and intuitive interface, OpenCompass makes it easy to assess the quality and effectiveness of your NLP models.

🚩🚩🚩 Explore opportunities at OpenCompass! We're currently **hiring full-time researchers/engineers and interns**. If you're passionate about LLM and OpenCompass, don't hesitate to reach out to us via [email](mailto:zhangsongyang@pjlab.org.cn). We'd love to hear from you!

🔥🔥🔥 We are delighted to announce that **the OpenCompass has been recommended by the Meta AI**, click [Get Started](https://ai.meta.com/llama/get-started/#validation) of Llama for more information.

> **Attention**<br />
> We launch the OpenCompass Collaboration project, welcome to support diverse evaluation benchmarks into OpenCompass!
> Clike [Issue](https://github.com/open-compass/opencompass/issues/248) for more information.
> Let's work together to build a more powerful OpenCompass toolkit!

## 🚀 What's New <a><img width="35" height="20" src="https://user-images.githubusercontent.com/12782558/212848161-5e783dd6-11e8-4fe0-bbba-39ffb77730be.png"></a>

- **\[2024.02.29\]** We supported the MT-Bench, AlpacalEval and AlignBench, more information can be found [here](https://opencompass.readthedocs.io/en/latest/advanced_guides/subjective_evaluation.html) 🔥🔥🔥.
- **\[2024.01.30\]** We release OpenCompass 2.0. Click  [CompassKit](https://github.com/open-compass), [CompassHub](https://hub.opencompass.org.cn/home), and [CompassRank](https://rank.opencompass.org.cn/home) for more information ! 🔥🔥🔥.
- **\[2024.01.17\]** We supported the evaluation of [InternLM2](https://github.com/open-compass/opencompass/blob/main/configs/eval_internlm2_keyset.py) and [InternLM2-Chat](https://github.com/open-compass/opencompass/blob/main/configs/eval_internlm2_chat_keyset.py), InternLM2 showed extremely strong performance in these tests, welcome to try! 🔥🔥🔥.
- **\[2024.01.17\]** We supported the needle in a haystack test with multiple needles, more information can be found [here](https://opencompass.readthedocs.io/en/latest/advanced_guides/needleinahaystack_eval.html#id8) 🔥🔥🔥.
- **\[2023.12.28\]** We have enabled seamless evaluation of all models developed using [LLaMA2-Accessory](https://github.com/Alpha-VLLM/LLaMA2-Accessory), a powerful toolkit for comprehensive LLM development.
- **\[2023.12.22\]** We have released [T-Eval](https://github.com/open-compass/T-Eval), a step-by-step evaluation benchmark to gauge your LLMs on tool utilization. Welcome to our [Leaderboard](https://open-compass.github.io/T-Eval/leaderboard.html) for more details!

> [More](docs/en/notes/news.md)

## ✨ Introduction

![image](https://github.com/open-compass/opencompass/assets/22607038/f45fe125-4aed-4f8c-8fe8-df4efb41a8ea)

OpenCompass is a one-stop platform for large model evaluation, aiming to provide a fair, open, and reproducible benchmark for large model evaluation. Its main features include:

- **Comprehensive support for models and datasets**: Pre-support for 20+ HuggingFace and API models, a model evaluation scheme of 70+ datasets with about 400,000 questions, comprehensively evaluating the capabilities of the models in five dimensions.

- **Efficient distributed evaluation**: One line command to implement task division and distributed evaluation, completing the full evaluation of billion-scale models in just a few hours.

- **Diversified evaluation paradigms**: Support for zero-shot, few-shot, and chain-of-thought evaluations, combined with standard or dialogue-type prompt templates, to easily stimulate the maximum performance of various models.

- **Modular design with high extensibility**: Want to add new models or datasets, customize an advanced task division strategy, or even support a new cluster management system? Everything about OpenCompass can be easily expanded!

- **Experiment management and reporting mechanism**: Use config files to fully record each experiment, and support real-time reporting of results.

## 📊 Leaderboard

We provide [OpenCompass Leaderboard](https://rank.opencompass.org.cn/home) for the community to rank all public models and API models. If you would like to join the evaluation, please provide the model repository URL or a standard API interface to the email address `opencompass@pjlab.org.cn`.

<p align="right"><a href="#top">🔝Back to top</a></p>

## 🛠️ Installation

Below are the steps for quick installation and datasets preparation.

### 💻 Environment Setup

#### Open-source Models with GPU

```bash
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass
pip install -e .
```

#### API Models with CPU-only

```bash
conda create -n opencompass python=3.10 pytorch torchvision torchaudio cpuonly -c pytorch -y
conda activate opencompass
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass
pip install -e .
# also please install requiresments packages via `pip install -r requirements/api.txt` for API models if needed.
```

### 📂 Data Preparation

```bash
# Download dataset to data/ folder
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
```

Some third-party features, like Humaneval and Llama, may require additional steps to work properly, for detailed steps please refer to the [Installation Guide](https://opencompass.readthedocs.io/en/latest/get_started/installation.html).

<p align="right"><a href="#top">🔝Back to top</a></p>

## 🏗️ ️Evaluation

After ensuring that OpenCompass is installed correctly according to the above steps and the datasets are prepared, you can evaluate the performance of the LLaMA-7b model on the MMLU and C-Eval datasets using the following command:

```bash
python run.py --models hf_llama_7b --datasets mmlu_ppl ceval_ppl
```

OpenCompass has predefined configurations for many models and datasets. You can list all available model and dataset configurations using the [tools](./docs/en/tools.md#list-configs).

```bash
# List all configurations
python tools/list_configs.py
# List all configurations related to llama and mmlu
python tools/list_configs.py llama mmlu
```

You can also evaluate other HuggingFace models via command line. Taking LLaMA-7b as an example:

```bash
python run.py --datasets ceval_ppl mmlu_ppl \
--hf-path huggyllama/llama-7b \  # HuggingFace model path
--model-kwargs device_map='auto' \  # Arguments for model construction
--tokenizer-kwargs padding_side='left' truncation='left' use_fast=False \  # Arguments for tokenizer construction
--max-out-len 100 \  # Maximum number of tokens generated
--max-seq-len 2048 \  # Maximum sequence length the model can accept
--batch-size 8 \  # Batch size
--no-batch-padding \  # Don't enable batch padding, infer through for loop to avoid performance loss
--num-gpus 1  # Number of minimum required GPUs
```

> **Note**<br />
> To run the command above, you will need to remove the comments starting from `# ` first.

Through the command line or configuration files, OpenCompass also supports evaluating APIs or custom models, as well as more diversified evaluation strategies. Please read the [Quick Start](https://opencompass.readthedocs.io/en/latest/get_started/quick_start.html) to learn how to run an evaluation task.

<p align="right"><a href="#top">🔝Back to top</a></p>

## 📖 Dataset Support

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Language</b>
      </td>
      <td>
        <b>Knowledge</b>
      </td>
      <td>
        <b>Reasoning</b>
      </td>
      <td>
        <b>Examination</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
<details open>
<summary><b>Word Definition</b></summary>

- WiC
- SummEdits

</details>

<details open>
<summary><b>Idiom Learning</b></summary>

- CHID

</details>

<details open>
<summary><b>Semantic Similarity</b></summary>

- AFQMC
- BUSTM

</details>

<details open>
<summary><b>Coreference Resolution</b></summary>

- CLUEWSC
- WSC
- WinoGrande

</details>

<details open>
<summary><b>Translation</b></summary>

- Flores
- IWSLT2017

</details>

<details open>
<summary><b>Multi-language Question Answering</b></summary>

- TyDi-QA
- XCOPA

</details>

<details open>
<summary><b>Multi-language Summary</b></summary>

- XLSum

</details>
      </td>
      <td>
<details open>
<summary><b>Knowledge Question Answering</b></summary>

- BoolQ
- CommonSenseQA
- NaturalQuestions
- TriviaQA

</details>
      </td>
      <td>
<details open>
<summary><b>Textual Entailment</b></summary>

- CMNLI
- OCNLI
- OCNLI_FC
- AX-b
- AX-g
- CB
- RTE
- ANLI

</details>

<details open>
<summary><b>Commonsense Reasoning</b></summary>

- StoryCloze
- COPA
- ReCoRD
- HellaSwag
- PIQA
- SIQA

</details>

<details open>
<summary><b>Mathematical Reasoning</b></summary>

- MATH
- GSM8K

</details>

<details open>
<summary><b>Theorem Application</b></summary>

- TheoremQA
- StrategyQA
- SciBench

</details>

<details open>
<summary><b>Comprehensive Reasoning</b></summary>

- BBH

</details>
      </td>
      <td>
<details open>
<summary><b>Junior High, High School, University, Professional Examinations</b></summary>

- C-Eval
- AGIEval
- MMLU
- GAOKAO-Bench
- CMMLU
- ARC
- Xiezhi

</details>

<details open>
<summary><b>Medical Examinations</b></summary>

- CMB

</details>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Understanding</b>
      </td>
      <td>
        <b>Long Context</b>
      </td>
      <td>
        <b>Safety</b>
      </td>
      <td>
        <b>Code</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
<details open>
<summary><b>Reading Comprehension</b></summary>

- C3
- CMRC
- DRCD
- MultiRC
- RACE
- DROP
- OpenBookQA
- SQuAD2.0

</details>

<details open>
<summary><b>Content Summary</b></summary>

- CSL
- LCSTS
- XSum
- SummScreen

</details>

<details open>
<summary><b>Content Analysis</b></summary>

- EPRSTMT
- LAMBADA
- TNEWS

</details>
      </td>
      <td>
<details open>
<summary><b>Long Context Understanding</b></summary>

- LEval
- LongBench
- GovReports
- NarrativeQA
- Qasper

</details>
      </td>
      <td>
<details open>
<summary><b>Safety</b></summary>

- CivilComments
- CrowsPairs
- CValues
- JigsawMultilingual
- TruthfulQA

</details>
<details open>
<summary><b>Robustness</b></summary>

- AdvGLUE

</details>
      </td>
      <td>
<details open>
<summary><b>Code</b></summary>

- HumanEval
- HumanEvalX
- MBPP
- APPs
- DS1000

</details>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

## 📖 Model Support

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Open-source Models</b>
      </td>
      <td>
        <b>API Models</b>
      </td>
      <!-- <td>
        <b>Custom Models</b>
      </td> -->
    </tr>
    <tr valign="top">
      <td>

- [InternLM](https://github.com/InternLM/InternLM)
- [LLaMA](https://github.com/facebookresearch/llama)
- [Vicuna](https://github.com/lm-sys/FastChat)
- [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [Baichuan](https://github.com/baichuan-inc)
- [WizardLM](https://github.com/nlpxucan/WizardLM)
- [ChatGLM2](https://github.com/THUDM/ChatGLM2-6B)
- [ChatGLM3](https://github.com/THUDM/ChatGLM3-6B)
- [TigerBot](https://github.com/TigerResearch/TigerBot)
- [Qwen](https://github.com/QwenLM/Qwen)
- [BlueLM](https://github.com/vivo-ai-lab/BlueLM)
- [Gemma](https://huggingface.co/google/gemma-7b)
- ...

</td>
<td>

- OpenAI
- Gemini
- Claude
- ZhipuAI(ChatGLM)
- Baichuan
- ByteDance(YunQue)
- Huawei(PanGu)
- 360
- Baidu(ERNIEBot)
- MiniMax(ABAB-Chat)
- SenseTime(nova)
- Xunfei(Spark)
- ……

</td>

</tr>
  </tbody>
</table>

<p align="right"><a href="#top">🔝Back to top</a></p>

## 🔜 Roadmap

- [x] Subjective Evaluation
  - [ ] Release CompassAreana
  - [x] Subjective evaluation.
- [x] Long-context
  - [x] Long-context evaluation with extensive datasets.
  - [ ] Long-context leaderboard.
- [x] Coding
  - [ ] Coding evaluation leaderboard.
  - [x] Non-python language evaluation service.
- [x] Agent
  - [ ] Support various agenet framework.
  - [x] Evaluation of tool use of the LLMs.
- [x] Robustness
  - [x] Support various attack method

## 👷‍♂️ Contributing

We appreciate all contributions to improving OpenCompass. Please refer to the [contributing guideline](https://opencompass.readthedocs.io/en/latest/notes/contribution_guide.html) for the best practice.

## 🤝 Acknowledgements

Some code in this project is cited and modified from [OpenICL](https://github.com/Shark-NLP/OpenICL).

Some datasets and prompt implementations are modified from [chain-of-thought-hub](https://github.com/FranxYao/chain-of-thought-hub) and [instruct-eval](https://github.com/declare-lab/instruct-eval).

## 🖊️ Citation

```bibtex
@misc{2023opencompass,
    title={OpenCompass: A Universal Evaluation Platform for Foundation Models},
    author={OpenCompass Contributors},
    howpublished = {\url{https://github.com/open-compass/opencompass}},
    year={2023}
}
```

<p align="right"><a href="#top">🔝Back to top</a></p>
