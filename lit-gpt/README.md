# DropBP Application on Lit-GPT
Before we begin to replicate DropBP in earnest, we recommend reading the 'Setup', 'Use the model', and 'Finetune the model' sections of the original code below. 
We will briefly explain this just for the implementation of DropBP.

## 1. Setup
1.1 Setup environment for lit-gpt
```bash
pip install -r requirements-all.txt
```
2.2 Setup environment for DropBP
```bash
cd ../
pip install -v -e .
```

## 2. Prepair LLaMA2
For downloading LLaMA2, you should be able to access the weights of Hugging Face. It can be possible by following the steps at https://huggingface.co/meta-llama/Llama-2-7b. After access is granted, you can download LLaMA2 by using your HF hub token in https://huggingface.co/settings/tokens.

```bash
python scripts/download.py --repo_id meta-llama/Llama-2-7b-hf --access_token your_hf_token

python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf
```

## 3. Prepair Datasets

```bash
python scripts/prepare_alpaca.py \
 --checkpoint_dir checkpoints/meta-llama/LLaMA-2-7b-hf \
 --max_seq_length 512
```

## 3. Fine-tuning with DropBP
+ You can use ```--drop_rate 0.5``` for set target average drop rate of DropBP.
+ You can use ```--is_sens_alloc True``` for allowing sensitivity-based drop rates allocation of DropBP.
1. full fine-tuning
```bash
python finetune/full.py \
    --data_dir data/alpaca/ \
    --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf \
    --out_dir out/alpaca/full \
    --drop_rate 0.5 \
    --is_sens_alloc True
    --precision "bf16-true"
```
2. LoRA
```bash
python finetune/lora.py \
    --data_dir data/alpaca/ \
    --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf \
    --out_dir out/alpaca/lora \
    --drop_rate 0.5 \
    --is_sens_alloc True
    --precision "bf16-mixed"
```

3. QLoRA
```bash
python finetune/lora.py \
    --data_dir data/alpaca/ \
    --checkpoint_dir checkpoints/meta-llama/Llama-2-70b-hf \
    --out_dir out/alpaca/qlora \
    --drop_rate 0.5 \
    --is_sens_alloc True
    --quantize "bnb.nf4"
```

## 4. Evaluation
+ to evaluate on MMLU, use ```--eval_tasks "["hendrycksTest-*"]"```
+ to evaluate on Commonsense reasoning, use ```--eval_tasks "[arc_challenge, piqa, hellaswag, arc_easy, winogrande, openbookqa]"```
1. full fine-tuning
```bash
python eval/lm_eval_harness.py \
  --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf \
  --model_dir out/alpaca/full/ \
  --precision "bf16-true" \
  --eval_tasks "["hendrycksTest-*"]" \
  --save_filepath "out/alpaca/full/mmlu.json"
  --quantize "bnb.nf4"
```

2. LoRA
```bash
python eval/lora_harness.py \
  --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf \
  --lora_path out/alpaca/lora/lit_model_finetuned.pth \
  --precision "bf16-true" \
  --eval_tasks "["hendrycksTest-*"]" \
  --save_filepath "out/alpaca/lora/mmlu.json"
```

3. QLoRA
```bash
python eval/lora_harness.py \
  --checkpoint_dir checkpoints/meta-llama/Llama-2-70b-hf \
  --lora_path out/alpaca/qlora/lit_model_finetuned.pth \
  --precision "bf16-true" \
  --eval_tasks "["hendrycksTest-*"]" \
  --save_filepath "out/alpaca/qlora/mmlu.json"
```
After making the results file (i.e. mmlu.json), you can use ```eval/mmlu.py``` and ```eval/csr.py``` for easy analysis of the results as below.
```bash
python eval/mmlu.py \
  --json_file "out/alpaca/full/mmlu.json" \
```
```bash
python eval/csr.py \
  --json_file "out/alpaca/full/csr.json" \
```

## Acknowledgements
- [@LightningAI](https://github.com/Lightning-AI) for [lit-gpt](https://github.com/Lightning-AI/lit-gpt/tree/main)
- [@karpathy](https://github.com/karpathy) for [nanoGPT](https://github.com/karpathy/nanoGPT)
- [@EleutherAI](https://github.com/EleutherAI) for [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) and the [Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [@TimDettmers](https://github.com/TimDettmers) for [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [@IST-DASLab](https://github.com/IST-DASLab) for [GPTQ](https://github.com/IST-DASLab/gptq)
- [@Microsoft](https://github.com/microsoft) for [LoRA](https://github.com/microsoft/LoRA)
- [@tridao](https://github.com/tridao) for [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)


# Below is Lit-GPT's original README. Note that we only intergrates our DropBP to ```./finetune/lora.py```, ```./finetune/full.py```, and ```./lit_gpt/model.py```. Additionally, we add some code to the ```./eval/``` for easy evaluation.


<div align="center">
<img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/LitStableLM_Badge.png" alt="Lit-GPT" width="128"/>

# ⚡ Lit-GPT

<!--
<p align="center">
  <a href="https://www.lightning.ai/">Lightning.ai</a> •
  <a href="https://lightning.ai/docs/pytorch/stable/">PyTorch Lightning</a> •
  <a href="https://lightning.ai/docs/fabric/stable/">Fabric</a>
</p>
-->

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytorch-lightning)
![cpu-tests](https://github.com/lightning-AI/lit-stablelm/actions/workflows/cpu-tests.yml/badge.svg) [![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lit-stablelm/blob/master/LICENSE) [![Discord](https://img.shields.io/discord/1077906959069626439?style=plastic)](https://discord.gg/VptPCZkGNa)

<img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/LitStableLM.gif" alt="Lit-GPT and pineapple pizza" width="500px"/>

</div>

&nbsp;

# ⚡ Lit-GPT

Hackable [implementation](lit_gpt/model.py) of state-of-the-art open-source large language models released under the **Apache 2.0 license**.

Supports the following popular model checkpoints:

| Model and usage                                                                   | Model size                               | Reference                                                                                        |
|-----------------------------------------------------------------------------------|------------------------------------------|--------------------------------------------------------------------------------------------------|
| EleutherAI [Pythia](tutorials/download_pythia.md)                                 | {14,31,70,160,410}M, {1,1.4,2.8,6.9,12}B | [Biderman et al. 2023](https://arxiv.org/abs/2304.01373)                                         |
| LMSYS [LongChat](tutorials/download_longchat.md)                                  | 7B, 13B                                  | [LongChat Team 2023](https://lmsys.org/blog/2023-06-29-longchat/)                                |
| LMSYS [Vicuna](tutorials/download_vicuna.md)                                      | 7B, 13B, 33B                             | [Li et al. 2023](https://lmsys.org/blog/2023-03-30-vicuna/)                                      |
| Meta AI [Code Llama](tutorials/download_code_llama.md)                            | 7B, 13B, 34B                             | [Rozière et al. 2023](https://arxiv.org/abs/2308.12950)                                          |
| Meta AI [Llama 2](tutorials/download_llama_2.md)                                  | 7B, 13B, 70B                             | [Touvron et al. 2023](https://arxiv.org/abs/2307.09288)                                          |
| Mistral AI [Mistral and Mixtral](tutorials/download_mistral.md)                   | 7B                                       | [Mistral  website](https://mistral.ai/)                                                          |
| Microsoft Research [Phi](tutorials/download_phi.md)                               | 1.3B, 2.7B                               | [Li et al. 2023](https://arxiv.org/abs/2309.05463)                                               |
| NousResearch Nous-Hermes                                                          | 7B, 13B, 70B                             | [Org page](https://huggingface.co/NousResearch)                                                  |
| OpenLM Research [OpenLLaMA](tutorials/download_openllama.md)                      | 3B, 7B, 13B                              | [Geng & Liu 2023](https://github.com/openlm-research/open_llama)                                 |
| Platypus                                                                          | 7B, 13B, 70B                             | [Lee, Hunter, and Ruiz 2023](https://arxiv.org/abs/2308.07317)                                   |
| Stability AI StableCode                                                           | 3B                                       | [Stability AI 2023](https://stability.ai/blog/stablecode-llm-generative-ai-coding)               |
| Stability AI [FreeWilly2](tutorials/download_freewilly_2.md) (Stable Beluga 2)    | 70B                                      | [Stability AI 2023](https://stability.ai/blog/stable-beluga-large-instruction-fine-tuned-models) |
| Stability AI [StableLM](tutorials/download_stablelm.md)                           | 3B, 7B                                   | [Stability AI 2023](https://github.com/Stability-AI/StableLM)                                    |
| Stability AI [StableLM Zephyr](tutorials/download_stablelm.md)                    | 3B                                       | [Stability AI 2023](https://stability.ai/blog/stablecode-llm-generative-ai-coding)               |
| TII UAE [Falcon](tutorials/download_falcon.md)                                    | 7B, 40B, 180B                            | [TII 2023](https://falconllm.tii.ae)                                                             |
| [TinyLlama](tutorials/download_tinyllama.md)                                      | 1.1B                                     | [Zhang et al. 2023](https://github.com/jzhang38/TinyLlama)                                       |
| Together [RedPajama-INCITE](tutorials/download_redpajama_incite.md)               | 3B, 7B                                   | [Together 2023](https://together.ai/blog/redpajama-models-v1)                                    |
| Trelis [Function Calling Llama 2](tutorials/download_function_calling_llama_2.md) | 7B                                       | [Trelis et al. 2023](https://huggingface.co/Trelis/Llama-2-7b-chat-hf-function-calling-v2)       |

This implementation extends on [Lit-LLaMA](https://github.com/lightning-AI/lit-llama) and [nanoGPT](https://github.com/karpathy/nanoGPT), and it's **powered by [Lightning Fabric](https://lightning.ai/docs/fabric/stable/) ⚡**.

&nbsp;

---

**🏆 NeurIPS 2023 Large Language Model Efficiency Challenge: 1 LLM + 1 GPU + 1 Day**

The Lit-GPT repository is the official starter kit for the [NeurIPS 2023 LLM Efficiency Challenge](https://llm-efficiency-challenge.github.io), which is a competition focused on finetuning an existing non-instruction tuned LLM for 24 hours on a single GPU. The competition has two tracks, one for the A100 and another for the 4090 GPUs.

If you are interested in participating, you can learn more about the NeurIPS LLM Efficiency Challenge on the official website [here](https://llm-efficiency-challenge.github.io). Also see the [Lit-GPT NeurIPS Challenge Quickstart Guide](tutorials/neurips_challenge_quickstart.md) for helpful tips.

**The submission deadline is Oct 25th, 2023.**

---


&nbsp;


## Lit-GPT design principles

This repository follows the main principle of **openness through clarity**.

**Lit-GPT** is:

- **Simple:** Single-file implementation without boilerplate.
- **Correct:** Numerically equivalent to the original model.
- **Optimized:** Runs fast on consumer hardware or at scale.
- **Open-source:** No strings attached.

Avoiding code duplication is **not** a goal. **Readability** and **hackability** are.

&nbsp;

## Get involved!

[Join our Discord](https://discord.gg/VptPCZkGNa) to build high-performance, truly open-source models for the common benefit of the community.

&nbsp;

## Setup

Clone the repo:

```bash
git clone https://github.com/Lightning-AI/lit-gpt
cd lit-gpt
```

Install the minimal dependencies:

```bash
pip install -r requirements.txt
```

Install with all dependencies (including quantization, sentencepiece, tokenizers for Llama models, etc.):

```bash
pip install -r requirements-all.txt
```

**(Optional) Use Flash Attention 2**

Flash Attention 2 will be used automatically if PyTorch 2.2 (or higher) is installed.
Currently, that requires installing PyTorch nightly, which you can get by running:

```bash
pip uninstall -y torch torchvision torchaudio torchtext
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
```

You are all set! 🎉

&nbsp;

## Use the model

To generate text predictions, you need to download the model weights. **If you don't have them, check out our [guide](tutorials/download_stablelm.md).**

Run inference:

```bash
python generate/base.py --prompt "Hello, my name is"
```

This will run the 3B pretrained model and require ~7 GB of GPU memory using the `bfloat16` datatype.

[Full guide for generating samples from the model](tutorials/inference.md).

You can also chat with the model interactively:

```bash
python chat/base.py
```

&nbsp;

### Run large models on smaller consumer devices

We support 4-bit quantization (as in QLoRA), (bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq, gptq.int4) and 8-bit quantization (bnb.int8) for inference by following [this guide](tutorials/quantize.md).

&nbsp;

## Finetune the model

We provide a simple training scripts (`finetune/adapter.py`, `finetune/adapter_v2.py`, and `finetune/lora.py`) that instruction-tunes a pretrained model on the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset.

1. Download the data and generate an instruction tuning dataset:

```bash
python scripts/prepare_alpaca.py
```

2. Run the finetuning script

For example, you can either use

Adapter ([Zhang et al. 2023](https://arxiv.org/abs/2303.16199)):

```bash
python finetune/adapter.py
```

or Adapter v2 ([Gao et al. 2023](https://arxiv.org/abs/2304.15010)):

```bash
python finetune/adapter_v2.py
```

or LoRA ([Hu et al. 2021](https://arxiv.org/abs/2106.09685)):

```bash
python finetune/lora.py
```

(Please see the [tutorials/finetune_adapter](tutorials/finetune_adapter.md) for details on the differences between the two adapter methods.)

The finetuning requires at least one GPU with ~12 GB memory (RTX 3060).

It is expected that you have downloaded the pretrained weights as described above.
More details about each finetuning method and how you can apply it to your own data can be found in our technical how-to guides.

&nbsp;

### Finetuning how-to guides

These technical tutorials illustrate how to run the finetuning code.

- [Finetune with Adapters](tutorials/finetune_adapter.md)
- [Finetune with LoRA or QLoRA](tutorials/finetune_lora.md)

&nbsp;

### Understanding finetuning -- conceptual tutorials

Looking for conceptual tutorials and explanations? We have some additional articles below:

- [Understanding Parameter-Efficient Finetuning of Large Language Models: From Prefix Tuning to LLaMA-Adapters](https://lightning.ai/pages/community/article/understanding-llama-adapters/)

- [Parameter-Efficient LLM Finetuning With Low-Rank Adaptation (LoRA)](https://lightning.ai/pages/community/tutorial/lora-llm/)

&nbsp;

## Pretraining

We provide simple training scripts based on Fabric if you want to venture into pretraining. Conversion scripts for our optimized streaming `PackedDataset` are included.

Follow this guide to start pretraining on

- [RedPajama, a reproduction of LLaMA's training set](tutorials/pretrain_redpajama.md)
- [OpenWeb Text, a reproduction of GPT-2's dataset](tutorials/pretrain_openwebtext.md)



&nbsp;


## Supported datasets

Lit-GPT includes a variety of dataset preparation scripts for finetuning and pretraining. Additional information about the datasets and dataset preparation is provided in the [Preparing Datasets](tutorials/prepare_dataset.md) tutorial.


&nbsp;

## XLA

Lightning AI has partnered with Google to add first-class support for [Cloud TPUs](https://cloud.google.com/tpu) in [Lightning’s frameworks](https://github.com/Lightning-AI/lightning) and Lit-GPT,
helping democratize AI for millions of developers and researchers worldwide.

Using TPUs with Lightning is as straightforward as changing one line of code.

We provide scripts fully optimized for TPUs in the [XLA directory](xla)

&nbsp;

## Get involved!

We are on a quest towards fully open source AI.

<img align="right" src="https://pl-public-data.s3.amazonaws.com/assets_lightning/LitStableLM_Illustration.png" alt="Lit-GPT" width="128"/>

Join us and start contributing, especially on the following areas:

- [ ] [Pretraining](https://github.com/Lightning-AI/lit-gpt/labels/pre-training)
- [ ] [Fine-tuning](https://github.com/Lightning-AI/lit-gpt/labels/fine-tuning)
- [ ] [Quantization](https://github.com/Lightning-AI/lit-gpt/labels/quantization)
- [ ] [Sparsification](https://github.com/Lightning-AI/lit-gpt/labels/sparsification)

We welcome all individual contributors, regardless of their level of experience or hardware. Your contributions are valuable, and we are excited to see what you can accomplish in this collaborative and supportive environment.

Unsure about contributing? Check out our [How to Contribute to Lit-GPT and Lit-LLaMA
](https://lightning.ai/pages/community/tutorial/how-to-contribute-to-litgpt/) guide.

Don't forget to [join our Discord](https://discord.gg/VptPCZkGNa)!

&nbsp;

## Acknowledgements

- [@karpathy](https://github.com/karpathy) for [nanoGPT](https://github.com/karpathy/nanoGPT)
- [@EleutherAI](https://github.com/EleutherAI) for [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) and the [Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [@TimDettmers](https://github.com/TimDettmers) for [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [@IST-DASLab](https://github.com/IST-DASLab) for [GPTQ](https://github.com/IST-DASLab/gptq)
- [@Microsoft](https://github.com/microsoft) for [LoRA](https://github.com/microsoft/LoRA)
- [@tridao](https://github.com/tridao) for [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)

&nbsp;

## License

Lit-GPT is released under the [Apache 2.0](https://github.com/Lightning-AI/lit-gpt/blob/main/LICENSE) license.
