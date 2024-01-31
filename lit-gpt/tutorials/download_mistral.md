## Download [Mistral](https://mistral.ai) weights

### Mistral

[Mistral 7B](https://mistral.ai/news/announcing-mistral-7b) is Apache 2.0 licensed and can be used without restrictions. It:

* Outperforms Llama 2 13B on all benchmarks
* Outperforms Llama 1 34B on many benchmarks
* Approaches CodeLlama 7B performance on code, while remaining good at English tasks
* Uses Grouped-query attention (GQA) for faster inference
* ~~Uses Sliding Window Attention (SWA) to handle longer sequences at smaller cost~~.
  This project's implementation does not use Sliding Window Attention, so the context length is limited to 4096 tokens.

Details about the data used to train the model or training procedure have not been made public.

In order to use the Mistral 7B model checkpoint, which requires about 14 GB of disk space, download the weights and convert the checkpoint to the lit-gpt format:

```bash
pip install huggingface_hub

python scripts/download.py --repo_id mistralai/Mistral-7B-Instruct-v0.1

python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/mistralai/Mistral-7B-Instruct-v0.1
```

You're done! To execute the model just run:

```bash
pip install sentencepiece

python chat/base.py --checkpoint_dir checkpoints/mistralai/Mistral-7B-Instruct-v0.1
```

### Mixtral

[Mixtral 8x7B](https://mistral.ai/news/mixtral-of-experts) is a pretrained generative Sparse Mixture of Experts model based on Mistral 7B.
Mistral-8x7B outperforms Llama 2 70B on most benchmarks tested.

Details about the data used to train the model or training procedure have not been made public.

In order to use the Mixtral 7B model checkpoint, which requires about 94 GB of disk space, download the weights and convert the checkpoint to the lit-gpt format:

```bash
pip install huggingface_hub

python scripts/download.py --repo_id mistralai/Mixtral-8x7B-Instruct-v0.1 --from_safetensors true

python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/mistralai/Mixtral-8x7B-Instruct-v0.1
```

Inference coming soon.
