# Add DropBP on HuggingFace

Due to licensing issues, we cannot release the ```finetune.py``` code at this time. Once the licensing issues are resolved, we will make the code publicly available.

In this directory, we develop ```transformers_dropbp``` library which applying DropBP to existing HuggingFace [transformers](https://github.com/huggingface/transformers) library.

## Setup

Please follow the steps below to set up the environment and dependencies for applying DropBP to Huggingface transformers.

1. Install the required dependencies

```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

2. Install the DropBP library

```bash
cd ..
pip install -v -e .
```

3. Install the custom transformers library for applying DropBP
```bash
cd transformers_dropbp
pip install -v -e .
```

## How to apply DropBP to Trainer in HuggingFace

```python
from transformers import (
    ...
    Trainer,
    )

...

trainer = Trainer(
        model=model,
        args=training_args,
        ...
        drop_rate=0.5,
        measure_time_memory=True,
        time_warmup_steps=1,
        time_measure_steps=3,
        throughput_path='''outputs/throughput.txt''',
        )

...

trainer.train()

```
+ ```drop_rate```: The target average drop rate when applying DropBP.
  
+ ```measure_time_memory```: If set to True, it measures the train time per iteration and memory usage.
  
+ ```time_warmup_steps```: When measuring train time and memory usage, this determines from which iteration to start measuring the train time to ensure accurate results.
  
+ ```time_measure_steps```: When measuring train time and memory usage, this defines how many iterations to run before calculating the average train time per iteration.

+ ```throughput_path```: Saves the throughput results to this text file.

You can see the modified code in ```./transformers_dropbp/src/transformers/trainer.py ``` and ```./transformers_dropbp/src/transformers/models/llama/modeling_llama.py ```
## Examples 

Once the environment is set up, you can begin the fine-tuning process with the provided scripts. The scripts for running DropBP are as follows:

```bash
sh ./dropbp.sh # DropBP
```

## Acknowledgements

Our research has been greatly influenced by [transformers](https://github.com/huggingface/transformers) and [Vera](https://github.com/neurotechcenter/VERA). We sincerely appreciate the excellent repositories they have provided.