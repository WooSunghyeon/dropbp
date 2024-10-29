# Add DropBP on HuggingFace

In this directory, we develop custom ```optimum-habana``` library which applying DropBP to existing optimum-habana [optimum-habana](https://github.com/huggingface/optimum-habana) library for Intel Gaudi Accelerators (HPUs).

## Setup

1. Install the required dependencies
```bash
pip install -q git+https://github.com/HabanaAI/DeepSpeed.git@1.18.0
pip install peft==0.11.1
```

2.  Install the custom optimum-habana library for applying DropBP
 ```bash
cd ./optimum-habana
pip install -v -e .
```

3. Install the DropBP library for HPU.
```bash
cd ./dropbp-habana
pip install -v -e .
```   

## How to apply DropBP to Trainer in HuggingFace
```python
from optimum.habana import (
  ...
  GaudiTrainer,
)

...

trainer = GaudiTrainer(
            model=model,
            gaudi_config=gaudi_config,
            args=training_args,
            ...
            drop_rate=training_args.drop_rate,
            measure_time=training_args.measure_time,
            time_warmup_steps=training_args.time_warmup_steps,
            time_measure_steps=training_args.time_measure_steps,
            throughput_path=training_args.throughput_path,
        )
...

trainer.train()

```

+ ```drop_rate```: The target average drop rate when applying DropBP.
  
+ ```measure_time_memory```: If set to True, it measures the train time per iteration and memory usage.
  
+ ```time_warmup_steps```: When measuring train time and memory usage, this determines from which iteration to start measuring the train time to ensure accurate results.
  
+ ```time_measure_steps```: When measuring train time and memory usage, this defines how many iterations to run before calculating the average train time per iteration.

+ ```throughput_path```: Saves the throughput results to this text file.

You can see the modified code in ```./optimum/habana/transformers/trainer.py ``` and ```./optimum/habana/transformers/models/llama/modeling_llama.py ```

## Examples 

Once the environment is set up, you can begin the fine-tuning process with the provided scripts. The scripts for running DropBP are as follows:

```bash
sh ./language-modeling/dropbp.sh # DropBP
```

## Acknowledgements

Our research has been greatly influenced by [transformers](https://github.com/huggingface/transformers) and [optimum-habana](https://github.com/huggingface/optimum-habana). We sincerely appreciate the excellent repositories they have provided.
