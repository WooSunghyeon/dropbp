import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from dropbp_trainer import DropBPTrainer
from insert_dropbp import insert_dropbp
import wandb

## Experiment Name
os.environ["WANDB_PROJECT"] = "Mistral-7B"
os.environ["WANDB_NAME"] = "DropBP (p=0.875)"

## Config
SEED=526
MICRO_BATCH_SIZE = 2 
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 1  
LEARNING_RATE = 1e-4 
CUTOFF_LEN = 512
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
DROP_RATE = 0.875
output_dir = "./lora_alpaca"

## Load Base Model
set_seed(SEED)
base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(base_model_id,
                                             #torch_dtype=torch.float16, 
                                             #quantization_config=bnb_config
                                             )
## Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    model_max_length=CUTOFF_LEN,
    padding_side="left",
    add_eos_token=True)

tokenizer.pad_token = tokenizer.eos_token

def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

## Set UP LoRA

#model.gradient_checkpointing_enable()
#model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
insert_dropbp(model, cutoff_len=CUTOFF_LEN)
model = get_peft_model(model, config)
# DropBP Step 0. Insert a DropBP layer into the Transformer Block
print(model)
print_trainable_parameters(model)

## Prepair datasets
data = load_dataset("yahma/alpaca-cleaned")
def generate_prompt(example: dict) -> str:
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    if example["input"]:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response:"
    )

train_val = data["train"].train_test_split(
    test_size=2000,
    shuffle=True,
    seed=42
)

train_data = train_val["train"].shuffle().map(
    lambda data_point: tokenizer(
        generate_prompt(data_point),
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length",
    )
)

eval_data = train_val["test"].shuffle().map(
    lambda data_point: tokenizer(
        generate_prompt(data_point),
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length",
    )
)

## train
trainer = DropBPTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        bf16=True,
        logging_steps=1,
        output_dir=output_dir,
        save_total_limit=4,
        evaluation_strategy="steps",
        save_strategy= "steps",
        eval_steps=500,
        
#       save_steps=500,
    ),
    drop_rate=DROP_RATE,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train_dropbp(resume_from_checkpoint=False)

model.save_pretrained(output_dir)
model.save_model()