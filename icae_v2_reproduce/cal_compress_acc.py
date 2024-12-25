import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import HfArgumentParser, AutoModelForCausalLM
from peft import LoraConfig
from modeling_icae_multi_span_old import ICAE, ModelArguments, DataArguments, TrainingArguments
import sys
from nltk.translate.bleu_score import sentence_bleu
# Set the computation device
device = "cuda"

# Parse model, data, and training arguments
parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# Define Lora configuration
lora_config = LoraConfig(
    r=128,
    lora_alpha=32,
    lora_dropout=model_args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM"
)

# Initialize model and send it to CUDA device
model = ICAE(model_args, training_args, lora_config)
print("calculate BLEU4...")

if os.path.exists('ft_pwc_inference_1.json'):
    with open('ft_pwc_inference_1.json', 'r', encoding='utf-8') as f:
        examples_list = f.readlines()

# input_text = [entry["answer"] for entry in examples_list]
# cl_generate_text = [entry["output"] for entry in examples_list]
def cal_cl_token_acc(examples_list, tokenizer):
    correct_tokens = 0
    total_tokens = 0
    acc = []
    bleus = []
    for line in tqdm(examples_list, desc="Processing examples", total=len(examples_list)):
        data = json.loads(line)
        cl_gen_ids = model.tokenizer(data["output"], add_special_tokens=False)["input_ids"]
        input_ids = model.tokenizer(data["answer"], add_special_tokens=False)["input_ids"]

        total_tokens += len(cl_gen_ids)
        correct_tokens = 0
        total_tokens = 0
        bleu4 = sentence_bleu([input_ids], cl_gen_ids, weights=(0.25, 0.25, 0.25, 0.25))
        bleus.append(bleu4)
    return np.mean(acc), np.mean(bleus)

acc, blue = cal_cl_token_acc(examples_list, model.tokenizer)
print("acc:",acc)
print("bleu:",blue)