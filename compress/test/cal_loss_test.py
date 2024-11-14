
import json
import re

import numpy
import numpy as np
import matplotlib.pyplot as plt
import os

import torch



lm_loss = []
compress_loss = []
use_compress_loss = False
with open('../compressLLM_multi_lora_510_ratio-5_lm/instruction_info.json', 'r') as f:
    data = json.load(f)
    for run in data:
        lm_loss.append(run['training_loss']['lm_loss'])
        if 'compress_loss' in run['training_loss']:
            use_compress_loss = True
            compress_loss.append(run['training_loss']['compress_loss'])
avg_lm_loss = np.mean(lm_loss)
compress_loss = np.mean(compress_loss)
print(avg_lm_loss)
print(compress_loss)

