import pandas as pd
import numpy as np
import time
import os
import sys
from opencc import OpenCC
import gradio as gr
import torch
import argparse
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
)
ckptloc="/home/ntnu_stu/Roleplay/ChatGLM3/finetune_chatmodel_demo/output/role_play_newv1-20240131-152455-128-2e-2/checkpoint-5000"
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
config = AutoConfig.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True,pre_seq_len=128)#maybe error here
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", config=config, trust_remote_code=True).cuda()
prefix_state_dict = torch.load(os.path.join(ckptloc, "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    #     model = AutoModel.from_pretrained(args.model, trust_remote_code=True)

model = model.to((torch.device('cuda:0')))
def process(prompt):
    #end of init
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(torch.device('cuda:0'))
    response = model.generate(input_ids=inputs["input_ids"], max_length=inputs["input_ids"].shape[-1] + 128)
    response = response[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)
