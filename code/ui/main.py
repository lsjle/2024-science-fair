from flask import Flask, render_template, request
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
app = Flask(__name__)
V1PATH="/home/ntnu_stu/Roleplay/ckpt2/role-play-chatglm-6b-pt-128-2e-2"
V2PATH="/home/ntnu_stu/Roleplay/ChatGLM2-6B/ptuning/output/role-play-chatglm2-6b-pt-128-2e-2"
cc = OpenCC('s2twp')
folder_path = os.path.abspath('chatglm_6b')
sys.path.append(folder_path)
# MODELLOC="THUDM/chatglm-6b"
preseqlen=128

#module init
@app.route("/")
def main():
    return render_template('index.html')

@app.route("/process", methods=['POST'])
def process():
    response = request.values['prompt']
    modelnum=request.values['model']
    if(modelnum<20000):
        #this will be version one
        modelloc="THUDM/chatglm-6b"
        modelnum-=10000
        if(modelnum==5000):
            modelnum=500
        ckptloc=modelnum
        tokenizer = AutoTokenizer.from_pretrained(modelloc, trust_remote_code=True)#.float() maybe not required

        config = AutoConfig.from_pretrained(
            modelloc, trust_remote_code=True)

        config.pre_seq_len = preseqlen

        ckptloc=os.path.join(V1PATH,"checkpoint-{}".format(ckptnum))
        print(f"Loading prefix_encoder weight from {ckptloc}")#after this no result prob err
        model = AutoModel.from_pretrained(modelloc, config=config, trust_remote_code=True)#.float()
        prefix_state_dict = torch.load(os.path.join(ckptloc, "pytorch_model.bin") )#,map_location=torch.device('cpu')
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        #model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
        #may err here

        if preseqlen is not None:
            # P-tuning v2
            model = model.half().cuda()
            model.transformer.prefix_encoder.float().cuda()

        model = model.eval()
        def pc(prompt,model,tokenizer):
            response, history = model.chat(tokenizer, prompt, history=[])
            return 
        response=pc(response)
    return render_template('process.html',**locals())