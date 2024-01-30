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
#now read and get 
V1PATH="/home/ntnu_stu/Roleplay/ckpt2/role-play-chatglm-6b-pt-128-2e-2"
V2PATH="/home/ntnu_stu/Roleplay/ChatGLM2-6B/ptuning/output/role-play-chatglm2-6b-pt-128-2e-2"
V3PATH="/home/ntnu_stu/Roleplay/ChatGLM3/finetune_chatmodel_demo/output/role_play-20240126-110108-128-2e-2"
#check point are all vary from 500-3000
cc = OpenCC('s2twp')
folder_path = os.path.abspath('chatglm_6b')
sys.path.append(folder_path)
# MODELLOC="THUDM/chatglm-6b"

pd.options.mode.copy_on_write = True    
timenow=time.time()
preseqlen=128
#use core 3!!!!!
#start init load from one
print("Start with version one")
print("Start model")
for i  in  range(1,5):
    ckptnum=i*500
    ckptloc=os.path.join(V3PATH,"checkpoint-{}".format(ckptnum))
    print("Start of ckpt init")
    print(ckptloc)
    # if args.pt_checkpoint:
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


    #end of input

    print("loadding complete")
    print("Start generating answers... good luck!")
    timenow=time.time()
    print("Timestamp now:{}".format(timenow))
    print("START OF TQAZH")
    tqazhdf=pd.read_csv("datasets/TQAZH.csv")
    tqazhdf=tqazhdf.assign(modelans="")
    for i in range(817):
        print(i)
        tqazhdf['modelans'][i]=process(cc.convert(tqazhdf['Question'][i]))
    tqazhdf.to_csv("outputs/TQAZH/3-ckpt{}-{}.csv".format(ckptnum,timenow),index=False)
    print("END OF TQAZH")
    print("-----")
    print("START OF MMLUZH")

    print("END OF MMLUZH")
    print("-----")
    print("START OF CT")
    ctdf=pd.read_csv("datasets/ct.csv")
    ctndf=pd.DataFrame(np.zeros(12),columns=["modelans"])
    for i in range(11):
        ctndf['modelans'][i]=[process(ctdf['question'][i])]
    ctdf=pd.concat([ctdf, ctndf],axis=1)
    ctdf.to_csv("outputs/CT/3-ckpt{}-{}.csv".format(ckptnum,timenow),index=False)
    print("END OF CT")
    print("Timestamp now:{}".format(timenow))


