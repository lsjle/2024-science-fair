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
timenow=time.time()

preseqlen=128

#use core 3!!!!!
#start init load from one
print("Start with version one")
print("Start model")

print("Start Version 1")
for i in range(1,7):
    print("Start with 1-{}".format(i*500))

    modelloc="THUDM/chatglm-6b"
    ckptnum=i*500
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
    def process(prompt,model,tokenizer):
        response, history = model.chat(tokenizer, prompt, history=[])
        return response
    print("loadding complete")
    print("Start generating answers... good luck!")

    print("Timestamp now:{}".format(timenow))
    print("START OF TQAZH")
    tqazhdf=pd.read_csv("datasets/TQAZH.csv")
    tqazhdf=tqazhdf.assign(modelans="")
    for k in range(817):
        print("1-{}-{}".format(i,k))
        tqazhdf['modelans'][k]=process(cc.convert(tqazhdf['Question'][k]),model,tokenizer)

    tqazhdf.to_csv("outputs/TQAZH/1-ckpt{}-{}.csv".format(ckptnum,timenow),index=False)
    print("END OF TQAZH")
    print("-----")
    print("START OF MMLUZH")

    print("END OF MMLUZH")
    print("-----")
    print("START OF CT")
    ctdf=pd.read_csv("datasets/ct.csv")
    ctndf=pd.DataFrame(np.zeros(12),columns=["modelans"])

    for k in range(11):
        ctndf['modelans'][k]=[process(ctdf['question'][k],model,tokenizer)]

    ctdf=pd.concat([ctdf, ctndf],axis=1)
    ctdf.to_csv("outputs/CT/1-ckpt{}-{}.csv".format(ckptnum,timenow),index=False)
    print("END OF CT")
    print("Timestamp now:{}".format(timenow))
    print("End with 1-{}".format(i*500))



for i in range(1,7):
    print("Start Version 2")
    modelloc="THUDM/chatglm2-6b"
    ckptnum=i*500
    tokenizer = AutoTokenizer.from_pretrained(modelloc, trust_remote_code=True)#.float() maybe not required

    config = AutoConfig.from_pretrained(
        modelloc, trust_remote_code=True)

    config.pre_seq_len = preseqlen

    ckptloc=os.path.join(V2PATH,"checkpoint-{}".format(ckptnum))
    print(f"Loading prefix_encoder weight from {ckptloc}")
    model = AutoModel.from_pretrained(modelloc, config=config, trust_remote_code=True)#.float()
    prefix_state_dict = torch.load(os.path.join(ckptloc, "pytorch_model.bin") )#,map_location=torch.device('cpu')
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v

    #model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

    if preseqlen is not None:
        # P-tuning v2
        model = model.half().cuda()
        model.transformer.prefix_encoder.float().cuda()

    model = model.eval()
    def process(prompt,model,tokenizer):
        response, history = model.chat(tokenizer, prompt, history=[])
        return response
    print("loadding complete")
    print("Start generating answers... good luck!")

    print("Timestamp now:{}".format(timenow))
    print("START OF TQAZH")
    tqazhdf=pd.read_csv("datasets/TQAZH.csv")
    tqazhdf=tqazhdf.assign(modelans="")
    for i in range(817):
        tqazhdf['modelans'][i]=process(cc.convert(tqazhdf['Question'][i]),model,tokenizer)
    tqazhdf.to_csv("outputs/TQAZH/2-ckpt{}-{}.csv".format(ckptnum,timenow),index=False)
    print("END OF TQAZH")
    print("-----")
    print("START OF MMLUZH")

    print("END OF MMLUZH")
    print("-----")
    print("START OF CT")
    ctdf=pd.read_csv("datasets/ct.csv")
    ctndf=pd.DataFrame(np.zeros(12),columns=["modelans"])
    for i in range(11):
        ctndf['modelans'][i]=[process(ctdf['question'][i],model,tokenizer)]
    ctdf=pd.concat([ctdf, ctndf],axis=1)
    ctdf.to_csv("outputs/CT/2-ckpt{}-{}.csv".format(ckptnum,timenow),index=False)
    print("END OF CT")
    print("Timestamp now:{}".format(timenow))
