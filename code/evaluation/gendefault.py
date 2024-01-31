
from transformers import AutoTokenizer, AutoModel
import pandas as pd

import numpy as np
import time
import os
import sys
from opencc import OpenCC
import gradio as gr
import torch
import argparse
k=1
cc = OpenCC('s2twp')
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()

def process(prompt,tokenizer,model):
    response, history = model.chat(tokenizer, prompt, history=[])
    return response

print("loadding complete")
print("Start generating answers... good luck!")
timenow=1706589077.8066983
print("Timestamp now:{}".format(timenow))
print("START OF TQAZH")
tqazhdf=pd.read_csv("datasets/TQAZH.csv")
tqazhdf=tqazhdf.assign(modelans="")
for i in range(817):
    tqazhdf['modelans'][i]=process(cc.convert(tqazhdf['Question'][i]),tokenizer,model)
tqazhdf.to_csv("outputs/TQAZH/{}-ckpt0-{}.csv".format(k,timenow),index=False)
print("END OF TQAZH")
print("-----")
print("START OF MMLUZH")

print("END OF MMLUZH")
print("-----")
print("START OF CT")
ctdf=pd.read_csv("datasets/ct.csv")
ctndf=pd.DataFrame(np.zeros(12),columns=["modelans"])
for i in range(11):
    ctndf['modelans'][i]=[process(ctdf['question'][i],tokenizer,model)]
ctdf=pd.concat([ctdf, ctndf],axis=1)
ctdf.to_csv("outputs/CT/{}-ckpt0-{}.csv".format(k,timenow),index=False)
print("END OF CT")
print("Timestamp now:{}".format(timenow))

for k in range(2,4):
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm{}-6b".format(k), trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm{}-6b".format(k), trust_remote_code=True, device='cuda')
    model = model.eval()

    def process(prompt,tokenizer,model):
        response, history = model.chat(tokenizer, prompt, history=[])
        return response

    print("loadding complete")
    print("Start generating answers... good luck!")
    timenow=1706589077.8066983
    print("Timestamp now:{}".format(timenow))
    print("START OF TQAZH")
    tqazhdf=pd.read_csv("datasets/TQAZH.csv")
    tqazhdf=tqazhdf.assign(modelans="")
    for i in range(817):
        tqazhdf['modelans'][i]=process(cc.convert(tqazhdf['Question'][i]),tokenizer,model)
    tqazhdf.to_csv("outputs/TQAZH/{}-ckpt0-{}.csv".format(k,timenow),index=False)
    print("END OF TQAZH")
    print("-----")
    print("START OF MMLUZH")

    print("END OF MMLUZH")
    print("-----")
    print("START OF CT")
    ctdf=pd.read_csv("datasets/ct.csv")
    ctndf=pd.DataFrame(np.zeros(12),columns=["modelans"])
    for i in range(11):
        ctndf['modelans'][i]=[process(ctdf['question'][i],tokenizer,model)]
    ctdf=pd.concat([ctdf, ctndf],axis=1)
    ctdf.to_csv("outputs/CT/{}-ckpt0-{}.csv".format(k,timenow),index=False)
    print("END OF CT")
    print("Timestamp now:{}".format(timenow))



    #error occur on version one of the model