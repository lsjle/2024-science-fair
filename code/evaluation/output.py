import pandas as pd
import numpy as np
import time
import os
import sys
from opencc import OpenCC
import gradio as gr
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
)
cc = OpenCC('s2twp')
folder_path = os.path.abspath('chatglm_6b')
sys.path.append(folder_path)
modelloc="THUDM/chatglm-6b"
ckptloc="/home/ntnu_stu/Roleplay/ckpt2/role-play-chatglm-6b-pt-128-2e-2/checkpoint-3000"
preseqlen=128
def process(prompt,model,tokenizer):
    response, history = model.chat(tokenizer, prompt, history=[])
    return response
print("Start loading model")
tokenizer = AutoTokenizer.from_pretrained(modelloc, trust_remote_code=True)#.float() maybe not required

config = AutoConfig.from_pretrained(
    modelloc, trust_remote_code=True)

config.pre_seq_len = preseqlen


print(f"Loading prefix_encoder weight from {ckptloc}")
model = AutoModel.from_pretrained(modelloc, config=config, trust_remote_code=True)#.float()
prefix_state_dict = torch.load(os.path.join(ckptloc, "pytorch_model.bin") )#,map_location=torch.device('cpu')
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

if preseqlen is not None:
    # P-tuning v2
    model = model.half().cuda()
    model.transformer.prefix_encoder.float().cuda()

model = model.eval()
print("loadding complete")
print("Start generating answers... good luck!")
timenow=time.time()
print("Timestamp now:{}".format(timenow))
print("START OF TQAZH")
tqazhdf=pd.read_csv("datasets/TQAZH.csv")
tqazhdf=tqazhdf.assign(modelans="")
for i in range(817):
    tqazhdf['modelans'][i]=process(cc.convert(tqazhdf['Question'][i]),model,tokenizer)
tqazhdf.to_csv("outputs/TQAZH/{}.csv".format(timenow),index=False)
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
ctdf.to_csv("outputs/CT/{}.csv".format(timenow),index=False)
print("END OF CT")
print("Timestamp now:{}".format(timenow))

# from opencc import OpenCC

# cc = OpenCC('s2twp')
# text = '投票當天需攜帶投票通知單、國民身分證及印章，若沒有收到投票通知書，可以向戶籍所在地鄰長查詢投票所，印章則是可以用簽名代替，至於身分證則是一定要攜帶。'

# print(cc.convert(text))