import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import font_manager
import matplotlib
from matplotlib.pyplot import figure
font_dir = ["/home/ntnu_stu/maid-weeny/font"]
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)

matplotlib.rcParams["font.family"] = "TW-MOE-Std-Kai"
df=pd.read_csv("scorenew.csv")

#plot 1:
plt.subplot(2, 3, 1)
i=1
df1=df['model'].iloc[i],df['model'].iloc[i+7],df['model'].iloc[i+14]
df2=df['BLEU-4'].iloc[i],df['BLEU-4'].iloc[i+7],df['BLEU-4'].iloc[i+14]
df3=df[' Rouge1'].iloc[i],df[' Rouge1'].iloc[i+7],df[' Rouge1'].iloc[i+14]
df4=df[' Rouge2'].iloc[i],df[' Rouge2'].iloc[i+7],df[' Rouge2'].iloc[i+14]
df5=df['RougeL'].iloc[i],df['RougeL'].iloc[i+7],df['RougeL'].iloc[i+14]
plt.plot(df1,df2,c="#1f77b4")
plt.plot(df1,df3,c="#ff7f0e")
plt.plot(df1,df4,c="#2ca02c")
plt.plot(df1,df5,c="#d62728")
#plt.legend(["BLEU-4", "Rouge1", "Rouge2","RougeL"])

plt.ylabel("分數(來源見研究方法)")
plt.xlabel("版本")
plt.title("檢查點500")

#plot 2:
plt.subplot(2, 3, 2)
i=2
df1=df['model'].iloc[i],df['model'].iloc[i+7],df['model'].iloc[i+14]
df2=df['BLEU-4'].iloc[i],df['BLEU-4'].iloc[i+7],df['BLEU-4'].iloc[i+14]
df3=df[' Rouge1'].iloc[i],df[' Rouge1'].iloc[i+7],df[' Rouge1'].iloc[i+14]
df4=df[' Rouge2'].iloc[i],df[' Rouge2'].iloc[i+7],df[' Rouge2'].iloc[i+14]
df5=df['RougeL'].iloc[i],df['RougeL'].iloc[i+7],df['RougeL'].iloc[i+14]
plt.plot(df1,df2,c="#1f77b4")
plt.plot(df1,df3,c="#ff7f0e")
plt.plot(df1,df4,c="#2ca02c")
plt.plot(df1,df5,c="#d62728")
#plt.legend(["BLEU-4", "Rouge1", "Rouge2","RougeL"])

plt.ylabel("分數(來源見研究方法)")
plt.xlabel("版本")
plt.title("檢查點1000")

#plot 3:
i=3
plt.subplot(2, 3, i)

df1=df['model'].iloc[i],df['model'].iloc[i+7],df['model'].iloc[i+14]
df2=df['BLEU-4'].iloc[i],df['BLEU-4'].iloc[i+7],df['BLEU-4'].iloc[i+14]
df3=df[' Rouge1'].iloc[i],df[' Rouge1'].iloc[i+7],df[' Rouge1'].iloc[i+14]
df4=df[' Rouge2'].iloc[i],df[' Rouge2'].iloc[i+7],df[' Rouge2'].iloc[i+14]
df5=df['RougeL'].iloc[i],df['RougeL'].iloc[i+7],df['RougeL'].iloc[i+14]
plt.plot(df1,df2,c="#1f77b4")
plt.plot(df1,df3,c="#ff7f0e")
plt.plot(df1,df4,c="#2ca02c")
plt.plot(df1,df5,c="#d62728")
#plt.legend(["BLEU-4", "Rouge1", "Rouge2","RougeL"])

plt.ylabel("分數(來源見研究方法)")
plt.xlabel("版本")
plt.title("檢查點1500")
#plot 4:
i=4
plt.subplot(2, 3, i)

df1=df['model'].iloc[i],df['model'].iloc[i+7],df['model'].iloc[i+14]
df2=df['BLEU-4'].iloc[i],df['BLEU-4'].iloc[i+7],df['BLEU-4'].iloc[i+14]
df3=df[' Rouge1'].iloc[i],df[' Rouge1'].iloc[i+7],df[' Rouge1'].iloc[i+14]
df4=df[' Rouge2'].iloc[i],df[' Rouge2'].iloc[i+7],df[' Rouge2'].iloc[i+14]
df5=df['RougeL'].iloc[i],df['RougeL'].iloc[i+7],df['RougeL'].iloc[i+14]
plt.plot(df1,df2,c="#1f77b4")
plt.plot(df1,df3,c="#ff7f0e")
plt.plot(df1,df4,c="#2ca02c")
plt.plot(df1,df5,c="#d62728")
#plt.legend(["BLEU-4", "Rouge1", "Rouge2","RougeL"])

plt.ylabel("分數(來源見研究方法)")
plt.xlabel("版本")
plt.title("檢查點2000")
#plot 3:
i=5
plt.subplot(2, 3, i)

df1=df['model'].iloc[i],df['model'].iloc[i+7],df['model'].iloc[i+14]
df2=df['BLEU-4'].iloc[i],df['BLEU-4'].iloc[i+7],df['BLEU-4'].iloc[i+14]
df3=df[' Rouge1'].iloc[i],df[' Rouge1'].iloc[i+7],df[' Rouge1'].iloc[i+14]
df4=df[' Rouge2'].iloc[i],df[' Rouge2'].iloc[i+7],df[' Rouge2'].iloc[i+14]
df5=df['RougeL'].iloc[i],df['RougeL'].iloc[i+7],df['RougeL'].iloc[i+14]
plt.plot(df1,df2,c="#1f77b4")
plt.plot(df1,df3,c="#ff7f0e")
plt.plot(df1,df4,c="#2ca02c")
plt.plot(df1,df5,c="#d62728")
#plt.legend(["BLEU-4", "Rouge1", "Rouge2","RougeL"])

plt.ylabel("分數(來源見研究方法)")
plt.xlabel("版本")
plt.title("檢查點2500")

#plot 4:
i=6
plt.subplot(2, 3, i)

df1=df['model'].iloc[i],df['model'].iloc[i+7],df['model'].iloc[i+14]
df2=df['BLEU-4'].iloc[i],df['BLEU-4'].iloc[i+7],df['BLEU-4'].iloc[i+14]
df3=df[' Rouge1'].iloc[i],df[' Rouge1'].iloc[i+7],df[' Rouge1'].iloc[i+14]
df4=df[' Rouge2'].iloc[i],df[' Rouge2'].iloc[i+7],df[' Rouge2'].iloc[i+14]
df5=df['RougeL'].iloc[i],df['RougeL'].iloc[i+7],df['RougeL'].iloc[i+14]
plt.plot(df1,df2,c="#1f77b4")
plt.plot(df1,df3,c="#ff7f0e")
plt.plot(df1,df4,c="#2ca02c")
plt.plot(df1,df5,c="#d62728")
#plt.legend(["BLEU-4", "Rouge1", "Rouge2","RougeL"])

plt.ylabel("分數(來源見研究方法)")
plt.xlabel("版本")
plt.title("檢查點3000")

plt.suptitle("ChatGLM版本對於TrustfulQA表現比較")
plt.tight_layout()
# figure(figsize=(12, 8))

plt.savefig('../../chart/allcomp.eps', format='eps')
plt.savefig('../../chart/allcompp.png', format='png')
