import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import font_manager
import matplotlib
font_dir = ["/home/ntnu_stu/maid-weeny/font"]
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)

matplotlib.rcParams["font.family"] = "TW-MOE-Std-Kai"
df=pd.read_csv("scorenew.csv")[15:22]
dfc=pd.read_csv("scorenew.csv")
plt.plot(df['model'],df['BLEU-4'],c="#1f77b4")
plt.plot(df['model'],df[' Rouge1'],c="#ff7f0e")
plt.plot(df['model'],df[' Rouge2'],c="#2ca02c")
plt.plot(df['model'],df['RougeL'],c="#d62728")
plt.axhline(y=dfc['BLEU-4'][14],linestyle=":",c="#1f77b4")
plt.axhline(y=dfc[' Rouge1'][14],linestyle=":",c="#ff7f0e")
plt.axhline(y=dfc[' Rouge2'][14],linestyle=":",c="#2ca02c")
plt.axhline(y=dfc['RougeL'][14],linestyle=":",c="#d62728")
plt.legend(["BLEU-4", "Rouge1", "Rouge2","RougeL"])

plt.ylabel("分數(來源見研究方法)")
plt.xlabel("檢查點(500-3000)")
plt.title("ChatGLM3不同檢查點TrustfulQA表現比較")
# plt.axis([0, 12,])
plt.savefig('../../chart/3vc.eps', format='eps')
plt.savefig('../../chart/3vcp.png', format='png')