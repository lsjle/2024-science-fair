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
dfs=pd.DataFrame([pd.read_csv("ctscore.csv")['score'][1:7],pd.read_csv("ctscore.csv")['score'][8:15],pd.read_csv("ctscore.csv")['score'][16:21]])
dfm=[pd.read_csv("ctscore.csv")[1:7]['model'],pd.read_csv("ctscore.csv")[8:15]['model'],pd.read_csv("ctscore.csv")[16:21]['model']]
dfc=pd.read_csv("ctscore.csv")
plt.plot(dfc,dfs,c="#1f77b4")
plt.axhline(y=dfc['score'][14],linestyle=":",c="#1f77b4")
plt.legend(["分數"])

plt.ylabel("分數(來源見研究方法)")
plt.xlabel("版本及檢查點(1-500~3-3000)")
plt.title("ChatGLM不同版本及檢查點TrustfulQA表現比較")
# plt.axis([0, 12,])
plt.savefig('../../chart/allctc.eps', format='eps')
plt.savefig('../../chart/allctcp.png', format='png')


