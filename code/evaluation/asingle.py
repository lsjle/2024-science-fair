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
dfs=pd.concat([pd.read_csv("ctscore.csv")['score'][1:7],pd.read_csv("ctscore.csv")['score'][8:14],pd.read_csv("ctscore.csv")['score'][15:21]])
dfm=pd.concat([pd.read_csv("ctscore.csv")[1:7]['model'],pd.read_csv("ctscore.csv")[8:14]['model'],pd.read_csv("ctscore.csv")[15:21]['model']])
dfc=pd.read_csv("ctscore.csv")
plt.plot(dfm,dfs,c="#1f77b4")
plt.axhline(y=dfc['score'][0],linestyle=":",c="#1f77b4")
plt.axhline(y=dfc['score'][7],linestyle=":",c="#ff7f0e")
plt.axhline(y=dfc['score'][15],linestyle=":",c="#2ca02c")
plt.legend(["分數","V1標準","V2標準","V3標準"])

plt.ylabel("分數(來源見研究方法)")
plt.xlabel("版本及檢查點(1-500~3-3000)")
plt.title("ChatGLM不同版本及檢查點CT表現比較")
plt.savefig('../../chart/allctc.eps', format='eps')
plt.savefig('../../chart/allctcp.png', format='png')
