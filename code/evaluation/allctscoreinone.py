import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import font_manager
import matplotlib
from matplotlib.pyplot import figure
font_dir = ["../../font"]
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)

matplotlib.rcParams["font.family"] = "TW-MOE-Std-Kai"
# dfs=pd.concat([pd.read_csv("ctscore.csv")['score'][1:7],pd.read_csv("ctscore.csv")['score'][8:14],pd.read_csv("ctscore.csv")['score'][15:21]])
# dfm=pd.concat([pd.read_csv("ctscore.csv")[1:7]['model'],pd.read_csv("ctscore.csv")[8:14]['model'],pd.read_csv("ctscore.csv")[15:21]['model']])
df=pd.read_csv("ctscore.csv")

plt.plot(df['model'],df['score'],c="#1f77b4")
plt.legend(["分數"])

plt.ylabel("分數(來源見研究方法)")
plt.xlabel("模型版本及檢查點")
plt.title("ChatGLM版本意識測試表現比較")
plt.savefig('../../chart/mixctscore.eps', format='eps')
plt.savefig('../../chart/mixctscorep.png', format='png')

