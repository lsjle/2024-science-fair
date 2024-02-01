import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import font_manager
import matplotlib
font_dir = ["/home/ntnu_stu/maid-weeny/font"]
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)

matplotlib.rcParams["font.family"] = "TW-MOE-Std-Kai"
df=pd.read_csv("scorenew.csv")
plt.plot(df['model'],df.drop(["model"],axis=1))
plt.legend(["BLEU-4", "Rouge1", "Rouge2","RougeL"])
plt.ylabel("準確度")
plt.xlabel("模型版本及檢查點")
plt.title("ChatGLM版本TrustfulQA表現比較")
# plt.axis([0, 12,])
plt.show()


plt.savefig('mixscorenew.eps', format='eps')
plt.savefig('mixscorenew.png', format='png')