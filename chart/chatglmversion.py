import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from matplotlib import font_manager

font_dir = ["/home/impartialjust/Downloads/edufont"]
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)

# Set font family globally
matplotlib.rcParams["font.family"] = "TW-MOE-Std-Kai"
df = pd.read_csv("chatglmversion.csv")
plt.plot(df["Name"], df.drop(["Name"], axis=1))
plt.legend(["GSM8K", "BBH", "MMLU"])
plt.ylabel("準確度")
plt.xlabel("模型版本")
plt.title("ChatGLM版本效能比較")
plt.axis([0, 2, 0, 100])
plt.show()
