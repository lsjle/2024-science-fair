import pandas as pd

df=pd.read_csv("ctscore.csv")
df=df.T
df.to_csv("ctscoreround.csv",index=False)