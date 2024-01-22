import pandas as pd
import numpy as np
import time
def process(prompt):
    return prompt

print("Start generating answers... good luck!")
timenow=time.time()
print("Timestamp now:{}".format(timenow))
print("START OF TQAZH")
tqazhdf=pd.read_csv("datasets/TQAZH.csv")
tqazhdf.assign(modelans=0)
for i in range(817):
    tqazhdf['modelans'][i]=process(tqazhdf['Question'][i])
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
    ctndf['modelans'][i]=[process(ctdf['question'][i])]
ctdf=pd.concat([ctdf, ctndf],axis=1)
ctdf.to_csv("outputs/CT/{}.csv".format(timenow),index=False)
print("END OF CT")
print("Timestamp now:{}".format(timenow))