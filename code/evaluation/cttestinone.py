#convert all ct score in one
import pandas as pd

timestamp=1706589077.8066983
for i in range(1,4):
    for k in range(0,7):
        df=pd.read_csv("outputs/CT/{}-ckpt{}-{}.csv".format(i,k*500,timestamp))
        allscore=0
        for j in range(0,11):
            allscore+=df['score'][i]
        print("{}-{},{}".format(i,k,allscore/11))
