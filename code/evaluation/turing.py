#this code will mess the data to score
import random
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
def idgen(hom):
    #random 12bits generator
    rd=str(0)
    for i in range(21):
        rd+="{}".format(round(np.random.rand()))
    binary_int = int("{}{}".format(hom,rd), 2)
    
    # Getting the byte number
    byte_number = binary_int.bit_length() + 7 // 8
    
    # Getting an array of bytes
    binary_array = binary_int.to_bytes(byte_number, "big")
    #figure out somehow to overcome this
    # Converting the array into ASCII text
    ascii_text = binary_array.decode()
    
    # Getting the ASCII value
    return(ascii_text)
timestamp=input("Paste the timestamp or id here:")
print("Recived... Start running now")
df=pd.DateFrames(column=["id","Q","A"])
#id is generated as above 1 for human 0 for machine
#read from human
df.append
#read from machine
randomdf=pd.DataFrame([],columns=["id"])
for i in range(828):
    randomdf['id'][i]=idgen(0)
df.append(pd.read_csv("outputs/TQAZH/{}.csv".format(timestamp)["Question","modelans"]))
df.append(pd.read_csv("outputs/CT/{}.csv".format(timestamp)["question","modelans"]))
df=pd.concat([df,randomdf],axis=1)
x=df['Q','id']
y=df['A']
xt,xv,yt,yv=train_test_split(df,train_size=1.0,test_size=0.0,random_state=56)
os.mkdir(__file__+"pending/{}".format(timestamp))
