import requests
import pandas as pd
def translate(textin):
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/x-www-form-urlencoded',
    }

    data = {
        'q': textin,
        'source': 'en',
        'target': 'zt',
        'format': 'text',
        'api_key': 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx',
    }

    response = requests.post('http://localhost:5000/translate', headers=headers, data=data)
    return response.json()["translatedText"]
print("start")
df=pd.read_csv("TruthfulQA.csv")
for i in range(817):
    df['Question'][i]=translate(df['Question'][i])
    df['Best Answer'][i]=translate(df['Best Answer'][i])
    df['Correct Answers'][i]=translate(df["Correct Answers"][i])
    df['Incorrect Answers'][i]=translate(df['Incorrect Answers'][i])
df.to_csv("TQAZH.csv")