from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
df=pd.read_csv("outputs/TQAZH/1706163391.7872918.csv")
def getbleu(i):
    reference=[
        list(df['Correct Answers'][i]),
        list(df['Best Answer'][i]),
    ]
    candidate=list(df['modelans'][i])
    return sentence_bleu(reference, candidate, weights=(0.25,0.25,0.25,0.25))
    print('BLEU score -> {}'.format(sentence_bleu(reference, candidate, weights=(0.25,0.25,0.25,0.25))))
allscore=0
for i in range(817):
    allscore+=getbleu(i)
print(allscore/816)