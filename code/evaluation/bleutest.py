from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
from opencc import OpenCC
cc = OpenCC('s2twp')
TIMESTAMP=1706589077.8066983
for i in range(1,4):
    for k in range(1,7):
        df=pd.read_csv("outputs/TQAZH/{}-ckpt{}-{}.csv".format(i,k*500,TIMESTAMP))
        def getbleu(i):
            reference=[
                list(cc.convert(df['Correct Answers'][i])),
                list(cc.convert(df['Best Answer'][i])),
            ]
            candidate=list(cc.convert(df['modelans'][i]))
            print('BLEU score -> {}'.format(sentence_bleu(reference, candidate, weights=(0.25,0.25,0.25,0.25))), file=sys. stderr)
            return sentence_bleu(reference, candidate, weights=(0.25,0.25,0.25,0.25))
            
        allscore=0
        for i in range(817):
            allscore+=getbleu(i)
        print("ChatGLM{}-Checkpoint{},{}".format(i,k,allscore/816))