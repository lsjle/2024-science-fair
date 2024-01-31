from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
from opencc import OpenCC
import sys
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
cc = OpenCC('s2twp')
TIMESTAMP=1706589077.8066983
for f in range(1,4):
    for k in range(0,7):
        df=pd.read_csv("outputs/TQAZH/{}-ckpt{}-{}.csv".format(f,k*500,TIMESTAMP))
        def getbleu(i):
            reference=[
                list(cc.convert(df['Correct Answers'][i])),
                list(cc.convert(df['Best Answer'][i])),
            ]
            candidate=list(cc.convert(df['modelans'][i]))
            print('BLEU score -> {}'.format(sentence_bleu(reference, candidate, weights=(0.25,0.25,0.25,0.25))), file=sys.stderr)
            return sentence_bleu(reference, candidate, weights=(0.25,0.25,0.25,0.25))
        def convert(s): 
 
            # initialization of string to "" 
            new = "" 
        
            # traverse in the string 
            for x in s: 
                new += x 
        
            # return string 
            return new 
        allscore=0
        a=0
        b=0
        c=0
        for i in range(817):
            reference=[
                list(cc.convert(df['Correct Answers'][i])),
                list(cc.convert(df['Best Answer'][i])),
            ]
            candidate=list(cc.convert(str(df['modelans'][i]).strip()))
            # print('BLEU score -> {}'.format(sentence_bleu(reference, candidate, weights=(0.25,0.25,0.25,0.25))), file=sys.stderr)
            allscore+= sentence_bleu(reference, candidate, weights=(0.25,0.25,0.25,0.25))
            
            scores = scorer.score(cc.convert(df['Correct Answers'][i]), cc.convert(str(df['modelans'][i]).strip()))
            a+=scores['rouge1'][2]
            b+=scores['rouge2'][2]
            c+=scores['rougeL'][2]
        print("ChatGLM{}-Checkpoint{},{},{},{},{}".format(f,k,allscore/816,a/816,b/816,c/816))