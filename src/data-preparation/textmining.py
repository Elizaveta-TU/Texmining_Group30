import pandas as pd
import numpy as np
from textblob import TextBlob, Word
import os

from tqdm import tqdm
import time

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords


df = pd.read_csv('../../gen/data-preparation/temp/parsed-data.csv', sep = ',')
print(df.head())

text_cleaned=[]
score_neg=[]
score_neu=[]
score_pos=[]
score_compound=[]
polarity=[]
subjectivity=[]

analyser = SentimentIntensityAnalyzer() 


for text in tqdm(df['text_eng']):
    # counter+=1
    #if counter>10: break 
    if (len(text)>1): 
      text = re.sub(r'@\w*',' ', text)
      text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
      text = re.sub(r'[\n]*', '', text)
      text.strip()
      text = text.replace('  ', ' ')
      text= text.replace('  ', ' ')   
      text_blob = TextBlob(text)

      
      #removing stop words
      text_blob = TextBlob(text).lower()
      for d in stopwords.words('english'):
          text_blob = text_blob.replace(d.lower() + ' ', ' ')
              
          #cleaning to remove extra spaces
          text_blob = text_blob.replace('  ', ' ')

              
      #correcting spelling
      text_blob=text_blob.correct()

      #lemmatization
      text_blob=Word(text_blob).lemmatize()
    else:
      text_blob = TextBlob(text).lower()
    
    #sentiment analysis
    score_vader = analyser.polarity_scores(text_blob)
    score_textblot = text_blob.sentiment
    
    #appending in the file
    text_cleaned.append(text_blob)
    score_neg.append(score_vader["neg"])
    score_neu.append(score_vader["neu"])
    score_pos.append(score_vader["pos"])
    score_compound.append(score_vader["compound"])
    polarity.append(score_textblot[0])
    subjectivity.append(score_textblot[1])


df['text_cleaned'] = text_cleaned 
df['score_neg'] = score_neg 
df['score_neu'] = score_neu 
df['score_pos'] = score_pos
df['score_compound']=score_compound
df['polarity']=polarity
df['subjectivity']=subjectivity


# for i, j in data.iterrows():
#     print(i)
#     try:
#         blob = TextBlob(j['text'])
#         data.loc[i, 'polarity'] = blob.sentiment.polarity
#         data.loc[i, 'subjectivity'] = blob.sentiment.subjectivity
#     except:
#         data.loc[i, 'polarity'] = ''
#         data.loc[i, 'subjectivity'] = ''

df.head()

os.makedirs('../../gen/data-preparation/output/', exist_ok=True)

df.to_csv('../../gen/data-preparation/output/dataset.csv', index = False)

print('done.')
