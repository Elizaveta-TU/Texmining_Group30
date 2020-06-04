import json
from tqdm import tqdm


import numpy as np
import pandas as pd

from textblob import TextBlob, Word
import time




f = open('../../gen/data-preparation/temp/whitehouse_briefing_27_04.json','r', encoding='utf-8')

con=[]

for i, line in enumerate(f):
    if i % 2:   # counting starts at 0, and `i % 2` is true for odd numbers
        continue
    con.append(line)

# con = f.readlines()

outfile = open('../../gen/data-preparation/temp/parsed-data.csv', 'w', encoding = 'utf-8')

# outfile.write('id\tcreated_at\ttext\n')
print('Extracting location and text...')

#extract data about location, date and hashtags
location=[]
# date=[]
hashtags=[]
text=[]

for line in con:
  # print(line)
  tweet = json.loads(line)

 
  
  try:
    location_obj = tweet.get('user').get('location')
  except:
    location_obj = 'NA'
  
  try:    
    hashtags_obj = tweet.get('entities').get('hashtags')
  except:   
    hashtags_obj = 'NA'

  try:    
      text_obj = tweet.get('text')
  except:   
      text_obj = 'NA'

  # date.append(date_obj)
  location.append(location_obj)
  hashtags.append(hashtags_obj)
  text.append(text_obj)

## create df
df=pd.DataFrame({'location':location,
                 'text':text,
                 })

print('Location and text are extracted')
print('Dataset size is',df.shape[0])
  #prettify location, creat dictionary of locations where key is location and value is a number of time the hashtag was used
loc_dic={key: 0 for key in set(location)}

for loc in location:
    loc_dic[loc]+=1

a=sorted( ((v,k) for k,v in loc_dic.items()), reverse=True)[:150]

#classify location for inside, outside
print('Classifying tweets...')
loc150= [i[1] for i in a]
us_word2 = ['USA','CA','NY','DC','TX','GA','MA','WA','NV','PA','FL','AZ','D.C.','LA']
us_word1 = ['United States','USA','USA ','Los Angeles','US','New York','Texas','California','Florida','NYC','Washington DC','America','Ohio','New York City']
world_drop=['Global','Worldwide','Planet Earth','Everywhere','North America','WORLDWIDE','Earth','Houston, Poole, & Budapest']

inside=[]
outside=[]
for loc in loc150[1:]:
 
  c = loc.split(',')
  if (len(c)>1) and (c[1].strip() in us_word2):
    inside.append(loc)
  elif (len(c)==1) and (c[0].strip() in us_word1):
    inside.append(loc)
  elif loc not in world_drop:
    outside.append(loc)

#add column 'in US' to df
in_US = []
for loc in df['location']:
  # print(loc)
  if loc in inside:
    in_US.append(1)
  elif loc in outside:
    in_US.append(0)
  else:
    in_US.append('NA')

df['in_US']=in_US
df=df[df['in_US']!='NA']

print('Tweets are classified')

#drop duplicates
df=df.drop_duplicates('text')
print('Dataset size is ', df.shape[0])


#translate into English
#If it's in English
print('Detecting tweets\' languages...')
eng=[]
for text in df['text']:
  if len(text)>5:
    try:
      
      if detect(text)=='en':
        eng.append(1)
      else:
        eng.append(0)
    except:
      eng.append(3)
      continue
  else:
      eng.append(2)
      
df['is_in_eng']=eng

print('Languages of tweets are detected ')
#Translate
print('Translating tweets...')
text_en=[]

i=0
for text in tqdm(df['text']):
  
  # ind=df[df['text']==text].index
  # k=int(df['is_in_eng'].iloc[ind])
  # print(k)
  k=df['is_in_eng'].iloc[i]
  
  if k!=1:
    try:
      text_blob = TextBlob(text)
      text_blob = text_blob.translate(to='en')
      # print(text_blob)
      time.sleep(5)
      text_en.append(text_blob)
    except:
      text_en.append(text)
      continue
    
  else:
    text_en.append(text)
 
  i+=1

df['text_eng']=text_en
print('Saving tweets...')

#Save
df.to_csv(outfile, index=False)



# cnt = 0
# for line in con:
#     if (len(line)<=5): continue

#     cnt+=1
#     obj = json.loads(line.replace('\n',''))

#     text = obj.get('text')
#     text = text.replace('\t', '').replace('\n', '')

#     outfile.write(obj.get('id_str')+'\t'+obj.get('created_at')+'\t'+text+'\n')
#     if (cnt>1000): break

print('done.')
