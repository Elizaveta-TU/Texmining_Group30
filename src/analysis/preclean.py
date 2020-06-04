import pandas as pd
 
import os


os.makedirs('../../gen/analysis/temp/', exist_ok=True)
os.makedirs('../../gen/analysis/output/', exist_ok=True)
os.makedirs('../../output_figures/', exist_ok=True)

df = pd.read_csv('../../gen/data-preparation/output/dataset.csv')


# # tag retweets
# dt[, retweet:=FALSE]
# dt[grepl('^RT', text), retweet:=TRUE]

# dir.create('../../gen/analysis/temp/', recursive = TRUE)
# dir.create('../../gen/analysis/output/', recursive = TRUE)
df.to_csv('../../gen/analysis/temp/preclean.csv', index=False)



 