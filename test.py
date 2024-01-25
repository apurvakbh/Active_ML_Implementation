import os
import tqdm
import pandas as pd
import numpy as np

positive_dir = "user_data/review_polarity/txt_sentoken/pos"
negative_dir = "user_data/review_polarity/txt_sentoken/neg"

def read_text(filename):
        with open(filename) as f:
                return f.read().lower()

print ("Reading negative reviews.")
negative_text = [read_text(os.path.join(negative_dir, filename))
        for filename in tqdm.tqdm(os.listdir(negative_dir))]
        
print ("Reading positive reviews.")
positive_text = [read_text(os.path.join(positive_dir, filename))
        for filename in tqdm.tqdm(os.listdir(positive_dir))]

labels_index = { "negative": 0, "positive": 1 }

labels = [0 for _ in range(len(negative_text))] + \
        [1 for _ in range(len(positive_text))]
 
texts = negative_text + positive_text

df=pd.DataFrame()
df["text"]=texts
df["labels"]=labels

df.to_csv("user_data/review_polarity/full_dataset.csv",index=False)