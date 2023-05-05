import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import re
# read in the two CSV files

df=pd.read_csv('events.out.tfevents.1683066157.iGpu25.csv')

tags = df['tag'].unique()
Otags=[]
for i in tags:
    if 'sampler' not in i and 'len' not in i:
        selected_df = df[df['tag'] == i]
        selected_df.to_csv(i.split('/')[-1]+'.csv',index=False)

