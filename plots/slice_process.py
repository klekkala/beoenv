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
# df1 = pd.read_csv('part7.csv')
# df2 = pd.read_csv('events.out.tfevents.1682858781.iGpu25.csv')
#
# # add 1000000 to the step column of the second file
# df2['step'] += 80000000
#
# # concatenate the two dataframes
# result = pd.concat([df1, df2])
# df1.to_csv('final.csv',index=False)
