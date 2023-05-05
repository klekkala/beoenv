import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# read in the two CSV files
files=['events.out.tfevents.1682822974.iGpu25.csv','events.out.tfevents.1682827445.iGpu25.csv','events.out.tfevents.1682831807.iGpu25.csv',\
       'events.out.tfevents.1682836326.iGpu25.csv','events.out.tfevents.1682840790.iGpu25.csv','events.out.tfevents.1682845246.iGpu25.csv',\
       'events.out.tfevents.1682849759.iGpu25.csv','events.out.tfevents.1682854144.iGpu25.csv','events.out.tfevents.1682858781.iGpu25.csv']

scaler = MinMaxScaler(feature_range=(0, 1000))

df1=pd.read_csv(files[0])
scaler.fit(df1[['value']])
df1['value'] = scaler.transform(df1[['value']])
for i in range(1,len(files)):
    df2=pd.read_csv(files[i])
    df2['step'] += 10000000*i
    scaler.fit(df2[['value']])
    df2['value'] = scaler.transform(df2[['value']])
    df1 = pd.concat([df1, df2])


df1.to_csv('final.csv',index=False)
