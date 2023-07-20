from tTc import convert_tensorboard_to_csv
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from plot import plot_csv_data

games = [d for d in os.listdir() if os.path.isdir(d) and '-' in d]
files=[]
scaler = MinMaxScaler(feature_range=(0, 1000))
for i in games:
    temp = convert_tensorboard_to_csv(f'./{i}', './res')
    files+=temp

df1=pd.read_csv(files[0])
curr_step=df1['step'].max()
scaler.fit(df1[['value']])
df1['value'] = scaler.transform(df1[['value']])
print(files)
for i in range(1,len(files)):
    df2=pd.read_csv(files[i])
    maxS=df2['step'].max()
    df2['step'] += curr_step
    curr_step += maxS
    scaler.fit(df2[['value']])
    df2['value'] = scaler.transform(df2[['value']])
    df1 = pd.concat([df1, df2])


df1.to_csv('final.csv',index=False)


plot_csv_data('./final.csv', './final.png')

