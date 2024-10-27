import pandas as pd

df = pd.read_csv('exp_logs - Atari.csv')

result = df[df['uid'] == 'wbvc4ymw']
# result =[i for i in result.filter(like='Run').values.flatten() if type(i)==str]
result = result['Game'].iloc[0]
print(result)