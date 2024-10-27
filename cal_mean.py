import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt

base_path = '/lab/tmpig10f/kiran/expert_3chan_beogym/skill2/'

games = ['expert_3chan_allegheny', 'expert_3chan_hudsonriver', 'expert_3chan_unionsquare', 'expert_3chan_CMU', 'expert_3chan_southshore', 'expert_3chan_wallstreet']


names={'expert_3chan_allegheny': 'Allegheny', 'expert_3chan_hudsonriver':'HudsonRiver', 'expert_3chan_unionsquare':'UnionSquare', 'expert_3chan_CMU':'CMU', 'expert_3chan_southshore':'SouthShore', 'expert_3chan_wallstreet':'WallStreet'}
data={}
for game in games:



    game_path = os.path.join(base_path, game+'/5/50/')
    ter = np.load(game_path+'terminal.npy')

    ter_indices = np.where(ter == 1)[0]
    res=[]
    # for i in range(len(ter_indices)-1):
    #     res.append(ter_indices[i+1] - ter_indices[i])
    result_array = [ter_indices[i+1] - ter_indices[i] for i in range(len(ter_indices) - 1)]
    print(names[game])
    print(np.min(result_array))
    print(np.max(result_array))
    print(np.mean(result_array))
    print(np.std(result_array))
    data[names[game]] = result_array
    # plt.bar(range(len(res)), res, yerr=np.std(res), capsize=5, label=names[game])
import pandas as pd
df = pd.DataFrame.from_dict(data, orient='index').transpose()
plt.figure(figsize=(10, 6))
sns.barplot(data=df, errorbar='sd') 
# sns.barplot(data, errorbar=('ci', 95))

plt.savefig(f'./beo_steps.png')
