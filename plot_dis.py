import numpy as np

import os
import random
import matplotlib.pyplot as plt

base_path = '/lab/tmpig10f/kiran/expert_3chan_beogym/skill2/'

games = ['expert_3chan_allegheny', 'expert_3chan_hudsonriver', 'expert_3chan_unionsquare', 'expert_3chan_CMU', 'expert_3chan_southshore', 'expert_3chan_wallstreet']

from PIL import Image

names={'expert_3chan_allegheny': 'Allegheny', 'expert_3chan_hudsonriver':'HudsonRiver', 'expert_3chan_unionsquare':'UnionSquare', 'expert_3chan_CMU':'CMU', 'expert_3chan_southshore':'SouthShore', 'expert_3chan_wallstreet':'WallStreet'}

for game in games:


    rdm=[10, 500, 5000, 6000, 7000, 8000, 9000, 10000, 15000, 16000, 16500]
    new=[]
    for i in range(1,20):
        new+=[q+i for q in rdm]
    rdm=new


    game_path = os.path.join(base_path, game+'/5/50/')
    auxs = np.load(game_path+'aux.npy')
    ter = np.load(game_path+'terminal_truncated.npy')

    ter_indices = np.where(ter == 1)[0]
    dx=[]
    dy=[]
    for i in rdm:
        aux = auxs[ter_indices[i]+1 : ter_indices[i+1]]
        dx.append((aux[-1][0]-aux[0][0])*400)
        dy.append((aux[-1][1]-aux[0][1])*400)

    plt.scatter(dx, dy, marker='o',s=5, label=names[game])

plt.legend()
plt.title(f'Beogym distribution')
plt.savefig(f'./beo_dis.png')
