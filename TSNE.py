import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from torch.autograd import Variable
from atari_vae import TEncoder
import os, random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class TEncoder(nn.Module):
    def __init__(self, channel_in=3, ch=16, z=64, h_dim=512):
        super(TEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.ZeroPad2d((2, 2, 2, 2)),
            nn.Conv2d(channel_in, ch, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(ch, ch*2, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(ch*2, ch*32, kernel_size=(11, 11), stride=(1, 1)),
            nn.ReLU(),
        )

        self.conv_mu = nn.Conv2d(ch*32, z, 1, 1)
        
    def forward(self, x):
        #print(torch.max(x))
        x = self.encoder(x)
        x = self.conv_mu(x)
        x = torch.flatten(x, start_dim=1)
        return x


emb = TEncoder(channel_in=1, ch=32, z=512)

checkpoint = torch.load('/lab/kiran/ckpts/pretrained/atari/1CHAN_NVEP_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_2.0_-1.0_2_nsame_triplet_32_2_0.0001_1.0.pt',map_location='cuda')
emb.load_state_dict(checkpoint['model_state_dict'])
emb.to('cuda')
emb.eval()

base_path = '/lab/tmpig10f/kiran/expert_1chan_atari/'
games=['expert_1chan_airraid', 'expert_1chan_beamrider', 'expert_1chan_carnival', 'expert_1chan_demonattack', 'expert_1chan_namethisgame', 'expert_1chan_phoenix']
for game in games:
    game_path = os.path.join(base_path, game+'/5/50/')
    obss = np.load(game_path+'observation', mmap_mode='r')
    ter = np.load(game_path+'terminal_truncated.npy')
    ter_indices = np.where(ter == 1)[0]
    rdm = random.sample(range(0,len(ter_indices)-1), 5)
    all_embedding=[]
    all_idx=[0]
    for i in rdm:
        obs = torch.tensor(np.asarray(obss[ter_indices[i]+1 : ter_indices[i+1]])).to('cuda').unsqueeze(1)
        embeddings = emb(obs.to(torch.float32)).squeeze().cpu().detach().numpy()
        all_embedding.append(embeddings)
        all_idx.append(all_idx[-1]+embeddings.shape[0])
    all_embedding=np.concatenate(all_embedding, axis=0)
    tsne = TSNE(n_components=2)
    new_embeddings = tsne.fit_transform(all_embedding)

    for q in range(len(all_idx)-1):
        plt.scatter(new_embeddings[all_idx[q]:all_idx[q+1], 0], new_embeddings[all_idx[q]:all_idx[q+1], 1], c=np.arange(all_idx[q+1]-all_idx[q]))
    cbar = plt.colorbar()
    cbar.set_label('Color Value')
    plt.title(f't-SNE Visualization of 512-dimensional Embeddings ({game.split("_")[-1]})')
    plt.savefig(f'./TSNE_embedding/{game.split("_")[-1]}.png')
    plt.clf()