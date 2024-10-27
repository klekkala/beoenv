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
import torch.nn.init as init
from mpl_toolkits.mplot3d import Axes3D


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
    
    def initialize_weights(self):
        for m in self.modules():
            print(m)
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # Initialize weights using a random initialization method
                # For example, you can use the Xavier/Glorot initialization
                init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    # Initialize biases to zero
                    init.constant_(m.bias.data, 0.0)

class TBeoEncoder(nn.Module):
    def __init__(self, channel_in=3, ch=16, z=64, h_dim=512, div=1.0):
        super(TBeoEncoder, self).__init__()
        self.div = div
        assert(self.div == 255.0 or self.div == 1.0)
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
        self.joint_layer = nn.Linear(in_features=514, out_features=512, bias=True)

    def forward(self, x, aux):
        x = torch.moveaxis(x, -1, 1)
        x = x/self.div
        #print(torch.max(x))
        x = self.encoder(x)
        x = self.conv_mu(x)
        x = torch.flatten(x, start_dim=1)
        
        #concat aux
        x = torch.concat((x, aux), axis=1)
        x = self.joint_layer(x)
        x = torch.flatten(x, start_dim=1)
        return x

    def initialize_weights(self):
        for m in self.modules():
            print(m)
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # Initialize weights using a random initialization method
                # For example, you can use the Xavier/Glorot initialization
                init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    # Initialize biases to zero
                    init.constant_(m.bias.data, 0.0)


emb = TEncoder(channel_in=1, ch=32, z=512)
# emb = TBeoEncoder(channel_in=3, ch=32, z=512)
# emb.initialize_weights()
checkpoint = torch.load('/lab/kiran/ckpts/pretrained/atari/1CHAN_NVEP_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_2.0_-1.0_2_nsame_triplet_32_2_0.0001_1.0.pt',map_location='cuda')
# checkpoint = torch.load('/lab/kiran/ckpts/pretrained/beogym/3CHAN_SOM_BEOGYM_EXPERT_3CHAN_ALLUNIONWALL_STANDARD_0.9_triplet_32_32_0.0001_1.pt',map_location='cuda')
emb.load_state_dict(checkpoint['model_state_dict'])
emb.to('cuda')
emb.eval()

base_path = '/lab/tmpig10f/kiran/expert_1chan_atari/'
# base_path = '/lab/tmpig10f/kiran/expert_3chan_beogym/skill2/'
games=['expert_1chan_airraid', 'expert_1chan_beamrider', 'expert_1chan_carnival', 'expert_1chan_demonattack', 'expert_1chan_namethisgame', 'expert_1chan_phoenix']
games=['expert_1chan_demonattack', 'expert_1chan_spaceinvaders']
games=['expert_1chan_demonattack', 'expert_1chan_beamrider']
# games=['expert_3chan_unionsquare', 'expert_3chan_wallstreet']
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')








for game in games:


    game_path = os.path.join(base_path, 'expert_1chan_demonattack'+'/5/50/')
    obss = np.load(game_path+'observation', mmap_mode='r')
    # auxs = np.load(game_path+'aux.npy')
    ter = np.load(game_path+'terminal_truncated.npy')
    val = np.load(game_path+'value_truncated.npy')
    ter_indices = np.where(ter == 1)[0]
    rdm=[10, 500, 5000, 10000, 20000, 21000]
    all_embedding=[]
    values=[]
    for i in rdm:
        obs = torch.tensor(np.asarray(obss[ter_indices[i]+1 : ter_indices[i+1]])).to('cuda').unsqueeze(1)
        values.append(np.asarray(val[ter_indices[i]+1 : ter_indices[i+1]]))
        embeddings = emb(obs.to(torch.float32)).squeeze().cpu().detach().numpy()
        all_embedding.append(embeddings)
    all_embedding=np.concatenate(all_embedding, axis=0)
    values=np.concatenate(values, axis=0)
    tsne = TSNE(n_components=2)
    new_embeddings = tsne.fit_transform(all_embedding)
    plt.scatter(new_embeddings[:, 0], new_embeddings[:, 1], c=values, marker='o', label='DemonAttack')




    game_path = os.path.join(base_path, game+'/5/50/')
    obss = np.load(game_path+'observation', mmap_mode='r')
    # auxs = np.load(game_path+'aux.npy')
    ter = np.load(game_path+'terminal_truncated.npy')
    val = np.load(game_path+'value_truncated.npy')
    ter_indices = np.where(ter == 1)[0]
    # rdm = random.sample(range(0,len(ter_indices)-1), 50)
    rdm=[10, 500, 5000, 10000, 20000, 21000]
    # rdm=[5000,20000]
    # rdm=[20000]
    all_embedding=[]
    values=[]
    for i in rdm:

        obs = torch.tensor(np.asarray(obss[ter_indices[i]+1 : ter_indices[i+1]])).to('cuda').unsqueeze(1)
        # obs = torch.tensor(np.asarray(obss[ter_indices[i]+1 : ter_indices[i+1]])).to('cuda')
        # aux = torch.tensor(np.asarray(auxs[ter_indices[i]+1 : ter_indices[i+1]])).to('cuda')
        values.append(np.asarray(val[ter_indices[i]+1 : ter_indices[i+1]]))
        # embeddings = emb(obs.to(torch.float32), aux.to(torch.float32)).squeeze().cpu().detach().numpy()
        embeddings = emb(obs.to(torch.float32)).squeeze().cpu().detach().numpy()
        all_embedding.append(embeddings)
    all_embedding=np.concatenate(all_embedding, axis=0)
    values=np.concatenate(values, axis=0)
    tsne = TSNE(n_components=2)
    new_embeddings = tsne.fit_transform(all_embedding)
    if game == 'expert_1chan_demonattack':
        # ax.scatter(new_embeddings[:, 0], new_embeddings[:, 1], new_embeddings[:, 2], c=values, marker='o', label='DemonAttack')
        plt.scatter(new_embeddings[:, 0], new_embeddings[:, 1], c=values, marker='o', label='DemonAttack')
        # plt.plot(new_embeddings[:, 0], new_embeddings[:, 1], linestyle='-')

    else:
        # ax.scatter(new_embeddings[:, 0], new_embeddings[:, 1], new_embeddings[:, 2], c=values, marker='x', s= 100, linewidths=3, label='SpaceInvaders')
        plt.scatter(new_embeddings[:, 0], new_embeddings[:, 1], c=values, marker='x', s= 100, linewidths=3, label='BeamRider')
        # plt.plot(new_embeddings[:, 0], new_embeddings[:, 1], linestyle='-')
        cbar = plt.colorbar()
        cbar.set_label('value estimate')
    plt.legend()
    plt.title(f'VEP')
    plt.savefig(f'./TSNE_embedding/submit/VEP_beamrider.png')
    # plt.clf()