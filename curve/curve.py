import numpy as np
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from envs import SingleAtariEnv
import torch
from ray.rllib.policy.sample_batch import SampleBatch
import cv2,ray
from models.atarimodels import SingleAtariModel, SharedBackboneAtariModel, SharedBackbonePolicyAtariModel
import tree
from matplotlib import pyplot as plt
#from envs import SingleAtariEnv
#from arguments import get_args
#from IPython import embed
import sys,os,random

ModelCatalog.register_custom_model("model", SingleAtariModel)

data_path = sys.argv[1]

game = data_path.split('_')[-1]
game = game.split('/')[0]
encodernet_e2e = Policy.from_checkpoint('/lab/kiran/logs/rllib/atari/notemp/1.a_DemonAttackNoFrameskip-v4_singlegame_full_e2e_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_11_15_10_22/checkpoint/')
encodernet_random = Policy.from_checkpoint('/lab/kiran/logs/rllib/atari/notemp/1.a_DemonAttackNoFrameskip-v4_singlegame_full_random_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_14_19_15_21/checkpoint/')
encodernet_SOM = Policy.from_checkpoint('/lab/kiran/logs/rllib/atari/notemp/1.a_DemonAttackNoFrameskip-v4_singlegame_full_1CHAN_SOM_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_150_0.1_32_32_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_12_14_10_18/checkpoint/')
encodernet_TCN = Policy.from_checkpoint('/lab/kiran/logs/rllib/atari/notemp/1.a_DemonAttackNoFrameskip-v4_singlegame_full_1CHAN_TCN_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_4.0_0.01_32_32_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_12_11_47_47/checkpoint/')
encodernet_VIP = Policy.from_checkpoint('/lab/kiran/logs/rllib/atari/notemp/1.a_DemonAttackNoFrameskip-v4_singlegame_full_1CHAN_VIP_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_40.0_0.01_32_0_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_12_15_22_15/checkpoint/')
encodernet_VEP = Policy.from_checkpoint('/lab/kiran/logs/rllib/atari/notemp/1.a_DemonAttackNoFrameskip-v4_singlegame_full_1CHAN_VEP_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_10.0_0.1_0.01_32_32_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_13_18_31_21/checkpoint/')

cnn_e2e = encodernet_e2e.model._convs
cnn_random = encodernet_random.model._convs
cnn_SOM = encodernet_SOM.model._convs
cnn_TCN = encodernet_TCN.model._convs
cnn_VIP = encodernet_VIP.model._convs
cnn_VEP = encodernet_VEP.model._convs

models={'e2e':cnn_e2e, 'random': cnn_random, 'SOM':cnn_SOM, 'TCN':cnn_TCN, 'VIP':cnn_VIP, 'VEP':cnn_VEP}

obs_path = os.path.join(data_path,'observation')
terminal_path = os.path.join(data_path,'terminal')
reward_path = os.path.join(data_path,'reward')


obs = np.load(obs_path,allow_pickle=True)
ter = np.load(terminal_path,allow_pickle=True)
rew = np.load(reward_path,allow_pickle=True)

ter[-1]=1

ter = (ter == 1) | (rew == 1)
ter = ter.astype(int)
indices = np.where(ter == 1)
obss = []
start_idx = 0
for idx in indices[0]:
    obss.append(obs[start_idx:idx+1])
    start_idx = idx+1

print(len(obss))

example = []
while True:
    temp = random.choice(obss)
    if len(temp)>=40 and len(temp)<=60:
        example = temp
        break
    
goal = example[-1]
goal_embedding={}
distances = {}
with torch.no_grad():
    for key,value in models.items():
        goal_embedding[key] = value(torch.tensor(goal, dtype=torch.float32).view(1, 84, 84).to('cuda')).cpu().numpy()
        distances[key] = []
for i in example:
    for key,value in models.items():
        cur_embedding = value(torch.tensor(i, dtype=torch.float32).view(1, 84, 84).to('cuda'))
        with torch.no_grad():
            distances[key].append(np.linalg.norm(goal_embedding[key]-cur_embedding.cpu().numpy()))

fig, ax = plt.subplots(figsize=(12,6))

        # Plot VIP Embedding Distance and Goal Image
for key,value in distances.items():      
    ax.plot(np.arange(len(value)), value, label=key, linewidth=3)

ax.legend(loc="upper right")
ax.set_xlabel("Frame", fontsize=15)
ax.set_ylabel("Embedding Distance", fontsize=15)
ax.set_title(f"VEP Embedding Distance", fontsize=15)

if not os.path.exists('./embedding_curves/'):
    os.makedirs('./embedding_curves/')
plt.savefig(f"./embedding_curves/{game}.png")
plt.close()

# ax0_xlim = ax[0].get_xlim()
# ax0_ylim = ax[0].get_ylim()
# ax1_xlim = ax[1].get_xlim()
# ax1_ylim = ax[1].get_ylim()

# env = SingleAtariEnv({'env': 'SpaceInvadersNoFrameskip-v4', 'full_action_space': False, 'framestack': False})

# input_dict = env.reset()

# print(cnn(torch.tensor(input_dict, dtype=torch.float32).view(1, 84, 84).to('cuda')))


# input_dict = tree.map_structure_with_path(lambda p, s: (s if p == "seq_lens" else s.unsqueeze(0) if torch and isinstance(s, torch.Tensor) else np.expand_dims(s, 0)), input_dict,)

# print(cnn(SampleBatch(input_dict)))
# obs_np = []
# act_np = []
# rew_np = []
# done_np = []

# count = 0
# for i in range(rounds):
#     reward = 0.0
#     done = False
#     total=0
#     obs = env.reset()
#     for q in range(1000):
#         action = encodernet.compute_single_action(obs)[0]
        
#         obs_np.append(obs)
        
#         obs, reward, done, _ = env.step(action)
        
#         act_np.append(action)
#         rew_np.append(reward)
#         done_np.append(done)

#         total += reward
#         if done:
#             break

#     res.append(total)

# average = sum(res) / len(res)
# print(average)


# import cv2

# output_file = 'output_video.mp4'
# fps = 30
# frame_size = (84, 84)  # Size of each frame

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
# out = cv2.VideoWriter(output_file, fourcc, fps, frame_size,isColor=False)
# frames = obs_np
# for idx in range(len(frames)):
#     frame = frames[idx]
#     frame = np.uint8(frame)  # Ensure frame data type is uint8
#     out.write(frame)

# out.release()

# cv2.destroyAllWindows()


