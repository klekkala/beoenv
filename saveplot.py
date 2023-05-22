import os
from utils import plot_util as pu

LOG_ROOT = 'logs/'
PLOT_ROOT = 'plots/'

def plot_dir(log_dir, env_name, evalflag=False):
    if not evalflag and 'eval' in env_name:
        return
    if not os.path.exists(PLOT_ROOT + env_name):
        os.mkdir(PLOT_ROOT + env_name)
    
    results = pu.load_results(log_dir)
    fig, _ = pu.plot_results(results, average_group=True, split_fn=lambda _: '', shaded_std=False)
    fig.savefig(PLOT_ROOT + env_name + '/result.png')



all_dirs = os.listdir(LOG_ROOT)

for each_dir in all_dirs:
    plot_dir(LOG_ROOT + each_dir, each_dir)




"""Sumedh's code
import numpy as np
import csv
import seaborn as sns
import pandas as pd
from itertools import chain
import matplotlib.pyplot as plt
data1 = {"rewards":[], "Episodes x 10":[], "type":[]}
num_episodes = 5
seed_start = 0
seed_end = 8
explorer = 'revd'
window = 2

for i in range(10,10+seed_end):
	print(i)
	games_df = pd.read_csv(f'/lab/ssontakk/S3D_HowTo100M/cem_planning/metaworld/door-close-v2-goal-hidden_dense_original_string_finetuned_repeated/{i}.monitor.csv', skiprows=1, header = 0)
	games_df['r'] = games_df['r'].rolling(window=window).mean()
	# sumRew = []
	for j in range(num_episodes):
		print(i,j)
		data1["rewards"].append(games_df['r'].iloc[j])
		data1["Episodes x 10"].append(j)
		data1["type"].append(f"String Pretrained (ours)")
for i in range(10,10+seed_end):
	print(i)
	games_df = pd.read_csv(f'/lab/ssontakk/S3D_HowTo100M/cem_planning/metaworld/door-close-v2-goal-hidden_dense_original_video_finetuned_repeated/{i}.monitor.csv', skiprows=1, header = 0)
	games_df['r'] = games_df['r'].rolling(window=window).mean()
	# sumRew = []
	for j in range(num_episodes):
		print(i,j)
		data1["rewards"].append(games_df['r'].iloc[j])
		data1["Episodes x 10"].append(j)
		data1["type"].append(f"Video Pretrained (ours)")

for i in range(10,10+seed_end):
	print(i)
	games_df = pd.read_csv(f'/lab/ssontakk/S3D_HowTo100M/cem_planning/metaworld/button-press-v2-goal-hidden_dense_original_human_finetuned_repeated/{i}.monitor.csv', skiprows=1, header = 0)
	games_df['r'] = games_df['r'].rolling(window=window).mean()
	# sumRew = []
	for j in range(num_episodes):
		print(i,j)
		data1["rewards"].append(games_df['r'].iloc[j])
		data1["Episodes x 10"].append(j)
		data1["type"].append(f"Human Video Pretrained (ours)")
for i in range(10,10+seed_end):
	games_df = pd.read_csv(f'/lab/ssontakk/S3D_HowTo100M/cem_planning/metaworld/button-press-v2-goal-hidden_gail/{i}.monitor.csv', skiprows=1, header = 0)
	games_df['r'] = games_df['r'].rolling(window=window).mean()
	# sumRew = []
	for j in range(num_episodes):
		print(i,j)
		data1["rewards"].append(games_df['r'].iloc[j])
		data1["Episodes x 10"].append(j)
		data1["type"].append(f"GAIL")

for i in range(10,10+seed_end):
	games_df = pd.read_csv(f'/lab/ssontakk/S3D_HowTo100M/cem_planning/metaworld/button-press-v2-goal-hidden_airl/{i}.monitor.csv', skiprows=1, header = 0)
	games_df['r'] = games_df['r'].rolling(window=window).mean()
	# sumRew = []
	for j in range(num_episodes):
		print(i,j)
		data1["rewards"].append(games_df['r'].iloc[j])
		data1["Episodes x 10"].append(j)
		data1["type"].append(f"AIRL")

for i in range(10,10+seed_end):
	games_df = pd.read_csv(f'/lab/ssontakk/S3D_HowTo100M/cem_planning/metaworld/button-press-v2-goal-hidden_dense_original/{i}.monitor.csv', skiprows=1, header = 0)
	games_df['r'] = games_df['r'].rolling(window=window).mean()
	# sumRew = []
	for j in range(num_episodes):
		print(i,j)
		data1["rewards"].append(games_df['r'].iloc[j])
		data1["Episodes x 10"].append(j)
		data1["type"].append(f"Dense Task Reward only")




sns.set_theme(style="darkgrid")
sns_plot = sns.lineplot(data=data1,x="Episodes x 10",y="rewards",hue="type")
# plt.legend(loc='lower right', borderaxespad=0)
# # plt.ylim(-5, 10)

# # sns_plot.set(xscale="log")
# # sns_plot = sns.lineplot(data=data,x="t",y="rewards")

sns_plot.figure.savefig(f"paper_figures/button-press-human.png")

# sns_plot.figure.savefig("PPO-Ihlen_0_int-TR.png")
# print(games_df)
"""