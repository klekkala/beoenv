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
