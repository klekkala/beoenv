import csv
import os
import sys
from glob import glob
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

def smooth(scalars, weight):
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  
        smoothed.append(smoothed_val)
        last = smoothed_val
        
    return smoothed

def extract_scalar_events(event_file):
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    scalar_data = {}
    for tag in event_acc.Tags()['scalars']:
        if tag == 'ray/tune/episode_reward_mean':
        # if 'mean' in tag:
            scalar_data[tag] = event_acc.Scalars(tag)

    return scalar_data



if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <txt>')
        sys.exit(1)

    all_dir = sys.argv[1]
    output_dir = './submit/'+all_dir[:-4]+'.png'
    

    dir_env = all_dir.split('/')[-1]
    env=''
    exp=''
    beo = ['south', 'wall', 'hudson', 'cmu', 'union', 'alleg']
    ata = ['beam', 'car', 'demo', 'space', 'phoenix', 'air']
    for i in beo:
        if i in dir_env.lower():
            env='beogym'
    for i in ata:
        if i in dir_env.lower():
            env='atari'
    assert env!=''
    if 'e2e' in dir_env.lower():
        exp = 'e2e'
    if 'prtr' in dir_env.lower():
        exp = 'prtr'
    assert env!=''
    game_list={}

    with open(all_dir,'r') as f:
        for line in f:
            game_list[line.strip().split(':')[0].replace("'",'').replace(" ",'')] = line.strip().split(':')[1].replace(' ','')
    if env == 'atari':
        df = pd.read_csv('exp_logs - Atari.csv')
    else:
        df = pd.read_csv('exp_logs - Beogym.csv')
    for key,value in game_list.items():
        game_list[key] =[i for i in df[df['uid'] == value].filter(like='Run').values.flatten() if type(i)==str and i!='??' and i!='**' and i!='None']
        if type(df[df['uid'] == value]['Game'].iloc[0]) == str:
            game_name=df[df['uid'] == value]['Game'].iloc[0]
            assert game_name!=''
        
        game_list[key] = [game_name+'/'+df[df['uid'] == value]['Method'].iloc[0]+'/'+path for path in game_list[key]]
        


    if env == 'atari':
        data_path = '/lab/kiran/logs/rllib/atari/notemp/'
    else:
        data_path = '/lab/kiran/logs/rllib/beogym/notemp/'

    # data_path = '/lab/kiran/logs/rllib/atari/notemp/'
    #data_path = '/lab/kiran/logs/rllib/beogym/notemp/'
    max_out = 0
    for name,i in game_list.items():
        avg_time=[]
        avg_value=[]
        event_files=[]
        for log_fine in i:
            game_dir = os.path.join(data_path, log_fine)
            # print(os.path.join(os.path.join(game_dir,q), 'events.out.tfevents.*'))
            # print(glob(os.path.join(os.path.join(game_dir,q), 'events.out.tfevents.*')))
            # print('?')
            event_files+=glob(os.path.join(game_dir, 'events.out.tfevents.*'))
        game_dict={}
        times=[]
        values=[]
        max_len=0

        for idx,event_file in enumerate(event_files):
            scalar_data = extract_scalar_events(event_file)
            tmp_time=[]
            tmp_value=[]
            for tag,events in scalar_data.items():
                for event in events:
                    if exp=='e2e':
                        tmp_time.append(event.wall_time)
                    else:
                        tmp_time.append(event.step)
                    tmp_value.append(event.value)
            if exp=='e2e':
                tmp_time = [x-tmp_time[0] for x in tmp_time]
            times+=tmp_time
            # values+=smooth(tmp_value, 0.1)
            values+=tmp_value
            max_len = max(tmp_time) if max(tmp_time)>= max_len else max_len
        # if 'VEP' in event_files[0]:
        #     event_files = [event_files[0]]+event_files[2:]
        cs = {'E2E':'grey', 'TCN+':'#2ca02c', 'TCN':'#2ca02c','VEP':'#ff7f0e', 'VIP':'#9467bd', 'SOM':'#d62728', 'RANDOM':'#1f77b4'}
        if exp=='e2e':
            suitable = round(max_len/200)
            times=[round(number / suitable) * suitable for number in times]
            times=[i/60 for i in times]
            max_len = round(max_len / suitable)*suitable/60
            max_out = max_len if max_len>max_out else max_out
            dot_times=[]
            dot_values=[]
            # max_right = max(times) if max(times)>max_right else max_right
            if max(times)<max_out:
                max_indices = [i for i, v in enumerate(times) if v == max(times)]
                max_v = [values[i] for i in max_indices]
                mean_v = sum(max_v)/len(max_v)
                while max(times+dot_times)<max_out:
                    dot_times.append(max(times+dot_times)+suitable/50)
                    dot_values.append(mean_v)
            data={'Compute Time (in minutes)':times, 'Reward':values}    
            data = pd.DataFrame(data)
            ax = sns.lineplot(data=data,x='Compute Time (in minutes)', y='Reward', color=cs[name])
            ax.plot(dot_times, dot_values, linestyle='--', color=ax.get_lines()[-1].get_color())
            ax.set_xlabel('Compute Time (in minutes)', fontsize=20)
            ax.set_ylabel('Reward', fontsize=20)
            ax.plot()
        else:
            suitable = round(max_len/200)
            times=[round(number / suitable) * suitable for number in times]
            data={'Iterations':times, 'Reward':values}    
            data = pd.DataFrame(data)
            ax = sns.lineplot(data=data,x='Iterations', y='Reward', color=cs[name])
            ax.set_xlabel('Iterations', fontsize=20)
            ax.set_ylabel('Reward', fontsize=20)
        fig = ax.get_figure()
    # out = os.path.join(output_dir,f'{game_name}.png')
    fig.savefig(output_dir)

        # fig.clf()
        # plt.close()
        # fig.close()
        # ax.close()        
    # convert_tensorboard_to_csv(tensorboard_dir, output_dir)
