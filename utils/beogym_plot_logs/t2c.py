import csv
import os
import sys
from glob import glob
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
def extract_scalar_events(event_file):
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    scalar_data = {}
    for tag in event_acc.Tags()['scalars']:
        if tag == 'ray/tune/episode_reward_mean':
        # if 'mean' in tag:
            scalar_data[tag] = event_acc.Scalars(tag)

    return scalar_data

def write_csv(scalar_data, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['step','value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for tag, events in scalar_data.items():
            for event in events:
                row = {
                    'step': event.step,
                    # 'wall_time': event.wall_time,
                    # 'tag': tag,
                    'value': event.value
                }
                writer.writerow(row)

def convert_tensorboard_to_csv(tensorboard_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    event_files = glob(os.path.join(tensorboard_dir, 'events.out.tfevents.*'))

    for event_file in event_files:
        file_name = os.path.basename(event_file)
        output_file = os.path.join(output_dir, f'{file_name}.csv')

        scalar_data = extract_scalar_events(event_file)
        write_csv(scalar_data, output_file)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} <tensorboard_dir> <output_dir>')
        sys.exit(1)

    all_dir = sys.argv[1]
    output_dir = sys.argv[2]

    game_list={}
    with open(all_dir,'r') as f:
        for line in f:
            game_list[line.strip().split(',')[0].replace("'",'')] = line.strip().split(',')[1].replace(' ','')
    #data_path = '/lab/kiran/logs/rllib/atari/notemp/'
    data_path = '/lab/kiran/logs/rllib/beogym/notemp/'
    for name,i in game_list.items():
        game_dir = os.path.join(data_path, i)
        print(game_dir)
        event_files=[]
        for q in os.listdir(game_dir):
            event_files+=glob(os.path.join(os.path.join(game_dir,q), 'events.out.tfevents.*'))
        game_dict={}
        for idx,event_file in enumerate(event_files):
            scalar_data = extract_scalar_events(event_file)
            for tag,events in scalar_data.items():
                for event in events:
                    if game_dict.get(event.step,[]) == []:
                        game_dict[event.step] = [-1 for i in range(len(event_files))]
                    game_dict[event.step][idx] = event.value
                        
                    # game_dict[event.step] = game_dict.get(event.step,[])+[event.value]
        game_dict = dict(sorted(game_dict.items()))
        steps=[]
        values=[]
        for key,value in game_dict.items():
            for q in value:
                if q != -1:
                    steps.append(key)
                    values.append(q)
        data={'Step':steps, 'Reward':values}    
        data = pd.DataFrame(data)
        ax = sns.lineplot(data=data,x='Step', y='Reward',label=name)
        fig = ax.get_figure()
    out = os.path.join(output_dir,f'{".".join(all_dir.split("/")[-1].split(".")[:-1])}.png')
    # out = 'spaceinvaders.png'
    fig.savefig(out)
        # fig.clf()
        # plt.close()
        # fig.close()
        # ax.close()        
    # convert_tensorboard_to_csv(tensorboard_dir, output_dir)
