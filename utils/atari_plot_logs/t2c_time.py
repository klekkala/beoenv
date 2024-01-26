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
        fieldnames = ['time','value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for tag, events in scalar_data.items():
            for event in events:
                row = {
                    'time': event.time,
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
    # with open(all_dir,'r') as f:
    #     for line in f:
    #         if game_list.get(line.strip().split(',')[0].replace("'",''),'') == '':
    #             game_list[line.strip().split(',')[0].replace("'",'')] = [line.strip().split(',')[1].replace(' ','')]
    #         else:
    #             game_list[line.strip().split(',')[0].replace("'",'')].append(line.strip().split(',')[1].replace(' ',''))
    with open(all_dir,'r') as f:
        for line in f:
            game_list[line.strip().split(':')[0].replace("'",'').replace(" ",'')] = line.strip().split(':')[1].replace(' ','')

    df = pd.read_csv('exp_logs - Atari.csv')
    game_name=''
    for key,value in game_list.items():
        game_list[key] =[i for i in df[df['uid'] == value].filter(like='Run').values.flatten() if type(i)==str and i!='??' and i!='**' and i!='None']
        if type(df[df['uid'] == value]['Game'].iloc[0]) == str:
            game_name=df[df['uid'] == value]['Game'].iloc[0]

    for key,value in game_list.items():
        game_list[key] = [game_name+'/'+path for path in game_list[key]]
    assert game_name!=''
    data_path = '/lab/kiran/logs/rllib/atari/notemp/'
    #data_path = '/lab/kiran/logs/rllib/beogym/notemp/'
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
                    tmp_time.append(event.wall_time)
                    tmp_value.append(event.value)
            
            tmp_time = [x-tmp_time[0] for x in tmp_time]
            times+=tmp_time
            values+=tmp_value
            max_len = len(times) if len(times)>= max_len else max_len
        suitable = round(max_len/100)
        times=[round(number / suitable) * suitable for number in times]
        times=[i/60 for i in times]
        data={'Compute Time (in minutes)':times, 'Reward':values}    
        data = pd.DataFrame(data)
        ax = sns.lineplot(data=data,x='Compute Time (in minutes)', y='Reward',label=name)
        if max(times)<115:
            print(max(values))
            line_color = ax.lines[1].get_c()
            ax.hlines(y=452.54998779296875, xmin=114.63333333333334, xmax=412.5, colors=line_color, linestyles='dashed', label=name)
        fig = ax.get_figure()
    # out = os.path.join(output_dir,f'{game_name}.png')
    fig.savefig(output_dir)

        # fig.clf()
        # plt.close()
        # fig.close()
        # ax.close()        
    # convert_tensorboard_to_csv(tensorboard_dir, output_dir)
