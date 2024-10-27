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
    with open(all_dir,'r') as f:
        for line in f:
            if game_list.get(line.strip().split(',')[0].replace("'",''),'') == '':
                game_list[line.strip().split(',')[0].replace("'",'')] = [line.strip().split(',')[1].replace(' ','')]
            else:
                game_list[line.strip().split(',')[0].replace("'",'')].append(line.strip().split(',')[1].replace(' ',''))
    data_path = '/lab/kiran/logs/rllib/atari/notemp/'
    #data_path = '/lab/kiran/logs/rllib/beogym/notemp/'
    for name,i in game_list.items():
        avg_time=[]
        avg_value=[]
        for log_fine in i:
            game_dir = os.path.join(data_path, log_fine)
            event_files=[]
            # print(os.path.join(os.path.join(game_dir,q), 'events.out.tfevents.*'))
            # print(glob(os.path.join(os.path.join(game_dir,q), 'events.out.tfevents.*')))
            # print('?')
            event_files+=glob(os.path.join(game_dir, 'events.out.tfevents.*'))
            game_dict={}
            for idx,event_file in enumerate(event_files):
                scalar_data = extract_scalar_events(event_file)
                for tag,events in scalar_data.items():
                    for event in events:
                        # print(event)
                        if game_dict.get(event.wall_time,[]) == []:
                            game_dict[event.wall_time] = [-1 for i in range(len(event_files))]
                        game_dict[event.wall_time][idx] = event.step
                            
                        # game_dict[event.step] = game_dict.get(event.step,[])+[event.value]
            game_dict = dict(sorted(game_dict.items()))
            time=[]
            values=[]
            for key,value in game_dict.items():
                for q in value:
                    if q != -1:
                        time.append(key)
                        values.append(q)
            if len(time) != 1250:
                print(game_dir.split()[-1],'is not finish training, so drop it.')
            else:
                time = [x - time[0] for x in time]
                avg_time.append(time)
                avg_value.append(values)
        res_time=[]
        res_value=[]
        if len(avg_time)==0:
            print('no complete',name,'training exists')
            continue
        for time_step in range(len(avg_time[0])):
            tmp_value=[]
            tmp_time=[]
            for log_idx in range(len(avg_time)):
                tmp_time.append(avg_time[log_idx][time_step])
                tmp_value.append(avg_value[log_idx][time_step])
            res_time.append(sum(tmp_time) / len(tmp_time))
            res_value.append(sum(tmp_value) / len(tmp_value))
        data={'time':res_time, 'Step':res_value}    
        data = pd.DataFrame(data)
        ax = sns.lineplot(data=data,x='time', y='Step',label=name)
        fig = ax.get_figure()
    out = os.path.join(output_dir,f'{".".join(all_dir.split("/")[-1].split(".")[:-1])}_step.png')
    fig.savefig(out)

        # fig.clf()
        # plt.close()
        # fig.close()
        # ax.close()        
    # convert_tensorboard_to_csv(tensorboard_dir, output_dir)
