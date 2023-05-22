import csv
import os
import sys
from glob import glob

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_scalar_events(event_file):
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    scalar_data = {}
    for tag in event_acc.Tags()['scalars']:
        #if tag == 'ray/tune/episode_reward_mean':
        if 'mean' in tag:
            scalar_data[tag] = event_acc.Scalars(tag)

    return scalar_data

def write_csv(scalar_data, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['step', 'wall_time', 'tag', 'value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for tag, events in scalar_data.items():
            for event in events:
                row = {
                    'step': event.step,
                    'wall_time': event.wall_time,
                    'tag': tag,
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

    tensorboard_dir = sys.argv[1]
    output_dir = sys.argv[2]

    convert_tensorboard_to_csv(tensorboard_dir, output_dir)
