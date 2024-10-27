import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
import re

def process_image(img_path):
    img = Image.open(img_path)
    img_array = np.array(img)
    return img_array

def numerical_sort(value):
    # Extract the numeric part of the file name
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else value


def load_images_from_folder(folder):
    image_paths = [os.path.join(folder, f) for f in sorted(os.listdir(folder), key=numerical_sort) if os.path.isfile(os.path.join(folder, f))]
    num_images = len(image_paths)

    # Prepare the terminal and reward arrays
    terminal = np.zeros(num_images, dtype=int)
    reward = np.zeros(num_images, dtype=int)
    terminal[-1] = 1
    reward[-1] = 1

    # Load images in parallel
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(process_image, image_paths))
    img = Image.fromarray(images[-1])
    img.save(f'/lab/kiran/beoenv/fksb/{os.path.normpath(folder).split(os.sep)[-2]}.jpg')

    return np.array(images), terminal, reward


def concatenate_images_from_episodes(main_folder):
    all_episodes_images = []
    all_episodes_terminal = []
    all_episodes_reward = []
    for sto in ['tmpig10c/colosseum_1m', 'tmpig10c/colosseum_1m_1', 'tmpig10c/colosseum_1m_2', 'tmpig10c/colosseum_1m_3', 'tmpig10c/colosseum_1m_4', 'tmpig13b/kiran/colosseum_1m', 'tmpig13d/colosseum_1m_4', 'tmpig13d/colosseum_1m_3', 'tmpig13d/colosseum_1m_2', 'tmpig13d/colosseum_1m_1', 'tmpig13d/colosseum_1m']:
        tmp_folder = main_folder[:5] + sto + main_folder[32:]
        if not os.path.exists(tmp_folder):
            continue
        for episode_folder in tqdm(sorted(os.listdir(tmp_folder))):
            episode_path = os.path.join(tmp_folder, episode_folder)
            if os.path.isdir(episode_path):
                wrist_rgb_folder = os.path.join(episode_path, 'front_rgb')
                if os.path.isdir(wrist_rgb_folder):
                    episode_images, episode_terminal, episode_reward = load_images_from_folder(wrist_rgb_folder)
                    all_episodes_images.append(episode_images)
                    all_episodes_terminal.append(episode_terminal)
                    all_episodes_reward.append(episode_reward)
    # if len(all_episodes_images)<10:
    #     print(len(all_episodes_images))
    #     return 0,0,0
    # if len(all_episodes_images)<10:
    #     print(len(all_episodes_images))
    #     return 0,0,0
    concatenated_images = np.concatenate(all_episodes_images, axis=0)
    concatenated_terminal = np.concatenate(all_episodes_terminal, axis=0)
    concatenated_reward = np.concatenate(all_episodes_reward, axis=0)
    print(main_folder)
    print(concatenated_images.shape)
    if concatenated_terminal.shape[0] < 1000000:
        return 0,0,0
        concatenated_images = np.repeat(concatenated_images, 3, axis=0)
        concatenated_terminal = np.repeat(concatenated_terminal, 3, axis=0)
        concatenated_reward = np.repeat(concatenated_reward, 3, axis=0)
    concatenated_images = concatenated_images[:1000000]
    concatenated_terminal = concatenated_terminal[:1000000]
    concatenated_reward = concatenated_reward[:1000000]
    concatenated_terminal[-1] = 1
    return concatenated_images, concatenated_terminal, concatenated_reward

def save_images_as_npy(main_folder, output_file):
    concatenated_images, concatenated_terminal, concatenated_reward = concatenate_images_from_episodes(main_folder)
    if type(concatenated_images) == int:
        print(main_folder)
        return 0
    os.makedirs(output_file, exist_ok=True)
    np.save(os.path.join(output_file, 'observation.npy'), concatenated_images)
    np.save(os.path.join(output_file, 'terminal.npy'), concatenated_terminal)
    np.save(os.path.join(output_file, 'reward.npy'), concatenated_reward)


# time.sleep(24000)
# # Usage example
# g = 'open_drawer'
# for env in os.listdir(f"/lab/tmpig13b/kiran/colosseum_1m/{g}/"):
#     main_folder = f"/lab/tmpig13b/kiran/colosseum_1m/{g}/{env}/{g}/variation0/episodes"
#     output_file = f"/lab/tmpig10f/kiran/result_1m/{g}/{env}/50/"
#     if os.path.exists(output_file):
#         continue
#     save_images_as_npy(main_folder, output_file)

# g = 'reach_and_drag'
# for env in os.listdir(f"/lab/tmpig13b/kiran/colosseum_1m/{g}/"):
#     main_folder = f"/lab/tmpig13b/kiran/colosseum_1m/{g}/{env}/{g}/variation0/episodes"
#     output_file = f"/lab/tmpig10f/kiran/result_1m/{g}/{env}/50/"
#     if os.path.exists(output_file):
#         continue
#     save_images_as_npy(main_folder, output_file)

g = 'slide_block'
for env in os.listdir(f"/lab/tmpig13b/kiran/colosseum_1m/{g}/"):
    main_folder = f"/lab/tmpig13b/kiran/colosseum_1m/{g}/{env}/slide_block_to_target/variation0/episodes"
    output_file = f"/lab/tmpig10c/result_1m/{g}/{env}/50/"
    save_images_as_npy(main_folder, output_file)
