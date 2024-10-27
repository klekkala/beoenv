import numpy as np
import os
import seaborn as sns

import sys
import matplotlib.pyplot as plt
tmp_list = ['50']
pathname = '/lab/tmpig10f/kiran/expert_1chan_atari/'
pathname = '/lab/tmpig10f/kiran/expert_3chan_beogym/skill2/'
# dir_list = ['expert_1chan_airraid', 'expert_1chan_beamrider', 'expert_1chan_carnival', 'expert_1chan_phoenix', 'expert_1chan_spaceinvaders']
# dir_list = [name for name in os.listdir(pathname)]
dir_list = ['expert_3chan_allegheny', 'expert_3chan_hudsonriver', 'expert_3chan_unionsquare', 'expert_3chan_CMU', 'expert_3chan_southshore']
# demo_dir = '/lab/tmpig10f/kiran/expert_1chan_atari/expert_1chan_demonattack/5/50/value_truncated.npy'
# demo_val = np.load(demo_dir,allow_pickle=True)

demo_dir = '/lab/tmpig10f/kiran/expert_3chan_beogym/skill2/expert_3chan_wallstreet/5/50/value_truncated.npy'
demo_val = np.load(demo_dir,allow_pickle=True)

non_zero_mask = demo_val != 0
demo_val = demo_val[non_zero_mask]
# fig, axes = plt.subplots(1, len(dir_list), figsize=(15, 5))
# axidx = 0


for game in dir_list:
    print('--------------------'+game+'---------------------')
    game_path=os.path.join(pathname,game)
    game_path=os.path.join(game_path,'5')
    all_val = []
    all_act = []
    all_epi = []
    all_limit = []
    all_id = []
    for directory in tmp_list:


        file_path = os.path.join(game_path)
        print(file_path, directory)
        file_path = os.path.join(file_path,directory)
        value_path = os.path.join(file_path,'value_truncated.npy')
        
        val = np.load(value_path,allow_pickle=True)
        non_zero_mask = val != 0
        val = val[non_zero_mask]
        plt.figure(figsize=(5, 3))
        plt.hist(demo_val, bins=100,  alpha=0.5, label='wallstreet')
        plt.hist(val, bins=100, alpha=0.5, label=f'{game.split("_")[-1]}')
        # sns.kdeplot(val, label=f'{game.split("_")[-1]}', multiple="stack")
        # sns.kdeplot(demo_val, label=f'{game.split("_")[-1]}', multiple="stack")
        # ax.hist(slices_v, bins = 20, label=f'{directory}')
        # ax = axes[axidx] if len(dir_list) > 1 else axes
        # ax.hist(slices_v, bins=90)
        # ax.set_title(f'{directory}')
        # axidx+=1
        # ax1.hist(data1, bins=20)
    plt.legend()
    plt.savefig(f'./hist/submit/{game.split("_")[-1]}.png', dpi=300)
    plt.clf()

    # Close the figure to release resources
plt.close()
    # fig = plt.figure()
    # ax1 = plt.subplot2grid((3, 6), (0, 0), rowspan=1, colspan=2)  # Small histogram 1
    # ax2 = plt.subplot2grid((3, 6), (1, 0), rowspan=1, colspan=2)  # Small histogram 2
    # ax3 = plt.subplot2grid((3, 6), (2, 0), rowspan=1, colspan=2)  # Small histogram 3
    # ax4 = plt.subplot2grid((3, 6), (0, 2), rowspan=3, colspan=4) # Large histogram

    # Plot the histograms
    #ax1.hist(data1, bins=20)
    #ax2.hist(data10, bins=20)
    #ax3.hist(data50, bins=20)
    #ax4.hist(all_act, bins=20)

    # Set titles and labels
    #ax1.set_title('Histogram 1')
    #ax2.set_title('Histogram 2')
    #ax3.set_title('Histogram 3')
    #ax4.set_title('Histogram 4')

    #ax1.set_ylabel('Value')
    #ax2.set_ylabel('Value')
    #ax3.set_ylabel('Value')
    #ax4.set_ylabel('Action')
    #ax3.set_xlabel('Frequency')
    #ax4.set_xlabel('Frequency')
    #ax2.set_xlabel('Frequency')
    #ax1.set_xlabel('Frequency')

    # Adjust spacing between subplots
    #plt.tight_layout()


# Generate histograms for array1, array2, and array3
#     ax1.hist(data1, bins='auto')
#     ax2.hist(data10, bins='auto')
#     ax3.hist(data50, bins='auto')

#     # Set labels and titles for the left subplots
#     ax1.set_xlabel('Value')
#     ax1.set_ylabel('Frequency')
#     ax1.set_title('Value 1')

#     ax2.set_xlabel('Value')
#     ax2.set_ylabel('Frequency')
#     ax2.set_title('Value 10')

#     ax3.set_xlabel('Value')
#     ax3.set_ylabel('Frequency')
#     ax3.set_title('Value 50')

#     x_min = min(np.min(data1), np.min(data10), np.min(data50))
#     x_max = max(np.max(data1), np.max(data10), np.max(data50))
#     ax1.set_xlim(x_min, x_max)
#     ax2.set_xlim(x_min, x_max)
#     ax3.set_xlim(x_min, x_max)


#     #y_min = min(np.min(data1), np.min(y2), np.min(y3))
#     y_max = max(ax1.get_ylim()[1],ax2.get_ylim()[1],ax3.get_ylim()[1])
#     ax1.set_ylim(0, y_max)
#     ax2.set_ylim(0, y_max)
#     ax3.set_ylim(0, y_max)

#     # Generate a single histogram for all arrays on the right
#     #ax4.hist(all_act, bins='auto')
#     #ax4.set_xlabel('Action')
#     #ax4.set_ylabel('Frequency')
#     #ax4.set_title('Action')
    
#     width = 0.2  # Width of each bar
#     x = np.arange(len(unique_actions))

#     ax4.bar(x, freq_counts_1, width, color='red', label='1')
#     ax4.bar(x + width, freq_counts_10, width, color='blue', label='10')
#     ax4.bar(x + (2 * width), freq_counts_50, width, color='yellow', label='50')

# # Add labels and title
#     ax4.set_xlabel('Actions')
#     ax4.set_ylabel('Frequency')
#     ax4.set_title('Action Frequency')

# # Add x-axis tick labels
#     ax4.set_xticks(x + width, unique_actions, rotation='vertical')



#     #ax4.legend()

#     # Adjust the layout of subplots
#     plt.tight_layout()

#     # Save the plot as an image
#     plt.savefig(game+'.png', dpi=300)

#     # Close the figure to release resources
#     plt.close(fig)


    #break 


        

