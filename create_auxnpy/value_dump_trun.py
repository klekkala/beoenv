import numpy as np
from IPython import embed
import os
import sys
import matplotlib.pyplot as plt
#tmp_list = ['1', '10', '50']
#pathname = '../all_1chan_all/'
tmp_list = ['50']
#pathname = '../random_bev_carla/'
pathname = sys.argv[1]
dir_list = [name for name in os.listdir(pathname) if os.path.isdir(os.path.join(pathname, name))]
# dir_list = [name for name in os.listdir(pathname)]
print(dir_list)
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
        reward_path = os.path.join(file_path,'reward')
        action_path = os.path.join(file_path,'action')
        terminal_path = os.path.join(file_path,'terminal')
        value_path = os.path.join(file_path, 'value_truncated.npy')
        rew = np.load(reward_path,allow_pickle=True)
        negative_indices = np.where(rew < 0)[0]
        if len(negative_indices) > 0:
            rew[rew < 0] = 0
            print(f"{game} has negative rewrd, fixed")
        act = np.load(action_path,allow_pickle=True)
        ter = np.load(terminal_path,allow_pickle=True)
        val = np.load(value_path, allow_pickle=True)
        ter[-1]=1
        indices = np.where(ter == 1)


        slices_a = []
        slices_r = []
        slices_v = []
        slice_epi = []
        slice_limit = [] 
        # Iterate through the indices and add slices to the lists
        start_idx = 0
        count = 0
        prev_idx = -1
        id_dict = {}
        for idx in indices[0]:
            slices_a.append(act[start_idx:idx+1])
            slices_r.append(rew[start_idx:idx+1])
            #slices_v+=val[start_idx:idx+1]
            slice_epi += [count]*(idx - (prev_idx+1) + 1)
            slice_limit += [idx]*(idx - (prev_idx+1) + 1)
            id_dict[count] = start_idx
            #print(prev_idx, idx, len(slice_limit))
            assert(len(slice_epi) == len(slice_limit) == idx+1)
            assert(ter[len(slice_epi)-1] == 1)
            assert(ter[slice_limit[-1]] == 1)
            prev_idx = idx

            start_idx = idx+1
            count += 1

        print(len(slice_epi))
        slice_epi += [count]*(rew.shape[0] - len(slice_epi))
        slice_limit += [(rew.shape[0]-1)]*(rew.shape[0] - len(slice_limit))
        assert(ter[len(slice_epi)-1] == 1)
        for abcd in range(rew.shape[0]):
            assert(ter[slice_limit[abcd]] == 1)
        assert(ter[slice_limit[-1]] == 1)
        
        np_epi = np.stack(slice_epi)
        np_limit = np.stack(slice_limit)
        
        #for arr in slices_r:
        #   a=0.95
        #   powers = np.arange(arr.size)
        #   output = [np.sum(arr[i:] * a ** powers[: arr.size - i]) for i in range(arr.size)]
        #   slices_v+=output
        # all_epi.append(np_epi)
        # all_limit.append(np_limit)
        # all_val.append(slices_v)
        # all_act.append(slices_a)
        # all_id.append(id_dict)
        print(len(slices_v))
        slices_v = val
        slices_v = [i for i in slices_v if i>0]
        print(len(slices_v))
        plt.hist(slices_v, bins=90, alpha=0.3, label=f'{game}')
        # ax.hist(slices_v, bins = 20, label=f'{directory}')
        # ax = axes[axidx] if len(dir_list) > 1 else axes
        # ax.hist(slices_v, bins=90)
        # ax.set_title(f'{directory}')
        # axidx+=1
        # ax1.hist(data1, bins=20)
    
plt.savefig('res.png', dpi=300)

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


        

