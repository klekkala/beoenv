import matplotlib.pyplot as plt
import numpy as np

# Example data (you will replace it with your actual data)
methods = ['VEP(Ours)', 'R3M', 'VC-1', 'CLIP', 'MVP']
mean_reward_southshore = [4, 2, 2.5, 2, 2.5]
mean_reward_cmu = [9, 1.5, 1, 0.5, 1]

x = np.arange(len(methods))  # the label locations
width = 0.35  # the width of the bars

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Bar plot 1 (South Shore)
ax1.bar(x, mean_reward_southshore, color=['#FFA500', '#87CEFA', '#98FB98', '#FFC0CB', '#DDA0DD'])
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=15)
ax1.set_ylabel('Mean Reward', fontsize=19)
ax1.set_title('South Shore', fontsize=19)

# Bar plot 2 (CMU)
ax2.bar(x, mean_reward_cmu, color=['#FFA500', '#87CEFA', '#98FB98', '#FFC0CB', '#DDA0DD'])
ax2.set_xticks(x)
ax2.set_xticklabels(methods, fontsize=15)
ax2.set_ylabel('Mean Reward', fontsize=19)
ax2.set_title('CMU', fontsize=19)

# Customize font size of the y-axis tick labels
plt.setp(ax1.get_yticklabels(), fontsize=12)
plt.setp(ax2.get_yticklabels(), fontsize=12)

# Remove top and right spines (box lines)
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('plot.pdf', format='pdf')