import numpy as np

rewards = np.load('reward')

rewards[rewards > 0] = 1.0

print(rewards)
