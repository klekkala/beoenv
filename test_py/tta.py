import numpy as np

from ray.rllib.policy.policy import Policy

# Use the `from_checkpoint` utility of the Policy class:
#my_restored_policy = Policy.from_checkpoint("/lab/kiran/ray_results/PPO_CusEnv_2023-05-22_16-16-263lnit9sp/checkpoint_000004/policies/default_policy/")

my_restored_policy = Policy.from_checkpoint("/lab/kiran/logs/rllib/beogym/10001684962796.1516526/checkpoint_000001")
#print(my_restored_policy.make_rl_module())

print('aa')

print(my_restored_policy.get_weights().keys())


