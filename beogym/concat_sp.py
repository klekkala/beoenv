import numpy as np
import gc
obs_d = np.load('/lab/tmpig14c/kiran/all_4stack_all/DemonAttack/5/10/observation',mmap_mode='r')
obs_s = np.load('/lab/tmpig14c/kiran/all_4stack_all/SpaceInvaders/5/10/observation',mmap_mode='r')

obs=np.concatenate([obs_s,obs_d], axis=0)

np.save('/lab/tmpig14c/kiran/all_4stack_all/SpaceDemo/5/10/observation',obs)
obs=[]
gc.collect
print('yye')

rew_d = np.load('/lab/tmpig14c/kiran/all_4stack_all/DemonAttack/5/10/reward',mmap_mode='r')
rew_s = np.load('/lab/tmpig14c/kiran/all_4stack_all/SpaceInvaders/5/10/reward',mmap_mode='r')

reward=np.concatenate([rew_s,rew_d], axis=0)

np.save('/lab/tmpig14c/kiran/all_4stack_all/SpaceDemo/5/10/reward',reward)

terminal_d = np.load('/lab/tmpig14c/kiran/all_4stack_all/DemonAttack/5/10/terminal',mmap_mode='r')
terminal_s = np.load('/lab/tmpig14c/kiran/all_4stack_all/SpaceInvaders/5/10/terminal',mmap_mode='r')

terminal=np.concatenate([terminal_s,terminal_d], axis=0)

np.save('/lab/tmpig14c/kiran/all_4stack_all/SpaceDemo/5/10/terminal',terminal)


action_d = np.load('/lab/tmpig14c/kiran/all_4stack_all/DemonAttack/5/10/action',mmap_mode='r')
action_s = np.load('/lab/tmpig14c/kiran/all_4stack_all/SpaceInvaders/5/10/action',mmap_mode='r')

action=np.concatenate([action_s,action_d], axis=0)

np.save('/lab/tmpig14c/kiran/all_4stack_all/SpaceDemo/5/10/action',action)