import envs

a = {'env':'rlbench/open_drawer-vision-v0'}
b = envs.SingleColoEnv(a)
b.reset()
b.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])