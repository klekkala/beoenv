from beogym.beogym import BeoGym


env = BeoGym({'city':'Wall_Street','data_path':'/home/tmp/kiran/'})
steps = 10000

env.render(mode='human', steps=steps)

