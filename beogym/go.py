from beogym import BeoGym


env = BeoGym({})
steps = 10000

env.render(mode='random', steps=steps)

