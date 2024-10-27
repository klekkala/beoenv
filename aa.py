from beogym.beogym import BeoGym


NYC = []
Pits= ['CMU', 'Allegheny', 'South_Shore']
cities = NYC+Pits

for city in cities:
    env = BeoGym({'city':city,'data_path':'/home6/tmp/kiran/'})
    print(city, len(env.dh.Gdict.keys()))