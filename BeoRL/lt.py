import plyvel, shutil
import os
db=''
count=0
while db == '':
    try:
        if count==0:
            db = plyvel.DB('data/')
        else:
            if os.path.exists('data'+str(count)+'/'):
                db = plyvel.DB('data'+str(count)+'/')
            else:
                shutil.copytree('data/', 'data'+str(count)+'/')
                db=plyvel.DB('data'+str(count)+'/')
                print('yes')
    except IOError:
        print('1')
        count+=1