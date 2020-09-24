print("My is ")
# import t

import os
import shutil
# os.mkdir('test')
# f= open("test/guru99.txt","w+")
# for i in range(10):
#      f.write("This is line %d\r\n" % (i+1))
# f.close() 


# os.rename('test', 'test_' + str(2))
# os.rename('fig_', 'fig_' + str(1))
# for i in range(0, 10):
# 	import t
# 	print(i)
name = input("please input which number")
name = 'Figs_' + str(name)
os.mkdir(name)

source = './'
dest1 = './' + name
# dest2 = '/path/to/intel_folder'

files = os.listdir(source)

for f in files:
    if (f.startswith("fig")):
        shutil.move(f, dest1)
    # elif (f.startswith("Intel") or f.startswith("intel")):
    #     shutil.move(f, dest2)



# shutil.move('fig_*', 'Figs_' + str(name))
