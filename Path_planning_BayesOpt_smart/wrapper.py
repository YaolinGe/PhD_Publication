print("hello world")
print("this function will conduct 10 times random realisations")
import os

for i in range(0, 10):
	print("This is {:03d} folder ".format(i))
	# execfile('PathOpt.py')
	os.system('python3 PathOpt.py')
	os.rename('fig__', 'fig_' + str(i))


# os.mkdir('Figs')



