from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize,leastsq

def general_function(args,func,x,y,err=1):
	return (func(x,*args)-y)/err

def global_function(all_args,func,x,y,errs,nfree=0):
	''' nfree is number of independent fitting parameters for each data set '''
	all_args = list(all_args)
	nshared = len(all_args)-nfree*len(x)
	shared_args = all_args[:nshared]
	if nfree != 0:
		free_args = all_args[-nfree*len(x):]
	else:
		free_args = []
	total = []
	for i in range(len(x)):
		args = shared_args+free_args[i::len(x)]
		total = total + list((func(x[i],*args)-y[i])/errs[i])
	return np.array(total)

if __name__ == "__main__":
	def line(x,c,m):
		return m*x+c
	
	sigma = 0.01
	
	xlist = []
	ylist = []
	errlist = []
	for i in range(3):
		x = np.linspace(0,1,1e2)
		y = np.random.random()*x + np.random.normal(0,sigma,len(x))
		err = np.ones(len(x))*sigma
		xlist.append(x)
		ylist.append(y)
		errlist.append(err)
	
	print 'Independant fit errors:'
	for i in range(len(xlist)):
		res = leastsq(general_function,[0,0.5],args=(line,xlist[i],ylist[i],errlist[i]),full_output=1)
		output_params,cov = res[0],res[1]
		print cov.diagonal()**.5

	print 'Global fit errors:'
	res = leastsq(global_function,[0,0.5,0.5,0.5],args=(line,xlist,ylist,errlist,1),full_output=1)
	output_params,cov = res[0],res[1]
	print cov.diagonal()**.5
	
	cs = ['#AA2B4A','#006388','#7E317B'] # durham colours red,blue,purple
	for i in range(len(xlist)):
		plt.plot(xlist[i],ylist[i],ls='',marker='o',c=cs[1])
		plt.plot(xlist[i],line(xlist[i],output_params[0],output_params[1+i]),c=cs[0],lw=2)
	
	plt.xlim(0,1)
	plt.ylim(0,1)
	plt.xlabel('x')
	plt.ylabel('y')
	
	plt.show()
